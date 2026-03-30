"""
VLMAuditModel
-------------
Thin wrapper around a pre-trained VLM (default: BLIP) that:
  1. Loads the model and processor from HuggingFace.
  2. Registers forward hooks on every cross-attention layer so the
     Extraction Layer can pull raw attention weights without modifying
     model internals.
  3. Exposes `forward()` for a standard image+caption batch.

"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from core.config import AuditConfig


class VLMAuditModel:
    """
    Wraps a HuggingFace VLM and exposes cross-attention internals.

    Usage
    -----
    config = AuditConfig()
    model  = VLMAuditModel(config)
    output = model.forward(images, captions)
    maps   = model.get_attention_cache()   # consumed by ExtractionLayer
    """

    def __init__(self, config: AuditConfig) -> None:
        self.config = config
        self.device = self._resolve_device(config.device)
        self._attention_cache: Dict[int, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

        self.processor = None
        self.model: Optional[nn.Module] = None
        self._cross_attention_layers: List[nn.Module] = []

        self._load_model()
        self._register_hooks()

    # ------------------------------------------------------------------
    # Internal helpers (to be implemented)
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device_name: str) -> torch.device:
        if device_name == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA requested but not available; falling back to CPU.")
            return torch.device("cpu")
        return torch.device(device_name)

    def _load_model(self) -> None:
        """Load processor and model from HuggingFace hub."""
        from transformers import AutoProcessor, BlipForImageTextRetrieval

        self.processor = AutoProcessor.from_pretrained(self.config.model_name)
        self.model = BlipForImageTextRetrieval.from_pretrained(self.config.model_name)
        self.model.to(self.device)
        self.model.eval()

        text_encoder = getattr(self.model, "text_encoder", None)
        encoder = getattr(text_encoder, "encoder", None)
        layers = getattr(encoder, "layer", None)
        if layers is None:
            raise ValueError("Unable to locate BLIP text encoder layers for cross-attention hooks.")

        self._cross_attention_layers = [
            layer.crossattention.self
            for layer in layers
            if getattr(layer, "crossattention", None) is not None
            and getattr(layer.crossattention, "self", None) is not None
        ]
        if not self._cross_attention_layers:
            raise ValueError("No BLIP cross-attention layers were found in the loaded model.")

    def _register_hooks(self) -> None:
        """
        Attach a forward hook to each cross-attention layer so that
        attention weight tensors are stored in self._attention_cache
        keyed by layer index.
        """
        self.remove_hooks()

        target_layers = self.config.target_layers or list(range(len(self._cross_attention_layers)))
        for layer_idx in target_layers:
            if layer_idx < 0 or layer_idx >= len(self._cross_attention_layers):
                raise ValueError(
                    f"Target layer {layer_idx} is out of range for "
                    f"{len(self._cross_attention_layers)} cross-attention layers."
                )

            handle = self._cross_attention_layers[layer_idx].register_forward_hook(
                self._hook_fn(layer_idx)
            )
            self._hooks.append(handle)

    def _hook_fn(self, layer_idx: int):
        """Returns a hook function that caches attention weights for `layer_idx`."""
        def hook(module, input, output):
            attention_weights = self._extract_attention_tensor(output)
            if attention_weights is not None:
                self._attention_cache[layer_idx] = attention_weights.detach()
        return hook

    @staticmethod
    def _extract_attention_tensor(output: Any) -> Optional[torch.Tensor]:
        if isinstance(output, torch.Tensor) and output.ndim == 4:
            return output

        if isinstance(output, Mapping):
            for key in ("attentions", "attention_probs", "attention_weights"):
                value = output.get(key)
                if isinstance(value, torch.Tensor) and value.ndim == 4:
                    return value
                if isinstance(value, (tuple, list)):
                    for item in value:
                        if isinstance(item, torch.Tensor) and item.ndim == 4:
                            return item

        if isinstance(output, (tuple, list)):
            for item in output:
                if isinstance(item, torch.Tensor) and item.ndim == 4:
                    return item

        return None

    @staticmethod
    def _move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
        moved: Dict[str, Any] = {}
        for key, value in batch.items():
            moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
        return moved

    @staticmethod
    def _extract_logits(output: Any) -> torch.Tensor:
        for field in ("itm_score", "logits", "logits_per_image", "scores"):
            value = getattr(output, field, None)
            if isinstance(value, torch.Tensor):
                return value
            if isinstance(output, Mapping):
                mapped_value = output.get(field)
                if isinstance(mapped_value, torch.Tensor):
                    return mapped_value
        raise KeyError("Could not find a logits-like tensor in the model output.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,
        captions: List[str],
    ) -> Dict:
        """
        Run a batched forward pass.

        Parameters
        ----------
        images   : pre-processed image tensor  (B, C, H, W)
        captions : list of caption strings, length B

        Returns
        -------
        dict with at least {"logits": Tensor, "loss": Tensor | None}
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model has not been loaded.")

        if torch.is_tensor(images):
            model_inputs = self.processor(
                text=captions,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            model_inputs["pixel_values"] = images
        else:
            model_inputs = self.processor(
                images=images,
                text=captions,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

        model_inputs = self._move_batch_to_device(model_inputs, self.device)
        output = self.model(
            **model_inputs,
            use_itm_head=True,
            output_attentions=True,
            return_dict=True,
        )

        loss = getattr(output, "loss", None)
        if loss is None and isinstance(output, Mapping):
            loss = output.get("loss")

        return {
            "logits": self._extract_logits(output),
            "loss": loss,
            "raw_output": output,
        }

    def get_attention_cache(self) -> Dict[int, torch.Tensor]:
        """
        Return cached attention weights from the most recent forward pass.

        Returns
        -------
        Dict mapping layer_index -> attention tensor
        Shape per tensor: (B, num_heads, seq_len_text, num_patches)
        """
        return dict(self._attention_cache)

    def clear_cache(self) -> None:
        """Clear attention cache between samples."""
        self._attention_cache.clear()

    def remove_hooks(self) -> None:
        """Clean up all registered forward hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @property
    def num_layers(self) -> int:
        """Number of cross-attention layers available for extraction."""
        return len(self._cross_attention_layers)

    @property
    def patch_grid(self) -> Tuple[int, int]:
        """
        (rows, cols) of the image patch grid.
        Derived from the vision encoder's patch size and image resolution.
        """
        if self.model is None:
            raise RuntimeError("Model has not been loaded.")

        vision_config = getattr(self.model.config, "vision_config", None)
        image_h, image_w = self.image_size
        patch_size = getattr(vision_config, "patch_size", None)
        if not isinstance(patch_size, int):
            raise ValueError("Could not determine vision patch size from the loaded model.")

        return (image_h // patch_size, image_w // patch_size)

    @property
    def image_size(self) -> Tuple[int, int]:
        if self.model is None:
            raise RuntimeError("Model has not been loaded.")

        vision_config = getattr(self.model.config, "vision_config", None)
        image_size = getattr(vision_config, "image_size", None)

        if isinstance(image_size, int):
            return (image_size, image_size)
        if isinstance(image_size, (tuple, list)) and len(image_size) == 2:
            return (int(image_size[0]), int(image_size[1]))
        if isinstance(image_size, dict):
            if "height" in image_size and "width" in image_size:
                return (int(image_size["height"]), int(image_size["width"]))
            if "shortest_edge" in image_size:
                edge = int(image_size["shortest_edge"])
                return (edge, edge)

        raise ValueError("Could not determine model image size from the loaded model.")

"""
VLMAuditModel
-------------
Thin wrapper around a pre-trained VLM (default: BLIP) that:
  1. Loads the model and processor from HuggingFace.
  2. Registers forward hooks on every cross-attention layer so the
     Extraction Layer can pull raw attention weights without modifying
     model internals.
  3. Exposes `forward()` for a standard image+caption batch.

Implementation notes
--------------------
BLIP text encoder layers each have a `crossattention.self` submodule
(BlipTextSelfAttention).  When called with output_attentions=True it
returns (context_layer, attn_probs) where attn_probs has shape
(B, num_heads, T_text, N_visual).  N_visual = num_patches + 1 because
the ViT vision encoder prepends a CLS token; we strip that token (index 0)
before caching so downstream code always sees (B, heads, T, N_patches).

patch_grid is derived from the vision encoder's config:
  n = image_size // patch_size  →  grid = (n, n)
For blip-itm-base-coco: image_size=384, patch_size=16 → (24, 24).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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
        self._attention_cache: Dict[int, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

        self.processor = None
        self.model: Optional[nn.Module] = None
        self._cross_attention_layers: List[nn.Module] = []

        self._load_model()
        self._register_hooks()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load processor and model from HuggingFace hub."""
        from transformers import BlipForImageTextRetrieval, BlipProcessor

        self.processor = BlipProcessor.from_pretrained(self.config.model_name)
        self.model = BlipForImageTextRetrieval.from_pretrained(self.config.model_name)
        self.model.to(self.config.device)
        self.model.eval()

        # BlipTextSelfAttention modules — one per text encoder layer.
        # These are the modules whose forward output contains attention weights.
        self._cross_attention_layers = [
            layer.crossattention.self
            for layer in self.model.text_encoder.encoder.layer
        ]

    def _register_hooks(self) -> None:
        """
        Attach a forward hook to each cross-attention layer so that
        attention weight tensors are stored in self._attention_cache
        keyed by layer index.
        """
        for idx, layer in enumerate(self._cross_attention_layers):
            h = layer.register_forward_hook(self._hook_fn(idx))
            self._hooks.append(h)

    def _hook_fn(self, layer_idx: int):
        """Returns a hook that caches detached attention weights for `layer_idx`."""
        def hook(module, input, output):
            # output is (context_layer, attn_probs) when output_attentions=True
            if isinstance(output, tuple) and len(output) > 1:
                attn = output[1]  # (B, heads, T_text, N_visual)
                # Drop CLS visual token so shape is (B, heads, T_text, N_patches)
                self._attention_cache[layer_idx] = attn[:, :, :, 1:].detach()
        return hook

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
        dict with at least {"logits": Tensor (B, 2), "loss": None}
        logits are ITM scores; take softmax[:, 1] for match probability.
        """
        text_inputs = self.processor.tokenizer(
            captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.config.device)

        images = images.to(self.config.device)

        outputs = self.model(
            pixel_values=images,
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            output_attentions=True,  # needed so hooks receive attn_probs
            return_dict=True,
        )

        return {"logits": outputs.itm_score, "loss": None}

    def get_attention_cache(self) -> Dict[int, torch.Tensor]:
        """
        Return cached attention weights from the most recent forward pass.

        Returns
        -------
        Dict mapping layer_index -> Tensor (B, num_heads, T_text, N_patches)
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
        (rows, cols) of the image patch grid, excluding the CLS token.
        For blip-itm-base-coco: image_size=384, patch_size=16 → (24, 24).
        """
        cfg = self.model.vision_model.config
        n = cfg.image_size // cfg.patch_size
        return (n, n)

    @property
    def image_size(self) -> Tuple[int, int]:
        """(H, W) of images expected by this model's processor."""
        s = self.model.vision_model.config.image_size
        return (s, s)

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

        self.processor = None   # TODO: load HuggingFace processor
        self.model: Optional[nn.Module] = None  # TODO: load HuggingFace model
        self._cross_attention_layers: List[nn.Module] = []  # TODO: populate after model load

        # self._load_model()
        # self._register_hooks()

    # ------------------------------------------------------------------
    # Internal helpers (to be implemented)
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load processor and model from HuggingFace hub."""
        # TODO: use transformers.AutoProcessor / BlipForImageTextRetrieval
        # Move model to self.config.device
        raise NotImplementedError

    def _register_hooks(self) -> None:
        """
        Attach a forward hook to each cross-attention layer so that
        attention weight tensors are stored in self._attention_cache
        keyed by layer index.
        """
        # TODO: iterate self._cross_attention_layers, attach hooks
        raise NotImplementedError

    def _hook_fn(self, layer_idx: int):
        """Returns a hook function that caches attention weights for `layer_idx`."""
        def hook(module, input, output):
            # TODO: extract attention weights from output tuple / dict
            # self._attention_cache[layer_idx] = attention_weights
            pass
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
        dict with at least {"logits": Tensor, "loss": Tensor | None}
        """
        # TODO: tokenize captions with self.processor
        # TODO: forward through self.model
        # Hooks populate self._attention_cache automatically
        raise NotImplementedError

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
        # TODO: return (H // patch_size, W // patch_size)
        raise NotImplementedError

"""
GradCAMExtractor
----------------
Computes Grad-CAM heatmaps over cross-attention layers of the VLM.

Grad-CAM uses the gradient of a scalar target (e.g. ITM logit for the
correct caption) with respect to each cross-attention feature map to
produce a class-discriminative localisation map.

Algorithm per layer:
  1. Retain the layer's activation A  (B, heads, T_text, N_patches)
     and its gradient dScore/dA.
  2. Compute importance weights α = GlobalAvgPool(dScore/dA)  → (B, heads)
  3. Weighted combination: L = ReLU(Σ_k α_k * A_k)  → (B, N_patches)
  4. Reshape + upsample to image size → (B, H_img, W_img)

"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from core.config import AuditConfig
from core.model import VLMAuditModel


class GradCAMExtractor:
    """
    Computes Grad-CAM heatmaps for selected cross-attention layers.

    Parameters
    ----------
    model      : VLMAuditModel  — needs access to the underlying nn.Module
                 for gradient hooks
    config     : AuditConfig
    image_size : (H, W) of input images in pixels

    Usage
    -----
    extractor = GradCAMExtractor(model, config, image_size=(384, 384))
    heatmaps  = extractor.compute(images, captions)
    # heatmaps: Dict[layer_idx -> Tensor (B, H, W)]
    """

    def __init__(
        self,
        model: VLMAuditModel,
        config: AuditConfig,
        image_size: Tuple[int, int],
    ) -> None:
        self.model = model
        self.config = config
        self.image_size = image_size

        self._activation_cache: Dict[int, torch.Tensor] = {}
        self._gradient_cache: Dict[int, torch.Tensor] = {}
        self._grad_hooks: List[torch.utils.hooks.RemovableHook] = []

    def compute(
        self,
        images: torch.Tensor,
        captions: List[str],
    ) -> Dict[int, torch.Tensor]:
        """
        Full Grad-CAM pipeline for one batch.

        Parameters
        ----------
        images   : (B, C, H, W) pre-processed image tensor
        captions : list of caption strings, length B

        Returns
        -------
        Dict[layer_idx -> Tensor (B, H_img, W_img)]  values in [0, 1]
        """
        self._register_grad_hooks()
        try:
            output = self.model.forward(images, captions)
            target_score = self._select_target_score(output)
            target_score.backward()

            target = self.config.target_layers or list(self._activation_cache.keys())
            heatmaps = {l: self._compute_layer(l) for l in target
                        if l in self._activation_cache}
        finally:
            # Always clean up even if an exception is raised
            self._remove_grad_hooks()
            self._clear_caches()

        return heatmaps

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _register_grad_hooks(self) -> None:
        """
        Register forward hooks (to cache activations) and backward hooks
        (to cache gradients) on each target cross-attention layer.
        """
        target = set(self.config.target_layers) if self.config.target_layers else None

        for idx, layer in enumerate(self.model._cross_attention_layers):
            if target is not None and idx not in target:
                continue
            h1 = layer.register_forward_hook(self._forward_hook(idx))
            h2 = layer.register_full_backward_hook(self._backward_hook(idx))
            self._grad_hooks.extend([h1, h2])

    def _forward_hook(self, layer_idx: int):
        """Caches the forward activation for `layer_idx`."""
        def hook(module, input, output):
            self._activation_cache[layer_idx] = output
        return hook

    def _backward_hook(self, layer_idx: int):
        """Caches the gradient w.r.t. activation for `layer_idx`."""
        def hook(module, grad_input, grad_output):
            self._gradient_cache[layer_idx] = grad_output[0]
        return hook

    def _remove_grad_hooks(self) -> None:
        for h in self._grad_hooks:
            h.remove()
        self._grad_hooks.clear()

    def _clear_caches(self) -> None:
        self._activation_cache.clear()
        self._gradient_cache.clear()

    # ------------------------------------------------------------------
    # Grad-CAM computation helpers
    # ------------------------------------------------------------------

    def _select_target_score(self, model_output: Dict) -> torch.Tensor:
        """
        Extract the scalar score to differentiate.
        Default: mean of ITM logits over the batch.

        Returns scalar Tensor with grad_fn attached.
        """
        return model_output["logits"].mean()

    def _compute_layer(self, layer_idx: int) -> torch.Tensor:
        """
        Apply Grad-CAM formula for one layer.

        Returns Tensor (B, H_img, W_img) in [0, 1].
        """
        A  = self._activation_cache[layer_idx]   # (B, heads, T, N)
        dA = self._gradient_cache[layer_idx]     # (B, heads, T, N)

        # Importance weight per head — average gradient over all token/patch positions
        alpha = dA.mean(dim=(-2, -1))                        # (B, heads)

        # Weighted sum of activations across heads
        L = (alpha[:, :, None, None] * A).sum(dim=1)        # (B, T, N)

        # Collapse text-token dimension
        L = L.mean(dim=1)                                    # (B, N)

        # Keep only positive contributions (patches that help the score)
        if self.config.gradcam_relu:
            L = L.clamp(min=0)

        # Reshape flat patches → spatial grid → full image resolution → [0, 1]
        B, N      = L.shape
        patch_grid = self.model.patch_grid                   # (ph, pw)
        L = L.view(B, *patch_grid)                           # (B, ph, pw)
        L = L.unsqueeze(1)                                   # (B, 1, ph, pw)
        L = torch.nn.functional.interpolate(
            L, size=self.image_size, mode="bilinear", align_corners=False
        )
        L = L.squeeze(1)                                     # (B, H, W)

        # Per-sample min-max normalisation
        flat  = L.view(B, -1)
        min_  = flat.min(dim=1).values.view(B, 1, 1)
        max_  = flat.max(dim=1).values.view(B, 1, 1)
        return (L - min_) / (max_ - min_ + 1e-8)

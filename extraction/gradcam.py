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
        # TODO:
        # 1. self._register_grad_hooks()
        # 2. output = self.model.forward(images, captions)
        # 3. target_score = _select_target_score(output)
        # 4. target_score.backward()
        # 5. heatmaps = {l: self._compute_layer(l) for l in target_layers}
        # 6. self._remove_grad_hooks(); self._clear_caches()
        # 7. return heatmaps
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _register_grad_hooks(self) -> None:
        """
        Register forward hooks (to cache activations) and backward hooks
        (to cache gradients) on each target cross-attention layer.
        """
        # TODO: iterate self.model._cross_attention_layers
        raise NotImplementedError

    def _forward_hook(self, layer_idx: int):
        """Caches the forward activation for `layer_idx`."""
        def hook(module, input, output):
            # TODO: store output (attention weights) in self._activation_cache
            pass
        return hook

    def _backward_hook(self, layer_idx: int):
        """Caches the gradient w.r.t. activation for `layer_idx`."""
        def hook(module, grad_input, grad_output):
            # TODO: store grad_output[0] in self._gradient_cache
            pass
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
        # TODO: return model_output["logits"].mean()
        raise NotImplementedError

    def _compute_layer(self, layer_idx: int) -> torch.Tensor:
        """
        Apply Grad-CAM formula for one layer.

        Returns Tensor (B, H_img, W_img) in [0, 1].
        """
        # A  = self._activation_cache[layer_idx]   # (B, heads, T, N)
        # dA = self._gradient_cache[layer_idx]      # (B, heads, T, N)
        # TODO:
        #   alpha  = dA.mean(dim=(-2, -1))           # (B, heads)
        #   L      = (alpha[:, :, None, None] * A).sum(dim=1)  # (B, T, N)
        #   L      = L.mean(dim=1)                   # (B, N)  avg over tokens
        #   if self.config.gradcam_relu: L = L.clamp(min=0)
        #   reshape → upsample → normalise
        raise NotImplementedError

"""
AttentionExtractor
------------------
Converts raw cross-attention weight tensors (stored in VLMAuditModel's cache)
into normalised 2-D spatial heatmaps aligned with the original image dimensions.

Pipeline per layer:
  raw weights  (B, H, T_text, N_patches)
    → head fusion  (mean/max over H)    → (B, T_text, N_patches)
    → token fusion (mean over T_text)   → (B, N_patches)
    → reshape to patch grid             → (B, ph, pw)
    → bilinear upsample to image size   → (B, img_H, img_W)
    → min-max normalise to [0, 1]

"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from core.config import AuditConfig


class AttentionExtractor:
    """
    Converts cached attention weights into image-space heatmaps.

    Parameters
    ----------
    config     : AuditConfig  — controls head_fusion strategy, target_layers
    patch_grid : (rows, cols) from VLMAuditModel.patch_grid
    image_size : (H, W) of the input images in pixels

    Usage
    -----
    extractor = AttentionExtractor(config, patch_grid, image_size)
    heatmaps  = extractor.extract(model.get_attention_cache())
    # heatmaps: Dict[layer_idx -> Tensor (B, H, W)]
    """

    def __init__(
        self,
        config: AuditConfig,
        patch_grid: Tuple[int, int],
        image_size: Tuple[int, int],
    ) -> None:
        self.config = config
        self.patch_grid = patch_grid    # (ph, pw)
        self.image_size = image_size    # (H, W)

    def extract(
        self,
        attention_cache: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        """
        Parameters
        ----------
        attention_cache : output of VLMAuditModel.get_attention_cache()
                          {layer_idx: Tensor (B, num_heads, T_text, N_patches)}

        Returns
        -------
        Dict[layer_idx -> Tensor (B, H_img, W_img)]  values in [0, 1]
        """
        target = self.config.target_layers or list(attention_cache.keys())
        result: Dict[int, torch.Tensor] = {}

        for layer_idx in target:
            if layer_idx not in attention_cache:
                continue
            weights = attention_cache[layer_idx]    # (B, heads, T, N)
            heatmap = self._process_layer(weights)  # (B, H, W)
            result[layer_idx] = heatmap

        return result

    # ------------------------------------------------------------------
    # Internal steps (implement each independently)
    # ------------------------------------------------------------------

    def _fuse_heads(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Reduce the head dimension.
        weights : (B, heads, T, N)  →  returns (B, T, N)

        Strategy from self.config.attention_head_fusion:
          "mean" | "max" | "min"
        """
        if self.config.attention_head_fusion == "mean":
            return torch.mean(weights, dim=1)
        elif self.config.attention_head_fusion == "max":
            return torch.max(weights, dim=1).values
        elif self.config.attention_head_fusion == "min":
            return torch.min(weights, dim=1).values
        else:
            raise ValueError(f"Unknown head fusion: {self.config.attention_head_fusion}")

    def _fuse_tokens(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Reduce text-token dimension by averaging.
        weights : (B, T, N)  →  returns (B, N)
        """
        return torch.mean(weights, dim=1)

    def _reshape_to_grid(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Reshape flat patch sequence to spatial grid.
        weights : (B, N)  →  returns (B, ph, pw)
        """
        # BLIP's ViT prepends a [CLS] token at position 0 — drop it before reshaping
        # to leave only the 576 spatial patch tokens (24×24)
        B, N = weights.shape
        weights = weights[:, 1:]
        return weights.view(B, *self.patch_grid)

    def _upsample(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Bilinear upsample patch grid to full image resolution.
        grid : (B, ph, pw)  →  returns (B, H_img, W_img)
        """
        grid = grid.unsqueeze(dim=1)
        up_sample = F.interpolate(grid, self.image_size, mode="bilinear", align_corners=False)
        return up_sample.squeeze(dim=1)

    def _normalise(self, heatmap: torch.Tensor) -> torch.Tensor:
        """
        Per-sample min-max normalisation → values in [0, 1].
        heatmap : (B, H, W)  →  returns (B, H, W)
        """
        B = heatmap.shape[0]
        flat = heatmap.view(B, -1)                 # (B, H*W)
        min_ = flat.min(dim=1).values.view(B, 1, 1)
        max_ = flat.max(dim=1).values.view(B, 1, 1)
        return (heatmap - min_) / (max_ - min_ + 1e-8)   # +1e-8 avoids div by zero

    def _process_layer(self, weights: torch.Tensor) -> torch.Tensor:
        """Compose the full per-layer pipeline."""
        weights = self._fuse_heads(weights)
        weights = self._fuse_tokens(weights)
        grid    = self._reshape_to_grid(weights)
        heatmap = self._upsample(grid)
        heatmap = self._normalise(heatmap)
        return heatmap

"""
Hybrid heatmap extraction
-------------------------
Blend attention and Grad-CAM heatmaps into a single map:

    hybrid = alpha * attention + (1 - alpha) * gradcam

where `alpha` is the attention weight. The blended map is normalised
per sample back to [0, 1] so it remains comparable to the individual
extractors for thresholding and masking-based evaluation.
"""

from __future__ import annotations

from typing import Dict

import torch


class HybridExtractor:
    """Blend already-upsampled attention and Grad-CAM heatmaps."""

    def blend(
        self,
        attention_heatmaps: Dict[int, torch.Tensor],
        gradcam_heatmaps: Dict[int, torch.Tensor],
        alpha: float,
    ) -> Dict[int, torch.Tensor]:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Hybrid alpha must be in [0, 1], got {alpha}.")

        blended: Dict[int, torch.Tensor] = {}
        for layer_idx in sorted(set(attention_heatmaps) & set(gradcam_heatmaps)):
            heatmap = alpha * attention_heatmaps[layer_idx] + (
                1.0 - alpha
            ) * gradcam_heatmaps[layer_idx]
            blended[layer_idx] = self._normalise(heatmap)
        return blended

    @staticmethod
    def _normalise(heatmap: torch.Tensor) -> torch.Tensor:
        if heatmap.ndim != 3:
            raise ValueError(
                f"Expected batched heatmap with shape (B, H, W), got {tuple(heatmap.shape)}."
            )

        batch_size = heatmap.shape[0]
        flat = heatmap.view(batch_size, -1)
        min_vals = flat.min(dim=1).values.view(batch_size, 1, 1)
        max_vals = flat.max(dim=1).values.view(batch_size, 1, 1)
        return (heatmap - min_vals) / (max_vals - min_vals + 1e-8)

"""
Extraction Layer
================
Owns: post-processing raw attention weights into spatial heatmaps,
      and computing Grad-CAM over cross-attention layers.
Consumes (from Core Module):
  - VLMAuditModel.get_attention_cache() -> Dict[int, Tensor]

Produces (for Evaluation Suite):
  - attention heatmaps : Dict[int, Tensor]  shape (B, H_img, W_img)
  - gradcam heatmaps   : Dict[int, Tensor]  shape (B, H_img, W_img)

Public interface:
  - AttentionExtractor
  - GradCAMExtractor
"""

from extraction.attention import AttentionExtractor
from extraction.gradcam import GradCAMExtractor

__all__ = ["AttentionExtractor", "GradCAMExtractor"]

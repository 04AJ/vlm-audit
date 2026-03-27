"""
Evaluation Suite
================
Owns: quantitative scoring of heatmaps against ground-truth annotations.
Sub-modules
-----------
grounding    : Pointing Game Accuracy + IoU vs Flickr30k bounding boxes
faithfulness : Sensitivity-n and SaCo via iterative pixel masking

Consumes (from Extraction Layer):
  heatmaps : Dict[layer_idx -> Tensor (B, H_img, W_img)]
  boxes    : bounding-box annotations from Flickr30kDataset

Produces:
  EvalResults dataclass — aggregated scores per layer + overall

Public interface:
  - GroundingEvaluator
  - FaithfulnessEvaluator
  - EvalResults
"""

from evaluation.grounding import GroundingEvaluator
from evaluation.faithfulness import FaithfulnessEvaluator
from evaluation.results import EvalResults

__all__ = ["GroundingEvaluator", "FaithfulnessEvaluator", "EvalResults"]

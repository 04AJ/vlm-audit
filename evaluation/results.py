"""
EvalResults
-----------
Dataclass for aggregated evaluation output consumed by the run script
and any downstream reporting/visualisation.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class LayerGroundingResult:
    layer_idx: int
    pointing_game_accuracy: float       # fraction of samples where peak pixel falls inside a GT box
    mean_iou: float                     # mean IoU between thresholded heatmap and GT boxes


@dataclass
class LayerFaithfulnessResult:
    layer_idx: int
    sensitivity_n_score: float          # avg confidence drop when top-n pixels are masked
    saco_auc: float                     # area under the SaCo confidence-drop curve


@dataclass
class EvalResults:
    """
    Top-level container returned after a full evaluation run.

    Attributes
    ----------
    grounding       : per-layer grounding scores
    faithfulness    : per-layer faithfulness scores
    config_snapshot : dict representation of the AuditConfig used
    """
    grounding: List[LayerGroundingResult] = field(default_factory=list)
    faithfulness: List[LayerFaithfulnessResult] = field(default_factory=list)
    config_snapshot: Dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience accessors (implement after grounding/faithfulness done)
    # ------------------------------------------------------------------

    def best_grounding_layer(self) -> Optional[LayerGroundingResult]:
        """Return the layer with highest mean IoU, or None if empty."""
        if not self.grounding:
            return None
        return max(self.grounding, key=lambda r: r.mean_iou)

    def best_faithfulness_layer(self) -> Optional[LayerFaithfulnessResult]:
        """Return the layer with highest SaCo AUC, or None if empty."""
        if not self.faithfulness:
            return None
        return max(self.faithfulness, key=lambda r: r.saco_auc)

    def summary(self) -> str:
        """Human-readable one-line summary for logging."""
        g = self.best_grounding_layer()
        f = self.best_faithfulness_layer()
        g_str = f"layer={g.layer_idx} mIoU={g.mean_iou:.3f}" if g else "N/A"
        f_str = f"layer={f.layer_idx} SaCo={f.saco_auc:.3f}" if f else "N/A"
        return f"Best grounding: [{g_str}]  |  Best faithfulness: [{f_str}]"

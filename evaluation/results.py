"""
EvalResults
-----------
Dataclass for aggregated evaluation output consumed by the run script
and any downstream reporting/visualisation.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


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
class HybridResult:
    alpha: float
    grounding: List[LayerGroundingResult] = field(default_factory=list)
    faithfulness: List[LayerFaithfulnessResult] = field(default_factory=list)


@dataclass
class EvalResults:
    """
    Top-level container returned after a full evaluation run.

    Attributes
    ----------
    grounding           : per-layer grounding scores (attention heatmaps)
    faithfulness        : per-layer faithfulness scores (attention heatmaps)
    grounding_grad      : per-layer grounding scores (Grad-CAM heatmaps)
    faithfulness_grad   : per-layer faithfulness scores (Grad-CAM heatmaps)
    hybrid              : per-alpha grounding/faithfulness scores for blended maps
    config_snapshot     : dict representation of the AuditConfig used
    """
    grounding: List[LayerGroundingResult] = field(default_factory=list)
    faithfulness: List[LayerFaithfulnessResult] = field(default_factory=list)
    grounding_grad: List[LayerGroundingResult] = field(default_factory=list)
    faithfulness_grad: List[LayerFaithfulnessResult] = field(default_factory=list)
    hybrid: List[HybridResult] = field(default_factory=list)
    config_snapshot: Dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def best_grounding_layer(self) -> Optional[LayerGroundingResult]:
        """Return the attention-heatmap layer with highest mean IoU, or None if empty."""
        if not self.grounding:
            return None
        return max(self.grounding, key=lambda r: r.mean_iou)

    def best_faithfulness_layer(self) -> Optional[LayerFaithfulnessResult]:
        """Return the attention-heatmap layer with highest SaCo AUC, or None if empty."""
        if not self.faithfulness:
            return None
        return max(self.faithfulness, key=lambda r: r.saco_auc)

    def best_grounding_layer_grad(self) -> Optional[LayerGroundingResult]:
        """Return the Grad-CAM layer with highest mean IoU, or None if empty."""
        if not self.grounding_grad:
            return None
        return max(self.grounding_grad, key=lambda r: r.mean_iou)

    def best_faithfulness_layer_grad(self) -> Optional[LayerFaithfulnessResult]:
        """Return the Grad-CAM layer with highest SaCo AUC, or None if empty."""
        if not self.faithfulness_grad:
            return None
        return max(self.faithfulness_grad, key=lambda r: r.saco_auc)

    def best_hybrid_grounding(self) -> Optional[Tuple[float, LayerGroundingResult]]:
        """Return (alpha, result) for the best hybrid map by mean IoU, or None if empty."""
        best: Optional[Tuple[float, LayerGroundingResult]] = None
        for hybrid_result in self.hybrid:
            if not hybrid_result.grounding:
                continue
            candidate = max(hybrid_result.grounding, key=lambda r: r.mean_iou)
            if best is None or candidate.mean_iou > best[1].mean_iou:
                best = (hybrid_result.alpha, candidate)
        return best

    def best_hybrid_faithfulness(self) -> Optional[Tuple[float, LayerFaithfulnessResult]]:
        """Return (alpha, result) for the best hybrid map by SaCo AUC, or None if empty."""
        best: Optional[Tuple[float, LayerFaithfulnessResult]] = None
        for hybrid_result in self.hybrid:
            if not hybrid_result.faithfulness:
                continue
            candidate = max(hybrid_result.faithfulness, key=lambda r: r.saco_auc)
            if best is None or candidate.saco_auc > best[1].saco_auc:
                best = (hybrid_result.alpha, candidate)
        return best

    def summary(self) -> str:
        """Human-readable summary comparing attention and Grad-CAM per metric."""
        g_attn = self.best_grounding_layer()
        f_attn = self.best_faithfulness_layer()
        g_grad = self.best_grounding_layer_grad()
        f_grad = self.best_faithfulness_layer_grad()
        g_hybrid = self.best_hybrid_grounding()
        f_hybrid = self.best_hybrid_faithfulness()

        def _g(r): return f"layer={r.layer_idx} mIoU={r.mean_iou:.3f}" if r else "N/A"
        def _f(r): return f"layer={r.layer_idx} SaCo={r.saco_auc:.3f}" if r else "N/A"
        def _gh(item):
            return f"alpha={item[0]:.2f} layer={item[1].layer_idx} mIoU={item[1].mean_iou:.3f}" if item else "N/A"
        def _fh(item):
            return f"alpha={item[0]:.2f} layer={item[1].layer_idx} SaCo={item[1].saco_auc:.3f}" if item else "N/A"

        return (
            f"Best grounding  — attn: [{_g(g_attn)}]  grad: [{_g(g_grad)}]\n"
            f"Best faithfulness — attn: [{_f(f_attn)}]  grad: [{_f(f_grad)}]\n"
            f"Best hybrid     — grounding: [{_gh(g_hybrid)}]  faithfulness: [{_fh(f_hybrid)}]"
        )

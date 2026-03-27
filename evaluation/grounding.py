"""
GroundingEvaluator
------------------
Measures spatial alignment between heatmaps and Flickr30k bounding-box
annotations using two metrics:

1. Pointing Game Accuracy (PGA)
   The peak-activation pixel is "correct" if it falls inside any ground-truth
   bounding box for the queried phrase.

2. Intersection-over-Union (IoU)
   Binarise the heatmap at a threshold (default 0.5) and compute IoU with
   the union of GT boxes.

Both metrics are computed per layer and averaged over the dataset.

"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from core.config import AuditConfig
from evaluation.results import LayerGroundingResult


class GroundingEvaluator:
    """
    Evaluate heatmap spatial grounding against GT bounding boxes.

    Parameters
    ----------
    config : AuditConfig — supplies iou_threshold

    Usage
    -----
    evaluator = GroundingEvaluator(config)

    for batch in dataloader:
        heatmaps = extractor.extract(...)       # Dict[layer -> (B, H, W)]
        boxes    = batch["boxes"]               # List[List[Dict]]
        evaluator.update(heatmaps, boxes, image_sizes=batch["image_sizes"])

    results = evaluator.compute()   # List[LayerGroundingResult]
    """

    def __init__(self, config: AuditConfig) -> None:
        self.config = config
        # Accumulators: layer_idx -> {"correct": int, "iou_sum": float, "n": int}
        self._state: Dict[int, Dict] = {}

    def reset(self) -> None:
        """Clear all accumulated statistics."""
        self._state.clear()

    def update(
        self,
        heatmaps: Dict[int, torch.Tensor],
        boxes: List[List[Dict]],
        image_sizes: List[Tuple[int, int]],
    ) -> None:
        """
        Accumulate statistics for one batch.

        Parameters
        ----------
        heatmaps    : Dict[layer_idx -> Tensor (B, H, W)] from ExtractionLayer
        boxes       : outer list = batch, inner list = per-image entity dicts
                      each dict has keys "phrase" and "box" [x1, y1, x2, y2]
                      in pixel coordinates
        image_sizes : list of (H, W) for each sample in the batch
        """
        for layer_idx, hmap in heatmaps.items():
            if layer_idx not in self._state:
                self._state[layer_idx] = {"correct": 0, "iou_sum": 0.0, "n": 0}

            # TODO: iterate over batch dimension
            # for i in range(hmap.shape[0]):
            #     correct, iou = self._score_sample(hmap[i], boxes[i], image_sizes[i])
            #     self._state[layer_idx]["correct"] += correct
            #     self._state[layer_idx]["iou_sum"] += iou
            #     self._state[layer_idx]["n"]       += 1
            raise NotImplementedError

    def compute(self) -> List[LayerGroundingResult]:
        """
        Finalise metrics over all accumulated batches.

        Returns
        -------
        List[LayerGroundingResult] sorted by layer_idx
        """
        results = []
        for layer_idx, acc in sorted(self._state.items()):
            n = acc["n"]
            if n == 0:
                continue
            results.append(LayerGroundingResult(
                layer_idx=layer_idx,
                pointing_game_accuracy=acc["correct"] / n,
                mean_iou=acc["iou_sum"] / n,
            ))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_sample(
        self,
        heatmap: torch.Tensor,
        boxes: List[Dict],
        image_size: Tuple[int, int],
    ) -> Tuple[int, float]:
        """
        Score a single heatmap against its GT boxes.

        Returns
        -------
        (pointing_game_correct: int, iou: float)
        """
        # TODO: find peak pixel, check containment (PGA)
        # TODO: binarise at self.config.iou_threshold, compute IoU
        raise NotImplementedError

    @staticmethod
    def _box_mask(
        box: List[float],
        h: int,
        w: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create a binary (H, W) mask that is 1 inside `box`.

        box : [x1, y1, x2, y2] in pixel coordinates
        """
        # TODO: implement
        raise NotImplementedError

    @staticmethod
    def _iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
        """
        Compute IoU between two binary masks of the same shape.
        Handles the zero-union edge case by returning 0.0.
        """
        # TODO: implement
        raise NotImplementedError

"""
FaithfulnessEvaluator
---------------------
Verifies that heatmap-highlighted regions are causally important to the
model's prediction via systematic masking analysis.

Two metrics
-----------
1. Sensitivity-n  (Sen-n)
   Mask the top-n% most-important pixels (by heatmap intensity) and measure
   the average drop in the model's confidence score.  A faithful explanation
   produces a large, immediate drop.

2. SaCo (Salience-guided Confidence)
   Iteratively mask pixels from highest to lowest salience and record the
   model's confidence after each step.  The area under this decay curve (AUC)
   reflects how well the ordering matches causal importance.

Both metrics are computed per layer and averaged over the dataset.

"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from core.config import AuditConfig
from core.model import VLMAuditModel
from evaluation.results import LayerFaithfulnessResult


class FaithfulnessEvaluator:
    """
    Evaluate heatmap faithfulness via iterative masking.

    Parameters
    ----------
    model  : VLMAuditModel — needed to re-score masked images
    config : AuditConfig   — supplies sensitivity_n, saco_steps

    Usage
    -----
    evaluator = FaithfulnessEvaluator(model, config)

    for batch in dataloader:
        heatmaps  = extractor.extract(...)           # Dict[layer -> (B, H, W)]
        images    = batch["image"]                   # (B, C, H, W)
        captions  = batch["caption"]                 # List[str]
        base_conf = model.forward(images, captions)  # baseline confidence

        evaluator.update(heatmaps, images, captions, base_conf)

    results = evaluator.compute()   # List[LayerFaithfulnessResult]
    """

    def __init__(self, model: VLMAuditModel, config: AuditConfig) -> None:
        self.model = model
        self.config = config
        # Accumulators: layer_idx -> {"sen_sum": float, "saco_sum": float, "n": int}
        self._state: Dict[int, Dict] = {}

    def reset(self) -> None:
        self._state.clear()

    def update(
        self,
        heatmaps: Dict[int, torch.Tensor],
        images: torch.Tensor,
        captions: List[str],
        base_confidences: torch.Tensor,
    ) -> None:
        """
        Accumulate faithfulness statistics for one batch.

        Parameters
        ----------
        heatmaps         : Dict[layer_idx -> Tensor (B, H, W)]
        images           : (B, C, H, W) original pre-processed images
        captions         : List[str] length B
        base_confidences : (B,) model confidence on unmasked images
        """
        for layer_idx, hmap in heatmaps.items():
            if layer_idx not in self._state:
                self._state[layer_idx] = {"sen_sum": 0.0, "saco_sum": 0.0, "n": 0}

            for i in range(hmap.shape[0]):
                base_conf_i = base_confidences[i].item()
                sen = self._sensitivity_n(hmap[i], images[i], captions[i], base_conf_i)
                saco = self._saco(hmap[i], images[i], captions[i], base_conf_i)
                self._state[layer_idx]["sen_sum"] += sen
                self._state[layer_idx]["saco_sum"] += saco
                self._state[layer_idx]["n"] += 1

    def compute(self) -> List[LayerFaithfulnessResult]:
        """
        Returns
        -------
        List[LayerFaithfulnessResult] sorted by layer_idx
        """
        results = []
        for layer_idx, acc in sorted(self._state.items()):
            n = acc["n"]
            if n == 0:
                continue
            results.append(LayerFaithfulnessResult(
                layer_idx=layer_idx,
                sensitivity_n_score=acc["sen_sum"] / n,
                saco_auc=acc["saco_sum"] / n,
            ))
        return results

    # Internal helpers

    def _sensitivity_n(
        self,
        heatmap: torch.Tensor,
        image: torch.Tensor,
        caption: str,
        base_confidence: float,
    ) -> float:
        """
        Mask top-n% pixels and return the confidence drop.

        Parameters
        ----------
        heatmap          : (H, W) normalised attention/GradCAM map
        image            : (C, H, W) single image tensor
        caption          : string
        base_confidence  : model score on the unmasked image

        Returns
        -------
        confidence_drop (float) — higher is better for faithfulness
        """
        H, W = heatmap.shape
        n_pixels = H * W
        k = max(1, int(n_pixels * self.config.sensitivity_n / 100))

        _, top_indices = heatmap.flatten().topk(k)
        masked_img = self._apply_mask(image, top_indices)

        with torch.no_grad():
            output = self.model.forward(masked_img.unsqueeze(0), [caption])
            masked_conf = self._extract_confidence(output)

        return base_confidence - masked_conf

    def _saco(
        self,
        heatmap: torch.Tensor,
        image: torch.Tensor,
        caption: str,
        base_confidence: float,
    ) -> float:
        """
        Iteratively mask pixels from highest to lowest salience.
        Record confidence at each of self.config.saco_steps steps.
        Return the area under the confidence-drop curve (normalised to [0, 1]).

        Returns
        -------
        AUC (float) — higher means faster / steeper confidence decay
        """
        H, W = heatmap.shape
        n_pixels = H * W
        sorted_indices = heatmap.flatten().argsort(descending=True)

        steps = self.config.saco_steps
        confidence_drops: List[float] = []
        x_vals: List[float] = []

        for k in range(1, steps + 1):
            n_mask = max(1, int(n_pixels * k / steps))
            top_indices = sorted_indices[:n_mask]
            masked_img = self._apply_mask(image, top_indices)

            with torch.no_grad():
                output = self.model.forward(masked_img.unsqueeze(0), [caption])
                score = self._extract_confidence(output)

            confidence_drops.append(base_confidence - score)
            x_vals.append(k / steps)

        # Trapezoidal AUC over the confidence-drop curve
        auc = torch.trapezoid(
            torch.tensor(confidence_drops, dtype=torch.float32),
            torch.tensor(x_vals, dtype=torch.float32),
        ).item()
        return auc

    @staticmethod
    def _apply_mask(
        image: torch.Tensor,
        flat_indices: torch.Tensor,
        fill_value: float = 0.0,
    ) -> torch.Tensor:
        """
        Zero-out (or fill) specified pixel positions in `image`.

        Parameters
        ----------
        image        : (C, H, W)
        flat_indices : 1-D LongTensor of flat spatial indices (into H*W)
        fill_value   : replacement pixel value (default 0)

        Returns
        -------
        masked image (C, H, W) — original not modified
        """
        C, H, W = image.shape
        # Build a boolean spatial mask, then use torch.where (no in-place ops)
        spatial_mask = torch.zeros(H * W, dtype=torch.bool, device=image.device)
        spatial_mask[flat_indices] = True
        spatial_mask = spatial_mask.view(H, W).unsqueeze(0).expand(C, -1, -1)  # (C, H, W)

        fill = torch.full_like(image, fill_value)
        return torch.where(spatial_mask, fill, image)

    def _extract_confidence(self, output: Dict) -> float:
        """
        Extract a scalar confidence score from model output.

        For BLIP ITM the logits are (B, 2); we take softmax and return the
        positive-class probability for the first (and only) sample.
        Falls back to sigmoid for single-logit outputs.
        """
        logits = output["logits"]
        if logits.dim() >= 2 and logits.shape[-1] == 2:
            return torch.softmax(logits, dim=-1)[0, 1].item()
        return torch.sigmoid(logits.flatten()[0]).item()

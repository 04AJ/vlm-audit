"""
run_audit.py — Main entry point for the VLM-Audit pipeline.

Wires together all three modules:
  Core → Extraction → Evaluation

Usage
-----
    python run_audit.py
    python run_audit.py --model Salesforce/blip-itm-base-coco \
                        --layers 6 7 8 \
                        --max-samples 500 \
                        --output-dir results/run_01
"""

import argparse
import dataclasses
import json
import os
from datetime import datetime
from pathlib import Path

import torch

from core.config import AuditConfig
from core.model import VLMAuditModel
from data.flickr30k import get_dataloader
from extraction.attention import AttentionExtractor
from extraction.gradcam import GradCAMExtractor
from extraction.hybrid import HybridExtractor
from evaluation.grounding import GroundingEvaluator
from evaluation.faithfulness import FaithfulnessEvaluator
from evaluation.results import EvalResults, HybridResult


DEFAULT_HYBRID_ALPHAS = [0.25, 0.5, 0.75]


def _validate_hybrid_alphas(alphas: list[float]) -> list[float]:
    validated: list[float] = []
    seen: set[float] = set()
    for alpha in alphas:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Hybrid alpha must be in [0, 1], got {alpha}.")
        rounded = round(alpha, 10)
        if rounded not in seen:
            validated.append(alpha)
            seen.add(rounded)
    return validated


def parse_args() -> AuditConfig:
    parser = argparse.ArgumentParser(description="VLM-Audit pipeline")
    parser.add_argument("--model",       default="Salesforce/blip-itm-base-coco")
    parser.add_argument("--gpu",          action="store_true", default=False)
    parser.add_argument("--layers",      nargs="*", type=int, default=[])
    parser.add_argument("--max-samples", type=int,  default=None)
    parser.add_argument("--batch-size",  type=int,  default=8)
    parser.add_argument("--iou-threshold",  type=float, default=0.5)
    parser.add_argument("--sensitivity-n",  type=int,   default=10)
    parser.add_argument("--saco-steps",     type=int,   default=20)
    parser.add_argument(
        "--hybrid-alphas",
        nargs="*",
        type=float,
        default=DEFAULT_HYBRID_ALPHAS,
        help="Attention weights for hybrid heatmaps: hybrid = alpha*attention + (1-alpha)*gradcam",
    )
    parser.add_argument("--output-dir",  default="results")
    args = parser.parse_args()

    return AuditConfig(
        model_name=args.model,
        device="cuda" if args.gpu else "cpu",
        target_layers=args.layers,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        hybrid_alphas=_validate_hybrid_alphas(args.hybrid_alphas),
        iou_threshold=args.iou_threshold,
        sensitivity_n=args.sensitivity_n,
        saco_steps=args.saco_steps,
        output_dir=args.output_dir,
    )


def main() -> None:
    config = parse_args()
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Core Module — load model                                         #
    # ------------------------------------------------------------------ #
    print(f"[core] Loading {config.model_name} on {config.device} ...")
    model = VLMAuditModel(config)

    # ------------------------------------------------------------------ #
    # 2. Data — build DataLoader                                          #
    # ------------------------------------------------------------------ #
    print(f"[data] Loading Flickr30k ({config.dataset_split} split) ...")
    loader = get_dataloader(config, processor=model.processor, batch_size=config.batch_size, num_workers=0)

    # ------------------------------------------------------------------ #
    # 3. Extraction — initialise extractors                               #
    # ------------------------------------------------------------------ #
    attn_extractor = AttentionExtractor(
        config=config,
        patch_grid=model.patch_grid,
        image_size=model.image_size,
    )
    grad_extractor = GradCAMExtractor(
        model=model,
        config=config,
        image_size=model.image_size,
    )
    hybrid_extractor = HybridExtractor()

    # ------------------------------------------------------------------ #
    # 4. Evaluation — initialise evaluators                               #
    # ------------------------------------------------------------------ #
    grounding_eval      = GroundingEvaluator(config)
    faithfulness_eval   = FaithfulnessEvaluator(model, config)
    grounding_eval_grad = GroundingEvaluator(config)
    faithfulness_eval_grad = FaithfulnessEvaluator(model, config)
    grounding_eval_hybrid = {alpha: GroundingEvaluator(config) for alpha in config.hybrid_alphas}
    faithfulness_eval_hybrid = {
        alpha: FaithfulnessEvaluator(model, config) for alpha in config.hybrid_alphas
    }

    # ------------------------------------------------------------------ #
    # 5. Main loop                                                        #
    # ------------------------------------------------------------------ #
    print("[audit] Starting evaluation loop ...")
    for batch_idx, batch in enumerate(loader):
        images   = batch["image"]
        captions = batch["caption"]
        boxes    = batch["boxes"]

        # --- Forward pass (populates attention cache) ---
        output = model.forward(images, captions)
        logits = output["logits"]
        if logits.dim() >= 2 and logits.shape[-1] == 2:
            base_conf = torch.softmax(logits, dim=-1)[:, 1]
        else:
            base_conf = torch.sigmoid(logits.view(logits.shape[0], -1)[:, 0])

        # --- Extract heatmaps ---
        attn_cache = model.get_attention_cache()
        attn_heatmaps = attn_extractor.extract(attn_cache)
        grad_heatmaps = grad_extractor.compute(images, captions)

        # --- Accumulate scores ---
        grounding_eval.update(attn_heatmaps, boxes, image_sizes=batch["image_size"])
        faithfulness_eval.update(attn_heatmaps, images, captions, base_conf)
        grounding_eval_grad.update(grad_heatmaps, boxes, image_sizes=batch["image_size"])
        faithfulness_eval_grad.update(grad_heatmaps, images, captions, base_conf)
        for alpha in config.hybrid_alphas:
            hybrid_heatmaps = hybrid_extractor.blend(attn_heatmaps, grad_heatmaps, alpha)
            grounding_eval_hybrid[alpha].update(
                hybrid_heatmaps, boxes, image_sizes=batch["image_size"]
            )
            faithfulness_eval_hybrid[alpha].update(
                hybrid_heatmaps, images, captions, base_conf
            )

        model.clear_cache()

        if (batch_idx + 1) % 10 == 0:
            print(f"  processed {(batch_idx + 1) * config.batch_size} samples ...")

    # ------------------------------------------------------------------ #
    # 6. Aggregate & save results                                         #
    # ------------------------------------------------------------------ #
    results = EvalResults(
        grounding=grounding_eval.compute(),
        faithfulness=faithfulness_eval.compute(),
        grounding_grad=grounding_eval_grad.compute(),
        faithfulness_grad=faithfulness_eval_grad.compute(),
        hybrid=[
            HybridResult(
                alpha=alpha,
                grounding=grounding_eval_hybrid[alpha].compute(),
                faithfulness=faithfulness_eval_hybrid[alpha].compute(),
            )
            for alpha in config.hybrid_alphas
        ],
        config_snapshot=vars(config),
    )

    print("\n" + results.summary())

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(config.output_dir) / f"results_{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(dataclasses.asdict(results), f, indent=2)
    print(f"[audit] Results saved to {out_path}")


if __name__ == "__main__":
    main()

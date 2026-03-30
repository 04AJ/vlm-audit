"""
run_audit.py — Main entry point for the VLM-Audit pipeline.

Wires together all three modules:
  Core → Extraction → Evaluation

Usage
-----
    python scripts/run_audit.py
    python scripts/run_audit.py --model Salesforce/blip-itm-base-coco \
                                --layers 6 7 8 \
                                --max-samples 500 \
                                --output-dir results/run_01
"""

import argparse
import json
import os
from pathlib import Path

from core.config import AuditConfig
from core.model import VLMAuditModel
from data.flickr30k import get_dataloader
from extraction.attention import AttentionExtractor
from extraction.gradcam import GradCAMExtractor
from evaluation.grounding import GroundingEvaluator
from evaluation.faithfulness import FaithfulnessEvaluator
from evaluation.results import EvalResults


def parse_args() -> AuditConfig:
    parser = argparse.ArgumentParser(description="VLM-Audit pipeline")
    parser.add_argument("--model",       default="Salesforce/blip-itm-base-coco")
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--layers",      nargs="*", type=int, default=[])
    parser.add_argument("--max-samples", type=int,  default=None)
    parser.add_argument("--batch-size",  type=int,  default=8)
    parser.add_argument("--iou-threshold",  type=float, default=0.5)
    parser.add_argument("--sensitivity-n",  type=int,   default=10)
    parser.add_argument("--saco-steps",     type=int,   default=20)
    parser.add_argument("--output-dir",  default="results")
    args = parser.parse_args()

    return AuditConfig(
        model_name=args.model,
        device=args.device,
        target_layers=args.layers,
        max_samples=args.max_samples,
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
    loader = get_dataloader(config, processor=model.processor, batch_size=8)

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

    # ------------------------------------------------------------------ #
    # 4. Evaluation — initialise evaluators                               #
    # ------------------------------------------------------------------ #
    grounding_eval    = GroundingEvaluator(config)
    faithfulness_eval = FaithfulnessEvaluator(model, config)

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
        base_conf = output["logits"]   # TODO: extract per-sample confidence

        # --- Extract heatmaps ---
        attn_cache = model.get_attention_cache()
        attn_heatmaps = attn_extractor.extract(attn_cache)
        grad_heatmaps = grad_extractor.compute(images, captions)

        # --- Accumulate scores (run both heatmap types) ---
        grounding_eval.update(attn_heatmaps, boxes, image_sizes=None)   # TODO: pass sizes
        faithfulness_eval.update(attn_heatmaps, images, captions, base_conf)

        model.clear_cache()

        if (batch_idx + 1) % 10 == 0:
            print(f"  processed {(batch_idx + 1) * 8} samples ...")

    # ------------------------------------------------------------------ #
    # 6. Aggregate & save results                                         #
    # ------------------------------------------------------------------ #
    results = EvalResults(
        grounding=grounding_eval.compute(),
        faithfulness=faithfulness_eval.compute(),
        config_snapshot=vars(config),
    )

    print("\n" + results.summary())

    out_path = Path(config.output_dir) / "results.json"
    # TODO: serialise EvalResults to JSON
    print(f"[audit] Results saved to {out_path}")


if __name__ == "__main__":
    main()

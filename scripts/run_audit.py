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
import dataclasses
import json
import os
from pathlib import Path

import torch

from core.config import AuditConfig
from core.model import VLMAuditModel
from data.flickr30k import get_dataloader
from extraction.attention import AttentionExtractor
from extraction.gradcam import GradCAMExtractor
from evaluation.grounding import GroundingEvaluator
from evaluation.faithfulness import FaithfulnessEvaluator


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
    parser.add_argument("--annotations-dir", default=None,
                        help="Path to Flickr30k Entities Annotations/ XML folder")
    parser.add_argument("--sentences-dir",   default=None,
                        help="Path to Flickr30k Entities Sentences/ txt folder")
    parser.add_argument("--split-file",      default=None,
                        help="Path to txt file of image IDs for the eval split")
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
        annotations_dir=args.annotations_dir,
        sentences_dir=args.sentences_dir,
        split_file=args.split_file,
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
    # 4. Evaluation — separate evaluators for each extraction method      #
    # ------------------------------------------------------------------ #
    attn_grounding_eval    = GroundingEvaluator(config)
    attn_faithfulness_eval = FaithfulnessEvaluator(model, config)
    grad_grounding_eval    = GroundingEvaluator(config)
    grad_faithfulness_eval = FaithfulnessEvaluator(model, config)

    # ------------------------------------------------------------------ #
    # 5. Main loop                                                        #
    # ------------------------------------------------------------------ #
    print("[audit] Starting evaluation loop ...")
    for batch_idx, batch in enumerate(loader):
        images   = batch["image"]
        captions = batch["caption"]
        boxes    = batch["boxes"]
        sizes    = batch["image_size"]

        # --- Forward pass (populates attention cache) ---
        output = model.forward(images, captions)
        base_conf = torch.softmax(output["logits"], dim=-1)[:, 1].detach().cpu()

        # --- Extract heatmaps ---
        attn_heatmaps = attn_extractor.extract(model.get_attention_cache())
        grad_heatmaps = grad_extractor.compute(images, captions)

        # --- Accumulate scores for both methods ---
        attn_grounding_eval.update(attn_heatmaps, boxes, image_sizes=sizes)
        attn_faithfulness_eval.update(attn_heatmaps, images, captions, base_conf)
        grad_grounding_eval.update(grad_heatmaps, boxes, image_sizes=sizes)
        grad_faithfulness_eval.update(grad_heatmaps, images, captions, base_conf)

        model.clear_cache()

        if (batch_idx + 1) % 10 == 0:
            print(f"  processed {(batch_idx + 1) * 8} samples ...")

    # ------------------------------------------------------------------ #
    # 6. Aggregate & save results                                         #
    # ------------------------------------------------------------------ #
    out_path = Path(config.output_dir) / "results.json"
    out_data = {
        "attention": {
            "grounding":    [dataclasses.asdict(r) for r in attn_grounding_eval.compute()],
            "faithfulness": [dataclasses.asdict(r) for r in attn_faithfulness_eval.compute()],
        },
        "gradcam": {
            "grounding":    [dataclasses.asdict(r) for r in grad_grounding_eval.compute()],
            "faithfulness": [dataclasses.asdict(r) for r in grad_faithfulness_eval.compute()],
        },
        "config": vars(config),
    }
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"[audit] Results saved to {out_path}")


if __name__ == "__main__":
    main()

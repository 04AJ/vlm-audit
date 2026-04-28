"""
Pipeline overview figure for the VLM-Audit paper (Figure 1).

3 rows × 4 cols:
    Original + GT boxes  |  Raw Attention  |  Grad-CAM  |  Hybrid (α=ALPHA)
Each heatmap panel has 4 per-image metric chips annotated at the bottom.
Chip colour: green = best of all three methods, red = not best.
The caption is shown above the input image.

Run from repo root:
    python -m visualization.visualise_pipeline
"""

import os
import torch
import matplotlib.pyplot as plt

from core.config import AuditConfig
from core.model import VLMAuditModel
from data.flickr30k import Flickr30kDataset
from extraction.attention import AttentionExtractor
from extraction.gradcam import GradCAMExtractor
from extraction.hybrid import HybridExtractor
from evaluation.grounding import GroundingEvaluator
from evaluation.faithfulness import FaithfulnessEvaluator
from visualization.visualise_maps import (
    ANNOTATIONS_DIR, SENTENCES_DIR, SPLIT_FILE, LAYER,
    to_numpy_image, overlay_heatmap_annotated, draw_boxes,
    config_image_size,
)

# config 

N_SEARCH   = 150
START_IDX  = 50
SACO_STEPS = 10    # increase to 20 for final version
ALPHA      = 0.25  # hybrid blend weight (attention fraction)

# metrics

def _get_base_confidence(model, image_t, caption):
    with torch.no_grad():
        out = model.forward(image_t.unsqueeze(0), [caption])
    logits = out["logits"]
    if logits.dim() >= 2 and logits.shape[-1] == 2:
        return torch.softmax(logits, dim=-1)[0, 1].item()
    return torch.sigmoid(logits.flatten()[0]).item()


def _score_sample(grounding_eval, faith_eval, heatmap, image_t, caption,
                  boxes, img_size, base_conf):
    pga, iou = grounding_eval._score_sample(heatmap, boxes, img_size)
    sen  = faith_eval._sensitivity_n(heatmap, image_t, caption, base_conf)
    saco = faith_eval._saco(heatmap, image_t, caption, base_conf)
    return {"pga": pga, "iou": iou, "sen": sen, "saco": saco}


def _annotate_metrics(ax, m_this, m_others):
    """4 metric chips at the bottom of ax. Green = best of all methods, red = not best."""
    chips = [
        ("PGA",  "✓" if m_this["pga"] else "✗",
         m_this["pga"] == 1 and m_this["pga"] >= max(m["pga"] for m in m_others)),
        ("IoU",  f'{m_this["iou"]:.3f}',
         m_this["iou"] >= max(m["iou"] for m in m_others)),
        ("Sens", f'{m_this["sen"]:.3f}',
         m_this["sen"] >= max(m["sen"] for m in m_others)),
        ("SaCo", f'{m_this["saco"]:.3f}',
         m_this["saco"] >= max(m["saco"] for m in m_others)),
    ]
    positions = [(0.03, 0.15), (0.53, 0.15), (0.03, 0.03), (0.53, 0.03)]
    for (x, y), (label, value, wins) in zip(positions, chips):
        colour = "#2ca02c" if wins else "#d62728"
        ax.text(x, y, f"{label}: {value}",
                transform=ax.transAxes,
                fontsize=7.5, fontweight="bold", color="white",
                va="bottom",
                bbox=dict(facecolor=colour, edgecolor="none", alpha=0.72,
                          boxstyle="round,pad=0.25"))


# image selection 

def _find_examples(model, dataset, attn_extractor, gradcam_extractor,
                   hybrid_extractor, grounding_eval, layer):
    """
    Scan for 3 examples with diverging method behaviour:
      1. Grad-CAM wins grounding (gradcam PGA=1, attn PGA=0)
      2. Both methods correct  (both PGA=1)
      3. Fallback (any remaining)
    """
    buckets = {"gradcam_wins": None, "both_correct": None, "fallback": None}
    end = min(START_IDX + N_SEARCH, len(dataset))

    for idx in range(START_IDX, end):
        if all(v is not None for v in buckets.values()):
            break

        sample   = dataset[idx]
        image_t  = sample["image"]
        caption  = sample["caption"]
        boxes    = sample["boxes"]
        img_size = sample["image_size"]

        if not boxes:
            continue

        images_b = image_t.unsqueeze(0)

        with torch.no_grad():
            model.forward(images_b, [caption])
        attn_cache    = model.get_attention_cache()
        attn_heatmaps = attn_extractor.extract(attn_cache)
        model.clear_cache()

        gradcam_heatmaps = gradcam_extractor.compute(images_b, [caption])

        if layer not in attn_heatmaps or layer not in gradcam_heatmaps:
            continue

        attn_map    = attn_heatmaps[layer][0]
        grad_map    = gradcam_heatmaps[layer][0]
        hybrid_map  = hybrid_extractor.blend(
            attn_heatmaps, gradcam_heatmaps, ALPHA
        )[layer][0]

        attn_pga, _ = grounding_eval._score_sample(attn_map, boxes, img_size)
        grad_pga, _ = grounding_eval._score_sample(grad_map, boxes, img_size)

        entry = {
            "image_t": image_t, "caption": caption, "boxes": boxes,
            "attn_map": attn_map, "grad_map": grad_map, "hybrid_map": hybrid_map,
            "image_size": img_size,
        }

        if grad_pga == 1 and attn_pga == 0 and buckets["gradcam_wins"] is None:
            buckets["gradcam_wins"] = entry
            print(f"[vis] gradcam_wins  → idx {idx}")
        elif grad_pga == 1 and attn_pga == 1 and buckets["both_correct"] is None:
            buckets["both_correct"] = entry
            print(f"[vis] both_correct  → idx {idx}")
        elif buckets["fallback"] is None:
            buckets["fallback"] = entry
            print(f"[vis] fallback      → idx {idx}")

    ordered = [buckets["gradcam_wins"], buckets["both_correct"], buckets["fallback"]]
    return [e for e in ordered if e is not None][:3]


# main figure 

def plot_pipeline(
    model, dataset, attn_extractor, gradcam_extractor, hybrid_extractor,
    grounding_eval, faith_eval,
    layer=LAYER,
    save_path=None,
):
    examples = _find_examples(
        model, dataset, attn_extractor, gradcam_extractor,
        hybrid_extractor, grounding_eval, layer,
    )
    if not examples:
        print("[vis] No suitable examples found.")
        return

    n_rows = len(examples)
    fig, axes = plt.subplots(
        n_rows, 4,
        figsize=(17, 4.2 * n_rows),
        gridspec_kw={"hspace": 0.12, "wspace": 0.03},
    )
    if n_rows == 1:
        axes = [axes]

    col_titles = [
        "Input",
        f"Raw Attention  (layer {layer})",
        f"Grad-CAM  (layer {layer})",
        f"Hybrid  α={ALPHA}  (layer {layer})",
    ]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)

    for row_idx, s in enumerate(examples):
        image_t  = s["image_t"]
        caption  = s["caption"]
        boxes    = s["boxes"]
        img_size = s["image_size"]
        image_np = to_numpy_image(image_t)

        print(f"[vis] Computing full metrics for row {row_idx + 1}...")
        base_conf = _get_base_confidence(model, image_t, caption)
        m_attn   = _score_sample(grounding_eval, faith_eval,
                                 s["attn_map"],   image_t, caption,
                                 boxes, img_size, base_conf)
        m_grad   = _score_sample(grounding_eval, faith_eval,
                                 s["grad_map"],   image_t, caption,
                                 boxes, img_size, base_conf)
        m_hybrid = _score_sample(grounding_eval, faith_eval,
                                 s["hybrid_map"], image_t, caption,
                                 boxes, img_size, base_conf)

        draw_boxes(axes[row_idx][0], image_np, boxes, caption,
                   original_size=img_size)
        overlay_heatmap_annotated(axes[row_idx][1], image_np, s["attn_map"],
                                  boxes, img_size)
        _annotate_metrics(axes[row_idx][1], m_attn,   [m_grad, m_hybrid])
        overlay_heatmap_annotated(axes[row_idx][2], image_np, s["grad_map"],
                                  boxes, img_size)
        _annotate_metrics(axes[row_idx][2], m_grad,   [m_attn, m_hybrid])
        overlay_heatmap_annotated(axes[row_idx][3], image_np, s["hybrid_map"],
                                  boxes, img_size)
        _annotate_metrics(axes[row_idx][3], m_hybrid, [m_attn, m_grad])

        for ax in axes[row_idx]:
            for spine in ax.spines.values():
                spine.set_edgecolor("#555555")
                spine.set_linewidth(1.2)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=180,
                    format="pdf", metadata={"Title": "VLM-Audit pipeline figure"})
        print(f"[vis] Saved → {save_path}")
    plt.close(fig)


# main

def main():
    config = AuditConfig(
        device="cpu",
        max_samples=START_IDX + N_SEARCH,
        target_layers=[LAYER],
        saco_steps=SACO_STEPS,
        annotations_dir=ANNOTATIONS_DIR,
        sentences_dir=SENTENCES_DIR,
        split_file=SPLIT_FILE,
    )

    model   = VLMAuditModel(config)
    dataset = Flickr30kDataset(config, processor=model.processor)

    img_size          = config_image_size(model)
    attn_extractor    = AttentionExtractor(config, model.patch_grid, img_size)
    gradcam_extractor = GradCAMExtractor(model, config, img_size)
    hybrid_extractor  = HybridExtractor()
    grounding_eval    = GroundingEvaluator(config)
    faith_eval        = FaithfulnessEvaluator(model, config)

    plot_pipeline(
        model, dataset, attn_extractor, gradcam_extractor, hybrid_extractor,
        grounding_eval, faith_eval,
        layer=LAYER,
        save_path="results/pipeline/fig1_pipeline.pdf",
    )


if __name__ == "__main__":
    main()

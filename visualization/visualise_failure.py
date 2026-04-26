"""
Failure comparison figure for the VLM-Audit paper.

Figure 2: 6 rows × 3 cols — 3 correct cases then 3 failure cases.
  Correct: attention peak lands inside a GT box
  Failure: attention peak misses all GT boxes
  Columns: original + GT boxes | attention | Grad-CAM

Saved as a compact letter-size PDF suitable for direct inclusion in a report.

Run from repo root:
    python -m visualization.visualise_failure
"""

import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from core.config import AuditConfig
from core.model import VLMAuditModel
from data.flickr30k import Flickr30kDataset
from extraction.attention import AttentionExtractor
from extraction.gradcam import GradCAMExtractor
from visualization.visualise_maps import (
    ANNOTATIONS_DIR, SENTENCES_DIR, SPLIT_FILE, LAYER,
    to_numpy_image, overlay_heatmap_annotated, draw_boxes,
    config_image_size, _peak_in_boxes,
)

# ------------------------------------------------------------------ config --

N_SEARCH     = 200
N_CASES      = 3     # correct cases and failure cases each

# ------------------------------------------------------------------- figure --

def plot_failure_comparison(
    model: VLMAuditModel,
    dataset: Flickr30kDataset,
    attn_extractor: AttentionExtractor,
    gradcam_extractor: GradCAMExtractor,
    layer: int = LAYER,
    save_path: str = None,
) -> None:
    """
    Scan dataset for N_CASES correct and N_CASES failure examples, then render
    a 6-row × 3-col figure saved as a compact portrait PDF.
    """
    correct_samples = []
    failure_samples = []

    START_IDX = 100
    for idx in range(START_IDX, min(START_IDX + N_SEARCH, len(dataset))):
        if len(correct_samples) >= N_CASES and len(failure_samples) >= N_CASES:
            break

        sample   = dataset[idx]
        image_t  = sample["image"]
        caption  = sample["caption"]
        boxes    = sample["boxes"]
        img_size = sample["image_size"]

        if not boxes:
            continue

        images_b   = image_t.unsqueeze(0)
        captions_b = [caption]

        with torch.no_grad():
            model.forward(images_b, captions_b)
        attn_cache    = model.get_attention_cache()
        attn_heatmaps = attn_extractor.extract(attn_cache)
        model.clear_cache()

        gradcam_heatmaps = gradcam_extractor.compute(images_b, captions_b)

        if layer not in attn_heatmaps or layer not in gradcam_heatmaps:
            continue

        attn_map     = attn_heatmaps[layer][0]
        grad_map     = gradcam_heatmaps[layer][0]
        attn_correct = _peak_in_boxes(attn_map, boxes, img_size)

        entry = {
            "image_t": image_t, "caption": caption, "boxes": boxes,
            "attn_map": attn_map, "grad_map": grad_map,
            "image_size": img_size,
        }

        if attn_correct and len(correct_samples) < N_CASES:
            correct_samples.append(entry)
        elif not attn_correct and len(failure_samples) < N_CASES:
            failure_samples.append(entry)

    if not correct_samples and not failure_samples:
        print("[vis] No samples with GT boxes found — skipping Figure 2.")
        return

    # Pad with duplicates if we couldn't find enough of one type
    while len(correct_samples) < N_CASES:
        correct_samples.append((correct_samples or failure_samples)[-1])
    while len(failure_samples) < N_CASES:
        failure_samples.append((failure_samples or correct_samples)[-1])

    n_rows = N_CASES * 2
    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(8.5, 11),
        gridspec_kw={"hspace": 0.02, "wspace": 0.02},
    )

    col_titles = [
        "Original + GT boxes",
        f"Attention  (layer {layer})",
        f"Grad-CAM  (layer {layer})",
    ]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=8, fontweight="bold", pad=4)

    row_colours = {"correct": "#2ca02c", "failure": "#d62728"}
    row_labels  = {"correct": "correct", "failure": "failure"}

    all_rows = [("correct", s) for s in correct_samples] + \
               [("failure", s) for s in failure_samples]

    for row_idx, (key, s) in enumerate(all_rows):
        image_np = to_numpy_image(s["image_t"])
        colour   = row_colours[key]

        draw_boxes(axes[row_idx][0], image_np, s["boxes"], "",
                   original_size=s["image_size"])
        overlay_heatmap_annotated(axes[row_idx][1], image_np, s["attn_map"],
                                  s["boxes"], s["image_size"])
        overlay_heatmap_annotated(axes[row_idx][2], image_np, s["grad_map"],
                                  s["boxes"], s["image_size"])

        axes[row_idx][0].set_ylabel(
            row_labels[key], fontsize=7, fontweight="bold",
            color=colour, labelpad=6,
        )

        for ax in axes[row_idx]:
            for spine in ax.spines.values():
                spine.set_edgecolor(colour)
                spine.set_linewidth(1.5)

        # Divider line between correct and failure groups
        if row_idx == N_CASES - 1:
            for ax in axes[row_idx]:
                ax.spines["bottom"].set_linewidth(3)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150,
                    format="pdf", metadata={"Title": "VLM-Audit failure comparison"})
        print(f"[vis] Saved → {save_path}")
    plt.close(fig)


# --------------------------------------------------------------------- main --

def main():
    config = AuditConfig(
        device="cpu",
        max_samples=N_SEARCH,
        target_layers=[LAYER],
        annotations_dir=ANNOTATIONS_DIR,
        sentences_dir=SENTENCES_DIR,
        split_file=SPLIT_FILE,
    )

    model   = VLMAuditModel(config)
    dataset = Flickr30kDataset(config, processor=model.processor)

    img_size          = config_image_size(model)
    attn_extractor    = AttentionExtractor(config, model.patch_grid, img_size)
    gradcam_extractor = GradCAMExtractor(model, config, img_size)

    os.makedirs("results/failure", exist_ok=True)
    plot_failure_comparison(
        model, dataset, attn_extractor, gradcam_extractor, layer=LAYER,
        save_path="results/failure/fig2_failure_comparison.pdf",
    )


if __name__ == "__main__":
    main()

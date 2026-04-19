"""
Heatmap visualisations for the VLM-Audit paper.

Three figures
-------------
0. Overview      — N images × [original, attention, Grad-CAM]   (existing)
1. Per-word      — 1 image  × [original, noun₁, noun₂, ...]    attention only
2. Failure comp  — 2 rows   × [original, attention, Grad-CAM]
                   row 0 = case where attention peak is inside GT box (correct)
                   row 1 = case where attention peak misses GT box   (failure)

Run from repo root with venv active:
    python -m extraction.visualise_maps
"""

import os
import re
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from core.config import AuditConfig
from core.model import VLMAuditModel
from data.flickr30k import Flickr30kDataset
from extraction.attention import AttentionExtractor
from extraction.gradcam import GradCAMExtractor

# ------------------------------------------------------------------ config --

N_IMAGES    = 3    # images shown in the overview figure
N_SEARCH    = 200  # images scanned when hunting for failure cases / per-word examples
MAX_NOUNS   = 4    # max noun columns in the per-word figure
LAYER       = 9    # cross-attention layer to visualise 
ALPHA       = 0.5  # heatmap overlay transparency

_REPO_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNOTATIONS_DIR = os.path.join(_REPO_ROOT, "data", "Annotations")
SENTENCES_DIR   = os.path.join(_REPO_ROOT, "data", "Sentences")
SPLIT_FILE      = os.path.join(_REPO_ROOT, "data", "test.txt")

# ------------------------------------------------------------------ helpers --

def to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) float tensor → (H, W, 3) uint8 numpy array."""
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return (img * 255).astype(np.uint8)


def overlay_heatmap(ax, image_np: np.ndarray, heatmap: torch.Tensor, title: str):
    """Draw image with heatmap overlaid in hot colourmap."""
    ax.imshow(image_np)
    ax.imshow(heatmap.detach().cpu().numpy(), cmap="hot", alpha=ALPHA,
              vmin=0, vmax=1, extent=[0, image_np.shape[1], image_np.shape[0], 0])
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def overlay_heatmap_annotated(
    ax,
    image_np: np.ndarray,
    heatmap: torch.Tensor,
    boxes: list,
    original_size: tuple,
) -> bool:
    """
    Draw heatmap overlay with GT boxes, peak pixel marker, and ✓/✗ label.

    GT boxes are scaled from original_size to the displayed image dimensions.
    Returns True if the peak pixel lands inside a GT box.
    """
    disp_h, disp_w = image_np.shape[:2]
    orig_h, orig_w = original_size
    x_scale = disp_w / orig_w
    y_scale = disp_h / orig_h

    ax.imshow(image_np)
    ax.imshow(heatmap.detach().cpu().numpy(), cmap="hot", alpha=ALPHA,
              vmin=0, vmax=1, extent=[0, disp_w, disp_h, 0])

    # Find peak pixel (heatmap is already at display resolution)
    hm_np   = heatmap.detach().cpu()
    flat_pk = hm_np.flatten().argmax().item()
    peak_y  = flat_pk // disp_w
    peak_x  = flat_pk  % disp_w

    # Check if peak is inside any GT box (in original coords, so unscale peak)
    inside = False
    for entity in boxes:
        x1, y1, x2, y2 = entity["box"]
        if x1 <= peak_x / x_scale <= x2 and y1 <= peak_y / y_scale <= y2:
            inside = True
            break

    # Draw GT boxes scaled to display size
    for entity in boxes:
        x1, y1, x2, y2 = entity["box"]
        x1, x2 = x1 * x_scale, x2 * x_scale
        y1, y2 = y1 * y_scale, y2 * y_scale
        ax.add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor="cyan", facecolor="none", linestyle="--",
        ))

    # Peak pixel marker
    marker_colour = "#2ca02c" if inside else "#d62728"
    ax.plot(peak_x, peak_y, marker="*", markersize=14,
            color=marker_colour, markeredgecolor="white", markeredgewidth=0.8)

    # ✓ / ✗ badge in top-right corner
    label      = "inside" if inside else "outside"
    badge_col  = "#2ca02c" if inside else "#d62728"
    ax.text(0.97, 0.97, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", color="white",
            ha="right", va="top",
            bbox=dict(facecolor=badge_col, edgecolor="none", pad=3, alpha=0.85))

    ax.axis("off")
    return inside


def draw_boxes(ax, image_np: np.ndarray, boxes: list, caption: str,
               original_size: tuple = None):
    """
    Draw original image with GT bounding boxes.

    Parameters
    ----------
    original_size : (H_orig, W_orig) of the image before preprocessing.
                    When provided, box coordinates are scaled from original
                    pixel space to the displayed image dimensions.
    """
    ax.imshow(image_np)
    disp_h, disp_w = image_np.shape[:2]

    if original_size is not None:
        orig_h, orig_w = original_size
        x_scale = disp_w / orig_w
        y_scale = disp_h / orig_h
    else:
        x_scale = y_scale = 1.0

    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    seen = {}
    for box_dict in boxes:
        label = box_dict["label"]
        if label not in seen:
            seen[label] = colours[len(seen) % len(colours)]
        x1, y1, x2, y2 = box_dict["box"]
        x1, x2 = x1 * x_scale, x2 * x_scale
        y1, y2 = y1 * y_scale, y2 * y_scale
        ax.add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor=seen[label], facecolor="none",
        ))
        ax.text(x1, y1 - 3, label, fontsize=6, color=seen[label],
                bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none"))
    wrapped = "\n".join(caption[i:i+55] for i in range(0, len(caption), 55))
    ax.set_title(wrapped, fontsize=7)
    ax.axis("off")


def config_image_size(model: VLMAuditModel):
    """Derive (H, W) from the model's expected input resolution."""
    ph, pw = model.patch_grid
    patch_size = 16
    return (ph * patch_size, pw * patch_size)


# ------------------------------------------------------- per-word attention --

def _load_entity_phrases(sentences_dir: str, filename: str, caption_idx: int = 4) -> dict:
    """Return {entity_id: phrase_text} parsed from the Flickr30k sentence file."""
    stem = os.path.splitext(filename)[0]
    path = os.path.join(sentences_dir, stem + ".txt")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]
    line = lines[caption_idx] if caption_idx < len(lines) else lines[0]
    pattern = re.compile(r'\[/EN#(\d+)/\w+\s+([^\]]+)\]')
    return {m.group(1): m.group(2).strip() for m in pattern.finditer(line)}


def _find_phrase_token_indices(tokenizer, caption: str, phrase: str) -> list:
    """
    Return token positions (1-based, to account for prepended [CLS]) in the
    full caption that correspond to `phrase`.  Falls back to [] if not found.
    """
    cap_ids    = tokenizer(caption, add_special_tokens=False)["input_ids"]
    phrase_ids = tokenizer(phrase,  add_special_tokens=False)["input_ids"]
    for i in range(len(cap_ids) - len(phrase_ids) + 1):
        if cap_ids[i : i + len(phrase_ids)] == phrase_ids:
            return [i + 1 + k for k in range(len(phrase_ids))]  # +1 for [CLS]
    return []


def _attn_heatmap_for_tokens(
    attn_weights: torch.Tensor,
    token_indices: list,
    patch_grid: tuple,
    image_size: tuple,
    head_fusion: str = "mean",
) -> torch.Tensor:
    """
    Build a spatial heatmap from attention over specific token positions.

    Parameters
    ----------
    attn_weights  : (heads, T, N) for a single sample
    token_indices : token positions to average over; falls back to all tokens
    patch_grid    : (ph, pw) from VLMAuditModel
    image_size    : (H, W) model input resolution

    Returns
    -------
    (H_img, W_img) tensor normalised to [0, 1]
    """
    fused = (attn_weights.mean(dim=0) if head_fusion == "mean"
             else attn_weights.max(dim=0).values)          # (T, N)
    indices = token_indices if token_indices else list(range(fused.shape[0]))
    pooled  = fused[indices].mean(dim=0)                   # (N,)
    pooled  = pooled[1:]                                   # drop CLS patch token
    ph, pw  = patch_grid
    grid    = pooled.view(ph, pw).unsqueeze(0).unsqueeze(0).float()
    heatmap = F.interpolate(grid, image_size, mode="bilinear",
                            align_corners=False).squeeze()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap


# -------------------------------------------------------- pointing game util --

def _peak_in_boxes(heatmap: torch.Tensor, boxes: list, image_size: tuple) -> bool:
    """True if the heatmap's peak pixel falls inside any GT bounding box."""
    h, w = image_size
    hm = F.interpolate(
        heatmap.unsqueeze(0).unsqueeze(0).float(),
        size=(h, w), mode="bilinear", align_corners=False,
    ).squeeze()
    flat_peak = hm.flatten().argmax()
    py = (flat_peak // w).item()
    px = (flat_peak  % w).item()
    for entity in boxes:
        x1, y1, x2, y2 = entity["box"]
        if x1 <= px <= x2 and y1 <= py <= y2:
            return True
    return False


# =========================================================== Figure 1 ========
# Attention per word: one image, one column per noun phrase

def plot_attention_per_word(
    model: VLMAuditModel,
    dataset: Flickr30kDataset,
    layer: int = LAYER,
    min_nouns: int = 3,
    n_examples: int = 10,
    save_dir: str = "results",
) -> None:
    """
    Figure 1 — Attention heatmap per entity noun (Option A).

    Scans the dataset for up to `n_examples` images that each have at least
    `min_nouns` unique entity phrases, saving a separate PDF for each.

    Layout per figure (1 row):
        "noun₁"  |  "noun₂"  |  "noun₃"  |  ...  (up to MAX_NOUNS)

    The full caption is shown at the top of each figure.
    """
    tokenizer = model.processor.tokenizer
    found = 0

    for idx in range(len(dataset)):
        if found >= n_examples:
            break

        s          = dataset[idx]
        phrase_map = _load_entity_phrases(SENTENCES_DIR, s["filename"])
        seen_labels = set()
        entities    = []
        for box in s["boxes"]:
            label = box["label"]
            if label in seen_labels:
                continue
            phrase = phrase_map.get(box["phrase"], label)
            token_indices = _find_phrase_token_indices(tokenizer, s["caption"], phrase)
            seen_labels.add(label)
            entities.append({"label": label, "phrase": phrase,
                              "token_indices": token_indices})
            if len(entities) >= MAX_NOUNS:
                break

        if len(entities) < min_nouns:
            continue

        image_t  = s["image"]
        caption  = s["caption"]
        image_np = to_numpy_image(image_t)

        with torch.no_grad():
            model.forward(image_t.unsqueeze(0), [caption])
        attn_cache = model.get_attention_cache()
        model.clear_cache()

        if layer not in attn_cache:
            continue

        raw_weights = attn_cache[layer][0]   # (heads, T, N)
        n_cols = len(entities)
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 5))
        if n_cols == 1:
            axes = [axes]

        for col, entity in enumerate(entities):
            matched = bool(entity["token_indices"])
            print(f"  [vis] phrase='{entity['phrase']}'  tokens={entity['token_indices']}  "
                  f"{'OK' if matched else 'NO MATCH — falling back to all tokens'}")
            heatmap = _attn_heatmap_for_tokens(
                raw_weights, entity["token_indices"],
                model.patch_grid, config_image_size(model),
            )
            overlay_heatmap(axes[col], image_np, heatmap, "")
            axes[col].set_title(f'"{entity["phrase"]}"', fontsize=9, fontweight="bold")

        # Full caption at the top, wrapped so it fits
        wrapped = "\n".join(caption[i:i+80] for i in range(0, len(caption), 80))
        fig.suptitle(
            f"{wrapped}\n\nlayer {layer}",
            fontsize=9, y=1.04, ha="center",
        )
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"fig1_attention_per_word_{found + 1:02d}.pdf")
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[vis] Saved → {save_path}")
        plt.close(fig)
        found += 1

    if found == 0:
        print(f"[vis] No samples with {min_nouns}+ nouns found — skipping Figure 1.")
    else:
        print(f"[vis] Figure 1: saved {found} examples to {save_dir}/")


# =========================================================== Figure 2 ========
# Failure comparison: attn vs Grad-CAM, correct row vs failure row

def plot_failure_comparison(
    model: VLMAuditModel,
    dataset: Flickr30kDataset,
    attn_extractor: AttentionExtractor,
    gradcam_extractor: GradCAMExtractor,
    layer: int = LAYER,
    save_path: str = None,
) -> None:
    """
    Figure 2 — Attention vs Grad-CAM failure comparison.

    Scans up to N_SEARCH images using the pointing game criterion to find:
      - a 'correct' case: attention peak lands inside a GT box
      - a 'failure' case: attention peak misses all GT boxes

    Layout (2 rows × 3 cols):
        Original + GT boxes  |  Attention  |  Grad-CAM
        (correct row)
        (failure row)
    """
    correct_sample = None
    failure_sample = None

    for idx in range(min(N_SEARCH, len(dataset))):
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

        attn_map = attn_heatmaps[layer][0]
        grad_map = gradcam_heatmaps[layer][0]
        attn_correct = _peak_in_boxes(attn_map, boxes, img_size)

        entry = {
            "image_t": image_t, "caption": caption, "boxes": boxes,
            "attn_map": attn_map, "grad_map": grad_map,
            "image_size": img_size,
        }

        if correct_sample is None and attn_correct:
            correct_sample = entry
        if failure_sample is None and not attn_correct:
            failure_sample = entry

        if correct_sample and failure_sample:
            break

    if correct_sample is None or failure_sample is None:
        print(f"[vis] Warning: could not find both cases in the first {N_SEARCH} "
              "images — using what was found.")
        correct_sample = correct_sample or failure_sample
        failure_sample = failure_sample or correct_sample

    if correct_sample is None:
        print("[vis] No samples with GT boxes found — skipping Figure 2.")
        return

    rows = [
        ("correct", "Attention peak inside GT box", correct_sample),
        ("failure", "Attention peak misses GT box",  failure_sample),
    ]
    row_colours = {"correct": "#2ca02c", "failure": "#d62728"}  # green / red

    fig, axes = plt.subplots(2, 3, figsize=(13, 8),
                             gridspec_kw={"hspace": 0.1})

    col_titles = [
        "Original + GT boxes",
        f"Attention  (layer {layer})",
        f"Grad-CAM  (layer {layer})",
    ]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=10, fontweight="bold")

    for row_idx, (key, row_label, s) in enumerate(rows):
        image_np = to_numpy_image(s["image_t"])
        colour   = row_colours[key]

        draw_boxes(axes[row_idx][0], image_np, s["boxes"], s["caption"],
                   original_size=s["image_size"])
        overlay_heatmap_annotated(axes[row_idx][1], image_np, s["attn_map"],
                                  s["boxes"], s["image_size"])
        overlay_heatmap_annotated(axes[row_idx][2], image_np, s["grad_map"],
                                  s["boxes"], s["image_size"])

        # Coloured row label on the left spine
        axes[row_idx][0].set_ylabel(
            row_label, fontsize=10, fontweight="bold",
            color=colour, labelpad=10,
        )
        # Coloured border around all three axes in this row
        for ax in axes[row_idx]:
            for spine in ax.spines.values():
                spine.set_edgecolor(colour)
                spine.set_linewidth(2.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[vis] Saved → {save_path}")
    plt.show()


# --------------------------------------------------------------------- main --

def main():
    config = AuditConfig(
        device="cpu",
        max_samples=N_SEARCH,      # load enough images for per-word examples + failure search
        target_layers=[LAYER],
        annotations_dir=ANNOTATIONS_DIR,
        sentences_dir=SENTENCES_DIR,
        split_file=SPLIT_FILE,
    )

    model   = VLMAuditModel(config)
    dataset = Flickr30kDataset(config, processor=model.processor)

    img_size = config_image_size(model)
    attn_extractor    = AttentionExtractor(config, model.patch_grid, img_size)
    gradcam_extractor = GradCAMExtractor(model, config, img_size)

    # ---- Figure 0: overview (one PDF per image) --------------------------------
    col_titles = ["Original + GT boxes", f"Attention  (layer {LAYER})", f"Grad-CAM  (layer {LAYER})"]

    for idx in range(N_IMAGES):
        sample   = dataset[idx]
        image_t  = sample["image"]
        caption  = sample["caption"]
        boxes    = sample["boxes"]
        image_np = to_numpy_image(image_t)

        images_batch   = image_t.unsqueeze(0)
        captions_batch = [caption]

        with torch.no_grad():
            model.forward(images_batch, captions_batch)
        attn_cache    = model.get_attention_cache()
        attn_heatmaps = attn_extractor.extract(attn_cache)
        attn_map      = attn_heatmaps[LAYER][0]
        model.clear_cache()

        gradcam_heatmaps = gradcam_extractor.compute(images_batch, captions_batch)
        gradcam_map      = gradcam_heatmaps[LAYER][0]

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        for ax, title in zip(axes, col_titles):
            ax.set_title(title, fontsize=10, fontweight="bold")

        draw_boxes(axes[0], image_np, boxes, caption, original_size=sample["image_size"])
        overlay_heatmap(axes[1], image_np, attn_map,    f"Attention  — {sample['filename']}")
        overlay_heatmap(axes[2], image_np, gradcam_map, f"Grad-CAM  — {sample['filename']}")

        plt.tight_layout()
        save_path = f"results/fig0_overview_{idx + 1:02d}.pdf"
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[vis] Saved → {save_path}")
        plt.close(fig)

    # ---- Figure 1: attention per word (first image with entities) -----------
    plot_attention_per_word(
        model, dataset, layer=LAYER, min_nouns=3, n_examples=10,
        save_dir="results",
    )

    # ---- Figure 2: failure comparison ---------------------------------------
    plot_failure_comparison(
        model, dataset, attn_extractor, gradcam_extractor, layer=LAYER,
        save_path="results/fig2_failure_comparison.pdf",
    )


if __name__ == "__main__":
    main()

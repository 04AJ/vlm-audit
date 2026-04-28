"""
Heatmap overview figure for the VLM-Audit paper.

Figure 0: N images × [original + GT boxes, attention, Grad-CAM]

Run from repo root:
    python -m visualization.visualise_maps
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from core.config import AuditConfig
from core.model import VLMAuditModel
from data.flickr30k import Flickr30kDataset
from extraction.attention import AttentionExtractor
from extraction.gradcam import GradCAMExtractor

# config

N_IMAGES = 3
LAYER    = 9
ALPHA    = 0.5

_REPO_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRATCH_DATA   = os.environ.get("DATA_DIR", os.path.join(_REPO_ROOT, "data"))
ANNOTATIONS_DIR = os.path.join(_SCRATCH_DATA, "Annotations")
SENTENCES_DIR   = os.path.join(_SCRATCH_DATA, "Sentences")
SPLIT_FILE      = os.path.join(_REPO_ROOT, "data", "test.txt")

# helpers 

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
    Draw heatmap overlay with GT boxes, peak pixel marker, and inside/outside label.
    Returns True if the peak pixel lands inside a GT box.
    """
    disp_h, disp_w = image_np.shape[:2]
    orig_h, orig_w = original_size
    x_scale = disp_w / orig_w
    y_scale = disp_h / orig_h

    ax.imshow(image_np)
    ax.imshow(heatmap.detach().cpu().numpy(), cmap="hot", alpha=ALPHA,
              vmin=0, vmax=1, extent=[0, disp_w, disp_h, 0])

    hm_np   = heatmap.detach().cpu()
    flat_pk = hm_np.flatten().argmax().item()
    peak_y  = flat_pk // disp_w
    peak_x  = flat_pk  % disp_w

    inside = False
    for entity in boxes:
        x1, y1, x2, y2 = entity["box"]
        if x1 <= peak_x / x_scale <= x2 and y1 <= peak_y / y_scale <= y2:
            inside = True
            break

    for entity in boxes:
        x1, y1, x2, y2 = entity["box"]
        x1, x2 = x1 * x_scale, x2 * x_scale
        y1, y2 = y1 * y_scale, y2 * y_scale
        ax.add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor="cyan", facecolor="none", linestyle="--",
        ))

    marker_colour = "#2ca02c" if inside else "#d62728"
    ax.plot(peak_x, peak_y, marker="*", markersize=14,
            color=marker_colour, markeredgecolor="white", markeredgewidth=0.8)

    label     = "inside" if inside else "outside"
    badge_col = "#2ca02c" if inside else "#d62728"
    ax.text(0.97, 0.97, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", color="white",
            ha="right", va="top",
            bbox=dict(facecolor=badge_col, edgecolor="none", pad=3, alpha=0.25))

    ax.axis("off")
    return inside


def draw_boxes(ax, image_np: np.ndarray, boxes: list, caption: str,
               original_size: tuple = None):
    """Draw original image with GT bounding boxes and wrapped caption title."""
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


def _peak_in_boxes(heatmap: torch.Tensor, boxes: list, image_size: tuple) -> bool:
    """True if the heatmap's peak pixel falls inside any GT bounding box."""
    import torch.nn.functional as F
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


def main():
    config = AuditConfig(
        device="cpu",
        max_samples=N_IMAGES,
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

    col_titles = [
        "Original + GT boxes",
        f"Attention  (layer {LAYER})",
        f"Grad-CAM  (layer {LAYER})",
    ]

    os.makedirs("results/overview", exist_ok=True)

    for idx in range(N_IMAGES):
        sample   = dataset[idx]
        image_t  = sample["image"]
        caption  = sample["caption"]
        boxes    = sample["boxes"]
        image_np = to_numpy_image(image_t)

        images_b   = image_t.unsqueeze(0)
        captions_b = [caption]

        with torch.no_grad():
            model.forward(images_b, captions_b)
        attn_cache    = model.get_attention_cache()
        attn_heatmaps = attn_extractor.extract(attn_cache)
        attn_map      = attn_heatmaps[LAYER][0]
        model.clear_cache()

        gradcam_heatmaps = gradcam_extractor.compute(images_b, captions_b)
        gradcam_map      = gradcam_heatmaps[LAYER][0]

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        for ax, title in zip(axes, col_titles):
            ax.set_title(title, fontsize=10, fontweight="bold")

        draw_boxes(axes[0], image_np, boxes, caption, original_size=sample["image_size"])
        overlay_heatmap(axes[1], image_np, attn_map,    f"Attention  — {sample['filename']}")
        overlay_heatmap(axes[2], image_np, gradcam_map, f"Grad-CAM  — {sample['filename']}")

        plt.tight_layout()
        save_path = f"results/overview/fig0_overview_{idx + 1:02d}.pdf"
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[vis] Saved → {save_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()

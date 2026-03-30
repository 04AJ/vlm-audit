"""
Side-by-side visualisation of Attention and Grad-CAM heatmaps.

Shows for each image:
  col 1 — original image with ground-truth bounding boxes
  col 2 — attention heatmap (layer 11) overlaid on image
  col 3 — Grad-CAM heatmap  (layer 11) overlaid on image

Run from repo root with venv active:
    python -m extraction.visualise_maps
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from core.config import AuditConfig
from core.model import VLMAuditModel
from data.flickr30k import Flickr30kDataset
from extraction.attention import AttentionExtractor
from extraction.gradcam import GradCAMExtractor

# ------------------------------------------------------------------ config --

N_IMAGES    = 3
LAYER       = 11          # which cross-attention layer to visualise
ALPHA       = 0.5         # heatmap overlay transparency

ANNOTATIONS_DIR = r"C:\vlm-audit\data\Annotations"
SENTENCES_DIR   = r"C:\vlm-audit\data\Sentences"
SPLIT_FILE      = r"C:\vlm-audit\data\test.txt"

# ------------------------------------------------------------------ helpers --

def to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) float tensor → (H, W, 3) uint8 numpy array."""
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return (img * 255).astype(np.uint8)


def overlay_heatmap(ax, image_np: np.ndarray, heatmap: torch.Tensor, title: str):
    """Draw image with heatmap overlaid in hot colourmap."""
    ax.imshow(image_np)
    ax.imshow(heatmap.cpu().numpy(), cmap="hot", alpha=ALPHA,
              vmin=0, vmax=1, extent=[0, image_np.shape[1], image_np.shape[0], 0])
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def draw_boxes(ax, image_np: np.ndarray, boxes: list, caption: str):
    """Draw original image with GT bounding boxes."""
    ax.imshow(image_np)
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    seen = {}
    for box_dict in boxes:
        label = box_dict["label"]
        if label not in seen:
            seen[label] = colours[len(seen) % len(colours)]
        x1, y1, x2, y2 = box_dict["box"]
        ax.add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor=seen[label], facecolor="none",
        ))
        ax.text(x1, y1 - 3, label, fontsize=6, color=seen[label],
                bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none"))
    wrapped = "\n".join(caption[i:i+55] for i in range(0, len(caption), 55))
    ax.set_title(wrapped, fontsize=7)
    ax.axis("off")


# --------------------------------------------------------------------- main --

def main():
    config = AuditConfig(
        device="cpu",
        max_samples=N_IMAGES,
        target_layers=[LAYER],
        annotations_dir=ANNOTATIONS_DIR,
        sentences_dir=SENTENCES_DIR,
        split_file=SPLIT_FILE,
    )

    # Load model (hooks registered inside _load_model / _register_hooks)
    model = VLMAuditModel(config)

    # Load dataset — processor converts PIL → tensor
    dataset = Flickr30kDataset(config, processor=model.processor)

    # Extractors
    attn_extractor   = AttentionExtractor(config, model.patch_grid, config_image_size(model))
    gradcam_extractor = GradCAMExtractor(model, config, config_image_size(model))

    fig, axes = plt.subplots(N_IMAGES, 3, figsize=(14, 5 * N_IMAGES))
    col_titles = ["Original + GT boxes", f"Attention  (layer {LAYER})", f"Grad-CAM  (layer {LAYER})"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=10, fontweight="bold")

    for row, idx in enumerate(range(N_IMAGES)):
        sample   = dataset[idx]
        image_t  = sample["image"]          # (C, H, W)
        caption  = sample["caption"]
        boxes    = sample["boxes"]
        img_h, img_w = sample["image_size"]

        images_batch   = image_t.unsqueeze(0)     # (1, C, H, W)
        captions_batch = [caption]
        image_np       = to_numpy_image(image_t)

        # --- Attention map ---
        with torch.no_grad():
            model.forward(images_batch, captions_batch)
        attn_cache    = model.get_attention_cache()
        attn_heatmaps = attn_extractor.extract(attn_cache)
        attn_map      = attn_heatmaps[LAYER][0]   # (H_img, W_img)
        model.clear_cache()

        # --- Grad-CAM map ---
        gradcam_heatmaps = gradcam_extractor.compute(images_batch, captions_batch)
        gradcam_map      = gradcam_heatmaps[LAYER][0]   # (H_img, W_img)

        # --- Plot ---
        draw_boxes(axes[row][0], image_np, boxes, caption)
        overlay_heatmap(axes[row][1], image_np, attn_map,    f"Attention  — {sample['filename']}")
        overlay_heatmap(axes[row][2], image_np, gradcam_map, f"Grad-CAM  — {sample['filename']}")

    plt.tight_layout()
    plt.show()


def config_image_size(model: VLMAuditModel):
    """Derive (H, W) from the model's expected input resolution."""
    ph, pw = model.patch_grid   # e.g. (24, 24)
    patch_size = 16             # ViT-B/16
    return (ph * patch_size, pw * patch_size)


if __name__ == "__main__":
    main()

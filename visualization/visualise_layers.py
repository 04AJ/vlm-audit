"""
Layer heatmap visualisation.

Generates a 3-panel overview (original + GT boxes | attention | Grad-CAM)
for N_IMAGES samples across selected layers, saving one PDF per image to
results/experimentation/.

Run from repo root:
    python -m visualization.visualise_layers                  # all layers
    python -m visualization.visualise_layers --layers 6 7 8   # specific layers
"""

import argparse
import os
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from core.config import AuditConfig
from core.model import VLMAuditModel
from data.flickr30k import Flickr30kDataset
from extraction.attention import AttentionExtractor
from extraction.gradcam import GradCAMExtractor
from visualization.visualise_maps import (
    to_numpy_image,
    overlay_heatmap,
    draw_boxes,
    config_image_size,
)

N_IMAGES    = 3
N_SEARCH    = 200
SAVE_DIR    = "results/experimentation"

_REPO_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRATCH_DATA   = os.environ.get("SCRATCH_DATA_DIR", "/scratch/comp-646-g9/data")
ANNOTATIONS_DIR = os.path.join(_SCRATCH_DATA, "Annotations")
SENTENCES_DIR   = os.path.join(_SCRATCH_DATA, "Sentences")
SPLIT_FILE      = os.path.join(_REPO_ROOT, "data", "test.txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layers", type=int, nargs="+", default=None,
        metavar="L",
        help="Layer indices to visualise (e.g. --layers 6 7 8). Defaults to all layers.",
    )
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)

    target_layers = args.layers or []
    config = AuditConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_samples=N_SEARCH,
        target_layers=target_layers,
        annotations_dir=ANNOTATIONS_DIR,
        sentences_dir=SENTENCES_DIR,
        split_file=SPLIT_FILE,
    )

    model   = VLMAuditModel(config)
    dataset = Flickr30kDataset(config, processor=model.processor)

    img_size          = config_image_size(model)
    attn_extractor    = AttentionExtractor(config, model.patch_grid, img_size)
    gradcam_extractor = GradCAMExtractor(model, config, img_size)

    n_layers     = model.num_layers
    layers_to_show = args.layers if args.layers else list(range(n_layers))
    print(f"[vis] Visualising layers: {layers_to_show}")

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
        attn_cache       = model.get_attention_cache()
        attn_heatmaps    = attn_extractor.extract(attn_cache)
        model.clear_cache()

        gradcam_heatmaps = gradcam_extractor.compute(images_batch, captions_batch)

        layer_tag = "_".join(str(l) for l in layers_to_show) if args.layers else "all"
        save_path = os.path.join(SAVE_DIR, f"fig{idx:02d}_layers_{layer_tag}.pdf")
        with PdfPages(save_path) as pdf:
            for layer in layers_to_show:
                if layer not in attn_heatmaps or layer not in gradcam_heatmaps:
                    print(f"[vis] Layer {layer} missing for image {idx + 1} — skipping.")
                    continue

                attn_map    = attn_heatmaps[layer][0]
                gradcam_map = gradcam_heatmaps[layer][0]

                fig, axes = plt.subplots(1, 3, figsize=(14, 5))
                fig.suptitle(f"Layer {layer} — {sample['filename']}", fontsize=11, fontweight="bold")

                axes[0].set_title("Original + GT boxes", fontsize=10)
                axes[1].set_title(f"Attention  (layer {layer})", fontsize=10)
                axes[2].set_title(f"Grad-CAM  (layer {layer})", fontsize=10)

                draw_boxes(axes[0], image_np, boxes, caption,
                           original_size=sample["image_size"])
                overlay_heatmap(axes[1], image_np, attn_map,
                                f"Attention — {sample['filename']}")
                overlay_heatmap(axes[2], image_np, gradcam_map,
                                f"Grad-CAM — {sample['filename']}")

                plt.tight_layout()
                pdf.savefig(fig, bbox_inches="tight", dpi=150)
                plt.close(fig)

        print(f"[vis] Saved → {save_path}")

    print(f"[vis] Done. Figures saved to {SAVE_DIR}/")


if __name__ == "__main__":
    main()

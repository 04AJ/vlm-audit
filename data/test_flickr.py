"""
Smoke test for data/flickr30k.py
Run from repo root with venv active:
    python -m data.test_flickr
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from core.config import AuditConfig
from data.flickr30k import Flickr30kDataset, get_dataloader

_REPO_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRATCH_DATA   = os.environ["DATA_DIR"]  # resolved by core.config via config.sh
ANNOTATIONS_DIR = os.path.join(_SCRATCH_DATA, "Annotations")
SENTENCES_DIR   = os.path.join(_SCRATCH_DATA, "Sentences")
SPLIT_FILE      = os.path.join(_REPO_ROOT, "data", "test.txt")


def _make_cfg(n: int) -> AuditConfig:
    return AuditConfig(
        max_samples=n,
        dataset_split="test",
        annotations_dir=ANNOTATIONS_DIR,
        sentences_dir=SENTENCES_DIR,
        split_file=SPLIT_FILE,
    )


def test_dataset():
    print("=== Dataset (no processor) ===")
    ds = Flickr30kDataset(config=_make_cfg(3))

    assert len(ds) == 3, f"Expected 3 samples, got {len(ds)}"
    print(f"Length:   {len(ds)}  OK")

    sample = ds[0]
    assert set(sample.keys()) == {"image", "caption", "boxes", "filename", "image_size"}
    print(f"Keys:     {list(sample.keys())}  OK")

    assert isinstance(sample["caption"], str)
    print(f"Caption:  {sample['caption'][:80]}")

    assert isinstance(sample["image_size"], tuple) and len(sample["image_size"]) == 2
    print(f"ImgSize:  {sample['image_size']}  (H, W)")

    print(f"Filename: {sample['filename']}")

    for box_dict in sample["boxes"]:
        assert "label" in box_dict and "box" in box_dict
        x1, y1, x2, y2 = box_dict["box"]
        h, w = sample["image_size"]
        assert x2 <= w and y2 <= h, "box coords should be pixel space"
    print(f"Boxes:    {len(sample['boxes'])} entities  OK")
    if sample["boxes"]:
        print(f"  eg: {sample['boxes'][0]}")


def test_dataloader():
    print("\n=== DataLoader (batch_size=3) ===")
    loader = get_dataloader(_make_cfg(3), processor=None, batch_size=3, num_workers=0)

    batch = next(iter(loader))
    assert len(batch["image"]) == 3
    print(f"Images:   {len(batch['image'])} items  OK")

    assert len(batch["caption"]) == 3
    print(f"Captions: {len(batch['caption'])} strings  OK")

    counts = [len(b) for b in batch["boxes"]]
    print(f"Boxes per image: {counts}  (variable length — collate OK)")


def visualise(n: int = 3):
    """Show the first n samples with caption and bounding boxes labelled by class."""
    ds = Flickr30kDataset(config=_make_cfg(n))

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, idx in zip(axes, range(n)):
        sample = ds[idx]
        print(f"\n--- {sample['filename']} ---")
        print(f"Caption: {sample['caption']}")
        print(f"Boxes:   {sample['boxes'][:3]}")

        ax.imshow(sample["image"])
        ax.axis("off")

        seen: dict = {}
        for box_dict in sample["boxes"]:
            label = box_dict["label"]
            if label not in seen:
                seen[label] = colours[len(seen) % len(colours)]
            colour = seen[label]

            x1, y1, x2, y2 = box_dict["box"]
            ax.add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=colour, facecolor="none",
            ))
            ax.text(
                x1, y1 - 4, label,
                fontsize=7, color=colour,
                bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none"),
            )

        caption = sample["caption"]
        wrapped = "\n".join(caption[i:i+60] for i in range(0, len(caption), 60))
        ax.set_title(wrapped, fontsize=8)

    save_dir = os.path.join(_REPO_ROOT, "results", "test_flickr")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "visualise.pdf")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    print(f"[vis] Saved → {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    test_dataset()
    test_dataloader()
    print("\nAll checks passed.")
    print("\nSaving visualisation...")
    visualise(n=3)

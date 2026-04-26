"""
Per-word attention figure for the VLM-Audit paper.

Figure 1: 1 image × one column per entity noun phrase, attention only.

Run from repo root:
    python -m visualization.visualise_per_word
"""

import os
import re
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from core.config import AuditConfig
from core.model import VLMAuditModel
from data.flickr30k import Flickr30kDataset
from visualization.visualise_maps import (
    ANNOTATIONS_DIR, SENTENCES_DIR, SPLIT_FILE, LAYER, ALPHA,
    to_numpy_image, overlay_heatmap, config_image_size,
)

# ------------------------------------------------------------------ config --

N_SEARCH  = 200
MAX_NOUNS = 4

# ------------------------------------------------------------------ helpers --

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
    """Return token positions (1-based) in the caption that correspond to phrase."""
    cap_ids    = tokenizer(caption, add_special_tokens=False)["input_ids"]
    phrase_ids = tokenizer(phrase,  add_special_tokens=False)["input_ids"]
    for i in range(len(cap_ids) - len(phrase_ids) + 1):
        if cap_ids[i : i + len(phrase_ids)] == phrase_ids:
            return [i + 1 + k for k in range(len(phrase_ids))]
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

    attn_weights  : (heads, T, N) for a single sample
    token_indices : token positions to average over; falls back to all tokens
    """
    fused = (attn_weights.mean(dim=0) if head_fusion == "mean"
             else attn_weights.max(dim=0).values)
    indices = token_indices if token_indices else list(range(fused.shape[0]))
    pooled  = fused[indices].mean(dim=0)
    pooled  = pooled[1:]                    # drop CLS patch token
    ph, pw  = patch_grid
    grid    = pooled.view(ph, pw).unsqueeze(0).unsqueeze(0).float()
    heatmap = F.interpolate(grid, image_size, mode="bilinear",
                            align_corners=False).squeeze()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap


# ------------------------------------------------------------------- figure --

def plot_attention_per_word(
    model: VLMAuditModel,
    dataset: Flickr30kDataset,
    layer: int = LAYER,
    min_nouns: int = 3,
    n_examples: int = 10,
    save_dir: str = "results",
) -> None:
    """
    Scan the dataset for images with at least min_nouns entity phrases and save
    a separate PDF per image showing one attention heatmap column per phrase.
    """
    tokenizer = model.processor.tokenizer
    found = 0

    for idx in range(len(dataset)):
        if found >= n_examples:
            break

        s           = dataset[idx]
        phrase_map  = _load_entity_phrases(SENTENCES_DIR, s["filename"])
        seen_labels = set()
        entities    = []

        for box in s["boxes"]:
            label = box["label"]
            if label in seen_labels:
                continue
            phrase        = phrase_map.get(box["phrase"], label)
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

        raw_weights = attn_cache[layer][0]    # (heads, T, N)
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

        wrapped = "\n".join(caption[i:i+80] for i in range(0, len(caption), 80))
        fig.suptitle(f"{wrapped}\n\nlayer {layer}", fontsize=9, y=1.04, ha="center")
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"fig1_attention_per_word_{found + 1:02d}.pdf")
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[vis] Saved → {save_path}")
        plt.close(fig)
        found += 1

    if found == 0:
        print(f"[vis] No samples with {min_nouns}+ nouns found in first {len(dataset)} images.")
    else:
        print(f"[vis] Figure 1: saved {found} examples to {save_dir}/")


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

    os.makedirs("results/wordwise", exist_ok=True)
    plot_attention_per_word(
        model, dataset, layer=LAYER, min_nouns=3, n_examples=10,
        save_dir="results/wordwise",
    )


if __name__ == "__main__":
    main()

"""
Smoke test for extraction/attention.py
Run from repo root with venv active:
    python -m extraction.test_attention
"""

import torch
from core.config import AuditConfig
from extraction.attention import AttentionExtractor

# BLIP-base constants
BATCH       = 2
NUM_HEADS   = 12
SEQ_LEN     = 10   # text tokens
NUM_PATCHES = 576  # 24x24
PATCH_GRID  = (24, 24)
IMAGE_SIZE  = (384, 384)


def make_cache(layers: list[int]) -> dict:
    """Fake attention cache with random values in [0, 1] (post-softmax range)."""
    return {
        i: torch.rand(BATCH, NUM_HEADS, SEQ_LEN, NUM_PATCHES)
        for i in layers
    }


def test_shapes():
    print("=== Shape checks ===")
    cfg       = AuditConfig(attention_head_fusion="mean")
    extractor = AttentionExtractor(cfg, PATCH_GRID, IMAGE_SIZE)
    cache     = make_cache(layers=[0, 5, 11])

    heatmaps = extractor.extract(cache)

    assert set(heatmaps.keys()) == {0, 5, 11}, "Wrong layer keys"
    for layer_idx, hmap in heatmaps.items():
        assert hmap.shape == (BATCH, *IMAGE_SIZE), (
            f"Layer {layer_idx}: expected {(BATCH, *IMAGE_SIZE)}, got {hmap.shape}"
        )
        print(f"  Layer {layer_idx}: {tuple(hmap.shape)}  OK")


def test_normalisation():
    print("\n=== Normalisation [0, 1] ===")
    cfg       = AuditConfig(attention_head_fusion="mean")
    extractor = AttentionExtractor(cfg, PATCH_GRID, IMAGE_SIZE)
    cache     = make_cache(layers=[11])

    hmap = extractor.extract(cache)[11]

    assert hmap.min() >= 0.0, f"Min below 0: {hmap.min()}"
    assert hmap.max() <= 1.0, f"Max above 1: {hmap.max()}"
    print(f"  min={hmap.min():.4f}  max={hmap.max():.4f}  OK")


def test_target_layers():
    print("\n=== target_layers filter ===")
    cfg       = AuditConfig(attention_head_fusion="mean", target_layers=[11])
    extractor = AttentionExtractor(cfg, PATCH_GRID, IMAGE_SIZE)
    cache     = make_cache(layers=[0, 5, 11])   # cache has 3 layers

    heatmaps = extractor.extract(cache)

    assert list(heatmaps.keys()) == [11], f"Expected only [11], got {list(heatmaps.keys())}"
    print(f"  Returned layers: {list(heatmaps.keys())}  OK")


def test_head_fusion_strategies():
    print("\n=== Head fusion strategies ===")
    cache = make_cache(layers=[11])

    for strategy in ("mean", "max", "min"):
        cfg       = AuditConfig(attention_head_fusion=strategy)
        extractor = AttentionExtractor(cfg, PATCH_GRID, IMAGE_SIZE)
        hmap      = extractor.extract(cache)[11]
        assert hmap.shape == (BATCH, *IMAGE_SIZE)
        print(f"  {strategy}: {tuple(hmap.shape)}  OK")


def test_flat_input():
    """Edge case: all patches attended equally — normalise must not produce NaN."""
    print("\n=== Flat input (all-equal attention) ===")
    cfg       = AuditConfig(attention_head_fusion="mean")
    extractor = AttentionExtractor(cfg, PATCH_GRID, IMAGE_SIZE)

    flat_cache = {11: torch.ones(BATCH, NUM_HEADS, SEQ_LEN, NUM_PATCHES)}
    hmap = extractor.extract(flat_cache)[11]

    assert not torch.isnan(hmap).any(), "NaN in flat-input heatmap"
    print(f"  No NaNs  OK  (values={hmap.unique().tolist()[:3]})")


def visualise():
    """
    Show heatmaps for all 12 layers using a structured random input.
    Each heatmap is overlaid on a grey placeholder image so spatial
    structure is visible even without a real image.
    """
    import matplotlib.pyplot as plt

    cfg       = AuditConfig(attention_head_fusion="mean")
    extractor = AttentionExtractor(cfg, PATCH_GRID, IMAGE_SIZE)

    # Structured input: attention peaks shift from top-left to bottom-right
    # across the 12 layers so the visualisation shows clear movement
    cache = {}
    for i in range(12):
        weights = torch.zeros(1, NUM_HEADS, SEQ_LEN, NUM_PATCHES)
        # Concentrate attention around a different patch for each layer
        peak = int((i / 11) * (NUM_PATCHES - 1))
        weights[:, :, :, peak] = 5.0
        weights = torch.softmax(weights, dim=-1)
        cache[i] = weights

    heatmaps = extractor.extract(cache)

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    fig.suptitle("AttentionExtractor output — layers 0–11 (random structured input)", fontsize=12)

    for layer_idx, ax in zip(range(12), axes.flat):
        hmap = heatmaps[layer_idx][0]          # take first item in batch → (H, W)
        im   = ax.imshow(hmap.numpy(), cmap="hot", vmin=0, vmax=1)
        ax.set_title(f"Layer {layer_idx}", fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_shapes()
    test_normalisation()
    test_target_layers()
    test_head_fusion_strategies()
    test_flat_input()
    print("\nAll checks passed.")
    print("\nOpening visualisation...")
    visualise()

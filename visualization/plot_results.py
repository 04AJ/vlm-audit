"""
Plot VLM-Audit best-layer results for Attention, Grad-CAM, and Hybrid α=0.25.

Loads two JSON files:
  - results_all_layers.json  : {"attention": {...}, "gradcam": {...}}
  - results_hybrid.json      : {"hybrid": [{"alpha": 0.25, ...}, ...]}

Produces a 2×2 figure:
  Top row    — Grounding    : Pointing Game Accuracy  |  Mean IoU
  Bottom row — Faithfulness : Sensitivity-n           |  SaCo AUC

Each subplot shows one bar per method at its single best layer.
The best layer index is annotated inside each bar.

Run from repo root:
    python -m visualization.plot_results
    python -m visualization.plot_results --all-layers results/results_all_layers.json \
                                         --hybrid     results/results_hybrid.json \
                                         --output     results/figures/best_layer_comparison.pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HYBRID_ALPHA = 0.25

METHODS = [
    ("Attention",        "#4878d0"),
    ("Grad-CAM",         "#ee854a"),
    (f"Hybrid α={HYBRID_ALPHA}", "#6acc65"),
]

PLOTS = [
    # (subplot title, section key, metric key, y-label)
    ("Pointing Game Accuracy", "grounding",    "pointing_game_accuracy", "Accuracy"),
    ("Mean IoU",               "grounding",    "mean_iou",               "IoU"),
    ("Sensitivity-n",          "faithfulness", "sensitivity_n_score",    "Score"),
    ("SaCo AUC",               "faithfulness", "saco_auc",               "AUC"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot best-layer VLM-Audit results.")
    parser.add_argument(
        "--all-layers",
        default="results/results_all_layers.json",
        help="Path to results_all_layers.json (attention + gradcam data).",
    )
    parser.add_argument(
        "--hybrid",
        default="results/results_hybrid.json",
        help="Path to results_hybrid.json (hybrid alpha sweep data).",
    )
    parser.add_argument(
        "--output",
        default="results/figures/best_layer_comparison.pdf",
        help="Output PDF path.",
    )
    return parser.parse_args()


def best(entries: list[dict], key: str) -> tuple[float, int]:
    """Return (best_value, best_layer_idx) for a metric key across all layers."""
    best_entry = max(entries, key=lambda e: e[key])
    return best_entry[key], best_entry["layer_idx"]


def load_method_data(all_layers_path: str, hybrid_path: str) -> list[tuple[str, dict]]:
    """
    Returns list of (method_label, {section: entries}) for the three methods.
    """
    with open(all_layers_path) as f:
        al = json.load(f)
    with open(hybrid_path) as f:
        hy = json.load(f)

    hybrid_entry = next(
        e for e in hy["hybrid"] if abs(e["alpha"] - HYBRID_ALPHA) < 1e-6
    )

    return [
        ("Attention",                  {"grounding": al["attention"]["grounding"],
                                        "faithfulness": al["attention"]["faithfulness"]}),
        ("Grad-CAM",                   {"grounding": al["gradcam"]["grounding"],
                                        "faithfulness": al["gradcam"]["faithfulness"]}),
        (f"Hybrid α={HYBRID_ALPHA}",   {"grounding": hybrid_entry["grounding"],
                                        "faithfulness": hybrid_entry["faithfulness"]}),
    ]


def main() -> None:
    args = parse_args()
    methods = load_method_data(args.all_layers, args.hybrid)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("VLM-Audit: Best-Layer Method Comparison", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    x      = np.arange(len(methods))
    width  = 0.5
    colors = [c for _, c in METHODS]

    for ax, (title, section, key, ylabel) in zip(axes, PLOTS):
        vals   = []
        layers = []
        for _, data in methods:
            v, l = best(data[section], key)
            vals.append(v)
            layers.append(l)

        bars = ax.bar(x, vals, width, color=colors, edgecolor="white", linewidth=0.8)

        # Annotate each bar with its best layer
        for bar, layer_idx in zip(bars, layers):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 0.5,
                f"L{layer_idx}",
                ha="center", va="center",
                fontsize=9, fontweight="bold", color="white",
            )

        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([label for label, _ in METHODS], fontsize=9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}"))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_ylim(0, max(vals) * 1.18)
        ax.spines[["top", "right"]].set_visible(False)

        # Value label on top of each bar
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.02,
                f"{v:.3f}",
                ha="center", va="bottom",
                fontsize=8, color="#333333",
            )

    # Group labels
    axes[0].set_title("Grounding — " + axes[0].get_title(), fontsize=11, fontweight="bold", pad=6)
    axes[1].set_title("Grounding — " + axes[1].get_title(), fontsize=11, fontweight="bold", pad=6)
    axes[2].set_title("Faithfulness — " + axes[2].get_title(), fontsize=11, fontweight="bold", pad=6)
    axes[3].set_title("Faithfulness — " + axes[3].get_title(), fontsize=11, fontweight="bold", pad=6)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=180)
    print(f"[plot] Saved → {output_path}")


if __name__ == "__main__":
    main()

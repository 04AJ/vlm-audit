import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot VLM-Audit results JSON.")
    parser.add_argument("results_path", help="Path to a results_*.json file.")
    parser.add_argument(
        "--output",
        default="visualization/results_barplot.pdf",
        help="Output PDF path for the generated figure.",
    )
    return parser.parse_args()


def load_methods(data: dict) -> tuple[list[int], list[tuple[str, dict]]]:
    layers = [entry["layer_idx"] for entry in data["grounding"]]
    methods: list[tuple[str, dict]] = [
        (
            "Attention",
            {
                "grounding": data["grounding"],
                "faithfulness": data["faithfulness"],
            },
        ),
        (
            "Grad-CAM",
            {
                "grounding": data["grounding_grad"],
                "faithfulness": data["faithfulness_grad"],
            },
        ),
    ]

    for hybrid_entry in data.get("hybrid", []):
        methods.append(
            (
                f"Hybrid α={hybrid_entry['alpha']:.2f}",
                {
                    "grounding": hybrid_entry["grounding"],
                    "faithfulness": hybrid_entry["faithfulness"],
                },
            )
        )

    return layers, methods


def metric_values(entries: list[dict], key: str, layers: list[int]) -> list[float]:
    by_layer = {entry["layer_idx"]: entry[key] for entry in entries}
    return [by_layer[layer] for layer in layers]


def main() -> None:
    args = parse_args()
    with open(args.results_path) as f:
        data = json.load(f)

    layers, methods = load_methods(data)
    x = np.arange(len(layers))
    width = 0.8 / max(len(methods), 1)
    labels = [f"Layer {layer}" for layer in layers]

    plots = [
        ("Pointing Game Accuracy", "grounding", "pointing_game_accuracy"),
        ("Mean IoU", "grounding", "mean_iou"),
        ("Sensitivity-n", "faithfulness", "sensitivity_n_score"),
        ("SaCo AUC", "faithfulness", "saco_auc"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("VLM-Audit Method Comparison", fontsize=15, fontweight="bold")
    axes = axes.flatten()
    cmap = plt.get_cmap("tab10")

    for ax, (title, section, key) in zip(axes, plots):
        for idx, (label, method) in enumerate(methods):
            offset = (idx - (len(methods) - 1) / 2) * width
            ax.bar(
                x + offset,
                metric_values(method[section], key, layers),
                width,
                label=label,
                color=cmap(idx % 10),
            )

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    axes[0].set_ylabel("Score")
    axes[2].set_ylabel("Score")
    axes[2].set_xlabel("Layer")
    axes[3].set_xlabel("Layer")

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=min(len(methods), 3))

    plt.tight_layout(rect=(0, 0, 1, 0.92))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

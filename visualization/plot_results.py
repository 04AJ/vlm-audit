import json
import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = "results/results_20260331_063826.json"

with open(RESULTS_PATH) as f:
    data = json.load(f)

layers = [e["layer_idx"] for e in data["grounding"]]
x = np.arange(len(layers))
width = 0.18
labels = [f"Layer {l}" for l in layers]

def get(section, key):
    return [e[key] for e in data[section]]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("VLM Audit — Attention vs GradCAM", fontsize=14, fontweight="bold")

# --- Grounding ---
ax = axes[0]
ax.bar(x - width*1.5, get("grounding", "pointing_game_accuracy"), width, label="Attn — Pointing Game Acc", color="#4C72B0")
ax.bar(x - width*0.5, get("grounding_grad", "pointing_game_accuracy"), width, label="Grad — Pointing Game Acc", color="#4C72B0", alpha=0.5, hatch="//")
ax.bar(x + width*0.5, get("grounding", "mean_iou"), width, label="Attn — Mean IoU", color="#DD8452")
ax.bar(x + width*1.5, get("grounding_grad", "mean_iou"), width, label="Grad — Mean IoU", color="#DD8452", alpha=0.5, hatch="//")

ax.set_yscale("log")
ax.set_title("Grounding")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Score (log scale)")
ax.legend(fontsize=8)
ax.grid(axis="y", linestyle="--", alpha=0.4)

# --- Faithfulness ---
ax = axes[1]
ax.bar(x - width*1.5, get("faithfulness", "saco_auc"), width, label="Attn — SaCo AUC", color="#55A868")
ax.bar(x - width*0.5, get("faithfulness_grad", "saco_auc"), width, label="Grad — SaCo AUC", color="#55A868", alpha=0.5, hatch="//")
ax.bar(x + width*0.5, get("faithfulness", "sensitivity_n_score"), width, label="Attn — Sensitivity-N", color="#C44E52")
ax.bar(x + width*1.5, get("faithfulness_grad", "sensitivity_n_score"), width, label="Grad — Sensitivity-N", color="#C44E52", alpha=0.5, hatch="//")

ax.set_yscale("log")
ax.set_title("Faithfulness")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Score (log scale)")
ax.legend(fontsize=8)
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("visualization/results_barplot.pdf", bbox_inches="tight")
print("Saved: visualization/results_barplot.pdf")
plt.show()

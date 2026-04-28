"""
AuditConfig
-----------
Single source of truth for all hyperparameters across the pipeline.
All three modules import this — add fields here rather than hard-coding values.
"""

import os
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_data_dir() -> str:
    if "DATA_DIR" in os.environ:
        return os.environ["DATA_DIR"]
    config_sh = os.path.join(_REPO_ROOT, "scripts", "config.sh")
    if os.path.exists(config_sh):
        result = subprocess.run(
            ["bash", "-c", f"source {config_sh} && echo $DATA_DIR"],
            capture_output=True, text=True,
        )
        data_dir = result.stdout.strip()
        if data_dir:
            os.environ["DATA_DIR"] = data_dir
            return data_dir
    raise EnvironmentError(
        "DATA_DIR is not set and scripts/config.sh could not be sourced. "
        "Update SCRATCH_DIR in scripts/config.sh then run: source scripts/config.sh"
    )


_SCRATCH_DATA = _resolve_data_dir()


@dataclass
class AuditConfig:
    # --- Model ---
    model_name: str = "Salesforce/blip-itm-base-coco"
    device: str = "cpu"                    # "cuda" | "cpu"

    # --- Which layers to audit ---
    # Indices into the cross-attention stack (0-based).
    # Leaving empty means "all layers".
    target_layers: List[int] = field(default_factory=list)

    # --- Data ---
    dataset_name: str = "nlphuji/flickr30k"
    dataset_split: str = "test"
    max_samples: Optional[int] = None       # None = use full split
    batch_size: int = 8
    annotations_dir: Optional[str] = field(default_factory=lambda: os.path.join(_SCRATCH_DATA, "Annotations"))
    sentences_dir: Optional[str] = field(default_factory=lambda: os.path.join(_SCRATCH_DATA, "Sentences"))
    split_file: Optional[str] = field(default_factory=lambda: os.path.join(_REPO_ROOT, "data", "test.txt"))

    # --- Extraction ---
    attention_head_fusion: str = "mean"     # "mean" | "max" | "min"
    gradcam_relu: bool = True
    hybrid_alphas: List[float] = field(default_factory=list)  # attention weight in alpha*attn + (1-alpha)*grad

    # --- Evaluation ---
    iou_threshold: float = 0.5
    sensitivity_n: int = 10                 # n for Sensitivity-n masking
    saco_steps: int = 20                    # steps for SaCo curve

    # --- Output ---
    output_dir: str = "results"
    save_heatmaps: bool = True
    hybrid_only: bool = False

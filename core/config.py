"""
AuditConfig
-----------
Single source of truth for all hyperparameters across the pipeline.
All three modules import this — add fields here rather than hard-coding values.
"""

from dataclasses import dataclass, field
from typing import List, Optional


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
    annotations_dir: Optional[str] = None  # path to Flickr30k Entities XML folder
    sentences_dir: Optional[str] = None   # path to Flickr30k Entities sentences folder
    split_file: Optional[str] = None       # path to txt file of image IDs for the split

    # --- Extraction ---
    attention_head_fusion: str = "mean"     # "mean" | "max" | "min"
    gradcam_relu: bool = True

    # --- Evaluation ---
    iou_threshold: float = 0.5
    sensitivity_n: int = 10                 # n for Sensitivity-n masking
    saco_steps: int = 20                    # steps for SaCo curve

    # --- Output ---
    output_dir: str = "results"
    save_heatmaps: bool = True

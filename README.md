# VLM-Audit

A diagnostic toolkit for auditing cross-attention faithfulness in Vision-Language Models (VLMs). Rather than relying on qualitative heatmap visualisations, VLM-Audit quantitatively measures how well a model's internal attention maps reflect its actual decision-making process.

## Motivation

Modern VLMs (e.g. BLIP) use cross-attention to align image regions with text tokens, but raw attention weights are often uninterpretable or misleading. This toolkit evaluates **grounding accuracy** (do the highlighted regions match ground-truth object locations?) and **faithfulness** (does removing those regions actually hurt the model's confidence?) across selected transformer layers.

## Pipeline

```
Input (Image + Caption)
        │
        ▼
┌───────────────┐
│  Core Module  │  Loads the VLM, registers hooks on cross-attention layers
└───────┬───────┘
        │  attention cache  Dict[layer → Tensor]
        ▼
┌──────────────────┐
│ Extraction Layer │  Converts raw weights → spatial heatmaps (Attention + Grad-CAM)
└────────┬─────────┘
         │  heatmaps  Dict[layer → Tensor (B, H, W)]
         ▼
┌──────────────────┐
│ Evaluation Suite │  Scores heatmaps against Flickr30k bounding-box annotations
└────────┬─────────┘
         │
         ▼
Output: reliability scorecard (IoU, Pointing Game, Sensitivity-n, SaCo)
```

## Repository Structure

```
vlm-audit/
├── core/                   # VLM wrapper and shared configuration
│   ├── config.py           #   AuditConfig — all hyperparameters in one place
│   └── model.py            #   VLMAuditModel — loads BLIP, registers attention hooks
│
├── data/                   # Dataset loading
│   └── flickr30k.py        #   Flickr30k Entities via HuggingFace datasets
│
├── extraction/             # Heatmap extraction from attention internals
│   ├── attention.py        #   Raw attention weights → upsampled spatial heatmaps
│   └── gradcam.py          #   Grad-CAM over cross-attention layers
│
├── evaluation/             # Quantitative scoring
│   ├── grounding.py        #   Pointing Game Accuracy + IoU vs GT bounding boxes
│   ├── faithfulness.py     #   Sensitivity-n and SaCo via iterative pixel masking
│   └── results.py          #   EvalResults dataclass
│
├── scripts/
│   └── run_audit.py        # Main entry point — wires all modules together
│
├── requirements.txt
└── pyproject.toml
```

## Module Interfaces

Each module has a narrow, explicit interface so they can be developed independently.

| Producer | Output | Consumer |
|---|---|---|
| `VLMAuditModel.get_attention_cache()` | `Dict[int, Tensor (B, heads, T, N)]` | `AttentionExtractor`, `GradCAMExtractor` |
| `AttentionExtractor.extract()` | `Dict[int, Tensor (B, H, W)]` | `GroundingEvaluator`, `FaithfulnessEvaluator` |
| `GradCAMExtractor.compute()` | `Dict[int, Tensor (B, H, W)]` | `GroundingEvaluator`, `FaithfulnessEvaluator` |
| `GroundingEvaluator.compute()` | `List[LayerGroundingResult]` | `EvalResults` |
| `FaithfulnessEvaluator.compute()` | `List[LayerFaithfulnessResult]` | `EvalResults` |

`AuditConfig` (`core/config.py`) is the single source of truth for all hyperparameters and is imported by every module.

## Dataset

[Flickr30k Entities](https://huggingface.co/datasets/nlphuji/flickr30k) is loaded automatically via HuggingFace `datasets`. It provides:
- Images paired with 5 reference captions each
- Bounding-box annotations per noun phrase — used by the Grounding evaluator

## Evaluation Metrics

**Grounding**
- *Pointing Game Accuracy* — is the peak heatmap pixel inside a ground-truth bounding box?
- *Mean IoU* — overlap between the thresholded heatmap and the union of GT boxes

**Faithfulness**
- *Sensitivity-n* — average confidence drop when the top-n% most salient pixels are masked
- *SaCo AUC* — area under the confidence-decay curve as pixels are masked from most to least salient

Both metrics are computed per layer, allowing identification of which transformer layers produce the most grounded and faithful explanations.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python scripts/run_audit.py
```

```bash
python scripts/run_audit.py \
  --model Salesforce/blip-itm-base-coco \
  --layers 6 7 8 \
  --max-samples 500 \
  --output-dir results/run_01
```


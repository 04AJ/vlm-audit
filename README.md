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

[Flickr30k Entities](https://github.com/BryanPlummer/flickr30k_entities) — 31k images, each with 5 reference captions and bounding-box annotations per noun phrase.

| Source | What it provides |
|---|---|
| HuggingFace `nlphuji/flickr30k` | Images (PIL) |
| Local `data/Sentences/` | Captions with entity class tags |
| Local `data/Annotations/` | Bounding boxes per entity (XML) |

## Evaluation Metrics

**Grounding**
- *Pointing Game Accuracy* — is the peak heatmap pixel inside a ground-truth bounding box?
- *Mean IoU* — overlap between the thresholded heatmap and the union of GT boxes

**Faithfulness**
- *Sensitivity-n* — average confidence drop when the top-n% most salient pixels are masked
- *SaCo AUC* — area under the confidence-decay curve as pixels are masked from most to least salient

Both metrics are computed per layer, allowing identification of which transformer layers produce the most grounded and faithful explanations.

## Setup

### 1. Create and activate a virtual environment

```bash
# Create (run once)
python -m venv .venv
```

Activate — pick the command for your terminal:

```bash
# Git Bash / bash
source .venv/Scripts/activate

# PowerShell
.venv\Scripts\Activate.ps1

# Command Prompt
.venv\Scripts\activate.bat
```

You should see `(.venv)` in your prompt.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Data

The pipeline uses [Flickr30k Entities](https://github.com/BryanPlummer/flickr30k_entities) for bounding-box annotations and captions, and HuggingFace for images.

### Setting up annotation files

1. Download the annotations archive from the [flickr30k_entities releases](https://github.com/BryanPlummer/flickr30k_entities/tree/master) page
2. Extract the zip
3. Copy the `Annotations` and `Sentences` folders into `data/`:

```
vlm-audit/
└── data/
    ├── Annotations/       ← XML bounding-box files (one per image)
    ├── Sentences/         ← Caption files with entity tags (one per image)
    └── test.txt           ← List of test-split image IDs
```

> `Annotations/` and `Sentences/` are git-ignored and must be set up locally by each team member.

### Testing the data loader

With your venv active and annotation files in place, run:

```bash
python -m data.test_flickr
```

This loads 3 images from the test split, checks captions and bounding boxes, and opens a matplotlib window showing each image with ground-truth boxes labelled by object class (e.g. `people`, `clothing`, `vehicles`).

## Usage

### 1. Activate the virtual environment

```bash
# Git Bash / bash
source .venv/Scripts/activate

# PowerShell
.venv\Scripts\Activate.ps1

# Command Prompt
.venv\Scripts\activate.bat
```

### 2. Run the audit

**CPU — quick test** (limited samples and SaCo steps to keep it fast):

```bash
python -m scripts.run_audit --max-samples 10 --layers 6 --saco-steps 3
```

**GPU — full run:**

```bash
python -m scripts.run_audit --gpu --layers 9 10 11 --max-samples 500
```

Results are saved to `results/results_<timestamp>.json`.

### All flags

```bash
python -m scripts.run_audit \
  --model Salesforce/blip-itm-base-coco \
  --gpu \
  --layers 9 10 11 \
  --max-samples 500 \
  --batch-size 8 \
  --iou-threshold 0.5 \
  --sensitivity-n 10 \
  --saco-steps 20 \
  --output-dir results/run_01
```


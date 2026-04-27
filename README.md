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
│   ├── config.sh           # Centralised path configuration
│   ├── conda_setup.sh      # Environment creation and package installation
│   └── run_audit.sh        # SLURM job script for the audit pipeline
│
├── run_audit.py            # Main entry point — wires all modules together
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

### 1. Create and activate a virtual environment (outside cluster environment)

```bash
# Create (run once)
python3 -m venv .venv
```

Activate — pick the command for your terminal:

```bash
# macOS / Linux (bash, zsh)
source .venv/bin/activate

# PowerShell
.venv\Scripts\Activate.ps1

# Command Prompt
.venv\Scripts\activate.bat
```

You should see `(.venv)` in your prompt.

### 2. Install dependencies (inside cluster environment)

```bash
module load Miniforge3/25.3.0-3
conda activate /scratch/comp-646-g9/vlm_audit_env
```

The environment already exists at `/scratch/comp-646-g9/vlm_audit_env` — do not run `--create`.

**Adding a new package:**

1. Add the package name to `requirements.txt` without a version number, e.g. `opencv-python`
2. Run the update job:
```bash
sbatch scripts/conda_setup.sh --update
```
3. Once installed, pin the exact version by running `pip show <package>` and updating the entry in `requirements.txt`, e.g.:
```bash
pip show opencv-python   # copy the Version field
# then update requirements.txt: opencv-python>=4.9.0
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

### Option 1 — SLURM batch job

Submit the audit as a SLURM job (runs unattended on a compute node):

```bash
sbatch scripts/run_audit.sh
```

Logs are written to `logs/audit<JOBID>.log` and `logs/audit<JOBID>.err`.

### Option 2 — Interactive shell

Request a GPU-enabled compute node and run the pipeline directly:

```bash
srun --pty --time=2:59:59 --gpus=1 --reservation=classroom --mem=64G $SHELL
```

Then activate the environment and run:

```bash
module load Miniforge3/25.3.0-3
conda activate /scratch/comp-646-g9/vlm_audit_env

python run_audit.py
```

```bash
python run_audit.py \
  --model Salesforce/blip-itm-base-coco \
  --layers 6 7 8 \
  --max-samples 500 \
  --output-dir results/run_01
```

Hybrid sweep across all 12 layers:

```bash
python run_audit.py \
  --layers 0 1 2 3 4 5 6 7 8 9 10 11 \
  --hybrid-alphas 0.25 0.5 0.75 \
  --max-samples 500 \
  --output-dir results/hybrid_all12
```

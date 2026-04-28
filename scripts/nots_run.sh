#!/bin/bash
#SBATCH --job-name=vlm-audit
#SBATCH --output=logs/vlm-audit-%j.out
#SBATCH --error=logs/vlm-audit-%j.err
#SBATCH --partition=commons
#SBATCH --gres=gpu:lovelace:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00

# ── paths ─────────────────────────────────────────────────────────────────────
SCRATCH_DIR=/scratch/$USER/vlm-audit
SHARED=/scratch/comp-646-g9

# ── annotation paths ──────────────────────────────────────────────────────────
ANNOTATIONS_DIR=$SHARED/data/Annotations
SENTENCES_DIR=$SHARED/data/Sentences
SPLIT_FILE=$SCRATCH_DIR/data/test.txt

# ── use our venv with torch 2.6.0+cu118 (compatible with CUDA driver 12.2) ────
source $SCRATCH_DIR/.venv/bin/activate
PYTHON=python

# ── run ───────────────────────────────────────────────────────────────────────
export HF_HOME=$SHARED/cache/huggingface
export HF_DATASETS_CACHE=$SCRATCH_DIR/cache/datasets
export PYTHONPATH=$SCRATCH_DIR

cd $SCRATCH_DIR
mkdir -p logs results/all-layers cache/datasets

# No --layers flag → defaults to all 12 layers (target_layers=[] means all).
# Faithfulness SaCo runs 20 masking passes × 12 layers × 500 samples = ~120k
# extra forward passes per method; 8hr wall is enough for both attn + GradCAM.
$PYTHON scripts/run_audit.py \
    --model Salesforce/blip-itm-base-coco \
    --device cuda \
    --max-samples 500 \
    --batch-size 8 \
    --annotations-dir "$ANNOTATIONS_DIR" \
    --sentences-dir   "$SENTENCES_DIR" \
    --split-file      "$SPLIT_FILE" \
    --output-dir results/all-layers

echo "Done. Results at $SCRATCH_DIR/results/all-layers/results.json"

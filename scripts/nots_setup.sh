#!/usr/bin/env bash
# One-time setup: rsync code to scratch, create venv, install deps.
# Run from your local machine:  bash scripts/nots_setup.sh
set -e

REMOTE=rgx1@nots.rice.edu
SCRATCH=/scratch/rgx1/vlm-audit
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> Syncing code to $REMOTE:$SCRATCH ..."
rsync -av --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='.venv' --exclude='results' \
    "$LOCAL_ROOT/" "$REMOTE:$SCRATCH/"

echo "==> Creating venv on NOTS ..."
ssh "$REMOTE" bash <<EOF
set -e
cd $SCRATCH
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Setup complete."
EOF

echo ""
echo "Done. To submit the job:"
echo "  ssh $REMOTE 'cd $SCRATCH && sbatch scripts/nots_run.sh'"
echo ""
echo "Make sure your annotations are at:"
echo "  $SCRATCH/data/Annotations/"
echo "  $SCRATCH/data/Sentences/"
echo "  $SCRATCH/data/test.txt"
echo "(or update the paths in scripts/nots_run.sh)"

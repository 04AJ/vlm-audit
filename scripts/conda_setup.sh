#!/bin/bash
#SBATCH --job-name=setup
#SBATCH --output=logs/setup_%j.log
#SBATCH --error=logs/setup_%j.err
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# Usage: sh scripts/mamba_setup.sh --create/update/recreate

source "$HOME/vlm-audit/scripts/config.sh"

mkdir -p $CACHE_DIR/{huggingface,pip,torch,conda_pkgs,matplotlib}

export HF_HOME="$CACHE_DIR/huggingface"
export PIP_CACHE_DIR="$CACHE_DIR/pip"
export TORCH_HOME="$CACHE_DIR/torch"
export MPLCONFIGDIR="$CACHE_DIR/matplotlib"
export CONDA_PKGS_DIRS="$CACHE_DIR/conda_pkgs"

set -e

module load Miniforge3/25.3.0-3
module load GCCcore/14.3.0
module load git/2.50.1

MODE=$1

if [[ -z "$MODE" ]]; then
    echo "Error: No mode specified."
    echo "Usage: sbatch mamba_setup.sh [--create | --update | --recreate]"
    exit 1
fi

eval "$(conda shell.bash hook)"

if [ "$MODE" == "--recreate" ]; then
    echo "Removing existing environment at $ENV_PATH..."
    rm -rf "$ENV_PATH"
fi

if [ "$MODE" == "--create" ] || [ "$MODE" == "--recreate" ]; then
    if [ -d "$ENV_PATH/conda-meta" ]; then
        echo "Environment exists. Skipping creation..."
    else
        echo "Creating mamba env at '$ENV_PATH'..."
        mamba create -p "$ENV_PATH" python=3.10 -y
    fi
fi

conda activate "$ENV_PATH"

echo "Installing PyTorch (cu121) ..."
pip install "torch>=2.6" torchvision --index-url https://download.pytorch.org/whl/cu124

echo "Installing requirements from requirements.txt..."
REQ_PATH="$HOME/vlm-audit/requirements.txt"

if [ -f "$REQ_PATH" ]; then
    pip install -r "$REQ_PATH"
else
    echo "Error: $REQ_PATH not found!"
    exit 1
fi

echo "Setup Task [$MODE] Complete."

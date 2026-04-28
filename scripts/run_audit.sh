#!/bin/bash
#SBATCH --job-name=audit
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --ntasks=1 
#SBATCH --output=logs/audit%j.log
#SBATCH --error=logs/audit%j.err
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:lovelace:1
#SBATCH --time=3:00:00


source "$HOME/vlm-audit/scripts/config.sh"

export HF_HOME="$CACHE_DIR/huggingface"
export PIP_CACHE_DIR="$CACHE_DIR/pip"
export TORCH_HOME="$CACHE_DIR/torch"
export MPLCONFIGDIR="$CACHE_DIR/matplotlib"
# DATA_DIR already exported by config.sh

module load Miniforge3/25.3.0-3
module load CUDA/12.4.0


eval "$(conda shell.bash hook)"
conda activate "$ENV_PATH"

cd "$PROJECT_DIR"

python run_audit.py --gpu "$@"

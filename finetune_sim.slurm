#!/bin/bash
#SBATCH --job-name=openpi-sim-lora
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e
set -o pipefail

############################
# Environment
############################
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export WANDB_ENTITY=jing-cao
export HF_DATASETS_DISABLE_PROGRESS_BARS=1

############################
# Paths
############################
HOME_SCRATCH=$HOME/orcd/scratch
PROJECT_DIR=$HOME/openpi

export WANDB_DIR=$HOME_SCRATCH/wandb
export HF_HOME=$HOME_SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME

# Point LeRobot at the dataset bundled in the repo so that
# repo_id="sim/droid" resolves to $PROJECT_DIR/data/lerobot/sim/droid
export HF_LEROBOT_HOME=$PROJECT_DIR/data/lerobot

mkdir -p \
  $HOME_SCRATCH \
  $WANDB_DIR \
  $HF_HOME \
  $PROJECT_DIR/logs

cd $PROJECT_DIR

uv lock --upgrade-package datasets
uv sync
source .venv/bin/activate

# sanity checks
python --version
uv --version
nvidia-smi
python - <<EOF
import torch, os
print("CUDA available:", torch.cuda.is_available())
print("Torch CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("HF_LEROBOT_HOME:", os.environ.get("HF_LEROBOT_HOME"))
EOF

# finetune pi0.5 with LoRA on sim data
uv run scripts/train.py \
  pi05_droid_lora_finetune \
  --exp-name=sim_lora_finetune \
  --overwrite

echo "JOB COMPLETED SUCCESSFULLY"

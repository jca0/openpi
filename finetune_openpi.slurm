#!/bin/bash
#SBATCH --job-name=openpi-droid-ft
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

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export WANDB_ENTITY=jing-cao
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export WANDB_DIR=/orcd/scratch/$USER/wandb
export HF_HOME=/orcd/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME

echo $SLURM_TMPDIR
############################
# Paths
############################
HOME_SCRATCH=$HOME/orcd/scratch
PROJECT_DIR=$HOME/openpi

DROID_RAW_DIR=$HOME_SCRATCH/droid_raw
DROID_ANN_DIR=$HOME_SCRATCH/droid_annotations
WANDB_DIR=$HOME_SCRATCH/wandb
HF_HOME=$HOME_SCRATCH/hf_cache

export WANDB_DIR
export HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME

mkdir -p \
  $HOME_SCRATCH \
  $DROID_RAW_DIR \
  $DROID_ANN_DIR \
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
import torch
print("CUDA available:", torch.cuda.is_available())
print("Torch CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
EOF

# download droid data
gsutil -m cp -r \
  gs://gresearch/robotics/droid_raw/1.0.1/IRIS/success/2023-12-04 \
  $DROID_RAW_DIR

gsutil -m cp -r \
  gs://gresearch/robotics/droid_raw/1.0.1/aggregated-annotations-030724.json \
  "$DROID_RAW_DIR/aggregated-annotations-030724.json"

# convert droid data to lerobot
uv run examples/droid/convert_droid_data_to_lerobot.py \
  --data_dir $DROID_RAW_DIR

# finetune openpi
uv run scripts/train.py \
  pi05_droid_lora_finetune \
  --exp-name=lora_finetune \
  --overwrite

echo "JOB COMPLETED SUCCESSFULLY"

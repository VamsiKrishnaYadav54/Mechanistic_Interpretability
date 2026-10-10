#!/bin/bash
#SBATCH --job-name=acdc_ioi
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --gres=gpu:1          
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --qos=dgx
#SBATCH --output=/home/cccp/25m2125/VAMSI/mech_interp/slurm_logs/%j_%x.out
#SBATCH --error=/home/cccp/25m2125/VAMSI/mech_interp/slurm_logs/%j_%x.err


# --- Environment setup ---
source ~/miniconda3/bin/activate autocircuit


# ------------------------------------------------------------------
# LOCAL HUGGING FACE CACHE
# ------------------------------------------------------------------
export HF_HOME=/home/cccp/25m2125/VAMSI/mech_interp/models/huggingface_cache
export HF_HUB_CACHE=/home/cccp/25m2125/VAMSI/mech_interp/models/huggingface_cache/hub
export TRANSFORMERS_CACHE=/home/cccp/25m2125/VAMSI/mech_interp/models/huggingface_cache
export HUGGINGFACE_HUB_VERBOSITY=info

# ------------------------------------------------------------------
# OFFLINE MODE
# ------------------------------------------------------------------
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# W&B local logging
export WANDB_MODE=offline

echo "=== $SLURM_JOB_ID on $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="

cd /home/cccp/25m2125/VAMSI/mech_interp/code
python scripts/run_experiment.py --config configs/ioi/acdc.yaml

echo "=== Done ==="
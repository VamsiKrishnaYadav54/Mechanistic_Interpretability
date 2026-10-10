#!/bin/bash
#SBATCH --job-name=ioi_eap
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --qos=a40
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
python scripts/run_experiment.py --config configs/ioi/eap.yaml

echo "=== Done ==="
#!/bin/bash
#SBATCH --job-name=phase0_final
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --qos=a40
#SBATCH --output=/home/cccp/25m2125/VAMSI/mech_interp/slurm_logs/%j_%x.out
#SBATCH --error=/home/cccp/25m2125/VAMSI/mech_interp/slurm_logs/%j_%x.err

# Phase 0 final run: EAP + ACDC through the single entry point, then figures.
# ACDC is loaded from its checkpoint -- the 68 h search is NOT repeated.

source ~/miniconda3/bin/activate autocircuit

export HF_HOME=/home/cccp/25m2125/VAMSI/mech_interp/models/huggingface_cache
export HF_HUB_CACHE=/home/cccp/25m2125/VAMSI/mech_interp/models/huggingface_cache
export TRANSFORMERS_CACHE=/home/cccp/25m2125/VAMSI/mech_interp/models/huggingface_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

set -e
echo "=== $SLURM_JOB_ID on $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
cd /home/cccp/25m2125/VAMSI/mech_interp

echo; echo "########## EAP ##########"
python code/scripts/run_experiment.py --config code/configs/ioi/eap.yaml

echo; echo "########## ACDC (from checkpoint) ##########"
python code/scripts/run_experiment.py --config code/configs/ioi/acdc.yaml

echo; echo "########## FIGURES ##########"
mkdir -p results/figures
python code/scripts/visualize_circuit.py \
    --scores results/ioi_eap_gpt2/scores_structural.json \
    --top 40 --out results/figures/eap_top40
python code/scripts/visualize_circuit.py \
    --scores results/ioi_acdc_gpt2/scores_structural.json \
    --top 40 --out results/figures/acdc_top40
python code/scripts/visualize_circuit.py \
    --compare results/ioi_eap_gpt2/scores_structural.json \
              results/ioi_acdc_gpt2/scores_structural.json \
    --top 40 --out results/figures/eap_vs_acdc

echo; echo "=== Done ==="

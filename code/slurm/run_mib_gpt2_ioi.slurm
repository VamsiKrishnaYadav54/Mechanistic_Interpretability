#!/bin/bash
#SBATCH --job-name=mib_gpt2_ioi
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --qos=a40
#SBATCH --output=/home/cccp/25m2125/VAMSI/mech_interp/slurm_logs/%j_%x.out
#SBATCH --error=/home/cccp/25m2125/VAMSI/mech_interp/slurm_logs/%j_%x.err

# Phase 1, step 1: reproduce MIB's published GPT-2 / IOI numbers.
# Runs in the separate "mib" env. Code lives in code/external/MIB-circuit-track
# (a git submodule, never edited); all outputs go to results/mib/ so the
# submodule stays clean.
# The model and dataset must already be in the cache (download on the login node).

source ~/miniconda3/bin/activate mib

export HF_HOME=/home/cccp/25m2125/VAMSI/mech_interp/models/huggingface_cache
export HF_HUB_CACHE=$HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

set -e
echo "=== $SLURM_JOB_ID on $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
MIB=/home/cccp/25m2125/VAMSI/mech_interp/code/external/MIB-circuit-track
OUT=/home/cccp/25m2125/VAMSI/mech_interp/results/mib
mkdir -p $OUT/circuits $OUT/results
cd $MIB

for METHOD in eap EAP-IG-inputs; do
    echo; echo "########## discovery: $METHOD ##########"
    python run_attribution.py --models gpt2 --tasks ioi --method $METHOD \
        --level edge --ablation patching --batch-size 20 \
        --circuit-dir $OUT/circuits/
done

echo; echo "########## evaluation ##########"
python run_evaluation.py --models gpt2 --tasks ioi --split validation \
    --level edge --ablation patching --batch-size 20 \
    --circuit-dir $OUT/circuits/ --output-dir $OUT/results/

echo; echo "########## CPR ##########"
python print_results.py --output-dir $OUT/results/ --split validation --metric cpr
echo; echo "########## CMD ##########"
python print_results.py --output-dir $OUT/results/ --split validation --metric cmd

echo; echo "=== Done ==="

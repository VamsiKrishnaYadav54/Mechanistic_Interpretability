# slurm/run_experiment.sh
#!/bin/bash
#SBATCH --job-name=mech_interp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1          # request specifically A100
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --qos=dgx
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err


# --- Environment setup ---
source ~/miniconda3/bin/activate mech_interp


# CRITICAL: disable internet on compute nodes
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=~/mech_interp/models/huggingface_cache
export WANDB_MODE=offline    # logs locally, sync manually later


# --- Run experiment ---
cd ~/mech_interp/code

python scripts/run_experiment.py \
    --config configs/ioi_baseline.yaml \
    --model gpt2 \
    --method acdc \
    --output_dir ~/mech_interp/results/$SLURM_JOB_ID

echo "Job $SLURM_JOB_ID complete"
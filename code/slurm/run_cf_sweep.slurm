#!/bin/bash
#SBATCH --job-name=cf_sweep
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --qos=a40
#SBATCH --output=/home/cccp/25m2125/VAMSI/mech_interp/slurm_logs/%j_%x.out
#SBATCH --error=/home/cccp/25m2125/VAMSI/mech_interp/slurm_logs/%j_%x.err

# Phase 2, experiment 1: does the circuit depend on the counterfactual?
# Same prompts, same method, 8 different counterfactuals -> 8 circuits -> compare.
#
#   sbatch code/slurm/run_cf_sweep.slurm                                      # gpt2, both methods
#   sbatch --export=ALL,MODEL=qwen2.5 code/slurm/run_cf_sweep.slurm
#
# Finished circuits are skipped, so it is safe to resubmit.

MODEL=${MODEL:-gpt2}
TASK=${TASK:-ioi}
METHODS=${METHODS:-"EAP EAP-IG-inputs"}
CFS="s2_io_flip_counterfactual abc_counterfactual s1_io_flip_counterfactual \
random_names_counterfactual random_names_s1_ioi_flip_counterfactual \
random_names_s2_ioi_flip_counterfactual s1_ioi_flip_s2_ioi_flip_counterfactual \
random_names_s1_ioi_flip_s2_ioi_flip_counterfactual"

source ~/miniconda3/bin/activate mib
export HF_HOME=/home/cccp/25m2125/VAMSI/mech_interp/models/huggingface_cache
export HF_HUB_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

ROOT=/home/cccp/25m2125/VAMSI/mech_interp
OUT=$ROOT/results/mib/circuits_cf
cd $ROOT
echo "=== $SLURM_JOB_ID on $(hostname) | model=$MODEL task=$TASK ==="

FAILED=""
for METHOD in $METHODS; do
  for CF in $CFS; do
    F=$OUT/$CF/${METHOD}_patching_edge/${TASK//_/-}_${MODEL}/importances.json
    if [[ -f "$F" ]]; then
        echo "---- $METHOD / $CF: already done, skipping"; continue
    fi
    echo; echo "########## $METHOD / $CF ##########"
    # one failing counterfactual must not stop the others
    if ! python code/scripts/run_attribution_cf.py --model $MODEL --task $TASK \
            --method $METHOD --counterfactual $CF --circuit-dir $OUT/$CF; then
        echo "!!!! FAILED: $METHOD / $CF"; FAILED="$FAILED $METHOD/$CF"
    fi
  done
done

for METHOD in $METHODS; do
    echo; echo "########## comparison: $METHOD ##########"
    python code/scripts/compare_cf_circuits.py --root $OUT --method $METHOD \
        --model $MODEL --task $TASK --k 32 162 324 \
        --out $ROOT/results/mib/cf_analysis/${METHOD}_${TASK}_${MODEL}
done

echo; echo "failed runs:${FAILED:- none}"
echo "=== Done ==="

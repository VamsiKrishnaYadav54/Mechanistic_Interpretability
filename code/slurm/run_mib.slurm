#!/bin/bash
#SBATCH --job-name=mib
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --qos=a40
#SBATCH --output=/home/cccp/25m2125/VAMSI/mech_interp/slurm_logs/%j_%x.out
#SBATCH --error=/home/cccp/25m2125/VAMSI/mech_interp/slurm_logs/%j_%x.err

# General MIB job: discovery + evaluation (CPR and CMD) for one model and task.
# Choose model / task when submitting (defaults: gpt2, ioi):
#
#   sbatch code/slurm/run_mib.slurm                                   # gpt2 ioi
#   sbatch --export=ALL,MODEL=qwen2.5 code/slurm/run_mib.slurm        # qwen ioi
#   sbatch --export=ALL,MODEL=qwen2.5,TASK=mcqa code/slurm/run_mib.slurm
#
# Anything already finished is skipped, so it is always safe to resubmit.
# "random" is the baseline: built from the EAP circuit's graph, never attributed.
# Model and dataset must already be cached (download on the login node).

MODEL=${MODEL:-gpt2}
TASK=${TASK:-ioi}
METHODS=${METHODS:-"EAP EAP-IG-inputs EAP-IG-activations random"}

source ~/miniconda3/bin/activate mib

export HF_HOME=/home/cccp/25m2125/VAMSI/mech_interp/models/huggingface_cache
export HF_HUB_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

set -e
ROOT=/home/cccp/25m2125/VAMSI/mech_interp
MIB=$ROOT/code/external/MIB-circuit-track
OUT=$ROOT/results/mib
mkdir -p $OUT/circuits $OUT/results
echo "=== $SLURM_JOB_ID on $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
echo "=== model=$MODEL task=$TASK methods=$METHODS ==="
cd $MIB

circuit() { echo "$OUT/circuits/${1}_patching_edge/${TASK}_${MODEL}/importances.json"; }

# ---------------------------------------------------------------- discovery
for METHOD in $METHODS; do
    F=$(circuit $METHOD)
    if [[ -f "$F" ]]; then
        echo; echo "########## discovery: $METHOD -- already done, skipping ##########"
        continue
    fi
    if [[ "$METHOD" == "random" ]]; then
        echo; echo "########## discovery: random (from EAP graph) ##########"
        python $ROOT/code/scripts/make_random_circuit.py \
            --like "$(circuit EAP)" --out "$F" --seed 0
        continue
    fi
    echo; echo "########## discovery: $METHOD ##########"
    python run_attribution.py --models $MODEL --tasks $TASK --method $METHOD \
        --level edge --ablation patching --batch-size 20 \
        --circuit-dir $OUT/circuits/
done

# --------------------------------------------------------------- evaluation
#   abs-False (signed scores)   -> CPR
#   abs-True  (absolute scores) -> CMD
TASKF=${TASK//_/-}
for METHOD in $METHODS; do
  for ABS in "" "--absolute"; do
    TAG=$([[ -z "$ABS" ]] && echo False || echo True)
    R=$OUT/results/${METHOD}_patching_edge/${TASKF}_${MODEL}_validation_abs-${TAG}.pkl
    if [[ -f "$R" ]]; then
        echo; echo "########## evaluation: $METHOD abs-$TAG -- already done, skipping ##########"
        continue
    fi
    echo; echo "########## evaluation: $METHOD abs-$TAG ##########"
    python run_evaluation.py --models $MODEL --tasks $TASK --method $METHOD $ABS \
        --split validation --level edge --ablation patching --batch-size 20 \
        --circuit-dir $OUT/circuits/ --output-dir $OUT/results/
  done
done

echo; echo "########## CPR ##########"
python print_results.py --output-dir $OUT/results/ --split validation --metric cpr
echo; echo "########## CMD ##########"
python print_results.py --output-dir $OUT/results/ --split validation --metric cmd
echo; echo "=== Done ==="

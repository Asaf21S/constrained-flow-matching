#!/bin/bash
#SBATCH --job-name=shared_budget
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/shared_budget_%A_%a.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/shared_budget_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --exclude=dgx04
#SBATCH --array=0-119%30
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Shared-N Functa vs fine-tuned few-shot on the v1k set. One array task = one budget N x one
# 50-constraint shard, both methods; task t runs budget N_VALUES[t / 20] on shard t % 20.
# Finished shards and per-constraint few-shot results are skipped, so resubmitting resumes.
#
#   sbatch scripts/run_shared_budget.sh
#   sbatch --array=7 scripts/run_shared_budget.sh                  # one failed task
#   sbatch --array=0 scripts/run_shared_budget.sh --end-idx 2 \
#          --outdir constrained_fm/baselines/shared_budget_v1k_smoke
#
# Then on the login node:
#   python3 -m constrained_fm.scripts.merge_val1k --outdir constrained_fm/baselines/shared_budget_v1k

set -eo pipefail

SHARD_SIZE="${SHARD_SIZE:-50}"
NUM_SHARDS="${NUM_SHARDS:-20}"
read -r -a BUDGETS <<< "${N_VALUES:-50 100 300 500 1000 2000}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
BUDGET_INDEX=$(( TASK_ID / NUM_SHARDS ))
if (( BUDGET_INDEX >= ${#BUDGETS[@]} )); then
    echo "task ${TASK_ID} is past the last budget (${#BUDGETS[@]} budgets x ${NUM_SHARDS} shards)"
    exit 1
fi
N="${BUDGETS[$BUDGET_INDEX]}"
START=$(( (TASK_ID % NUM_SHARDS) * SHARD_SIZE ))
END=$(( START + SHARD_SIZE ))
EXTRA="$*"

echo "array task ${TASK_ID}: constraints [${START}, ${END}) | N=${N} | extra: ${EXTRA:-none}"

export ENROOT_CACHE_PATH=/users/rosenbaum/asolomiak/.enroot_cache
mkdir -p "$ENROOT_CACHE_PATH"

srun --container-image=/users/rosenbaum/asolomiak/nvidia+pytorch+24.03-py3.sqsh \
     --container-mounts=/users/rosenbaum/asolomiak/constrained-flow-matching:/workspace \
     bash -c "set -eo pipefail && \
              export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
              cd /workspace && \
              pip install --user -q -r requirements.txt && \
              python -m constrained_fm.scripts.shared_budget_v1k \
                     --start-idx ${START} --end-idx ${END} --num-points ${N} ${EXTRA}"

echo "task ${TASK_ID} finished."

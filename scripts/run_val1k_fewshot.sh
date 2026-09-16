#!/bin/bash
#SBATCH --job-name=val1k_fewshot
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/val1k_fewshot_%A_%a.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/val1k_fewshot_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=0-19%5
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Few-shot baseline on the v1k set: one unconditional model trained from scratch per
# constraint, 50 constraints per array task, 1000 models over the full array.
#
# Per-constraint results are checkpointed before the shard file is written, so resubmitting
# a task resumes rather than retraining. Shards land in the v1k shards/ directory under the
# method name METHOD, which must be unique per shot budget or merge_val1k will read two
# budgets as one method covering every constraint twice.
#
#   N=2000 sbatch scripts/run_val1k_fewshot.sh
#   sbatch --array=7 scripts/run_val1k_fewshot.sh              # one failed shard
#   N=100 METHOD=fewshot_N100 sbatch scripts/run_val1k_fewshot.sh

set -eo pipefail

SHARD_SIZE="${SHARD_SIZE:-50}"
N="${N:-2000}"
METHOD="${METHOD:-fewshot}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
START=$(( TASK_ID * SHARD_SIZE ))
END=$(( START + SHARD_SIZE ))
EXTRA="$*"

echo "array task ${TASK_ID}: constraints [${START}, ${END}) | N=${N} | method=${METHOD}"

export ENROOT_CACHE_PATH=/users/rosenbaum/asolomiak/.enroot_cache
mkdir -p "$ENROOT_CACHE_PATH"

srun --container-image=/users/rosenbaum/asolomiak/nvidia+pytorch+24.03-py3.sqsh \
     --container-mounts=/users/rosenbaum/asolomiak/constrained-flow-matching:/workspace \
     bash -c "set -eo pipefail && \
              export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
              cd /workspace && \
              pip install --user -q -r requirements.txt && \
              python -m constrained_fm.scripts.few_shot_val1k \
                     --start-idx ${START} --end-idx ${END} --num-points ${N} \
                     --method ${METHOD} ${EXTRA}"

echo "shard [${START}, ${END}) finished."

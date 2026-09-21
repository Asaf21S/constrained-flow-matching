#!/bin/bash
#SBATCH --job-name=bench1k_eval
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/bench1k_eval_%A_%a.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/bench1k_eval_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=0-19%5
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Stage 2: scores every method a problem supports on 50 constraints per array task. Each task
# writes constrained_fm/baselines/bench1k/<problem>/shards/.
#
# Scores are keyed to the global constraint index, so re-cutting the shards does not change
# any number -- scripts/run_shard_invariance.sh is what enforces that. To re-run a failed
# task, resubmit with --array=<id> and the same SHARD_SIZE.
#
#   PROBLEM=kinematics6d sbatch scripts/run_bench1k_eval.sh
#   PROBLEM=bump2d sbatch scripts/run_bench1k_eval.sh --methods gt eci hardflow
#   PROBLEM=bump2d sbatch --array=7 scripts/run_bench1k_eval.sh
#   SHARD_SIZE=25 PROBLEM=bump2d sbatch --array=0-39 scripts/run_bench1k_eval.sh

set -eo pipefail

PROBLEM="${PROBLEM:-kinematics6d}"
SHARD_SIZE="${SHARD_SIZE:-50}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
START=$(( TASK_ID * SHARD_SIZE ))
END=$(( START + SHARD_SIZE ))
EXTRA="$*"

echo "array task ${TASK_ID}: ${PROBLEM} constraints [${START}, ${END})"

export ENROOT_CACHE_PATH=/users/rosenbaum/asolomiak/.enroot_cache
mkdir -p "$ENROOT_CACHE_PATH"

srun --container-image=/users/rosenbaum/asolomiak/nvidia+pytorch+24.03-py3.sqsh \
     --container-mounts=/users/rosenbaum/asolomiak/constrained-flow-matching:/workspace \
     bash -c "set -eo pipefail && \
              export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
              cd /workspace && \
              pip install --user -q -r requirements.txt && \
              python -m constrained_fm.scripts.eval_bench1k \
                     --problem ${PROBLEM} \
                     --start-idx ${START} --end-idx ${END} ${EXTRA}"

echo "${PROBLEM} shard [${START}, ${END}) finished."

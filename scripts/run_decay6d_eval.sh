#!/bin/bash
#SBATCH --job-name=decay6d_eval
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/decay6d_eval_%A_%a.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/decay6d_eval_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=0-27
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=FAIL

# decay6d IS shards: tasks 0-19 draw q blocks (box = id / 4), tasks 20-27 draw p_uncon blocks.
# Merge afterwards with scripts/run_decay6d_merge.sh (same flags).
#
#   sbatch scripts/run_decay6d_eval.sh
#   sbatch --array=0-5 scripts/run_decay6d_eval.sh --smoke

set -eo pipefail

EXTRA="$*"

export ENROOT_CACHE_PATH=/users/rosenbaum/asolomiak/.enroot_cache
mkdir -p "$ENROOT_CACHE_PATH"

srun --container-image=/users/rosenbaum/asolomiak/nvidia+pytorch+24.03-py3.sqsh \
     --container-mounts=/users/rosenbaum/asolomiak/constrained-flow-matching:/workspace \
     bash -c "set -eo pipefail && \
              cd /workspace && \
              pip install --user -q -r requirements.txt && \
              python -m constrained_fm.scripts.eval_decay6d_is --stage shard \
                  --task-id ${SLURM_ARRAY_TASK_ID} ${EXTRA}"

echo "decay6d eval task ${SLURM_ARRAY_TASK_ID} finished."

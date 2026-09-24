#!/bin/bash
#SBATCH --job-name=tune
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/tune_%A_%a.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/tune_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=FAIL

# Sampler hyperparameter selection on the held-out tuning split. As an array, each task
# scores one configuration of constrained_fm/scripts/tune_bench1k.py:grid (59 for bump2d,
# 77 for kinematics6d); without --array, pass --select to pick one per method and write
# constrained_fm/baselines/tuning/<problem>/selected.json.
#
#   PROBLEM=bump2d sbatch --array=0-58%16 scripts/run_tune.sh
#   PROBLEM=kinematics6d sbatch --array=0-76%16 scripts/run_tune.sh
#   PROBLEM=kinematics6d sbatch --dependency=afterok:<array id> scripts/run_tune.sh --select

set -eo pipefail

PROBLEM="${PROBLEM:-kinematics6d}"
TASK="${SLURM_ARRAY_TASK_ID:+--task-id ${SLURM_ARRAY_TASK_ID}}"
EXTRA="$*"

export ENROOT_CACHE_PATH=/users/rosenbaum/asolomiak/.enroot_cache
mkdir -p "$ENROOT_CACHE_PATH"

srun --container-image=/users/rosenbaum/asolomiak/nvidia+pytorch+24.03-py3.sqsh \
     --container-mounts=/users/rosenbaum/asolomiak/constrained-flow-matching:/workspace \
     bash -c "set -eo pipefail && \
              export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
              cd /workspace && \
              pip install --user -q -r requirements.txt && \
              python -m constrained_fm.scripts.tune_bench1k --problem ${PROBLEM} ${TASK} ${EXTRA}"

echo "tune ${PROBLEM} ${TASK} ${EXTRA} finished."

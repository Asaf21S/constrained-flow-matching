#!/bin/bash
#SBATCH --job-name=fm_train
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/train_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/train_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Stage 2 + 3. Trains one config and evaluates it in the same job, so a completed job
# always leaves runs/<run_id>/{ckpt.pt,losses.npy,metrics.json,figures/} behind.
#
#   sbatch scripts/run_train.sh constrained_fm/configs/baseline.yaml
#   sbatch --time=00:40:00 scripts/run_train.sh constrained_fm/configs/baseline.yaml --smoke
#   sbatch scripts/run_train.sh constrained_fm/configs/baseline.yaml --resume
#
# Requires the conditioning pool to exist already (scripts/run_pool.sh). Pass --build-pool
# to build it in-job instead; never do that for concurrent sweep jobs sharing one pool.

set -eo pipefail

CONFIG=${1:?usage: run_train.sh <config.yaml> [--smoke] [--resume] [--build-pool]}
shift

SMOKE=""
TRAIN_ARGS=""
for arg in "$@"; do
    [ "$arg" = "--smoke" ] && SMOKE="--smoke"
    TRAIN_ARGS="$TRAIN_ARGS $arg"
done

echo "config ${CONFIG} | train args:${TRAIN_ARGS:-none}"

export ENROOT_CACHE_PATH=/users/rosenbaum/asolomiak/.enroot_cache
mkdir -p "$ENROOT_CACHE_PATH"

srun --container-image=/users/rosenbaum/asolomiak/nvidia+pytorch+24.03-py3.sqsh \
     --container-mounts=/users/rosenbaum/asolomiak/constrained-flow-matching:/workspace \
     bash -c "set -eo pipefail && \
              export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
              cd /workspace && \
              pip install --user -q -r requirements.txt && \
              python -m constrained_fm.scripts.train_fm --config ${CONFIG} ${TRAIN_ARGS} && \
              python -m constrained_fm.scripts.eval_fm --config ${CONFIG} ${SMOKE}"

echo "Training job finished."

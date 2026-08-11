#!/bin/bash
# Opens an interactive shell on a compute node, inside the project container.
#
# Runs on the login node; blocks until the allocation is granted, then hands over a
# prompt. Use this instead of a job submission when the question is "what shape is this
# tensor" -- a live REPL with the model in memory answers it in seconds.
#
#   scripts/run_dev.sh            # 2 hours
#   scripts/run_dev.sh 00:30:00
#
# Once inside:
#   pip install --user -q -r requirements.txt
#   python -m constrained_fm.scripts.eval_fm --run-id <id> --no-figures
#   python           # then import from constrained_fm.src... interactively

set -eo pipefail

TIME_LIMIT=${1:-02:00:00}

export ENROOT_CACHE_PATH=/users/rosenbaum/asolomiak/.enroot_cache
mkdir -p "$ENROOT_CACHE_PATH"

echo "requesting an interactive GPU node for ${TIME_LIMIT}..."

srun --job-name=fm_dev \
     --partition=dlc \
     --gpus-per-node=1 \
     --cpus-per-task=4 \
     --mem=64G \
     --time="${TIME_LIMIT}" \
     --container-image=/users/rosenbaum/asolomiak/nvidia+pytorch+24.03-py3.sqsh \
     --container-mounts=/users/rosenbaum/asolomiak/constrained-flow-matching:/workspace \
     --container-workdir=/workspace \
     --pty bash

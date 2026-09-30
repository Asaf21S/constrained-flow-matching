#!/bin/bash
#SBATCH --job-name=decay6d_boxes
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/decay6d_boxes_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/decay6d_boxes_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=END,FAIL

# Fixed decay6d evaluation boxes (bisected to target P(B)) and simulator ground truth.
#
#   sbatch scripts/run_decay6d_boxes.sh
#   sbatch scripts/run_decay6d_boxes.sh --smoke

set -eo pipefail

EXTRA="$*"

export ENROOT_CACHE_PATH=/users/rosenbaum/asolomiak/.enroot_cache
mkdir -p "$ENROOT_CACHE_PATH"

srun --container-image=/users/rosenbaum/asolomiak/nvidia+pytorch+24.03-py3.sqsh \
     --container-mounts=/users/rosenbaum/asolomiak/constrained-flow-matching:/workspace \
     bash -c "set -eo pipefail && \
              cd /workspace && \
              pip install --user -q -r requirements.txt && \
              python -m constrained_fm.scripts.build_decay6d_boxes ${EXTRA}"

echo "decay6d boxes finished."

#!/bin/bash
#SBATCH --job-name=val1k_build
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/val1k_build_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/val1k_build_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Stage 1: builds the v1k benchmark -- 1000 polynomial constraints stratified over 20 equal
# mass bins in [0.02, 0.98], with exact masses from one fixed million-point GMM pool -- plus
# the frozen ground-truth points every NLL/KLD is averaged over.
#
# Writes constrained_fm/benchmark/{validation_set_v1k.pt,nll_eval_points_v1k.pt}.
# Both are pure functions of their seeds; run once, then never again.
#
#   sbatch scripts/run_val1k_build.sh
#   sbatch scripts/run_val1k_build.sh --rebuild

set -eo pipefail

EXTRA="$*"

export ENROOT_CACHE_PATH=/users/rosenbaum/asolomiak/.enroot_cache
mkdir -p "$ENROOT_CACHE_PATH"

srun --container-image=/users/rosenbaum/asolomiak/nvidia+pytorch+24.03-py3.sqsh \
     --container-mounts=/users/rosenbaum/asolomiak/constrained-flow-matching:/workspace \
     bash -c "set -eo pipefail && \
              export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
              cd /workspace && \
              pip install --user -q -r requirements.txt && \
              python -m constrained_fm.scripts.build_val1k ${EXTRA}"

echo "v1k benchmark built."

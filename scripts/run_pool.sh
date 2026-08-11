#!/bin/bash
#SBATCH --job-name=fm_pool
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/pool_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/pool_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Stage 1. Builds the Functa conditioning pool for a config and caches it under
# constrained_fm/functa_dataset/pools/, keyed by a fingerprint of the SIREN weights and
# the extraction settings. Every training run with matching settings reuses it.
#
#   sbatch scripts/run_pool.sh constrained_fm/configs/baseline.yaml
#   sbatch --time=00:15:00 scripts/run_pool.sh constrained_fm/configs/baseline.yaml --smoke

set -eo pipefail

CONFIG=${1:?usage: run_pool.sh <config.yaml> [--smoke] [--force]}
shift
EXTRA="$*"

export ENROOT_CACHE_PATH=/users/rosenbaum/asolomiak/.enroot_cache
mkdir -p "$ENROOT_CACHE_PATH"

srun --container-image=/users/rosenbaum/asolomiak/nvidia+pytorch+24.03-py3.sqsh \
     --container-mounts=/users/rosenbaum/asolomiak/constrained-flow-matching:/workspace \
     bash -c "set -eo pipefail && \
              export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
              cd /workspace && \
              pip install --user -q -r requirements.txt && \
              python -m constrained_fm.scripts.build_pool --config ${CONFIG} ${EXTRA}"

echo "Pool job finished."

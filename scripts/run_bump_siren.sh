#!/bin/bash
#SBATCH --job-name=bump_siren
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/bump_siren_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/bump_siren_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# CAVIA meta-training of the polygon SIREN encoder. Exits non-zero if the mass-IoU gate fails.
# Writes constrained_fm/functa_dataset/bump_siren_best.pt.
#
#   sbatch scripts/run_bump_siren.sh
#   sbatch scripts/run_bump_siren.sh --epochs 40 --steps-per-epoch 50   # smoke test

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
              python -m constrained_fm.scripts.train_bump_siren ${EXTRA}"

echo "polygon SIREN meta-training finished."

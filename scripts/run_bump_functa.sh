#!/bin/bash
#SBATCH --job-name=bump_functa
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/bump_functa_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/bump_functa_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Builds the bump2d Functa pool if missing, then trains the amortised constrained FM.
# Requires constrained_fm/functa_dataset/bump_siren_best.pt from run_bump_siren.sh.
#
#   sbatch scripts/run_bump_functa.sh
#   sbatch scripts/run_bump_functa.sh --no-siren-feature

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
              python -m constrained_fm.scripts.build_bump_pool && \
              python -m constrained_fm.scripts.train_bump_functa ${EXTRA}"

echo "bump2d Functa FM finished."

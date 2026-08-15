#!/bin/bash
#SBATCH --job-name=fm_showcase
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/showcase_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/showcase_%j.err
#SBATCH --time=00:40:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# Qualitative figures on freshly sampled polynomials that are in neither the training pool
# nor the frozen validation benchmark. Writes into constrained_fm/images/functa/showcase/.
#
#   sbatch scripts/run_showcase.sh siren-uniform-8d6375ab
#   sbatch scripts/run_showcase.sh siren-uniform-8d6375ab --num-shapes 6 --seed 7

set -eo pipefail

RUN_ID=${1:?usage: run_showcase.sh <run_id> [extra args]}
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
              python -m constrained_fm.scripts.showcase_samples --run-id ${RUN_ID} ${EXTRA}"

echo "Showcase job finished."

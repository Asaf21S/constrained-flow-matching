#!/bin/bash
#SBATCH --job-name=fm_functa_plot
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/functa_plot_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/functa_plot_%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Regenerates constrained_fm/images/functa/polynomial_functa.png.
#
# Pick a run whose SIREN checkpoint still exists; siren_best.pt was dropped in f89a54e,
# so baseline / mass-power / no-siren-feature runs no longer resolve.
#
#   sbatch scripts/run_functa_plot.sh siren-uniform-8d6375ab
#   sbatch scripts/run_functa_plot.sh siren-uniform-8d6375ab --resolution 800 --num-shapes 6

set -eo pipefail

RUN_ID=${1:?usage: run_functa_plot.sh <run_id> [extra args]}
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
              python -m constrained_fm.scripts.plot_functa_extraction --run-id ${RUN_ID} ${EXTRA}"

echo "Functa plot job finished."

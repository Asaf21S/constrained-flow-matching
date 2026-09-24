#!/bin/bash
#SBATCH --job-name=poly_transfer
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/poly_transfer_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/poly_transfer_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclude=dgx04
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Zero-shot polygon constraints on the frozen polynomial-trained SIREN + Functa FM.
# Arrays -> constrained_fm/baselines/polygon_transfer/artifacts/,
# figures -> constrained_fm/images/thesis_pool/polygon_transfer/{siren_encoder,samples}/.
#
#   sbatch scripts/run_polygon_transfer.sh
#   sbatch scripts/run_polygon_transfer.sh --field-gain 0.5 --outdir constrained_fm/baselines/polygon_transfer_g05
#   sbatch scripts/run_polygon_transfer.sh --plot-only --legend

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
              python -m constrained_fm.scripts.polygon_transfer ${EXTRA}"

echo "Polygon transfer finished."

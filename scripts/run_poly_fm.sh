#!/bin/bash
#SBATCH --job-name=poly_fm
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/poly_fm_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/poly_fm_%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Rebuilds the coefficient-conditioned baseline the Functa model is compared against; its
# original checkpoint was never saved. Trains, then evaluates on the frozen validation set
# with NLL/KLD. Writes constrained_fm/baselines/poly_fm/.
#
#   sbatch scripts/run_poly_fm.sh
#   sbatch scripts/run_poly_fm.sh --skip-train
#   sbatch scripts/run_poly_fm.sh --iterations 5001

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
              python -m constrained_fm.scripts.train_poly_fm ${EXTRA}"

echo "Coefficient-conditioned baseline finished."

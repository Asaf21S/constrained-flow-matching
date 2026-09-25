#!/bin/bash
#SBATCH --job-name=joint_interp
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/joint_interp_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/joint_interp_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclude=dgx04
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Polygon -> polynomial latent interpolation on the joint SIREN.
# Arrays -> constrained_fm/baselines/joint_interpolation/artifacts/,
# figures -> constrained_fm/images/thesis_pool/joint_interpolation/interpolation/.
#
#   sbatch scripts/run_joint_interp.sh
#   sbatch scripts/run_joint_interp.sh --plot-only --colorbar

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
              python -m constrained_fm.scripts.joint_latent_interpolation ${EXTRA}"

echo "Joint interpolation finished."

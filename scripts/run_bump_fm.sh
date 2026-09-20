#!/bin/bash
#SBATCH --job-name=bump2d_fm
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/bump2d_fm_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/bump2d_fm_%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Trains the unconstrained bump2d flow matcher that the ECI and HardFlow baselines steer.
# Writes constrained_fm/baselines/bump2d_base_fm/.
#
#   sbatch scripts/run_bump_fm.sh
#   sbatch scripts/run_bump_fm.sh --skip-train

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
              python -m constrained_fm.scripts.train_bump_fm ${EXTRA}"

echo "bump2d base model finished."

#!/bin/bash
#SBATCH --job-name=eci_proj
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/eci_proj_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/eci_proj_%j.err
#SBATCH --time=01:30:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=END,FAIL

# Sweeps the ECI Newton-projection budget on the hardest bump2d polygons. Read-only.
#
#   sbatch scripts/run_eci_projection_check.sh
#   sbatch scripts/run_eci_projection_check.sh --iters 64 256 --damping 1.0

set -eo pipefail

EXTRA="$*"

export ENROOT_CACHE_PATH=/users/rosenbaum/asolomiak/.enroot_cache
mkdir -p "$ENROOT_CACHE_PATH"

srun --container-image=/users/rosenbaum/asolomiak/nvidia+pytorch+24.03-py3.sqsh \
     --container-mounts=/users/rosenbaum/asolomiak/constrained-flow-matching:/workspace \
     bash -c "set -eo pipefail && \
              cd /workspace && \
              pip install --user -q -r requirements.txt && \
              python -m constrained_fm.scripts.check_eci_projection ${EXTRA}"

echo "ECI projection sweep finished."

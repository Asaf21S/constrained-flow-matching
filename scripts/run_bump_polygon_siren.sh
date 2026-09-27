#!/bin/bash
#SBATCH --job-name=bump_poly_siren
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/bump_poly_siren_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/bump_poly_siren_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -eo pipefail
EXTRA="$*"

export ENROOT_CACHE_PATH=/users/rosenbaum/asolomiak/.enroot_cache
mkdir -p "$ENROOT_CACHE_PATH"

srun --container-image=/users/rosenbaum/asolomiak/nvidia+pytorch+24.03-py3.sqsh \
     --container-mounts=/users/rosenbaum/asolomiak/constrained-flow-matching:/workspace \
     bash -c "set -eo pipefail && cd /workspace && pip install --user -q -r requirements.txt && \
              python -m constrained_fm.scripts.plot_bump_polygon_siren ${EXTRA}"
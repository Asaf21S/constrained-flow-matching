#!/bin/bash
#SBATCH --job-name=decay6d_check
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/decay6d_check_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/decay6d_check_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=END,FAIL

# decay6d gate: moments, quadrature convergence, split PIT, normalization, box-filter table.
#
#   sbatch scripts/run_decay6d_check.sh

set -eo pipefail

EXTRA="$*"

export ENROOT_CACHE_PATH=/users/rosenbaum/asolomiak/.enroot_cache
mkdir -p "$ENROOT_CACHE_PATH"

srun --container-image=/users/rosenbaum/asolomiak/nvidia+pytorch+24.03-py3.sqsh \
     --container-mounts=/users/rosenbaum/asolomiak/constrained-flow-matching:/workspace \
     bash -c "set -eo pipefail && \
              cd /workspace && \
              pip install --user -q -r requirements.txt && \
              python -m constrained_fm.scripts.check_decay6d ${EXTRA}"

echo "decay6d check finished."

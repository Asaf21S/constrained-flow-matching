#!/bin/bash
#SBATCH --job-name=bump_signal
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/bump_signal_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/bump_signal_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Walks the chain from the target to the amortized model and reports the signal fraction
# each stage recovers inside the band, to locate where the 1% component is lost.
#
#   sbatch scripts/run_bump_signal_check.sh

set -eo pipefail

EXTRA="$*"

export ENROOT_CACHE_PATH=/users/rosenbaum/asolomiak/.enroot_cache
mkdir -p "$ENROOT_CACHE_PATH"

srun --container-image=/users/rosenbaum/asolomiak/nvidia+pytorch+24.03-py3.sqsh \
     --container-mounts=/users/rosenbaum/asolomiak/constrained-flow-matching:/workspace \
     bash -c "set -eo pipefail && \
              cd /workspace && \
              pip install --user -q -r requirements.txt && \
              python -m constrained_fm.scripts.check_bump_signal ${EXTRA}"

echo "signal audit done."

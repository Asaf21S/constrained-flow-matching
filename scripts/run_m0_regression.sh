#!/bin/bash
#SBATCH --job-name=m0_regression
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/m0_regression_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/m0_regression_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Verifies that routing ECI and HardFlow through the Constraint interface changed no numbers.
# Read-only: compares against constrained_fm/baselines/{eci,hardflow}/metrics.json and writes
# nothing under baselines/ or runs/.
#
#   sbatch scripts/run_m0_regression.sh
#   sbatch scripts/run_m0_regression.sh --num-polys 100

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
              python -m constrained_fm.scripts.check_m0_regression ${EXTRA}"

echo "M0 regression gate finished."

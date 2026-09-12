#!/bin/bash
#SBATCH --job-name=val1k_plots
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/val1k_plots_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/val1k_plots_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Stages 3 and 4: merges the shards into constrained_fm/baselines/val1k/metrics.json, then
# renders the trend and parity figures into constrained_fm/images/thesis_pool/val1k/.
# No checkpoint, no ODE -- it reads the merged metric arrays only.
#
# The merge is pure stdlib and also runs on the login node:
#   python3 -m constrained_fm.scripts.merge_val1k
#
#   sbatch scripts/run_val1k_plots.sh
#   sbatch scripts/run_val1k_plots.sh --window 150 --step 50

set -eo pipefail

EXTRA="$*"

export ENROOT_CACHE_PATH=/users/rosenbaum/asolomiak/.enroot_cache
mkdir -p "$ENROOT_CACHE_PATH"

srun --container-image=/users/rosenbaum/asolomiak/nvidia+pytorch+24.03-py3.sqsh \
     --container-mounts=/users/rosenbaum/asolomiak/constrained-flow-matching:/workspace \
     bash -c "set -eo pipefail && \
              cd /workspace && \
              pip install --user -q -r requirements.txt && \
              python -m constrained_fm.scripts.merge_val1k && \
              python -m constrained_fm.scripts.plot_val1k ${EXTRA}"

echo "v1k figures rendered."

#!/bin/bash
#SBATCH --job-name=shared_budget_plots
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/shared_budget_plots_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/shared_budget_plots_%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --exclude=dgx04
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=END,FAIL

# Renders the shared-budget curves from the merged metrics.json. No checkpoint and no ODE;
# the container is needed only because the login node has no numpy/matplotlib.
#
# Merge first (login node, pure stdlib):
#   python3 -m constrained_fm.scripts.merge_val1k --outdir constrained_fm/baselines/shared_budget_v1k
#
#   sbatch scripts/run_shared_budget_plots.sh
#   sbatch scripts/run_shared_budget_plots.sh --ncols 4 --min-inside 10

set -eo pipefail

EXTRA="$*"

export ENROOT_CACHE_PATH=/users/rosenbaum/asolomiak/.enroot_cache
mkdir -p "$ENROOT_CACHE_PATH"

srun --container-image=/users/rosenbaum/asolomiak/nvidia+pytorch+24.03-py3.sqsh \
     --container-mounts=/users/rosenbaum/asolomiak/constrained-flow-matching:/workspace \
     bash -c "set -eo pipefail && \
              cd /workspace && \
              pip install --user -q -r requirements.txt && \
              python -m constrained_fm.scripts.plot_shared_budget ${EXTRA}"

echo "shared-budget figure finished."

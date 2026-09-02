#!/bin/bash
#SBATCH --job-name=fm_plot
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/plot_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/plot_%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=FAIL

# Figure-only stage. Redraws a run's figures from runs/<id>/artifacts/ without loading a
# checkpoint or integrating an ODE, so paper figures can be re-styled in minutes.
#
#   sbatch scripts/run_plots.sh baseline-1a2b3c4d
#   sbatch scripts/run_plots.sh --all
#   sbatch scripts/run_plots.sh --root constrained_fm/baselines/poly_fm

set -eo pipefail

TARGET=${1:?usage: run_plots.sh <run_id ... | --all | --root <dir> ...>}
shift
EXTRA="$*"

case "$TARGET" in
    --*) SELECTOR="$TARGET" ;;
    *)   SELECTOR="--run-id $TARGET" ;;
esac

export ENROOT_CACHE_PATH=/users/rosenbaum/asolomiak/.enroot_cache
mkdir -p "$ENROOT_CACHE_PATH"

srun --container-image=/users/rosenbaum/asolomiak/nvidia+pytorch+24.03-py3.sqsh \
     --container-mounts=/users/rosenbaum/asolomiak/constrained-flow-matching:/workspace \
     bash -c "set -eo pipefail && \
              cd /workspace && \
              pip install --user -q -r requirements.txt && \
              python -m constrained_fm.scripts.plot_run ${SELECTOR} ${EXTRA}"

echo "Plot job finished."

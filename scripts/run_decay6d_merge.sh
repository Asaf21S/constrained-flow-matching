#!/bin/bash
#SBATCH --job-name=decay6d_merge
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/decay6d_merge_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/decay6d_merge_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=END,FAIL

# Merges the decay6d IS shards into metrics + artifacts, then draws the figures.
# Pass the same flags as the eval array; chain it with --dependency=afterok:<array job id>.
#
#   sbatch --dependency=afterok:<jobid> scripts/run_decay6d_merge.sh --steps 32
#   sbatch --dependency=afterok:<jobid> scripts/run_decay6d_merge.sh --smoke --steps 8
#   sbatch scripts/run_decay6d_merge.sh --plot-only --steps 32

set -eo pipefail

EXTRA="$*"
PLOT_EXTRA=""
[[ " ${EXTRA} " == *" --smoke "* ]] && PLOT_EXTRA="--smoke"
if [[ " ${EXTRA} " =~ " --steps "([0-9]+)" " ]]; then
    PLOT_EXTRA="${PLOT_EXTRA} --steps ${BASH_REMATCH[1]}"
fi
# --plot-only redraws figures from saved artifacts without re-merging.
MERGE_CMD="python -m constrained_fm.scripts.eval_decay6d_is --stage merge ${EXTRA}"
if [[ " ${EXTRA} " == *" --plot-only "* ]]; then
    MERGE_CMD="true"
fi

export ENROOT_CACHE_PATH=/users/rosenbaum/asolomiak/.enroot_cache
mkdir -p "$ENROOT_CACHE_PATH"

srun --container-image=/users/rosenbaum/asolomiak/nvidia+pytorch+24.03-py3.sqsh \
     --container-mounts=/users/rosenbaum/asolomiak/constrained-flow-matching:/workspace \
     bash -c "set -eo pipefail && \
              cd /workspace && \
              pip install --user -q -r requirements.txt && \
              ${MERGE_CMD} && \
              python -m constrained_fm.scripts.plot_decay6d_is ${PLOT_EXTRA}"

echo "decay6d merge finished."

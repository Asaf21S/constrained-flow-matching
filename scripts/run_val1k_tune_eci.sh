#!/bin/bash
#SBATCH --job-name=val1k_tune_eci
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/val1k_tune_eci_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/val1k_tune_eci_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Selects ECI's mixing iterations M and noise-redraw interval R on 20 tuning constraints
# disjoint from the v1k set. Writes constrained_fm/baselines/val1k_v2/tuning/selected.json,
# which run_val1k_eval.sh consumes through --eci-selected.
#
#   JOB=$(sbatch --parsable scripts/run_val1k_tune_eci.sh)
#   sbatch --dependency=afterok:$JOB scripts/run_val1k_eval.sh --methods eci hardflow \
#       --outdir constrained_fm/baselines/val1k_v2 \
#       --eci-selected constrained_fm/baselines/val1k_v2/tuning/selected.json

set -eo pipefail

EXTRA="$*"

export ENROOT_CACHE_PATH=/users/rosenbaum/asolomiak/.enroot_cache
mkdir -p "$ENROOT_CACHE_PATH"

srun --container-image=/users/rosenbaum/asolomiak/nvidia+pytorch+24.03-py3.sqsh \
     --container-mounts=/users/rosenbaum/asolomiak/constrained-flow-matching:/workspace \
     bash -c "set -eo pipefail && \
              cd /workspace && \
              pip install --user -q -r requirements.txt && \
              python -m constrained_fm.scripts.tune_val1k_eci ${EXTRA}"

STATUS=$?
[ $STATUS -ne 0 ] && echo "FAILED (exit ${STATUS})" && exit $STATUS
echo "Done."

#!/bin/bash
#SBATCH --job-name=bench1k_build
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/bench1k_build_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/bench1k_build_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Stage 1: freezes the 1000-constraint benchmark for both problems, stratified over 20 mass
# bins and paired with the start points every method integrates from.
#
#   sbatch scripts/run_bench1k_build.sh
#   sbatch scripts/run_bench1k_build.sh --split tune
#   sbatch scripts/run_bench1k_build.sh --problem bump2d --rebuild

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
              python -m constrained_fm.scripts.build_bench1k ${EXTRA}"

STATUS=$?
[ $STATUS -ne 0 ] && echo "FAILED (exit ${STATUS})" && exit $STATUS
echo "Done."

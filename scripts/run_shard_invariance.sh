#!/bin/bash
#SBATCH --job-name=shard_inv
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/shard_inv_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/shard_inv_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# The bench1k gate: scores one constraint range as a single shard and again as two, and
# requires the rows to agree exactly. Must pass before the 20-task array is worth launching.
#
#   sbatch scripts/run_shard_invariance.sh
#   sbatch scripts/run_shard_invariance.sh --problem kinematics6d --num-constraints 12

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
              python -m constrained_fm.scripts.check_shard_invariance ${EXTRA}"

STATUS=$?
[ $STATUS -ne 0 ] && echo "SHARD INVARIANCE FAILED (exit ${STATUS})" && exit $STATUS
echo "Shard invariance passed."

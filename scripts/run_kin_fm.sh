#!/bin/bash
#SBATCH --job-name=kin_fm
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/kin_fm_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/kin_fm_%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=END,FAIL

# Unconstrained 6D base flow matcher for kinematics6d. Every M5 sampler starts from this
# checkpoint, so it is trained once and reused.
#
#   sbatch scripts/run_kin_fm.sh
#   sbatch scripts/run_kin_fm.sh --skip-train

set -eo pipefail

EXTRA="$*"

export ENROOT_CACHE_PATH=/users/rosenbaum/asolomiak/.enroot_cache
mkdir -p "$ENROOT_CACHE_PATH"

srun --container-image=/users/rosenbaum/asolomiak/nvidia+pytorch+24.03-py3.sqsh \
     --container-mounts=/users/rosenbaum/asolomiak/constrained-flow-matching:/workspace \
     bash -c "set -eo pipefail && \
              cd /workspace && \
              pip install --user -q -r requirements.txt && \
              python -m constrained_fm.scripts.train_kin_fm ${EXTRA}"

echo "kinematics6d base FM finished."

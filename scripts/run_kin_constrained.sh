#!/bin/bash
#SBATCH --job-name=kin_expl
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/kin_expl_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/kin_expl_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=END,FAIL

# Explicitly conditioned flow matcher on the kinematics6d mass shells.
#
#   sbatch scripts/run_kin_constrained.sh
#   sbatch scripts/run_kin_constrained.sh --iterations 2001 --num-shells 10

set -eo pipefail

EXTRA="$*"

export ENROOT_CACHE_PATH=/users/rosenbaum/asolomiak/.enroot_cache
mkdir -p "$ENROOT_CACHE_PATH"

srun --container-image=/users/rosenbaum/asolomiak/nvidia+pytorch+24.03-py3.sqsh \
     --container-mounts=/users/rosenbaum/asolomiak/constrained-flow-matching:/workspace \
     bash -c "set -eo pipefail && \
              cd /workspace && \
              pip install --user -q -r requirements.txt && \
              python -m constrained_fm.scripts.train_kin_constrained ${EXTRA}"

echo "kinematics6d explicit FM finished."

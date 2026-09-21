#!/bin/bash
#SBATCH --job-name=kin_eci
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/kin_eci_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/kin_eci_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=END,FAIL

# ECI and HardFlow on the kinematics6d mass shells.
#
#   sbatch scripts/run_kin_eci_hardflow.sh
#   sbatch scripts/run_kin_eci_hardflow.sh --damping-sweep 1.0 0.5 0.25 0.1

set -eo pipefail

EXTRA="$*"

export ENROOT_CACHE_PATH=/users/rosenbaum/asolomiak/.enroot_cache
mkdir -p "$ENROOT_CACHE_PATH"

srun --container-image=/users/rosenbaum/asolomiak/nvidia+pytorch+24.03-py3.sqsh \
     --container-mounts=/users/rosenbaum/asolomiak/constrained-flow-matching:/workspace \
     bash -c "set -eo pipefail && \
              cd /workspace && \
              pip install --user -q -r requirements.txt && \
              python -m constrained_fm.scripts.kin_eci_hardflow ${EXTRA}"

echo "kinematics6d ECI/HardFlow finished."

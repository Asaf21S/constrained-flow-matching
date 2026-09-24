#!/bin/bash
#SBATCH --job-name=hardflow_check
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/hardflow_check_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/hardflow_check_%j.err
#SBATCH --time=01:30:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Diagnoses HardFlow on bump2d (why the signal is sometimes over-represented) and on
# kinematics6d (guidance-scale / step / preconditioning sweep). Changes no benchmark number.
#
#   sbatch scripts/run_hardflow_check.sh
#   sbatch scripts/run_hardflow_check.sh --problems kinematics6d

set -eo pipefail

EXTRA="$*"

export ENROOT_CACHE_PATH=/users/rosenbaum/asolomiak/.enroot_cache
mkdir -p "$ENROOT_CACHE_PATH"

srun --container-image=/users/rosenbaum/asolomiak/nvidia+pytorch+24.03-py3.sqsh \
     --container-mounts=/users/rosenbaum/asolomiak/constrained-flow-matching:/workspace \
     bash -c "set -eo pipefail && \
              cd /workspace && \
              pip install --user -q -r requirements.txt && \
              python -m constrained_fm.scripts.check_hardflow ${EXTRA}"

echo "hardflow check done."

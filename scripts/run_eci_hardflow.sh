#!/bin/bash
#SBATCH --job-name=eci_hardflow
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/eci_hardflow_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/eci_hardflow_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Scores the ECI and HardFlow inference-time baselines on the 100-polynomial validation set.
# Requires constrained_fm/baselines/base_fm/ckpt.pt from run_base_fm.sh. No training here.
# Writes constrained_fm/baselines/{eci,hardflow}/.
#
#   sbatch scripts/run_eci_hardflow.sh
#   sbatch scripts/run_eci_hardflow.sh --methods hardflow --guidance-scale 20
#   sbatch scripts/run_eci_hardflow.sh --methods eci --correction-loops 5

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
              python -m constrained_fm.scripts.eci_hardflow ${EXTRA}"

echo "ECI / HardFlow baselines finished."

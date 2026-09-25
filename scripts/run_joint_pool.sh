#!/bin/bash
#SBATCH --job-name=joint_pool
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/joint_pool_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/joint_pool_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclude=dgx04
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Extracts z_pos / z_neg (constraint and exact complement) for the joint pool.
# pool.pt + metrics.json -> <siren-dir>/pool/.
#
#   sbatch scripts/run_joint_pool.sh
#   sbatch scripts/run_joint_pool.sh --siren-dir constrained_fm/functa_dataset/joint_siren_smoke --pool-size 256

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
              python -m constrained_fm.scripts.build_joint_pool ${EXTRA}"

echo "Joint pool finished."

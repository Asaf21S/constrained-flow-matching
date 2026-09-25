#!/bin/bash
#SBATCH --job-name=joint_siren
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/joint_siren_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/joint_siren_%j.err
#SBATCH --time=120:00:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclude=dgx04
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Meta-trains one SIREN on a 50/50 polynomial + bump2d-polygon pool with random complements.
# Checkpoints + metrics.json -> constrained_fm/functa_dataset/joint_siren/.
#
#   sbatch scripts/run_joint_siren.sh
#   sbatch scripts/run_joint_siren.sh --smoke --outdir constrained_fm/functa_dataset/joint_siren_smoke

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
              python -m constrained_fm.scripts.train_joint_siren ${EXTRA}"

echo "Joint SIREN training finished."

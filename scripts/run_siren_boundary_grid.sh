#!/bin/bash
#SBATCH --job-name=siren_grid
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/siren_grid_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/siren_grid_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclude=dgx04
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# SIREN boundary grids: several rows x cols panels of decoded-field heatmaps, one shape per
# panel, mass-diverse within each grid. Needs runs/<functa-run>/{config.yaml,siren weights}.
# Arrays land in constrained_fm/baselines/siren_boundary_grid/artifacts/ and figures in
# constrained_fm/images/thesis_pool/siren_encoder/boundary_grid/. Reselecting/restyling needs
# no GPU but still needs the container:
#
#   sbatch scripts/run_siren_boundary_grid.sh
#   sbatch scripts/run_siren_boundary_grid.sh --rows 2 --cols 3 --num-grids 4
#   sbatch scripts/run_siren_boundary_grid.sh --plot-only --no-colorbar --seed 1

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
              python -m constrained_fm.scripts.plot_siren_boundary_grid ${EXTRA}"

echo "SIREN boundary grid figures finished."

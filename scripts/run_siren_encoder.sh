#!/bin/bash
#SBATCH --job-name=siren_fig
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/siren_fig_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/siren_fig_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclude=dgx04
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Paper figures for the SIREN/CAVIA encoder: one decoded-field panel per sampled constraint
# plus 1xK latent-interpolation strips. Needs runs/<functa-run>/{config.yaml,siren weights}.
# Arrays land in constrained_fm/baselines/siren_encoder_figures/artifacts/ and figures in
# constrained_fm/images/thesis_pool/siren_encoder/{encoder,interpolation}/. Restyling needs
# no GPU but still needs the container:
#
#   sbatch scripts/run_siren_encoder.sh
#   sbatch scripts/run_siren_encoder.sh --num-shapes 8 --pairs 0:1 2:3 4:5 6:7
#   sbatch scripts/run_siren_encoder.sh --plot-only --legend --pred-color blue

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
              python -m constrained_fm.scripts.plot_siren_encoder ${EXTRA}"

echo "SIREN encoder figures finished."

#!/bin/bash
#SBATCH --job-name=feas_fid
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/feas_fid_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/feas_fid_%j.err
#SBATCH --time=01:30:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclude=dgx04
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Feasibility-vs-fidelity paper figure for one constraint:
# GT | ECI | HardFlow | Functa | Coefficients, in four panel compositions.
# Needs constrained_fm/baselines/{base_fm,poly_fm}/ckpt.pt and runs/<functa-run>/ckpt.pt.
# Writes arrays to constrained_fm/baselines/feasibility_fidelity/poly<id>/ and figures to
# constrained_fm/images/thesis_pool/feasibility_fidelity/<variant>/. Restyling needs no GPU:
#
#   sbatch scripts/run_feasibility_fidelity.sh
#   sbatch scripts/run_feasibility_fidelity.sh --poly-id 13 --boundary-profile
#   python -m constrained_fm.scripts.plot_feasibility_fidelity --plot-only --variants 5panel_all

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
              python -m constrained_fm.scripts.plot_feasibility_fidelity ${EXTRA}"

echo "Feasibility-vs-fidelity figure finished."

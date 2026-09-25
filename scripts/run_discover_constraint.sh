#!/bin/bash
#SBATCH --job-name=discover_cstr
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/discover_cstr_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/discover_cstr_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclude=dgx04
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Test-time FM-loss optimisation of a constraint isolating 3 of the 4 GMM modes (frozen FM + SIREN).
# Arrays -> constrained_fm/baselines/constraint_discovery/<param>_exclude<mode>/artifacts/,
# figures -> constrained_fm/images/thesis_pool/constraint_discovery/<param>_exclude<mode>/.
#
#   sbatch scripts/run_discover_constraint.sh --smoke
#   sbatch scripts/run_discover_constraint.sh
#   sbatch scripts/run_discover_constraint.sh --param coeffs
#   sbatch scripts/run_discover_constraint.sh --plot-only

set -eo pipefail

EXTRA="$*"

export ENROOT_CACHE_PATH=/users/rosenbaum/asolomiak/.enroot_cache
mkdir -p "$ENROOT_CACHE_PATH"

srun --container-image=/users/rosenbaum/asolomiak/nvidia+pytorch+24.03-py3.sqsh \
     --container-mounts=/users/rosenbaum/asolomiak/constrained-flow-matching:/workspace \
     bash -c "set -eo pipefail && \
              cd /workspace && \
              pip install --user -q -r requirements.txt && \
              python -m constrained_fm.scripts.discover_constraint ${EXTRA}"

echo "Constraint discovery finished."

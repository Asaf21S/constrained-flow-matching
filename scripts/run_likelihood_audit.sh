#!/bin/bash
#SBATCH --job-name=lik_audit
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/lik_audit_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/lik_audit_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=dlc
#SBATCH --exclude=dgx04
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Audits the exact-likelihood pipeline: SIREN-feature Jacobian completeness, density
# normalization (log Z), ODE step-size convergence and the truncated-GMM mass constant.
# Frozen checkpoint, no retraining.
#
#   sbatch scripts/run_likelihood_audit.sh --run-id siren-uniform-8d6375ab

export ENROOT_CACHE_PATH=/users/rosenbaum/asolomiak/.enroot_cache
mkdir -p $ENROOT_CACHE_PATH

srun --container-image=/users/rosenbaum/asolomiak/nvidia+pytorch+24.03-py3.sqsh \
     --container-mounts=/users/rosenbaum/asolomiak/constrained-flow-matching:/workspace \
     bash -c "set -eo pipefail && \
              cd /workspace && \
              pip install --user -q -r requirements.txt && \
              python -m constrained_fm.scripts.audit_likelihood $*"

STATUS=$?
[ $STATUS -ne 0 ] && echo "FAILED (exit ${STATUS})" && exit $STATUS
echo "Likelihood audit finished."

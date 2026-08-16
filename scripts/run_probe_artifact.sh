#!/bin/bash
#SBATCH --job-name=probe_extraction
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/probe_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/probe_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Separates extraction budget / optimizer / query-sample size / SIREN capacity as causes
# of poor boundary fidelity. Frozen SIREN, no retraining. Runs in a couple of minutes.

export ENROOT_CACHE_PATH=/users/rosenbaum/asolomiak/.enroot_cache
mkdir -p $ENROOT_CACHE_PATH

srun --container-image=/users/rosenbaum/asolomiak/nvidia+pytorch+24.03-py3.sqsh \
     --container-mounts=/users/rosenbaum/asolomiak/constrained-flow-matching:/workspace \
     bash -c "set -eo pipefail && \
              cd /workspace && \
              pip install --user -q -r requirements.txt && \
              python -m constrained_fm.scripts.probe_likelihood_artifact --run-id siren-uniform-8d6375ab"

STATUS=$?
[ $STATUS -ne 0 ] && echo "FAILED (exit ${STATUS})" && exit $STATUS
echo "Probe finished."


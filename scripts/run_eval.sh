#!/bin/bash
#SBATCH --job-name=fm_eval
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/eval_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/eval_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Stage 3 on its own. Re-scores and re-plots an already-trained checkpoint, which is the
# loop to use while iterating on metrics or diagnostic figures. Never retrains.
#
#   sbatch scripts/run_eval.sh baseline-1a2b3c4d
#   sbatch scripts/run_eval.sh constrained_fm/configs/baseline.yaml
#   sbatch scripts/run_eval.sh constrained_fm/configs/baseline.yaml --smoke

set -eo pipefail

TARGET=${1:?usage: run_eval.sh <run_id | config.yaml> [--smoke] [--no-figures]}
shift
EXTRA="$*"

case "$TARGET" in
    *.yaml|*.yml) SELECTOR="--config $TARGET" ;;
    *)            SELECTOR="--run-id $TARGET" ;;
esac

export ENROOT_CACHE_PATH=/users/rosenbaum/asolomiak/.enroot_cache
mkdir -p "$ENROOT_CACHE_PATH"

srun --container-image=/users/rosenbaum/asolomiak/nvidia+pytorch+24.03-py3.sqsh \
     --container-mounts=/users/rosenbaum/asolomiak/constrained-flow-matching:/workspace \
     bash -c "set -eo pipefail && \
              export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
              cd /workspace && \
              pip install --user -q -r requirements.txt && \
              python -m constrained_fm.scripts.eval_fm ${SELECTOR} ${EXTRA}"

echo "Evaluation job finished."

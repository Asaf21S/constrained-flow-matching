#!/bin/bash
#SBATCH --job-name=few_shot
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/few_shot_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/few_shot_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Few-shot unconstrained baseline: trains one flow matcher per (polynomial, N) pair.
# Work is sharded, so the full benchmark runs as concurrent jobs. Per-item results are
# written individually and completed items are skipped, so shards are resumable.
#
#   sbatch scripts/run_few_shot.sh                         # 4-shape subset, single shard
#   sbatch scripts/run_few_shot.sh --shard 0 --num-shards 8 --all-shapes
#   sbatch scripts/run_few_shot.sh --plot-only

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
              python -m constrained_fm.scripts.few_shot_unconstrained ${EXTRA}"

echo "Few-shot baseline finished."

#!/bin/bash
#SBATCH --job-name=fm_ablate_pts
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/ablate_pts_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/ablate_pts_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Inference-time query-budget ablation. The FM half re-samples the full validation set once
# per N value; a single full eval is ~17 min, so the default 6-value sweep runs ~2-2.5h.
# Writes runs/<run_id>/ablations/query_points/.
#
#   sbatch scripts/run_ablate_points.sh siren-uniform-8d6375ab
#   sbatch scripts/run_ablate_points.sh siren-uniform-8d6375ab --no-flow-matching
#   sbatch scripts/run_ablate_points.sh siren-uniform-8d6375ab --num-points 50 200 1000

set -eo pipefail

RUN_ID=${1:?usage: run_ablate_points.sh <run_id> [extra args]}
shift
EXTRA="$*"

export ENROOT_CACHE_PATH=/users/rosenbaum/asolomiak/.enroot_cache
mkdir -p "$ENROOT_CACHE_PATH"

srun --container-image=/users/rosenbaum/asolomiak/nvidia+pytorch+24.03-py3.sqsh \
     --container-mounts=/users/rosenbaum/asolomiak/constrained-flow-matching:/workspace \
     bash -c "set -eo pipefail && \
              export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
              cd /workspace && \
              pip install --user -q -r requirements.txt && \
              python -m constrained_fm.scripts.ablate_query_points --run-id ${RUN_ID} ${EXTRA}"

echo "Query-budget ablation finished."

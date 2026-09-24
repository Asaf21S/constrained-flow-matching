#!/bin/bash
#SBATCH --job-name=bumphunt_plots
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/bumphunt_plots_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/bumphunt_plots_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Stage 4 and 5 for both bump-hunting problems: merges the evaluation shards, renders the
# benchmark comparison suite, which reads only the merged metric arrays, then the dataset and
# headline panels, which do load the checkpoints and integrate the ODE.
#
#   sbatch scripts/run_bumphunt_plots.sh
#   sbatch --dependency=afterok:<eval array id> scripts/run_bumphunt_plots.sh
#   sbatch scripts/run_bumphunt_plots.sh --budget 50000

set -eo pipefail

EXTRA="$*"

export ENROOT_CACHE_PATH=/users/rosenbaum/asolomiak/.enroot_cache
mkdir -p "$ENROOT_CACHE_PATH"

srun --container-image=/users/rosenbaum/asolomiak/nvidia+pytorch+24.03-py3.sqsh \
     --container-mounts=/users/rosenbaum/asolomiak/constrained-flow-matching:/workspace \
     bash -c "set -eo pipefail && \
              cd /workspace && \
              pip install --user -q -r requirements.txt && \
              python -m constrained_fm.scripts.merge_bench1k --problem bump2d kinematics6d && \
              python -m constrained_fm.scripts.plot_bench1k --problem bump2d && \
              python -m constrained_fm.scripts.plot_bench1k --problem kinematics6d && \
              python -m constrained_fm.scripts.plot_bumphunt ${EXTRA} && \
              python -m constrained_fm.scripts.plot_kinematics ${EXTRA}"

echo "bump-hunting figures rendered."

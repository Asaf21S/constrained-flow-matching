#!/bin/bash
#SBATCH --job-name=constraint_grid
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/constraint_grid_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/constraint_grid_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

export ENROOT_CACHE_PATH=/users/rosenbaum/asolomiak/.enroot_cache
mkdir -p "$ENROOT_CACHE_PATH"

# Forwarded to plot_constraint_grid.py, e.g. --num-variants 5 --seed-start 0
ARGS="$@"

srun --container-image=/users/rosenbaum/asolomiak/nvidia+pytorch+24.03-py3.sqsh \
     --container-mounts=/users/rosenbaum/asolomiak/constrained-flow-matching:/workspace \
     bash -c "cd /workspace && python -m constrained_fm.scripts.plot_constraint_grid $ARGS"

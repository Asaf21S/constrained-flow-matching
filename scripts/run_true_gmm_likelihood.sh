#!/bin/bash
#SBATCH --job-name=true_gmm_likelihood
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/true_gmm_likelihood_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/true_gmm_likelihood_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

export ENROOT_CACHE_PATH=/users/rosenbaum/asolomiak/.enroot_cache
mkdir -p "$ENROOT_CACHE_PATH"

srun --container-image=/users/rosenbaum/asolomiak/nvidia+pytorch+24.03-py3.sqsh \
     --container-mounts=/users/rosenbaum/asolomiak/constrained-flow-matching:/workspace \
     bash -c "cd /workspace && python -c \"from constrained_fm.src.visualization.density import visualize_true_gmm_likelihood; visualize_true_gmm_likelihood(save_path='constrained_fm/images/thesis_pool/true_gmm_likelihood', show=False)\""
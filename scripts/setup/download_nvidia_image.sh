#!/bin/bash
#SBATCH --job-name=pull_container
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/download_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/download_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# One-off: rebuilds /users/rosenbaum/asolomiak/nvidia+pytorch+24.03-py3.sqsh if the image is lost.
# Only run this if that file no longer exists; every training job mounts it directly.

# 1. Point the temporary cache to the DGX node's ultra-fast local storage
export ENROOT_CACHE_PATH=/tmp/$USER/.enroot_cache
export ENROOT_DATA_PATH=/tmp/$USER/.enroot_data
export ENROOT_TEMP_PATH=/tmp/$USER/.enroot_temp

# 2. Create the temporary local directories
mkdir -p $ENROOT_CACHE_PATH $ENROOT_DATA_PATH $ENROOT_TEMP_PATH

# 3. Navigate to your home folder so the final .sqsh file saves here
cd /users/rosenbaum/asolomiak/

echo "Starting high-speed local container download..."

# 4. Pull the image
enroot import 'docker://nvcr.io#nvidia/pytorch:24.03-py3'

echo "Download complete! Cleaning up local node storage..."

# 5. Clean up the temporary local files so we don't clutter the DGX node
rm -rf /tmp/$USER/.enroot_cache /tmp/$USER/.enroot_data /tmp/$USER/.enroot_temp

echo "Done!"
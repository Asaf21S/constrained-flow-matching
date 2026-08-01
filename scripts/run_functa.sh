#!/bin/bash
#SBATCH --job-name=train_functa
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/functa_train_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/functa_train_%j.err
#SBATCH --time=120:00:00
#SBATCH --partition=dlc
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclude=dgx04

echo "Starting Functa meta-learning training..."

export ENROOT_CACHE_PATH=/users/rosenbaum/asolomiak/.enroot_cache
mkdir -p $ENROOT_CACHE_PATH

srun --container-image=/users/rosenbaum/asolomiak/nvidia+pytorch+24.03-py3.sqsh \
     --container-mounts=/users/rosenbaum/asolomiak/constrained-flow-matching:/workspace \
     bash -c "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && cd /workspace/ && pip install --user -r requirements.txt && python -m constrained_fm.scripts.train_functa"

echo "Job finished."

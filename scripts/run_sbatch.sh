#!/bin/bash
#SBATCH --job-name=train_fm
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# --- Container Configuration ---
#SBATCH --container-image=nvcr.io/nvidia/pytorch:24.03-py3
#SBATCH --container-mounts=/home/asolomiak/constrained-flow-matching:/workspace

# --- Execution Commands ---
pip install -r requirements.txt
cd /workspace
python my_script_name.py --param 1

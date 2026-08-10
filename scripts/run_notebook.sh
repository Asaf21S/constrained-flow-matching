#!/bin/bash
#SBATCH --job-name=run_fm_notebook
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/notebook_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/notebook_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=asafucho@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclude=dgx04

# Executes the paired .py as a notebook, then writes two artifacts to outputs/:
#   .ipynb - the notebook with every cell's stdout and figures stored inline
#   .html  - a standalone render, viewable in a browser with no Jupyter install
#
#   smoke test : sbatch --export=ALL,SMOKE_TEST=1 --time=00:40:00 scripts/run_notebook.sh
#   full run   : sbatch scripts/run_notebook.sh
#
# logs/ must already exist: SBATCH resolves the output paths before this script runs.

NOTEBOOK=${NOTEBOOK:-constrained_fm_2d_gmm_functa}
SMOKE_TEST=${SMOKE_TEST:-0}
STAMP=$(date +%Y%m%d_%H%M%S)

if [ "$SMOKE_TEST" = "1" ]; then
    OUT_NB="outputs/${NOTEBOOK}_smoke_${STAMP}.ipynb"
else
    OUT_NB="outputs/${NOTEBOOK}_full_${STAMP}.ipynb"
fi

echo "Executing ${NOTEBOOK} | SMOKE_TEST=${SMOKE_TEST} | -> ${OUT_NB}"

export ENROOT_CACHE_PATH=/users/rosenbaum/asolomiak/.enroot_cache
mkdir -p $ENROOT_CACHE_PATH

srun --container-image=/users/rosenbaum/asolomiak/nvidia+pytorch+24.03-py3.sqsh \
     --container-mounts=/users/rosenbaum/asolomiak/constrained-flow-matching:/workspace \
     bash -c "set -eo pipefail && \
              export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
              export MPLBACKEND=Agg && \
              export SMOKE_TEST=${SMOKE_TEST} && \
              export PATH=\"\$HOME/.local/bin:\$PATH\" && \
              cd /workspace && \
              pip install --user -q -r requirements.txt jupytext nbconvert ipykernel && \
              python -m ipykernel install --user --name python3 && \
              mkdir -p outputs && \
              python -m jupytext --to notebook \
                     --output ${OUT_NB} constrained_fm/notebooks/${NOTEBOOK}.py && \
              python -m nbconvert --to notebook --execute --inplace \
                     --ExecutePreprocessor.timeout=-1 \
                     --ExecutePreprocessor.kernel_name=python3 \
                     ${OUT_NB} && \
              python -m nbconvert --to html ${OUT_NB}"

STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "FAILED (exit ${STATUS}). Partial notebook: ${OUT_NB}"
    exit $STATUS
fi

echo "Job finished. Artifacts: ${OUT_NB} and ${OUT_NB%.ipynb}.html"

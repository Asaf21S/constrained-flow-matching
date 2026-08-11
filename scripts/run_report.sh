#!/bin/bash
#SBATCH --job-name=fm_report
#SBATCH --output=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/report_%j.out
#SBATCH --error=/users/rosenbaum/asolomiak/constrained-flow-matching/logs/report_%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=dlc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Renders the report notebook into outputs/ as .ipynb and standalone .html.
#
# The report only reads runs/*/{config.yaml,metrics.json,losses.npy,figures/*.png}, so it
# executes in seconds and never touches a model. It exists because the login node has no
# numpy; nothing here needs a GPU.
#
#   sbatch scripts/run_report.sh
#   NOTEBOOK=constrained_fm_2d_gmm_functa sbatch scripts/run_report.sh

NOTEBOOK=${NOTEBOOK:-constrained_fm_2d_gmm_functa}
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_NB="outputs/${NOTEBOOK}_report_${STAMP}.ipynb"

echo "Rendering ${NOTEBOOK} -> ${OUT_NB}"

export ENROOT_CACHE_PATH=/users/rosenbaum/asolomiak/.enroot_cache
mkdir -p "$ENROOT_CACHE_PATH"

srun --container-image=/users/rosenbaum/asolomiak/nvidia+pytorch+24.03-py3.sqsh \
     --container-mounts=/users/rosenbaum/asolomiak/constrained-flow-matching:/workspace \
     bash -c "set -eo pipefail && \
              export PATH=\"\$HOME/.local/bin:\$PATH\" && \
              cd /workspace && \
              pip install --user -q -r requirements.txt jupytext nbconvert ipykernel && \
              python -m ipykernel install --user --name python3 && \
              mkdir -p outputs && \
              python -m jupytext --to notebook \
                     --output ${OUT_NB} constrained_fm/notebooks/${NOTEBOOK}.py && \
              python -m nbconvert --to notebook --execute --inplace \
                     --ExecutePreprocessor.timeout=1800 \
                     --ExecutePreprocessor.kernel_name=python3 \
                     ${OUT_NB} && \
              python -m nbconvert --to html ${OUT_NB} && \
              python -m jupytext --to notebook \
                     --output constrained_fm/notebooks/${NOTEBOOK}.ipynb \
                     constrained_fm/notebooks/${NOTEBOOK}.py"

STATUS=$?
[ $STATUS -ne 0 ] && echo "FAILED (exit ${STATUS}). Partial notebook: ${OUT_NB}" && exit $STATUS

echo "Report ready: ${OUT_NB} and ${OUT_NB%.ipynb}.html"

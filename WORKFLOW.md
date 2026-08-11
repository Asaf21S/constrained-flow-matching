# Workflow

Every stage is one `sbatch`. Nothing heavy ever runs on the login node.

| # | stage | command |
| --- | --- | --- |
| 0 | check a config (login node, instant) | `python3 -m constrained_fm.scripts.check_config constrained_fm/configs/baseline.yaml` |
| 1 | build the Functa pool (once per SIREN) | `sbatch scripts/run_pool.sh constrained_fm/configs/baseline.yaml` |
| 2 | train **+** evaluate | `sbatch scripts/run_train.sh constrained_fm/configs/baseline.yaml` |
| 3 | re-evaluate an existing checkpoint | `sbatch scripts/run_eval.sh <run_id>` |
| 4 | render the report | `sbatch scripts/run_report.sh` |

Extras:

| task | command |
| --- | --- |
| several configs at once, overnight | `scripts/run_sweep.sh constrained_fm/configs/*.yaml` |
| fast end-to-end test (~1 min) | add `--smoke`, or `SMOKE=1` for the sweep |
| interactive shell on a GPU node | `scripts/run_dev.sh 02:00:00` |
| watch the queue | `squeue -u $USER` |
| read a job's log | `tail -f logs/train_<jobid>.out` |

Results land in `runs/<run_id>/`: `metrics.json`, `figures/*.png`, `ckpt.pt`, `config.yaml`.
Open the PNGs directly in VS Code — no notebook needed.

New experiment: copy a config in `constrained_fm/configs/`, keep `extends:` and change only
the lines you are testing, then run stage 0 and stage 2.

**Never retrain to change a plot or a metric.** Edit `constrained_fm/scripts/eval_fm.py` or
`constrained_fm/src/visualization/diagnostics.py` and re-run stage 3.

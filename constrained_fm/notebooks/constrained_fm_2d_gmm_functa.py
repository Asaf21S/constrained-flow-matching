# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Functa-Conditioned Flow Matching — Report
#
# This notebook **reports** finished runs. It does not train, sample, or touch a model.
# Everything it renders was produced by `constrained_fm.scripts.eval_fm` and lives under
# `runs/<run_id>/`, so the whole notebook executes in seconds.
#
# | stage | command |
# | :--- | :--- |
# | validate a config | `python3 -m constrained_fm.scripts.check_config constrained_fm/configs/*.yaml` |
# | build the pool | `sbatch scripts/run_pool.sh <config.yaml>` |
# | train + evaluate | `sbatch scripts/run_train.sh <config.yaml>` |
# | re-evaluate a checkpoint | `sbatch scripts/run_eval.sh <run_id>` |
# | sweep, concurrently | `scripts/run_sweep.sh constrained_fm/configs/*.yaml` |
# | interactive debugging | `scripts/run_dev.sh` |
# | render this report | `sbatch scripts/run_report.sh` |

# %%
# %matplotlib inline

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image, Markdown, display

repo_path = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "constrained_fm").is_dir())
if str(repo_path) not in sys.path:
    sys.path.insert(0, str(repo_path))

from constrained_fm.src.experiment.registry import (FIGURES_DIR, comparison_table, latest_run,
                                                    list_runs, load_config, load_metrics,
                                                    readme_table, run_dir)

print(f"repo {repo_path}")


# %%
def show_figure(run_id: str, filename: str, caption: str = "") -> None:
    """Displays a figure produced by eval_fm, or says so when the run predates it."""
    path = run_dir(run_id) / FIGURES_DIR / filename
    if not path.exists():
        display(Markdown(f"*missing: `{filename}`*"))
        return
    if caption:
        display(Markdown(f"**{caption}**"))
    display(Image(filename=str(path)))


# %% [markdown]
# ## Run registry
#
# Every run on disk, then every evaluated run ranked by median success rate. `run_id` is
# `<name>-<fingerprint>`, where the fingerprint covers the SIREN weights, the extraction
# settings, the pool, the architecture, and every training hyperparameter — two ids differ
# only when something that actually changes the trained model differs.

# %%
runs = list_runs()
print(f"{len(runs)} run(s) on disk\n")
for record in runs:
    flag = "ckpt" if record["has_checkpoint"] else "----"
    print(f"[{flag}] {record['run_id']:<32} {record['status']:<10} "
          f"iter {str(record['iteration']):>6}  {record['finished_at']}")

display(Markdown(comparison_table()))


# %% [markdown]
# ## Focus run
#
# Set `REPORT_RUN_ID` in the environment to pin a specific run; otherwise the most
# recently evaluated one is used.

# %%
evaluated = [r["run_id"] for r in runs if r["summary"]]
if not evaluated:
    raise RuntimeError("no evaluated runs found — submit scripts/run_train.sh first")

run_id = os.environ.get("REPORT_RUN_ID") or latest_run()

cfg = load_config(run_id)
metrics = load_metrics(run_id)
per_shape = metrics["per_shape"]
summary = metrics["summary"]

print(f"run_id      {run_id}")
print(f"description {cfg.description.strip()}")
print(f"iteration   {metrics['iteration']}")
print(f"evaluated   {metrics['evaluated_at']}")
print()
print(f"siren       {Path(cfg.siren.checkpoint).name} | latent {cfg.siren.latent_dim}")
print(f"extraction  {cfg.extraction.steps} SGD steps @ lr {cfg.extraction.lr} | "
      f"{cfg.extraction.points_per_shape} pts | gmm fraction {cfg.extraction.query_gmm_fraction}")
print(f"pool        {cfg.pool.size} shapes | area {cfg.pool.min_area}-{cfg.pool.max_area}")
print(f"fm          {cfg.fm.hidden_dim} wide x {cfg.fm.num_blocks} blocks | "
      f"siren feature {cfg.fm.use_siren_feature}")
print(f"train       {cfg.train.iterations} iters | bs {cfg.train.batch_size} | "
      f"lr {cfg.train.lr} | mass power {cfg.train.mass_weight_power}")


# %% [markdown]
# ## Validation-set metrics
#
# Scored on the static validation polynomials with fresh CAVIA extraction, reporting
# median / mean / worst-5% instead of single-shape anecdotes.

# %%
display(Markdown(readme_table(summary)))


# %%
show_figure(run_id, "loss_curve.png", "Training loss")


# %% [markdown]
# ## Typical behaviour
#
# The shape at the median success rate, chosen so these panels describe the usual case
# rather than the tail.

# %%
show_figure(run_id, "typical_trajectory.png", "Inference progression (t = 0 to t = 1)")
show_figure(run_id, "typical_samples.png", "Final samples")
show_figure(run_id, "typical_likelihood.png", "Exact model likelihood")


# %% [markdown]
# ## Failure tail
#
# The ten worst constraints by success rate, alongside each one's valid GMM mass (its
# training exposure) and mass-weighted region IoU (how faithfully the SIREN decodes the
# constraint from z).

# %%
order = np.argsort(per_shape["success_rate"])[:10]

print(f"{'rank':>4} {'SR':>7} {'mass':>7} {'massIoU':>8} {'swd':>8} {'jsd':>8} {'extrMSE':>9}")
for rank, i in enumerate(order):
    print(f"{rank:>4} {per_shape['success_rate'][i]:7.2f} {per_shape['mass'][i]:7.3f} "
          f"{per_shape['mass_iou'][i]:8.3f} {per_shape['swd'][i]:8.4f} "
          f"{per_shape['jsd'][i]:8.4f} {per_shape['extraction_mse'][i]:9.5f}")

print()
print(f"corr(success_rate, mass)     = {summary['corr_success_mass']:+.3f}")
print(f"corr(success_rate, mass_IoU) = {summary['corr_success_mass_iou']:+.3f}")


# %%
show_figure(run_id, "success_vs_fidelity.png",
            "Success rate against constraint mass and decoded-region fidelity")


# %% [markdown]
# ### Believed region vs. true region
#
# The flow matcher only ever sees z, so it can at best fill the region the SIREN decodes
# from z. Plotting that decoded boundary (blue) against the true $P(x) = 0$ curve (red)
# separates *the flow matcher fails to respect its conditioning* from *the conditioning
# describes the wrong region*.

# %%
show_figure(run_id, "worst_believed_vs_true.png", "Worst-tail constraints")


# %% [markdown]
# ## Cross-run comparison
#
# Ranks every evaluated run on the same validation set. Because the evaluation block is
# excluded from the run fingerprint, re-evaluating never forks a run and these rows stay
# directly comparable.

# %%
if len(evaluated) > 1:
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    panels = [("success_rate_median", "success rate median (%)", "seagreen"),
              ("success_rate_p5", "success rate worst 5% (%)", "darkseagreen"),
              ("swd_median", "SWD median (lower is better)", "indianred")]

    for ax, (key, label, color) in zip(axs, panels):
        values = [load_metrics(r)["summary"].get(key, np.nan) for r in evaluated]
        ax.barh(range(len(evaluated)), values, color=color)
        ax.set_yticks(range(len(evaluated)))
        ax.set_yticklabels(evaluated, fontsize=8)
        ax.set_title(label)
        ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.show()
else:
    display(Markdown("*only one evaluated run — nothing to compare yet*"))


# %%
for other in evaluated:
    other_summary = load_metrics(other)["summary"]
    headline = load_config(other).description.strip().splitlines()
    print(f"{other:<34} SR {other_summary.get('success_rate_median', float('nan')):6.2f} "
          f"| massIoU {other_summary.get('mass_iou_mean', float('nan')):5.3f} "
          f"| {headline[0] if headline else ''}")


# %% [markdown]
# ## Functa encoder reference
#
# The SIREN is trained separately by `constrained_fm/scripts/train_functa.py` and frozen
# for every run above. Its meta-training curves and qualitative reconstructions bound
# everything the flow matcher can achieve: a run's `mass_iou_mean` cannot exceed what this
# encoder is able to represent.

# %%
functa_dir = repo_path / "constrained_fm" / "functa_dataset"
train_loss_path = functa_dir / "loss_history.npy"
val_loss_path = functa_dir / "val_loss_history.npy"

if train_loss_path.exists() and val_loss_path.exists():
    loss_history = np.load(train_loss_path)
    val_loss_history = np.load(val_loss_path)

    plt.figure(figsize=(10, 4.5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, label="Training loss (MSE + L2)",
             color="royalblue", alpha=0.8)
    plt.plot(val_loss_history[:, 0], val_loss_history[:, 1], label="Validation loss (MSE)",
             color="crimson", marker="o", markersize=3, linewidth=1.8)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("CAVIA meta-training curves")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.show()
else:
    display(Markdown("*SIREN loss histories not found*"))


# %%
for name, caption in [("polynomial_functa.png", "Extracted Functa reconstructions"),
                      ("polynomial_functa_interpolation.png", "Latent-space interpolation")]:
    path = repo_path / "constrained_fm" / "images" / "functa" / name
    if path.exists():
        display(Markdown(f"**{caption}**"))
        display(Image(filename=str(path)))


# %% [markdown]
# ## Provenance
#
# The exact configuration and artifact digests behind the focus run, sufficient to
# reproduce it from a clean checkout.

# %%
print(json.dumps(json.loads((run_dir(run_id) / "provenance.json").read_text()), indent=2))

# -*- coding: utf-8 -*-
"""Draws the decay6d importance-sampling figures from saved artifacts; never runs a model.

    python -m constrained_fm.scripts.plot_decay6d_is --steps 32
    python -m constrained_fm.scripts.plot_decay6d_is --steps 8 --smoke
    python -m constrained_fm.scripts.plot_decay6d_is --steps 64 --paper --paper-n 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from constrained_fm.src.experiment import artifacts  # noqa: E402
from constrained_fm.src.visualization import decay6d_is as viz  # noqa: E402

EVAL_DIR = "constrained_fm/baselines/decay6d_is/eval"
SMOKE_EVAL_DIR = "constrained_fm/baselines/decay6d_is/smoke/eval"
FIGURE_DIR = "constrained_fm/images/thesis_pool/decay6d_is"
SMOKE_FIGURE_DIR = "constrained_fm/baselines/decay6d_is/smoke/figures"
PAPER_SUBDIR = "paper"
OBSERVABLE_LABELS = {"p2_norm": r"$E[|\vec p_2|\,|\,\mathcal{B}]$",
                     "p2_z": r"$E[p_{2z}\,|\,\mathcal{B}]$",
                     "p2_tail": r"$P(|\vec p_2| > \tau_{\mathcal{B}}\,|\,\mathcal{B})$"}
SUMMARY_OBSERVABLE = "p2_norm"
SUMMARY_LABEL = r"$\Vert\vec p_2\Vert$ RMSE"
RMSE_LABELS = {"p2_norm": SUMMARY_LABEL, "p2_z": r"$p_{2z}$ RMSE",
               "p2_tail": r"$P(\Vert\vec p_2\Vert > \tau_{\mathcal{B}})$ RMSE"}
MASS_ESTIMATORS = ("is_learned", "rej_equal_time", "rej_equal_nfe", "rej_equal_n")
TIME_ESTIMATORS = ("is_learned", "rej_equal_time", "rej_equal_nfe")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="decay6d IS figures from artifacts.")
    parser.add_argument("--steps", type=int, default=32, help="midpoint steps of the evaluation")
    parser.add_argument("--eval-dir", default=None)
    parser.add_argument("--figure-dir", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--paper", action="store_true",
                        help="only the paper row figures, into <figure root>/paper")
    parser.add_argument("--paper-n", type=int, default=1000, help="budget N of the paper mass row")
    return parser


def save(fig, out: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out / name}.{{png,pdf}}")


def _mass_rmse(boxes: dict, n: int) -> dict[str, dict[str, np.ndarray]]:
    """``[observable][estimator] -> (boxes,)`` RMSE at budget ``n``."""
    return {obs: {e: np.array([b["by_n"][str(n)]["estimators"][e][obs]["rmse"]
                               for b in boxes.values()]) for e in MASS_ESTIMATORS}
            for obs in RMSE_LABELS}


def _time_frontier(box: dict, n_values: list[int]):
    """Mean seconds, ``[observable][estimator]`` RMSE and draws per estimate, each ``(N,)``."""
    by_n = [box["by_n"][str(n)] for n in n_values]
    seconds = {e: np.array([c["cost"][e]["seconds"]["mean"] for c in by_n])
               for e in TIME_ESTIMATORS}
    rmse = {obs: {e: np.array([c["estimators"][e][obs]["rmse"] for c in by_n])
                  for e in TIME_ESTIMATORS} for obs in RMSE_LABELS}
    draws = {e: np.array([c["cost"][e]["q_draws"] + c["cost"][e]["uncon_draws"] for c in by_n])
             for e in TIME_ESTIMATORS}
    return seconds, rmse, draws


def plot_summaries(metrics: dict, out: Path) -> None:
    """Rare-event and cost-frontier summaries, read from the merged ``metrics.json``."""
    boxes = metrics["boxes"]
    names = list(boxes)
    n_values = metrics["n_values"]
    masses = np.array([boxes[b]["gt_mass"] for b in names])
    by_n = {n: _mass_rmse(boxes, n) for n in n_values}
    rmse = {obs: {n: by_n[n][obs] for n in n_values} for obs in RMSE_LABELS}
    title = f"{metrics['steps']} midpoint steps per ODE solve"
    save(viz.plot_rmse_vs_mass_grid(masses, rmse, RMSE_LABELS, title), out, "rmse_vs_mass_grid")

    rarest = names[int(np.argmin(masses))]
    seconds, rmse, draws = _time_frontier(boxes[rarest], n_values)
    save(viz.plot_rmse_vs_time(seconds, rmse[SUMMARY_OBSERVABLE], draws, SUMMARY_LABEL, rarest),
         out, f"rmse_vs_time_{rarest}_{SUMMARY_OBSERVABLE}")


def plot_paper(metrics: dict, out: Path, n: int) -> None:
    """The two paper figures: RMSE vs mass at budget ``n`` and the rarest-box frontier, as rows."""
    boxes = metrics["boxes"]
    names = list(boxes)
    steps = metrics["steps"]
    masses = np.array([boxes[b]["gt_mass"] for b in names])
    save(viz.plot_rmse_vs_mass_row(masses, _mass_rmse(boxes, n), RMSE_LABELS),
         out, f"rmse_vs_mass_n{n}_steps{steps}")

    rarest = names[int(np.argmin(masses))]
    seconds, rmse, draws = _time_frontier(boxes[rarest], metrics["n_values"])
    save(viz.plot_rmse_vs_time_row(seconds, rmse, draws, RMSE_LABELS),
         out, f"rmse_vs_time_{rarest}_steps{steps}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    steps_dir = f"steps{args.steps}"
    eval_dir = Path(args.eval_dir or Path(SMOKE_EVAL_DIR if args.smoke else EVAL_DIR) / steps_dir)
    figure_root = Path(SMOKE_FIGURE_DIR if args.smoke else FIGURE_DIR)
    if args.paper:
        out = Path(args.figure_dir or figure_root / PAPER_SUBDIR)
        out.mkdir(parents=True, exist_ok=True)
        plot_paper(json.loads((eval_dir / "metrics.json").read_text()), out, args.paper_n)
        return 0
    out = Path(args.figure_dir or figure_root / steps_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = artifacts.load_manifest(eval_dir)
    if not manifest:
        raise FileNotFoundError(f"no artifact manifest under {eval_dir}; run the merge stage")
    boxes, estimators = manifest["boxes"], manifest["estimators"]
    weights, observables = manifest["weights"], manifest["observables"]
    bench_dir = Path(manifest["benchmark_dir"])
    arr = {k: artifacts.load_array(eval_dir, k) for k in
           ("estimates", "gt_mean", "gt_se", "n_values", "ess_frac", "max_weight")}
    metrics_path = eval_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing {metrics_path}; run the merge stage")
    plot_summaries(json.loads(metrics_path.read_text()), out)

    for k, name in enumerate(observables):
        save(viz.plot_error_vs_n(arr["n_values"], arr["estimates"], arr["gt_mean"], arr["gt_se"],
                                 boxes, estimators, k, OBSERVABLE_LABELS[name]),
             out, f"error_vs_n_{name}")

    log_w = {kind: [artifacts.load_array(eval_dir, f"log_w_{kind}_box{b}")
                    for b in range(len(boxes))] for kind in weights}
    save(viz.plot_weight_histograms(log_w, boxes), out, "weight_histograms")
    save(viz.plot_ess(arr["n_values"], arr["ess_frac"], arr["max_weight"], boxes, weights),
         out, "ess_vs_n")

    for b, name in enumerate(boxes):
        fig = viz.plot_p2_marginals(artifacts.load_array(bench_dir, f"gt_p2_box{b}"),
                                    artifacts.load_array(eval_dir, f"q_p2_box{b}"),
                                    artifacts.load_array(eval_dir, f"q_inside_box{b}"),
                                    {kind: log_w[kind][b] for kind in weights}, name)
        save(fig, out, f"p2_marginals_box{b}_{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

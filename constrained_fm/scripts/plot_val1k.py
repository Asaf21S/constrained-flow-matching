# -*- coding: utf-8 -*-
"""Stage 4 of the v1k pipeline: the fidelity and density figure suite.

Reads only the merged metrics.json, persists the arrays the figures consume under
``artifacts/``, and renders:

* **Format A, trend lines** -- one figure per metric, all methods on shared axes.
  SWD / MMD / JSD are drawn against the *ground truth's own value* for that metric, which is
  the achievable noise floor at that constraint, so the GT curve is the parity line and the
  vertical gap to it is the excess discrepancy. Success rate is drawn against the true
  constraint mass. NLL and KLD are drawn against the true constraint mass for the
  coefficient and Functa models only -- ECI and HardFlow alter the state outside the
  probability-flow ODE, so no density of theirs is defined, and rejection sampling's KLD is
  identically zero by construction.
* **Format B, head-to-head parity scatter** -- SWD / MMD / JSD, our two learned methods
  against each inference-time baseline, one point per constraint.

    sbatch scripts/run_val1k_plots.sh
    python -m constrained_fm.scripts.plot_val1k --num-bins 16 --no-panels
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from constrained_fm.src.experiment import artifacts
from constrained_fm.src.visualization.comparison import (METRIC_SPECS, plot_metric_trend,
                                                         plot_parity, plot_parity_grid,
                                                         save_figure, short_label, win_rate)

DEFAULT_OUTDIR = "constrained_fm/baselines/val1k"
DEFAULT_FIGURE_DIR = "constrained_fm/images/thesis_pool/val1k"

TREND_ALL_METHODS = ("gt", "coeff", "functa", "eci", "hardflow")
# Metrics drawn against the ground truth's own value for that metric.
GT_AXIS_METRICS = ("swd", "mmd", "jsd")
# Metrics whose density only exists for the two ODE-faithful methods.
DENSITY_METRICS = ("nll", "kld")
# Baseline on x, our method on y.
PARITY_PAIRS = (("gt", "coeff"), ("gt", "functa"), ("eci", "coeff"), ("eci", "functa"),
                ("hardflow", "coeff"), ("hardflow", "functa"))
MASS_AXIS_LABEL = "True constraint mass (%)"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the v1k comparison figure suite.")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--figure-dir", default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--num-bins", type=int, default=12, help="equal-count bins per trend line")
    parser.add_argument("--no-panels", action="store_true",
                        help="render only the combined parity grids, not the standalone panels")
    return parser


def load_metrics(out: Path) -> dict:
    path = out / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- run `python3 -m constrained_fm.scripts.merge_val1k` first")
    return json.loads(path.read_text())


def metric_arrays(payload: dict) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray]:
    """{method: {metric: array}} plus the shared true-mass array, in constraint order."""
    num = payload["num_constraints"]
    by_method: dict[str, dict[str, np.ndarray]] = {}
    for method, merged in payload["methods"].items():
        by_method[method] = {key: np.asarray(values, dtype=float)
                             for key, values in merged["per_shape"].items()}

    mass = payload.get("mass")
    if mass is None:
        mass = next(m["mass"] for m in by_method.values() if "mass" in m)
    return by_method, np.asarray(mass, dtype=float).reshape(num)


def persist_arrays(out: Path, payload: dict, by_method: dict[str, dict[str, np.ndarray]],
                   mass: np.ndarray) -> None:
    """Every figure below is a function of exactly these arrays."""
    arrays = {"mass": mass}
    for method, metrics in by_method.items():
        for key, values in metrics.items():
            if key != "mass":
                arrays[f"{method}__{key}"] = values
    artifacts.save_arrays(out, **arrays)
    artifacts.write_manifest(out, run_id=payload["methods"][next(iter(payload["methods"]))]["run_id"],
                             method="val1k", validation_set="v1k",
                             poly_digest=payload["poly_digest"],
                             num_constraints=payload["num_constraints"],
                             methods=sorted(by_method))


def render_trends(by_method: dict[str, dict[str, np.ndarray]], mass: np.ndarray,
                  figure_dir: Path, num_bins: int) -> list[Path]:
    written = []
    mass_pct = mass * 100.0

    series = {name: (mass_pct, metrics["success_rate"])
              for name, metrics in by_method.items()
              if name in TREND_ALL_METHODS and "success_rate" in metrics}
    if series:
        written.append(save_figure(
            plot_metric_trend(series, MASS_AXIS_LABEL, "Success Rate (%)",
                              "Feasibility vs constraint mass", num_bins=num_bins),
            figure_dir / "trend_success_rate.png"))

    gt_metrics = by_method.get("gt", {})
    for metric in GT_AXIS_METRICS:
        short, axis_label, log = METRIC_SPECS[metric]
        if metric not in gt_metrics:
            print(f"skipping trend_{metric}: ground truth has no {short}")
            continue
        series = {name: (gt_metrics[metric], metrics[metric])
                  for name, metrics in by_method.items()
                  if name in TREND_ALL_METHODS and metric in metrics}
        written.append(save_figure(
            plot_metric_trend(series, f"Ground truth {short} (noise floor)", axis_label,
                              f"{short} against the achievable noise floor",
                              logx=log, logy=log, num_bins=num_bins, identity=True),
            figure_dir / f"trend_{metric}.png"))

    for metric in DENSITY_METRICS:
        short, axis_label, log = METRIC_SPECS[metric]
        series = {name: (mass_pct, by_method[name][metric])
                  for name in ("coeff", "functa")
                  if name in by_method and metric in by_method[name]}
        if not series:
            print(f"skipping trend_{metric}: no method reports it")
            continue
        written.append(save_figure(
            plot_metric_trend(series, MASS_AXIS_LABEL, axis_label,
                              f"{short} vs constraint mass (exact likelihood)",
                              logy=log, num_bins=num_bins),
            figure_dir / f"trend_{metric}.png"))

    return written


def parity_caption(baseline: str, x: np.ndarray, y: np.ndarray) -> str:
    """GT is the noise floor rather than a competitor, so its panels are read differently."""
    finite = np.isfinite(x) & np.isfinite(y)
    rate = win_rate(x[finite], y[finite])
    if baseline == "gt":
        return f"gap to noise floor\nat or below it: {rate:.1f}% of {int(finite.sum())}"
    return f"win rate (below the line)\n{rate:.1f}% of {int(finite.sum())}"


def render_parity(by_method: dict[str, dict[str, np.ndarray]], mass: np.ndarray,
                  figure_dir: Path, panels_too: bool) -> list[Path]:
    written = []
    mass_pct = mass * 100.0

    for metric in GT_AXIS_METRICS:
        short, _, log = METRIC_SPECS[metric]
        panels = []
        for baseline, ours in PARITY_PAIRS:
            if metric not in by_method.get(baseline, {}) or metric not in by_method.get(ours, {}):
                print(f"skipping parity {short} {baseline} vs {ours}: metric missing")
                continue
            x, y = by_method[baseline][metric], by_method[ours][metric]
            role = "noise floor" if baseline == "gt" else "baseline"
            panels.append({
                "x": x, "y": y, "color_by": mass_pct,
                "xlabel": f"{short_label(baseline)} {short}  ({role})",
                "ylabel": f"{short_label(ours)} {short}",
                "title": f"{short_label(ours)}  vs  {short_label(baseline)}",
                "annotate": parity_caption(baseline, x, y),
                "pair": (baseline, ours),
            })

        if not panels:
            continue

        written.append(save_figure(
            plot_parity_grid(panels, f"{short}: head-to-head, one point per constraint", log=log),
            figure_dir / f"parity_{metric}.png"))

        if panels_too:
            for panel in panels:
                baseline, ours = panel["pair"]
                written.append(save_figure(
                    plot_parity(panel["x"], panel["y"], panel["xlabel"], panel["ylabel"],
                                panel["title"], color_by=panel["color_by"], log=log,
                                annotate=panel["annotate"]),
                    figure_dir / "panels" / f"parity_{metric}__{baseline}_vs_{ours}.png"))

    return written


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.outdir)
    figure_dir = Path(args.figure_dir)

    payload = load_metrics(out)
    by_method, mass = metric_arrays(payload)
    print(f"validation set {payload['validation_set']} | digest {payload['poly_digest']} | "
          f"{payload['num_constraints']} constraints | methods {sorted(by_method)}")

    persist_arrays(out, payload, by_method, mass)
    written = render_trends(by_method, mass, figure_dir, args.num_bins)
    written += render_parity(by_method, mass, figure_dir, not args.no_panels)

    for path in written:
        print(f"wrote {path}")
    print(f"{len(written)} figures in {figure_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

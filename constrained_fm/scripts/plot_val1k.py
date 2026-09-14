# -*- coding: utf-8 -*-
"""Stage 4 of the v1k pipeline: the fidelity and density figure suite.

Reads only the merged metrics.json, persists the arrays the figures consume under
``artifacts/``, and renders:

* **Format A, trend lines** -- one figure per metric, all methods on shared axes, each point a
  rolling median over a fixed number of constraints sorted along the x axis.
  SWD / MMD / JSD are drawn against the *ground truth's own value* for that metric, which is
  the achievable noise floor at that constraint. No empirical ground-truth series is drawn
  there: on that axis the ground truth *is* the y = x line, so it is rendered as the parity
  reference and the vertical gap to it is the excess discrepancy. Acceptance rate is drawn
  against the true constraint mass, where sampling the target GMM unconditionally traces
  y = x. NLL and KLD are drawn against the true constraint mass for the methods sampled by
  an intact probability-flow ODE -- ECI and HardFlow alter the state outside it, so no
  density of theirs is defined, and rejection sampling's KLD is identically zero.
  Every trend metric is exported twice: once including the few-shot baseline
  (``*_with_fewshot``) and once without it.
* **Format B, head-to-head parity scatter** -- SWD / MMD / JSD, our two learned methods
  against each inference-time baseline, one point per constraint.

    sbatch scripts/run_val1k_plots.sh
    python -m constrained_fm.scripts.plot_val1k --window 150 --step 50 --no-panels
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from constrained_fm.src.experiment import artifacts
from constrained_fm.src.visualization.comparison import (METHOD_COLORS, METRIC_SPECS,
                                                         plot_metric_trend, plot_parity,
                                                         plot_parity_grid, save_figure,
                                                         short_label, win_rate)

DEFAULT_OUTDIR = "constrained_fm/baselines/val1k"
DEFAULT_FIGURE_DIR = "constrained_fm/images/thesis_pool/val1k"

TREND_ALL_METHODS = ("coeff", "functa", "fewshot", "eci", "hardflow")
# Metrics drawn against the ground truth's own value for that metric.
GT_AXIS_METRICS = ("swd", "mmd", "jsd")
# Metrics whose density only exists for methods sampled by an intact probability-flow ODE.
DENSITY_METRICS = ("nll", "kld")
DENSITY_TREND_METHODS = ("coeff", "functa", "fewshot")
ACCEPTANCE_TREND_METHODS = ("coeff", "functa", "fewshot")
FEWSHOT = "fewshot"
# Baseline on x, our method on y.
PARITY_PAIRS = (("gt", "coeff"), ("gt", "functa"), ("eci", "coeff"), ("eci", "functa"),
                ("hardflow", "coeff"), ("hardflow", "functa"))
MASS_AXIS_LABEL = "True constraint mass (%)"
ACCEPTANCE_AXIS_LABEL = "Acceptance Rate (%)"
# Sampling the target GMM unconditionally accepts exactly the constraint's mass, hence y = x.
GMM_REFERENCE_LABEL = "Unconstrained Target GMM"
# On a ground-truth x axis the parity line *is* the ground truth, so no empirical series is drawn.
GT_REFERENCE_LABEL = "Ground Truth"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the v1k comparison figure suite.")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--figure-dir", default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--window", type=int, default=100,
                        help="constraints summarised by each trend point")
    parser.add_argument("--step", type=int, default=25,
                        help="constraints the trend window advances between points")
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


def fewshot_label(payload: dict) -> str | None:
    """The shot budget is part of the baseline's identity, so it belongs in the legend."""
    merged = payload.get("methods", {}).get(FEWSHOT)
    if merged is None:
        return None
    n_points = merged.get("eval", {}).get("n_points")
    return f"Few-Shot ($N{{=}}{n_points}$)" if n_points else "Few-Shot"


def save_both_variants(figure_factory, figure_dir: Path, stem: str,
                       has_fewshot: bool) -> list[Path]:
    """Version B never shows the few-shot line; version A is added only when it exists."""
    written = [save_figure(figure_factory(False), figure_dir / f"{stem}.png")]
    if has_fewshot:
        written.append(save_figure(figure_factory(True),
                                   figure_dir / f"{stem}_with_fewshot.png"))
    return written


def render_trends(by_method: dict[str, dict[str, np.ndarray]], mass: np.ndarray,
                  figure_dir: Path, window: int, step: int,
                  fewshot_name: str | None = None) -> list[Path]:
    written: list[Path] = []
    mass_pct = mass * 100.0
    has_fewshot = FEWSHOT in by_method
    labels = {FEWSHOT: fewshot_name} if fewshot_name else None

    def pick(names, include_fewshot: bool, metric: str) -> list[str]:
        return [n for n in names if n in by_method and metric in by_method[n]
                and (include_fewshot or n != FEWSHOT)]

    def acceptance(include_fewshot: bool):
        series = {n: (mass_pct, by_method[n]["success_rate"])
                  for n in pick(ACCEPTANCE_TREND_METHODS, include_fewshot, "success_rate")}
        return plot_metric_trend(series, MASS_AXIS_LABEL, ACCEPTANCE_AXIS_LABEL, "",
                                 window=window, step=step, identity=True,
                                 identity_label=GMM_REFERENCE_LABEL,
                                 identity_color=METHOD_COLORS["gt"], labels=labels)

    if pick(ACCEPTANCE_TREND_METHODS, True, "success_rate"):
        written += save_both_variants(acceptance, figure_dir, "trend_success_rate", has_fewshot)

    gt_metrics = by_method.get("gt", {})
    for metric in GT_AXIS_METRICS:
        short, axis_label, log = METRIC_SPECS[metric]
        if metric not in gt_metrics:
            print(f"skipping trend_{metric}: ground truth has no {short}")
            continue

        def gt_axis(include_fewshot: bool, metric=metric, short=short,
                    axis_label=axis_label, log=log):
            series = {n: (gt_metrics[metric], by_method[n][metric])
                      for n in pick(TREND_ALL_METHODS, include_fewshot, metric)}
            return plot_metric_trend(series, f"Ground truth {short} (noise floor)", axis_label,
                                     "", logx=log, logy=log, window=window, step=step,
                                     identity=True, identity_label=GT_REFERENCE_LABEL,
                                     identity_color=METHOD_COLORS["gt"], labels=labels)

        written += save_both_variants(gt_axis, figure_dir, f"trend_{metric}", has_fewshot)

    for metric in DENSITY_METRICS:
        short, axis_label, log = METRIC_SPECS[metric]
        if not pick(DENSITY_TREND_METHODS, True, metric):
            print(f"skipping trend_{metric}: no method reports it")
            continue

        def density(include_fewshot: bool, metric=metric, axis_label=axis_label, log=log):
            series = {n: (mass_pct, by_method[n][metric])
                      for n in pick(DENSITY_TREND_METHODS, include_fewshot, metric)}
            return plot_metric_trend(series, MASS_AXIS_LABEL, axis_label, "",
                                     logy=log, window=window, step=step, labels=labels)

        written += save_both_variants(density, figure_dir, f"trend_{metric}", has_fewshot)

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
    written = render_trends(by_method, mass, figure_dir, args.window, args.step,
                            fewshot_name=fewshot_label(payload))
    written += render_parity(by_method, mass, figure_dir, not args.no_panels)

    for path in written:
        print(f"wrote {path}")
    print(f"{len(written)} figures in {figure_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

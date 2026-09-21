# -*- coding: utf-8 -*-
"""Stage 4 of the bench1k pipeline: the 1000-constraint comparison figure suite.

Reads only the merged metrics.json, persists the arrays the figures consume under
``artifacts/``, and renders:

* **Excess discrepancy** -- SWD / MMD / JSD divided by that constraint's own noise floor,
  against the true constraint mass. The floor is the same array for every method (one truth
  pool, one seed), so the ratio is the only scale on which two constraints of different mass
  are comparable at all: a raw distance shrinks with the feasible region regardless of how
  well a method tracks it. Unity is a perfect sampler.
* **Acceptance rate** against the true constraint mass, with ground truth drawn as the
  attainable ceiling rather than assumed to be 100%.
* **Density metrics** -- only the method whose probability-flow ODE is left intact reports
  NLL and KLD. The projection baselines move the state off the ODE, so no density of theirs
  is defined.
* **Head-to-head parity scatter** -- our method against each baseline, one point per
  constraint, shaded by constraint mass.

Also emits ``table.md``: the summary table the write-up quotes, so no number is hand-copied.

    python3 -m constrained_fm.scripts.plot_bench1k --problem bump2d
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from constrained_fm.src.experiment import artifacts
from constrained_fm.src.visualization.comparison import (METHOD_COLORS, METRIC_SPECS,
                                                         plot_metric_trend, plot_parity_grid,
                                                         save_figure, short_label)

DEFAULT_OUTDIR = "constrained_fm/baselines/bench1k"
DEFAULT_FIGURE_DIR = "constrained_fm/images/bench1k"

# The amortized method under test, per problem. Everything else is a baseline.
OURS = {"bump2d": "functa", "kinematics6d": "explicit"}
BASELINES = ("eci", "hardflow")
# Metrics that only mean something relative to the finite-sample floor at that constraint.
RATIO_METRICS = ("swd", "mmd", "jsd")
DENSITY_METRICS = ("nll", "kld")
# Acceptance is tabulated separately as a median and a 5th percentile, so it is absent here.
TABLE_METRICS = ("swd", "mmd", "jsd", "kld", "in_support_fraction")

MASS_AXIS_LABEL = "True constraint mass (%)"
FLOOR_LABEL = "Ground truth (noise floor)"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the bench1k comparison figure suite.")
    parser.add_argument("--problem", choices=sorted(OURS), required=True)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--figure-dir", default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--window", type=int, default=150,
                        help="constraints summarised by each trend point")
    parser.add_argument("--step", type=int, default=50,
                        help="constraints the trend window advances between points")
    return parser


def load_metrics(out: Path) -> dict:
    path = out / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- run `python3 -m constrained_fm.scripts.merge_bench1k` first")
    return json.loads(path.read_text())


def metric_arrays(payload: dict) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray]:
    """{method: {metric: array}} plus the shared true-mass array, in constraint order."""
    by_method = {method: {key: np.asarray(values, dtype=float)
                          for key, values in merged["per_shape"].items()}
                 for method, merged in payload["methods"].items()}
    return by_method, np.asarray(payload["mass"], dtype=float)


def persist_arrays(out: Path, payload: dict, by_method: dict[str, dict[str, np.ndarray]],
                   mass: np.ndarray) -> None:
    """Every figure below is a function of exactly these arrays."""
    arrays = {"mass": mass}
    for method, metrics in by_method.items():
        for key, values in metrics.items():
            if key != "mass":
                arrays[f"{method}__{key}"] = values
    artifacts.save_arrays(out, **arrays)
    artifacts.write_manifest(out, method="bench1k", problem=payload["problem"],
                             benchmark_digest=payload["benchmark_digest"],
                             num_constraints=payload["num_constraints"],
                             methods=sorted(by_method))


def excess(metrics: dict[str, np.ndarray], metric: str) -> np.ndarray | None:
    """metric / its own per-constraint noise floor, or None when either is absent."""
    floor = metrics.get(f"{metric}_noise_floor")
    if metric not in metrics or floor is None:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(floor > 0, metrics[metric] / floor, np.nan)


def render_excess(by_method: dict[str, dict[str, np.ndarray]], mass: np.ndarray,
                  figure_dir: Path, window: int, step: int) -> list[Path]:
    written: list[Path] = []
    mass_pct = mass * 100.0
    for metric in RATIO_METRICS:
        short, _, _ = METRIC_SPECS[metric]
        series = {}
        for method, metrics in by_method.items():
            if method == "gt":
                continue
            ratio = excess(metrics, metric)
            if ratio is not None:
                series[method] = (mass_pct, ratio)
        if not series:
            continue
        fig = plot_metric_trend(series, MASS_AXIS_LABEL,
                                f"{short} / noise floor", "", logy=True,
                                window=window, step=step, hline=1.0,
                                hline_label=FLOOR_LABEL, legend_loc="upper right")
        written.append(save_figure(fig, figure_dir / f"excess_{metric}.png"))
    return written


def render_acceptance(by_method: dict[str, dict[str, np.ndarray]], mass: np.ndarray,
                      figure_dir: Path, window: int, step: int) -> list[Path]:
    series = {method: (mass * 100.0, metrics["success_rate"])
              for method, metrics in by_method.items() if "success_rate" in metrics}
    if not series:
        return []
    _, axis_label, _ = METRIC_SPECS["success_rate"]
    fig = plot_metric_trend(series, MASS_AXIS_LABEL, axis_label, "", window=window,
                            step=step, legend_loc="lower right")
    return [save_figure(fig, figure_dir / "trend_success_rate.png")]


def render_density(by_method: dict[str, dict[str, np.ndarray]], mass: np.ndarray,
                   figure_dir: Path, window: int, step: int) -> list[Path]:
    written: list[Path] = []
    for metric in DENSITY_METRICS:
        _, axis_label, log = METRIC_SPECS[metric]
        series = {method: (mass * 100.0, metrics[metric])
                  for method, metrics in by_method.items()
                  if metric in metrics and np.isfinite(metrics[metric]).any()}
        if not series:
            print(f"skipping trend_{metric}: no method reports it")
            continue
        fig = plot_metric_trend(series, MASS_AXIS_LABEL, axis_label, "", logy=log,
                                window=window, step=step, legend_loc="upper right")
        written.append(save_figure(fig, figure_dir / f"trend_{metric}.png"))
    return written


def render_parity(by_method: dict[str, dict[str, np.ndarray]], mass: np.ndarray,
                  ours: str, figure_dir: Path) -> list[Path]:
    written: list[Path] = []
    if ours not in by_method:
        return written
    mass_pct = mass * 100.0
    for metric in RATIO_METRICS:
        short, _, _ = METRIC_SPECS[metric]
        mine = excess(by_method[ours], metric)
        if mine is None:
            continue
        panels = []
        for baseline in BASELINES:
            if baseline not in by_method:
                continue
            theirs = excess(by_method[baseline], metric)
            if theirs is None:
                continue
            panels.append({"x": theirs, "y": mine,
                           "xlabel": f"{short_label(baseline)} excess {short}",
                           "ylabel": f"{short_label(ours)} excess {short}",
                           "title": "", "color_by": mass_pct})
        if panels:
            fig = plot_parity_grid(panels, "", ncols=len(panels))
            written.append(save_figure(fig, figure_dir / f"parity_{metric}.png"))
    return written


def summary_table(payload: dict, by_method: dict[str, dict[str, np.ndarray]],
                  ours: str) -> str:
    """The write-up's table, generated so that no figure and no number can drift apart."""
    order = [m for m in ("gt", ours, *BASELINES) if m in by_method]
    present = [key for key in TABLE_METRICS
               if any(key in by_method[m] and np.isfinite(by_method[m][key]).any()
                      for m in order)]

    header = ["method", "AR median", "AR p5"]
    for key in present:
        if key in RATIO_METRICS:
            short, _, _ = METRIC_SPECS[key]
            header.append(f"{short} (x floor)")
        elif key == "kld":
            header.append("KLD median")
        elif key == "in_support_fraction":
            header.append("in support (%)")
    rows = ["| " + " | ".join(header) + " |",
            "|" + "|".join([":---"] * len(header)) + "|"]

    for method in order:
        metrics = by_method[method]
        cells = [short_label(method),
                 f"{np.nanmedian(metrics['success_rate']):.3f}",
                 f"{np.nanpercentile(metrics['success_rate'], 5):.3f}"]
        for key in present:
            if key in RATIO_METRICS:
                ratio = excess(metrics, key)
                cells.append("--" if ratio is None
                             else f"{np.nanmedian(metrics[key]):.4g} ({np.nanmedian(ratio):.1f}x)")
            elif key in metrics and np.isfinite(metrics[key]).any():
                cells.append(f"{np.nanmedian(metrics[key]):.4f}")
            else:
                cells.append("--")
        rows.append("| " + " | ".join(cells) + " |")

    return (f"### {payload['problem']} -- {payload['num_constraints']} constraints\n\n"
            + "\n".join(rows) + "\n")


def stratified_table(by_method: dict[str, dict[str, np.ndarray]], mass: np.ndarray,
                     ours: str, metric: str = "mmd",
                     edges: tuple[float, ...] = (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.01)) -> str:
    """Excess discrepancy per mass band: the median alone hides where the methods cross."""
    order = [m for m in (ours, *BASELINES) if m in by_method]
    ratios = {m: excess(by_method[m], metric) for m in order}
    order = [m for m in order if ratios[m] is not None]
    short, _, _ = METRIC_SPECS[metric]

    rows = ["| mass band | " + " | ".join(short_label(m) for m in order) + " |",
            "|" + "|".join([":---"] * (len(order) + 1)) + "|"]
    for lo, hi in zip(edges, edges[1:]):
        inside = (mass >= lo) & (mass < hi)
        if not inside.any():
            continue
        cells = [f"[{lo:.1f}, {min(hi, 1.0):.1f})"]
        cells += [f"{np.nanmedian(ratios[m][inside]):.1f}x" for m in order]
        rows.append("| " + " | ".join(cells) + " |")

    return f"\n### Excess {short} by constraint mass\n\n" + "\n".join(rows) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    out = Path(args.outdir) / args.problem
    figure_dir = Path(args.figure_dir) / args.problem
    figure_dir.mkdir(parents=True, exist_ok=True)

    payload = load_metrics(out)
    by_method, mass = metric_arrays(payload)
    ours = OURS[args.problem]
    persist_arrays(out, payload, by_method, mass)

    written = render_excess(by_method, mass, figure_dir, args.window, args.step)
    written += render_acceptance(by_method, mass, figure_dir, args.window, args.step)
    written += render_density(by_method, mass, figure_dir, args.window, args.step)
    written += render_parity(by_method, mass, ours, figure_dir)

    table = summary_table(payload, by_method, ours) + \
        stratified_table(by_method, mass, ours, "mmd") + \
        stratified_table(by_method, mass, ours, "swd")
    table_path = figure_dir / "table.md"
    table_path.write_text(table)

    print(table)
    for path in written:
        print(f"wrote {path}")
    print(f"wrote {table_path}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Renders the query-budget bar chart from the merged metrics.

Reads only ``metrics.json``, persists the arrays the figure consumes under ``artifacts/``,
and writes the figure as PNG (for the README) and PDF (for the paper).

    sbatch scripts/run_query_budget_plots.sh
    python -m constrained_fm.scripts.plot_query_budget --ncols 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from constrained_fm.src.experiment import artifacts
from constrained_fm.src.visualization.query_budget import (BarPanel, panel_statistics,
                                                           plot_query_budget_bars, save_figure)

DEFAULT_OUTDIR = "constrained_fm/baselines/query_budget"
DEFAULT_FIGURE_DIR = "constrained_fm/images/thesis_pool/query_budget"
XLABEL = "Inference query points $N$"

PANELS = (
    BarPanel("mass_iou", "Mass IoU"),
    BarPanel("extraction_mse", "Extraction MSE", log_y=True),
    BarPanel("success_rate", "Acceptance Rate (%)"),
    BarPanel("swd", "SWD"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the query-budget ablation bar chart.")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--figure-dir", default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--name", default="query_budget_bars", help="figure stem")
    parser.add_argument("--ncols", type=int, default=2, help="panels per row")
    parser.add_argument("--panel-width", type=float, default=6.4)
    parser.add_argument("--panel-height", type=float, default=5.2)
    parser.add_argument("--reference-n", type=int, default=None,
                        help="budget to highlight (default: the meta-training budget)")
    return parser


def load_metrics(out: Path) -> dict:
    path = out / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- run `python3 -m constrained_fm.scripts.merge_query_budget` first")
    return json.loads(path.read_text())


def metric_series(payload: dict) -> tuple[list[int], dict[str, np.ndarray]]:
    """metric key -> (num_budgets, num_constraints) array, rows ordered by budget."""
    n_values = payload["n_values"]
    keys = {key for n in n_values for key in payload["budgets"][str(n)]["per_shape"]
            if key != "mass"}
    series = {key: np.asarray([payload["budgets"][str(n)]["per_shape"][key] for n in n_values],
                              dtype=float)
              for key in keys
              if all(key in payload["budgets"][str(n)]["per_shape"] for n in n_values)}
    return n_values, series


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.outdir)
    payload = load_metrics(out)
    n_values, series = metric_series(payload)

    reference_n = args.reference_n
    if reference_n is None:
        reference_n = payload["budgets"][str(n_values[0])]["eval"]["meta_trained_points_per_shape"]

    artifacts.save_arrays(out, n_values=np.asarray(n_values),
                          mass=np.asarray(payload["mass"], dtype=float),
                          **series)
    artifacts.write_manifest(out, run_id=payload["budgets"][str(n_values[0])]["run_id"],
                             method="query_budget",
                             validation_set=payload["validation_set"],
                             poly_digest=payload["poly_digest"],
                             num_constraints=payload["num_constraints"],
                             n_values=n_values)

    fig = plot_query_budget_bars(
        n_values, series, xlabel=XLABEL, panels=PANELS, reference_n=reference_n,
        ncols=args.ncols, panel_size=(args.panel_width, args.panel_height),
        reference_label=f"Meta-training budget ($N{{=}}{reference_n}$)")
    path = save_figure(fig, Path(args.figure_dir) / f"{args.name}.png")
    print(f"wrote {path} and {path.with_suffix('.pdf')}")

    print(f"\n{payload['num_constraints']} constraints, mean +- SD")
    header = "".join(f"{panel.label:>26}" for panel in PANELS if panel.key in series)
    print(f"{'N':>6}{header}")
    for row, n in enumerate(n_values):
        cells = ""
        for panel in PANELS:
            if panel.key not in series:
                continue
            mean, std = panel_statistics(series[panel.key])
            fmt = ".2e" if panel.log_y else ".4f"
            cells += f"{format(mean[row], fmt) + ' +- ' + format(std[row], fmt):>26}"
        print(f"{n:>6}{cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

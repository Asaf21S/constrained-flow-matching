# -*- coding: utf-8 -*-
"""Renders the shared-budget curves (Functa vs fine-tuned few-shot) from the merged metrics.

Reads only ``metrics.json``, persists the arrays the figure consumes under ``artifacts/``,
and writes the figure as PNG and PDF.

    sbatch scripts/run_shared_budget_plots.sh
    sbatch scripts/run_shared_budget_plots.sh --ncols 4 --min-inside 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from constrained_fm.scripts import shared_budget_results as results
from constrained_fm.src.experiment import artifacts
from constrained_fm.src.visualization.comparison import METHOD_COLORS
from constrained_fm.src.visualization.query_budget import SPREAD_MODES, BarPanel, save_figure
from constrained_fm.src.visualization.shared_budget import plot_budget_curves

DEFAULT_FIGURE_DIR = "constrained_fm/images/thesis_pool/shared_budget"
XLABEL = "Shared points $N$"
COLORS = {results.FUNCTA: METHOD_COLORS["functa"], results.FEWSHOT: METHOD_COLORS["fewshot"]}

PANELS = (
    BarPanel("success_rate", "Acceptance Rate (%)", vmin=0.0, vmax=100.0),
    BarPanel("swd", "SWD", log_y=True, vmin=0.0),
    BarPanel("mmd", "MMD", log_y=True, vmin=0.0),
    BarPanel("kld", "KLD"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the shared-budget comparison curves.")
    parser.add_argument("--metrics", default=results.DEFAULT_METRICS, help="merged sweep")
    parser.add_argument("--min-inside", type=int, default=results.DEFAULT_MIN_INSIDE,
                        help="inside points a constraint needs at the smallest budget")
    parser.add_argument("--figure-dir", default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--name", default="shared_budget_curves", help="figure stem")
    parser.add_argument("--ncols", type=int, default=2, help="panels per row")
    parser.add_argument("--panel-width", type=float, default=7.2)
    parser.add_argument("--panel-height", type=float, default=5.2)
    parser.add_argument("--spread", nargs="+", default=["iqr"], choices=sorted(SPREAD_MODES),
                        help="band each curve shows; one figure is rendered per mode")
    return parser


def metric_series(by_method: dict, n_values: list[int],
                  indices: list[int]) -> dict[str, dict[str, np.ndarray]]:
    """metric key -> method -> (num_budgets, num_eligible) array."""
    return {panel.key: {method: np.asarray([results.column(by_method[method][n], panel.key,
                                                           indices) for n in n_values])
                        for method in results.METHODS}
            for panel in PANELS}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = results.load(args.metrics)
    by_method = results.budgets(payload)
    n_values = sorted(by_method[results.FUNCTA])
    indices = results.eligible(by_method, args.min_inside)
    series = metric_series(by_method, n_values, indices)
    n_inside = np.asarray([results.column(by_method[results.FUNCTA][n], "n_inside", indices)
                           for n in n_values])

    out = Path(args.metrics).parent
    artifacts.save_arrays(out, n_values=np.asarray(n_values), indices=np.asarray(indices),
                          mass=np.asarray(payload["mass"], dtype=float)[indices],
                          n_inside=n_inside,
                          **{f"{method}__{key}": values for key, by_key in series.items()
                             for method, values in by_key.items()})
    run_id = next(iter(by_method[results.FUNCTA].values()))["run_id"]
    artifacts.write_manifest(out, run_id=run_id, method="shared_budget_v1k",
                             validation_set=payload["validation_set"],
                             poly_digest=payload["poly_digest"],
                             num_constraints=payload["num_constraints"],
                             num_eligible=len(indices), min_inside=args.min_inside,
                             n_values=n_values)

    print(f"{len(indices)}/{payload['num_constraints']} constraints with >= {args.min_inside} "
          f"inside points at N={n_values[0]}")
    for spread in args.spread:
        suffix = "" if spread == args.spread[0] else f"_{spread}"
        fig = plot_budget_curves(n_values, series, PANELS, labels=results.LABELS,
                                 colors=COLORS, xlabel=XLABEL, ncols=args.ncols,
                                 panel_size=(args.panel_width, args.panel_height),
                                 spread=spread)
        path = save_figure(fig, Path(args.figure_dir) / f"{args.name}{suffix}.png")
        print(f"wrote {path} and {path.with_suffix('.pdf')}")

    print(f"\nmedians, {' / '.join(results.METHODS)}")
    print(f"{'N':>6}{'N_in':>7}" + "".join(f"{panel.key:>24}" for panel in PANELS))
    for row, n in enumerate(n_values):
        cells = ""
        for panel in PANELS:
            pair = " / ".join(f"{np.nanmedian(series[panel.key][method][row]):.4g}"
                              for method in results.METHODS)
            cells += f"{pair:>24}"
        print(f"{n:>6}{np.median(n_inside[row]):>7.0f}{cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

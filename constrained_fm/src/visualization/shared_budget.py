# -*- coding: utf-8 -*-
"""Metric-vs-budget curves for the shared-budget sweep: one line per method, one panel per metric.

Each line is the per-budget median across constraints with a percentile band, so the two
methods are read off the same x positions. N is drawn on a log axis with a tick at every
budget, since the budgets span two orders of magnitude.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullLocator

from constrained_fm.src.visualization.query_budget import (BAR_RC, SPREAD_MODES, BarPanel,
                                                           panel_statistics)

BAND_ALPHA = 0.18


def plot_budget_curves(n_values: Sequence[int], series: dict[str, dict[str, np.ndarray]],
                       panels: Sequence[BarPanel], labels: dict[str, str],
                       colors: dict[str, str], xlabel: str, ncols: int = 2,
                       panel_size: tuple[float, float] = (7.2, 5.2),
                       spread: str = "iqr") -> Figure:
    """Grid of panels, each overlaying every method's median curve and spread band.

    Args:
        n_values: the budgets, ascending.
        series: metric key -> method -> (num_budgets, num_constraints) array.
        panels: ``BarPanel`` specs, in order; those missing from ``series`` are skipped.
        labels: legend text per method, in drawing order.
        colors: line colour per method.
        xlabel: shared x-axis label.
        ncols: panels per row.
        panel_size: (width, height) in inches per panel.
        spread: key into ``SPREAD_MODES`` selecting the band.
    """
    if spread not in SPREAD_MODES:
        raise ValueError(f"unknown spread '{spread}'; expected one of {sorted(SPREAD_MODES)}")
    drawn = [panel for panel in panels if panel.key in series]
    if not drawn:
        raise ValueError(f"none of {[p.key for p in panels]} are present in the series")

    ncols = max(1, min(ncols, len(drawn)))
    nrows = -(-len(drawn) // ncols)
    x = np.asarray(n_values, dtype=float)

    with plt.rc_context(BAR_RC):
        fig, axs = plt.subplots(nrows, ncols, squeeze=False,
                                figsize=(panel_size[0] * ncols, panel_size[1] * nrows))
        flat = [ax for row in axs for ax in row]

        for position, (ax, panel) in enumerate(zip(flat, drawn)):
            for method in labels:
                centre, arms = panel_statistics(np.asarray(series[panel.key][method], float),
                                                spread=spread, panel=panel)
                color = colors[method]
                ax.fill_between(x, centre - arms[0], centre + arms[1], color=color,
                                alpha=BAND_ALPHA, linewidth=0)
                ax.plot(x, centre, color=color, linewidth=2.6, marker="o", markersize=7)

            ax.set_xscale("log")
            if panel.log_y:
                ax.set_yscale("log")
            ax.xaxis.set_major_locator(FixedLocator(x))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xticklabels([str(n) for n in n_values])
            if panel.vmax is not None and not panel.log_y:
                ax.set_ylim(top=min(ax.get_ylim()[1], panel.vmax * 1.005))
            if position >= len(drawn) - ncols:
                ax.set_xlabel(xlabel)
            ax.set_ylabel(panel.label)
            ax.grid(True, which="major", alpha=0.25)
            ax.set_axisbelow(True)

        for ax in flat[len(drawn):]:
            ax.set_visible(False)

        handles = [Line2D([0], [0], color=colors[method], lw=2.6, marker="o", markersize=7,
                          label=label) for method, label in labels.items()]
        handles.append(Patch(facecolor="#9ca3af", alpha=BAND_ALPHA * 2,
                             label=SPREAD_MODES[spread][3]))
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
                   bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=(0, 0.06 / nrows, 1, 1))

    return fig


__all__ = ["plot_budget_curves"]

# -*- coding: utf-8 -*-
"""Bar chart for the inference-time CAVIA query-budget ablation.

One panel per metric, one bar per budget N, with a spread interval across the validation
constraints. Bars on a categorical axis rather than a line on a log axis: N is picked from a
shortlist rather than tuned continuously, and the categorical axis gives the small budgets
the same width as the large ones instead of crowding them into the left margin, which is
where the claim -- that inference needs far fewer points than meta-training -- actually lives.

The per-constraint distributions are bounded (IoU <= 1, acceptance <= 100%, SWD >= 0) and
heavily skewed, because the v1k set is stratified uniformly over constraint mass and its
small-mass constraints form a long lower tail. A symmetric mean +- SD interval therefore
draws whiskers at values that cannot occur, so the default interval is built from percentiles
instead: order statistics of the observed constraints, which can never leave the achievable
range and which follow the skew rather than imposing a symmetry the data does not have.

The budget the SIREN was meta-trained at is drawn in a contrasting colour so the reader can
see directly that the bars to its left match it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from constrained_fm.src.visualization.style import PAPER_RC

BAR_COLOR = "#4C72B0"
REFERENCE_COLOR = "#C44E52"
ERROR_COLOR = "#1A1A1A"

# The figure is authored wide and lands about 7in across in a two-column layout, so every
# glyph shrinks by roughly a fifth on top of the reduction PAPER_RC already budgets for.
BAR_RC = {
    **PAPER_RC,
    "font.size": 20,
    "axes.labelsize": 24,
    # Seven categories, the widest of them four digits: any larger and 1000 touches 2000.
    "xtick.labelsize": 18,
    "ytick.labelsize": 20,
    "legend.fontsize": 21,
    # cmr10 has no upright glyphs for the log-axis exponents, so route them through mathtext.
    "axes.formatter.use_mathtext": True,
}

# name -> (centre statistic, lower percentile, upper percentile, legend text)
SPREAD_MODES = {
    "iqr": ("median", 25.0, 75.0, "Median, interquartile range"),
    "p5p95": ("median", 5.0, 95.0, "Median, 5th-95th percentile"),
    "sd": ("mean", None, None, r"Mean $\pm$ 1 SD, clipped to range"),
}


@dataclass(frozen=True)
class BarPanel:
    """One metric panel: which merged key it reads, and the bounds its axis must respect."""

    key: str
    label: str
    log_y: bool = False
    vmin: float | None = None
    vmax: float | None = None


def panel_statistics(values: np.ndarray, spread: str = "iqr",
                     panel: BarPanel | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-budget bar height and (2, K) whisker arm lengths.

    ``values`` is (num_budgets, num_constraints).
    """
    centre_name, low_q, high_q = SPREAD_MODES[spread][:3]
    finite = np.where(np.isfinite(values), values, np.nan)

    if centre_name == "median":
        centre = np.nanmedian(finite, axis=1)
        lower = np.nanpercentile(finite, low_q, axis=1)
        upper = np.nanpercentile(finite, high_q, axis=1)
    else:
        centre = np.nanmean(finite, axis=1)
        sd = np.nanstd(finite, axis=1)
        lower, upper = centre - sd, centre + sd
        if panel is not None:
            if panel.vmin is not None:
                lower = np.maximum(lower, panel.vmin)
            if panel.vmax is not None:
                upper = np.minimum(upper, panel.vmax)

    return centre, np.vstack([centre - lower, upper - centre])


def _axis_limits(centre: np.ndarray, arms: np.ndarray, panel: BarPanel,
                 zero_based: bool) -> tuple[float, float]:
    """Limits framing the bars and their whiskers.

    A log panel cannot be zero-based; a linear one is by default, because a bar whose baseline
    sits off-screen invites the reader to compare visible heights that are not proportional to
    the values. ``zero_based=False`` restores the truncated view.
    """
    high = float(np.max(centre + arms[1]))
    if panel.log_y:
        low = float(np.min(centre - arms[0]))
        return max(low, float(np.min(centre)) * 0.1) * 0.6, high * 1.6

    top = high + 0.08 * max(high, 1e-12)
    if panel.vmax is not None:
        top = min(top, panel.vmax * 1.02)
    if zero_based:
        return (panel.vmin or 0.0), top

    low = float(np.min(centre - arms[0]))
    return low - 0.12 * max(high - low, 1e-12), top


def plot_query_budget_bars(n_values, series, xlabel: str, panels, reference_n: int | None = None,
                           ncols: int = 2, panel_size: tuple[float, float] = (7.2, 5.2),
                           reference_label: str | None = None, spread: str = "iqr",
                           zero_based: bool = True) -> Figure:
    """Grid of bar panels over the query budgets.

    Args:
        n_values: the budgets, in plotting order.
        series: metric key -> (num_budgets, num_constraints) array.
        xlabel: shared x-axis label.
        panels: ``BarPanel`` specs, in order; those missing from ``series`` are skipped.
        reference_n: budget to highlight, typically the meta-training one.
        ncols: panels per row.
        panel_size: (width, height) in inches per panel.
        reference_label: legend text for the highlighted bar.
        spread: key into ``SPREAD_MODES`` selecting the interval the whiskers show.
        zero_based: anchor the linear panels' axes at the metric's floor.
    """
    if spread not in SPREAD_MODES:
        raise ValueError(f"unknown spread '{spread}'; expected one of {sorted(SPREAD_MODES)}")

    drawn = [panel for panel in panels if panel.key in series]
    if not drawn:
        raise ValueError(f"none of {[p.key for p in panels]} are present in the merged metrics")

    ncols = max(1, min(ncols, len(drawn)))
    nrows = -(-len(drawn) // ncols)
    positions = np.arange(len(n_values), dtype=float)

    with plt.rc_context(BAR_RC):
        fig, axs = plt.subplots(nrows, ncols, squeeze=False,
                                figsize=(panel_size[0] * ncols, panel_size[1] * nrows))
        flat = [ax for row in axs for ax in row]

        for position_in_grid, (ax, panel) in enumerate(zip(flat, drawn)):
            centre, arms = panel_statistics(np.asarray(series[panel.key], dtype=float),
                                            spread=spread, panel=panel)
            colors = [REFERENCE_COLOR if n == reference_n else BAR_COLOR for n in n_values]

            ax.bar(positions, centre, width=0.72, color=colors, edgecolor="black", linewidth=0.9,
                   yerr=arms, capsize=6,
                   error_kw={"ecolor": ERROR_COLOR, "elinewidth": 1.8, "capthick": 1.8},
                   zorder=3)

            if panel.log_y:
                ax.set_yscale("log")
            ax.set_ylim(*_axis_limits(centre, arms, panel, zero_based))
            ax.set_xticks(positions)
            ax.set_xticklabels([str(n) for n in n_values])
            ax.set_xlim(positions[0] - 0.6, positions[-1] + 0.6)
            # Every panel shares the same axis, so only the bottom of each column names it.
            if position_in_grid >= len(drawn) - ncols:
                ax.set_xlabel(xlabel)
            ax.set_ylabel(panel.label)
            ax.yaxis.grid(True, alpha=0.25, zorder=0)
            ax.set_axisbelow(True)

        for ax in flat[len(drawn):]:
            ax.set_visible(False)

        handles = [Line2D([0], [0], color=ERROR_COLOR, lw=1.8, marker="_", markersize=12,
                          label=SPREAD_MODES[spread][3])]
        if reference_n in n_values and reference_label:
            handles.insert(0, Patch(facecolor=REFERENCE_COLOR, edgecolor="black",
                                    label=reference_label))
            handles.insert(0, Patch(facecolor=BAR_COLOR, edgecolor="black",
                                    label="Inference budget"))

        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
                   bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=(0, 0.05 / nrows, 1, 1))

    return fig


def save_figure(fig: Figure, path: str | Path, dpi: int = 200) -> Path:
    """Writes the PNG the README embeds and the vector PDF the paper includes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


__all__ = ["BarPanel", "SPREAD_MODES", "panel_statistics", "plot_query_budget_bars",
           "save_figure"]

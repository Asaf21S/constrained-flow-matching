# -*- coding: utf-8 -*-
"""Bar chart for the inference-time CAVIA query-budget ablation.

One panel per metric, one bar per budget N, height = mean over the validation constraints
with +-1 standard deviation whiskers. Bars on a categorical axis rather than a line on a log
axis: N is picked from a shortlist rather than tuned continuously, and the categorical axis
gives the small budgets the same width as the large ones instead of crowding them into the
left margin, which is where the claim -- that inference needs far fewer points than
meta-training -- actually lives.

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
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 21,
    # cmr10 has no upright glyphs for the log-axis exponents, so route them through mathtext.
    "axes.formatter.use_mathtext": True,
}


@dataclass(frozen=True)
class BarPanel:
    """One metric panel: which merged key it reads and how its axis is drawn."""

    key: str
    label: str
    log_y: bool = False


def panel_statistics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-budget mean and population standard deviation, ignoring non-finite constraints.

    ``values`` is (num_budgets, num_constraints).
    """
    finite = np.where(np.isfinite(values), values, np.nan)
    return np.nanmean(finite, axis=1), np.nanstd(finite, axis=1)


def _error_arms(mean: np.ndarray, std: np.ndarray, log_y: bool) -> np.ndarray:
    """Asymmetric (2, K) whisker lengths.

    On a log axis a symmetric arm can reach zero or below, which Matplotlib cannot draw; the
    lower arm is then capped so the whisker stops just short of the axis instead of vanishing.
    """
    lower = std.copy()
    if log_y:
        lower = np.minimum(lower, mean * 0.9)
    return np.vstack([lower, std])


def _axis_limits(mean: np.ndarray, std: np.ndarray, log_y: bool) -> tuple[float, float]:
    """Limits framing the mean +- std band.

    The bars are deliberately not anchored at zero: the whole point of the sweep is that the
    budgets agree to within a few thousandths, which a zero-based axis renders as four
    identical rectangles.
    """
    low, high = float(np.min(mean - std)), float(np.max(mean + std))
    if log_y:
        low = max(low, float(np.min(mean)) * 0.1)
        return low * 0.6, high * 1.6

    pad = 0.12 * max(high - low, 1e-12)
    return low - pad, high + pad


def plot_query_budget_bars(n_values, series, xlabel: str, panels, reference_n: int | None = None,
                           ncols: int = 2, panel_size: tuple[float, float] = (6.4, 5.2),
                           reference_label: str | None = None) -> Figure:
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
    """
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
            mean, std = panel_statistics(np.asarray(series[panel.key], dtype=float))
            colors = [REFERENCE_COLOR if n == reference_n else BAR_COLOR for n in n_values]

            ax.bar(positions, mean, width=0.72, color=colors, edgecolor="black", linewidth=0.9,
                   yerr=_error_arms(mean, std, panel.log_y), capsize=6,
                   error_kw={"ecolor": ERROR_COLOR, "elinewidth": 1.8, "capthick": 1.8},
                   zorder=3)

            if panel.log_y:
                ax.set_yscale("log")
            ax.set_ylim(*_axis_limits(mean, std, panel.log_y))
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
                          label=r"Mean $\pm$ 1 SD")]
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


__all__ = ["BarPanel", "panel_statistics", "plot_query_budget_bars", "save_figure"]

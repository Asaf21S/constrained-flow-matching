# -*- coding: utf-8 -*-
"""Cross-method comparison figures for the v1k benchmark.

Two formats, both pure functions of the merged metric arrays:

* **Trend lines** -- one panel per metric, every method on the same axes, drawn as a median
  through a window sliding along sorted x with an interquartile band. Every point summarises
  the same number of constraints, so the band's width reports spread rather than how many
  constraints happened to fall in a bin.
* **Parity scatter** -- one method's metric against another's, constraint by constraint,
  with a dashed y = x line. Below the line means the y method won that constraint.

Nothing here loads a checkpoint, samples, or integrates an ODE.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

METHOD_ORDER = ("gt", "coeff", "functa", "eci", "hardflow")

METHOD_LABELS = {
    "gt": "Ground Truth (rejection sampling)",
    "coeff": "Coefficients (ours)",
    "functa": "Functa (ours)",
    "eci": "ECI",
    "hardflow": "HardFlow",
}

# Used wherever panels sit side by side and the full legend labels would collide.
METHOD_SHORT = {
    "gt": "Ground Truth",
    "coeff": "Coefficients",
    "functa": "Functa (ours)",
    "eci": "ECI",
    "hardflow": "HardFlow",
}

METHOD_COLORS = {
    "gt": "#111827",
    "coeff": "#2563eb",
    "functa": "#dc2626",
    "eci": "#0d9488",
    "hardflow": "#d97706",
}

METHOD_STYLES = {"gt": (0, (6, 4)), "coeff": "-", "functa": "-", "eci": "-", "hardflow": "-"}

# label, axis title, log scale. SWD/MMD/JSD span orders of magnitude across the mass range;
# NLL and KLD do not, and KLD dips below zero on the finite-sample estimate.
METRIC_SPECS = {
    "success_rate": ("AR", "Acceptance Rate (%)", False),
    "swd": ("SWD", "Sliced Wasserstein Distance", True),
    "mmd": ("MMD", "Maximum Mean Discrepancy", True),
    "jsd": ("JSD", "Jensen-Shannon Divergence", True),
    "nll": ("NLL", "Negative Log-Likelihood", False),
    "kld": ("KLD", r"$\mathrm{KL}(p_{\mathrm{true}} \Vert p_{\mathrm{model}})$", False),
}

GRID_STYLE = {"color": "#d1d5db", "linewidth": 0.6, "alpha": 0.7}


def label_for(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def short_label(method: str) -> str:
    return METHOD_SHORT.get(method, method)


def finite_pairs(x, y) -> tuple[np.ndarray, np.ndarray]:
    """Rows where both series are finite; every plot here needs the pairing intact."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def positive_pairs(x, y) -> tuple[np.ndarray, np.ndarray]:
    """Finite rows that are also strictly positive, for log-scaled axes."""
    x, y = finite_pairs(x, y)
    keep = (x > 0) & (y > 0)
    return x[keep], y[keep]


def rolling_trend(x, y, window: int = 100, step: int = 25
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Median and interquartile band of y over a window sliding along sorted x.

    Returns (centre, median, lower, upper). Every point summarises exactly ``window``
    constraints, so the band's width reports spread and never sample scarcity.
    """
    x, y = finite_pairs(x, y)
    if x.size == 0:
        empty = np.empty(0)
        return empty, empty, empty, empty

    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    window = min(window, x.size)

    starts = list(range(0, x.size - window + 1, max(step, 1)))
    # The final partial step would otherwise drop the upper tail entirely.
    if starts[-1] != x.size - window:
        starts.append(x.size - window)

    centre, median, lower, upper = [], [], [], []
    for start in starts:
        xs, ys = x[start:start + window], y[start:start + window]
        centre.append(float(np.median(xs)))
        median.append(float(np.median(ys)))
        lower.append(float(np.quantile(ys, 0.25)))
        upper.append(float(np.quantile(ys, 0.75)))

    return (np.asarray(centre), np.asarray(median), np.asarray(lower), np.asarray(upper))


def plot_metric_trend(series: dict[str, tuple[np.ndarray, np.ndarray]], xlabel: str,
                      ylabel: str, title: str, logx: bool = False, logy: bool = False,
                      window: int = 100, step: int = 25, identity: bool = False,
                      identity_label: str = "parity ($y = x$)",
                      identity_color: str = "#9ca3af",
                      labels: dict[str, str] | None = None,
                      figsize: tuple[float, float] = (7.6, 5.2)) -> Figure:
    """One rolling-median line with an interquartile band per method, on shared axes.

    ``series`` maps a method name to its (x, y) arrays; methods are drawn in METHOD_ORDER so
    the legend reads the same on every figure. ``labels`` overrides legend text per method.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ordered = [m for m in METHOD_ORDER if m in series] + \
              [m for m in series if m not in METHOD_ORDER]
    labels = labels or {}

    for method in ordered:
        x, y = series[method]
        x, y = (positive_pairs(x, y) if logy else finite_pairs(x, y))
        centre, median, lower, upper = rolling_trend(x, y, window=window, step=step)
        if centre.size == 0:
            continue
        color = METHOD_COLORS.get(method, "#6b7280")
        ax.fill_between(centre, lower, upper, color=color, alpha=0.15, linewidth=0)
        ax.plot(centre, median, color=color, linewidth=2.0,
                linestyle=METHOD_STYLES.get(method, "-"), marker="o", markersize=3.0,
                label=labels.get(method, label_for(method)))

    if identity:
        lo, hi = ax.get_xlim()
        span = np.geomspace(max(lo, 1e-12), hi, 64) if logx else np.linspace(lo, hi, 64)
        ax.plot(span, span, color=identity_color, linestyle=(0, (5, 5)), linewidth=1.4,
                zorder=0, label=identity_label)
        ax.set_xlim(lo, hi)

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", **GRID_STYLE)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def win_rate(x: np.ndarray, y: np.ndarray) -> float:
    """Fraction of constraints where the y method scored strictly lower than the x method."""
    return float(np.mean(y < x) * 100.0) if x.size else float("nan")


def plot_parity(x, y, xlabel: str, ylabel: str, title: str, color_by=None,
                color_label: str = "True constraint mass (%)", log: bool = True,
                annotate: str | None = None, ax=None,
                figsize: tuple[float, float] = (5.4, 5.2)) -> Figure:
    """One constraint per point, our metric against a baseline's, with a dashed y = x line.

    Points below the line are constraints the y method handled better. Axes share one range
    so the diagonal is visually at 45 degrees and the comparison is not distorted.
    """
    own_figure = ax is None
    fig, ax = plt.subplots(figsize=figsize) if own_figure else (ax.figure, ax)

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    keep = np.isfinite(x_arr) & np.isfinite(y_arr)
    if log:
        keep &= (x_arr > 0) & (y_arr > 0)
    x_arr, y_arr = x_arr[keep], y_arr[keep]
    shading = None if color_by is None else np.asarray(color_by, dtype=float)[keep]

    scatter = ax.scatter(x_arr, y_arr, c=shading if shading is not None else "#2563eb",
                         cmap="viridis" if shading is not None else None, s=13,
                         alpha=0.75, linewidths=0.0)

    if x_arr.size:
        lo = float(min(x_arr.min(), y_arr.min()))
        hi = float(max(x_arr.max(), y_arr.max()))
        pad = (hi / lo) ** 0.05 if log and lo > 0 else (hi - lo) * 0.05
        lo, hi = (lo / pad, hi * pad) if log and lo > 0 else (lo - pad, hi + pad)
        ax.plot([lo, hi], [lo, hi], color="#111827", linestyle=(0, (5, 5)), linewidth=1.2,
                zorder=3)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", **GRID_STYLE)
    ax.set_axisbelow(True)

    caption = annotate if annotate is not None else \
        f"below the line: {win_rate(x_arr, y_arr):.1f}% of {x_arr.size}"
    ax.text(0.03, 0.97, caption, transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#d1d5db",
                      alpha=0.9))

    if own_figure:
        if shading is not None:
            fig.colorbar(scatter, ax=ax, label=color_label, fraction=0.046, pad=0.04)
        fig.tight_layout()
    return fig


def plot_parity_grid(panels: list[dict], suptitle: str, ncols: int = 2, log: bool = True,
                     color_label: str = "True constraint mass (%)",
                     panel_size: tuple[float, float] = (4.6, 4.4)) -> Figure:
    """The head-to-head panels of one metric on a shared grid, with a single colourbar.

    Each entry of ``panels`` carries x, y, xlabel, ylabel, title and optional color_by.
    """
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
                             squeeze=False)
    flat = axes.ravel()

    mappable = None
    for ax, panel in zip(flat, panels):
        plot_parity(panel["x"], panel["y"], panel["xlabel"], panel["ylabel"], panel["title"],
                    color_by=panel.get("color_by"), log=log, annotate=panel.get("annotate"),
                    ax=ax)
        if panel.get("color_by") is not None and ax.collections:
            mappable = ax.collections[0]
    for ax in flat[len(panels):]:
        ax.axis("off")

    fig.tight_layout(rect=(0, 0, 0.92 if mappable is not None else 1.0, 0.97))
    if mappable is not None:
        cbar_ax = fig.add_axes((0.94, 0.12, 0.015, 0.76))
        fig.colorbar(mappable, cax=cbar_ax, label=color_label)
    return fig


def save_figure(fig: Figure, path: str | Path, dpi: int = 200) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


__all__ = ["METHOD_ORDER", "METHOD_LABELS", "METHOD_SHORT", "METHOD_COLORS", "METRIC_SPECS",
           "label_for", "short_label", "finite_pairs", "positive_pairs", "rolling_trend",
           "plot_metric_trend", "win_rate", "plot_parity", "plot_parity_grid", "save_figure"]

# -*- coding: utf-8 -*-
"""Figures for the 6D two-body kinematics problem.

Every function here is a pure function of numpy arrays: nothing loads a checkpoint, samples,
or integrates an ODE.

The state is the pair of massless three-momenta
:math:`x = (p_{x1}, p_{y1}, p_{z1}, p_{x2}, p_{y2}, p_{z2})`, and the constrained quantity is
the invariant mass of the pair,

.. math::
    M(x) = \\sqrt{2 p_{T1} p_{T2} (\\cosh \\Delta\\eta - \\cos \\Delta\\phi)},

so the feasible set :math:`\\{x : |M(x) - M_\\star| \\le \\epsilon\\}` is a curved shell of
codimension one rather than a convex body. A shell has no interior point in the sense a
projection method wants, which is why its behaviour here differs from the polygon case.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from constrained_fm.src.visualization.style import PAPER_RC

plt.rcParams.update(PAPER_RC)

GRID_STYLE = {"color": "#d1d5db", "linewidth": 0.6, "alpha": 0.7}
WINDOW_FACE = "#fca5a5"

COMPONENT_LABELS = ("$p_{x1}$", "$p_{y1}$", "$p_{z1}$",
                    "$p_{x2}$", "$p_{y2}$", "$p_{z2}$")

SERIES_COLORS = {
    "target": "#94a3b8",
    "truth": "#111827",
    "explicit": "#dc2626",
    "eci": "#0d9488",
    "hardflow": "#d97706",
}

SERIES_LABELS = {
    "target": "Unconstrained target",
    "truth": "Exact conditional",
    "explicit": "Explicit conditioning (ours)",
    "eci": "ECI",
    "hardflow": "HardFlow",
}


def _color(name: str) -> str:
    return SERIES_COLORS.get(name, "#6b7280")


def _label(name: str) -> str:
    return SERIES_LABELS.get(name, name)


def plot_mass_spectrum(mass: dict[str, np.ndarray], window: tuple[float, float],
                       bins: int = 120, zoom_bins: int = 60, zoom_halfwidth: float = 2.0,
                       zoom_headroom: float = 2.5,
                       figsize: tuple[float, float] = (15.0, 5.4)) -> Figure:
    """Invariant-mass spectra: the full range on a log axis, and a linear zoom on the window.

    Every histogram is normalised by its own sample count, so the height of a bar is
    comparable across methods and panels regardless of how many points fall in view. The
    zoom spans the window centre :math:`\\pm` ``zoom_halfwidth`` half-widths and drops the
    unconstrained target, whose density there is negligible; the exact conditional is drawn
    as a filled black line so the methods can be read against it. The zoom's vertical axis
    stops at ``zoom_headroom`` times the exact conditional's tallest bar, and any method
    exceeding it is labelled with its clipped peak height.
    """
    fig, (full, zoom) = plt.subplots(1, 2, figsize=figsize)
    finite = {name: values[np.isfinite(values)] for name, values in mass.items()}
    pooled = np.concatenate(list(finite.values()))
    centre, half = 0.5 * (window[0] + window[1]), 0.5 * (window[1] - window[0])
    ranges = {"full": np.linspace(0.0, float(np.quantile(pooled, 0.995)), bins + 1),
              "zoom": np.linspace(centre - zoom_halfwidth * half,
                                  centre + zoom_halfwidth * half, zoom_bins + 1)}

    peaks: dict[str, float] = {}
    for ax, key in ((full, "full"), (zoom, "zoom")):
        edges = ranges[key]
        width = float(edges[1] - edges[0])
        ax.axvspan(window[0], window[1], color=WINDOW_FACE, alpha=0.45, zorder=0,
                   label="requested window")
        for name, values in finite.items():
            if key == "zoom" and name == "target":
                continue
            weights = np.full(values.shape, 1.0 / (max(values.size, 1) * width))
            if key == "zoom" and name == "truth":
                heights, _, _ = ax.hist(values, bins=edges, weights=weights, color=_color(name),
                                        histtype="stepfilled", alpha=0.15, linewidth=0.0)
                ax.hist(values, bins=edges, weights=weights, color=_color(name),
                        histtype="step", linewidth=2.6, zorder=4)
            else:
                heights, _, _ = ax.hist(values, bins=edges, weights=weights,
                                        color=_color(name), label=_label(name),
                                        histtype="step", linewidth=2.2)
            if key == "zoom":
                peaks[name] = float(np.max(heights, initial=0.0))
        ax.set_xlabel("Invariant mass $M$ [GeV]")
        ax.grid(True, which="both", **GRID_STYLE)
        ax.set_axisbelow(True)

    full.set_yscale("log")
    full.set_ylabel("density")
    full.set_title("full range (log scale)", fontsize=15)
    zoom.set_xlim(ranges["zoom"][0], ranges["zoom"][-1])
    if peaks.get("truth", 0.0) > 0.0:
        top = zoom_headroom * peaks["truth"]
        zoom.set_ylim(0.0, top)
        clipped = [f"{_label(name)} peak {peak:.2f} (clipped)"
                   for name, peak in peaks.items() if peak > top]
        if clipped:
            zoom.text(0.02, 0.97, "\n".join(clipped), transform=zoom.transAxes, va="top",
                      ha="left", fontsize=12)
    zoom.set_title("zoom on the window (linear scale)", fontsize=15)
    full.legend(loc="upper right", frameon=False, labelspacing=0.35)
    fig.tight_layout()
    return fig


def plot_kinematic_marginals(series: dict[str, dict[str, np.ndarray]], bins: int = 80,
                             figsize: tuple[float, float] = (14.0, 4.4)) -> Figure:
    """The :math:`(p_T, \\eta, \\phi)` marginals of the leading particle.

    A method can land every sample inside the mass window and still be wrong here: the window
    fixes one scalar function of six coordinates and leaves the rest of the distribution free.
    """
    names = ["pt", "eta", "phi"]
    axis_labels = {"pt": "$p_{T1}$ [GeV]", "eta": "$\\eta_1$", "phi": "$\\phi_1$"}
    fig, axes = plt.subplots(1, len(names), figsize=figsize, squeeze=False)

    for ax, key in zip(axes.ravel(), names):
        pooled = np.concatenate([values[key] for values in series.values()])
        edges = np.linspace(float(np.quantile(pooled, 0.002)),
                            float(np.quantile(pooled, 0.998)), bins + 1)
        for name, values in series.items():
            ax.hist(values[key], bins=edges, density=True, histtype="step", linewidth=2.2,
                    color=_color(name), label=_label(name))
        ax.set_xlabel(axis_labels[key])
        ax.grid(True, **GRID_STYLE)
        ax.set_axisbelow(True)
    axes.ravel()[0].set_ylabel("density")
    axes.ravel()[0].legend(loc="upper right", frameon=False, fontsize=13, labelspacing=0.3)
    fig.tight_layout()
    return fig


def plot_corner(samples: dict[str, np.ndarray], bins: int = 60,
                labels: Sequence[str] = COMPONENT_LABELS,
                panel_size: float = 1.65) -> Figure:
    """Lower-triangle corner plot of the six Cartesian momentum components.

    The diagonal holds 1D marginals and the off-diagonal panels hold contours of a 2D
    histogram, one contour set per series, so a method that matches every 1D marginal while
    breaking the correlations is still visibly wrong.
    """
    dim = next(iter(samples.values())).shape[1]
    fig, axes = plt.subplots(dim, dim, figsize=(panel_size * dim, panel_size * dim),
                             squeeze=False)

    pooled = np.concatenate(list(samples.values()), axis=0)
    limits = [(float(np.quantile(pooled[:, d], 0.005)),
               float(np.quantile(pooled[:, d], 0.995))) for d in range(dim)]

    for row in range(dim):
        for col in range(dim):
            ax = axes[row][col]
            if col > row:
                ax.axis("off")
                continue

            if row == col:
                edges = np.linspace(*limits[row], bins + 1)
                for name, values in samples.items():
                    ax.hist(values[:, row], bins=edges, density=True, histtype="step",
                            linewidth=1.8, color=_color(name))
                ax.set_yticks([])
            else:
                for name, values in samples.items():
                    counts, xedges, yedges = np.histogram2d(
                        values[:, col], values[:, row], bins=bins,
                        range=[limits[col], limits[row]])
                    if counts.max() <= 0:
                        continue
                    centres_x = 0.5 * (xedges[:-1] + xedges[1:])
                    centres_y = 0.5 * (yedges[:-1] + yedges[1:])
                    ax.contour(centres_x, centres_y, counts.T / counts.max(),
                               levels=(0.1, 0.4, 0.75), colors=_color(name),
                               linewidths=1.2)
                ax.set_ylim(*limits[row])
            ax.set_xlim(*limits[col])

            ax.tick_params(labelsize=10)
            if row != dim - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(labels[col], fontsize=13)
            if col != 0 or row == 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(labels[row], fontsize=13)

    handles = [plt.Line2D([], [], color=_color(name), linewidth=2.0, label=_label(name))
               for name in samples]
    fig.legend(handles=handles, loc="upper right", frameon=False,
               bbox_to_anchor=(0.98, 0.98), fontsize=14)
    fig.tight_layout()
    return fig


__all__ = ["plot_mass_spectrum", "plot_kinematic_marginals", "plot_corner",
           "COMPONENT_LABELS", "SERIES_COLORS", "SERIES_LABELS"]

# -*- coding: utf-8 -*-
"""Figures for the 2D bump-hunting problem.

Every function here is a pure function of numpy arrays: nothing loads a checkpoint, samples,
or integrates an ODE. The caller computes the arrays once and persists them under
``artifacts/``, so a figure can be restyled without re-running the model.

The target is the mixture

.. math::
    p(x) = (1 - w)\\,p_{\\mathrm{bg}}(x) + w\\,\\mathcal{N}(x; \\mu_s, \\Sigma_s),

with :math:`w = 0.01` and a background that factorises into two truncated exponentials on
:math:`[0, L]^2`. The signal sits far out in the :math:`x_2` tail of the background, so it
carries a negligible share of the total mass and is invisible in an unconditional marginal;
it becomes the dominant component only inside a polygon that cuts away the bulk.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon as PolygonPatch

from constrained_fm.src.visualization.style import PAPER_RC

plt.rcParams.update(PAPER_RC)

DENSITY_CMAP = "magma"
POLYGON_EDGE = "#f8fafc"
SIGNAL_MARK = "#22d3ee"
GRID_STYLE = {"color": "#d1d5db", "linewidth": 0.6, "alpha": 0.7}

SERIES_COLORS = {
    "target": "#111827",
    "filtered": "#d97706",
    "generated": "#dc2626",
    "truth": "#0d9488",
    "eci": "#7c3aed",
    "hardflow": "#0284c7",
}


def half_plane_vertices(normals: np.ndarray, offsets: np.ndarray,
                        rel_tol: float = 1e-5) -> np.ndarray:
    """Counter-clockwise vertices of the convex polygon :math:`\\{x : Ax \\le b\\}`.

    Every pair of half-planes is intersected and the points that satisfy all of the
    remaining constraints are kept, which is :math:`O(K^3)` but exact and needs no external
    geometry dependency at the handful of half-planes this problem uses.

    The inputs usually arrive as float32, whose rounding alone puts a true vertex up to
    :math:`\\sim 10^{-6} L` outside its neighbouring faces, so the feasibility test is
    done in float64 with a tolerance relative to the offsets' scale.
    """
    normals = np.asarray(normals, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    tol = rel_tol * max(1.0, float(np.abs(offsets).max()))
    points: list[np.ndarray] = []
    num = normals.shape[0]
    for i in range(num):
        for j in range(i + 1, num):
            matrix = np.stack([normals[i], normals[j]])
            if abs(np.linalg.det(matrix)) < 1e-12:
                continue
            point = np.linalg.solve(matrix, np.array([offsets[i], offsets[j]]))
            if np.all(normals @ point <= offsets + tol):
                points.append(point)
    if not points:
        return np.empty((0, 2))

    hull = np.unique(np.round(np.stack(points), 9), axis=0)
    angle = np.arctan2(hull[:, 1] - hull[:, 1].mean(), hull[:, 0] - hull[:, 0].mean())
    return hull[np.argsort(angle)]


def _draw_polygon(ax, normals: np.ndarray, offsets: np.ndarray, color: str = POLYGON_EDGE,
                  linewidth: float = 2.0, fill: bool = False) -> None:
    vertices = half_plane_vertices(normals, offsets)
    if vertices.shape[0] < 3:
        return
    ax.add_patch(PolygonPatch(vertices, closed=True, fill=fill,
                              facecolor=color if fill else "none", edgecolor=color,
                              linewidth=linewidth, alpha=0.25 if fill else 1.0, zorder=4))


def plot_target_density(log_density: np.ndarray, domain: float, signal_mean: Sequence[float],
                        floor: float = 1e-8,
                        figsize: tuple[float, float] = (12.4, 5.4)) -> Figure:
    """Heat maps of :math:`p(x)` over :math:`[0, L]^2`, linear on the left and log on the right.

    The linear panel is what the eye sees in the data: the bump is a faint smudge a few
    percent of the peak height. The log panel shows the full dynamic range.
    """
    density = np.maximum(np.exp(log_density), floor)
    norms = {"linear": matplotlib.colors.Normalize(vmin=0.0, vmax=float(density.max())),
             "log": matplotlib.colors.LogNorm(vmin=float(np.quantile(density, 0.02)),
                                              vmax=float(density.max()))}
    fig, axes = plt.subplots(1, 2, figsize=figsize, squeeze=False)
    for ax, (scale, norm) in zip(axes.ravel(), norms.items()):
        mesh = ax.imshow(density, origin="lower", extent=(0.0, domain, 0.0, domain),
                         cmap=DENSITY_CMAP, norm=norm)
        ax.scatter([signal_mean[0]], [signal_mean[1]], s=90, facecolors="none",
                   edgecolors=SIGNAL_MARK, linewidths=2.0, zorder=5)
        ax.annotate("signal", xy=(signal_mean[0], signal_mean[1]),
                    xytext=(signal_mean[0] + 1.4, signal_mean[1] + 1.1), color=SIGNAL_MARK,
                    fontsize=15, arrowprops=dict(arrowstyle="->", color=SIGNAL_MARK, lw=1.6))
        ax.set_title(f"{scale} colour scale", fontsize=15)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        fig.colorbar(mesh, ax=ax, label="$p(x)$", fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_polygon_gallery(polygons: Sequence[tuple[np.ndarray, np.ndarray]],
                         masses: Sequence[float], background: np.ndarray,
                         domain: float, nrows: int = 2, ncols: int = 5,
                         panel_size: float = 2.9) -> Figure:
    """A grid of sampled constraints over a thinned draw from the target.

    Captions carry the constraint mass, which is the only scale on which two polygons are
    comparable: area alone ignores where the target puts its probability.
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size * ncols, panel_size * nrows),
                             squeeze=False)
    for ax, (normals, offsets), mass in zip(axes.ravel(), polygons, masses):
        ax.scatter(background[:, 0], background[:, 1], s=1.4, c="#cbd5e1", alpha=0.55,
                   linewidths=0.0)
        _draw_polygon(ax, normals, offsets, color="#dc2626", fill=True)
        ax.set_xlim(0.0, domain)
        ax.set_ylim(0.0, domain)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"mass {mass * 100:.1f}%", fontsize=14)
    for ax in axes.ravel()[len(polygons):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_signal_heatmaps(panels: Sequence[dict], normals: np.ndarray, offsets: np.ndarray,
                         extent: tuple[float, float, float, float],
                         signal_mean: Sequence[float], signal_radius: float, vmax: float,
                         panel_size: float = 3.4) -> Figure:
    """One conditional density estimate per method, inside one polygon, on one colour scale.

    Each entry of ``panels`` carries ``label``, ``density`` (a 2D histogram over ``extent``,
    rows along :math:`x_2`) and an optional multi-line ``caption``. All panels share a
    linear colour axis capped at ``vmax``; brighter pixels saturate rather than rescaling
    the others. The dashed circle marks the signal at radius ``signal_radius``. Captions sit
    under the panels, whose width follows the aspect ratio of ``extent``.
    """
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=vmax)
    aspect = (extent[1] - extent[0]) / (extent[3] - extent[2])
    width = max(panel_size * aspect, 2.4)
    fig, axes = plt.subplots(1, len(panels), figsize=(width * len(panels) + 1.2,
                                                      panel_size + 1.0),
                             squeeze=False, layout="constrained")
    for ax, panel in zip(axes.ravel(), panels):
        mesh = ax.imshow(panel["density"], origin="lower", extent=extent, cmap=DENSITY_CMAP,
                         norm=norm, aspect="equal", interpolation="nearest")
        _draw_polygon(ax, normals, offsets, color=POLYGON_EDGE, linewidth=1.8)
        ax.add_patch(Circle(tuple(signal_mean), signal_radius, fill=False,
                            edgecolor=SIGNAL_MARK, linestyle="--", linewidth=1.6, zorder=5))
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(panel["label"], fontsize=15)
        if panel.get("caption"):
            ax.set_xlabel(panel["caption"], fontsize=11)
    fig.colorbar(mesh, ax=axes.ravel().tolist(), extend="max", shrink=0.85,
                 label="conditional density")
    return fig


def plot_conditional_panels(panels: Sequence[dict], domain: float,
                            panel_size: float = 3.5) -> Figure:
    """One column per method for a single constraint, so the failure modes are side by side.

    Each entry carries ``label``, ``samples`` and the polygon ``normals``/``offsets``;
    ``caption`` is drawn in the corner and normally holds that method's success rate.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(panel_size * len(panels), panel_size),
                             squeeze=False)
    for ax, panel in zip(axes.ravel(), panels):
        samples = panel["samples"]
        ax.scatter(samples[:, 0], samples[:, 1], s=2.0, c=SERIES_COLORS["generated"],
                   alpha=0.35, linewidths=0.0)
        _draw_polygon(ax, panel["normals"], panel["offsets"], color="#111827", linewidth=1.8)
        ax.set_xlim(0.0, domain)
        ax.set_ylim(0.0, domain)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(panel["label"], fontsize=15)
        if panel.get("caption"):
            ax.text(0.04, 0.96, panel["caption"], transform=ax.transAxes, va="top",
                    ha="left", fontsize=13,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="#d1d5db", alpha=0.9))
    fig.tight_layout()
    return fig


__all__ = ["half_plane_vertices", "plot_target_density", "plot_polygon_gallery",
           "plot_signal_heatmaps", "plot_conditional_panels", "SERIES_COLORS"]

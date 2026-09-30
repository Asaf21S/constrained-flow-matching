# -*- coding: utf-8 -*-
"""Figures for test-time constraint discovery: a decoded boundary wrapping a subset of GMM modes.

Pure consumers of arrays. Fields follow ``meshgrid(axis, axis, indexing="ij")``:
``field[i, j]`` is the value at ``(x = axis[j], y = axis[i])``; the feasible side is ``field <= 0``.

* :func:`plot_discovery_frame` -- one snapshot: GMM scatter, shaded region, decoded zero level set.
* :func:`plot_discovery_strip` -- 2xK strip: scatter + boundary over the optimisation, and below
  each panel the x-marginal of the GMM points inside the boundary against the 3-mode target.
* :func:`plot_likelihood_map` -- FM density of one latent over the whole domain.
* :func:`plot_discovery_history` -- FM loss and per-mode inside fraction against step.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from constrained_fm.src.visualization.diagnostics import smooth_field
from constrained_fm.src.visualization.style import SERIF_RC


@dataclass
class DiscoveryStyle:
    """Every visual knob of the discovery figures."""

    target_color: str = "#1f77b4"
    excluded_color: str = "#d62728"
    point_size: float = 3.0
    point_alpha: float = 0.35
    region_color: str = "#2ca02c"
    region_alpha: float = 0.15
    boundary_color: str = "black"
    boundary_linewidth: float = 2.0
    smooth_sigma: float = 2.0      # blur before tracing; the w0=30 ripple fragments raw contours
    likelihood_cmap: str = "viridis"
    hist_ratio: float = 0.42       # histogram row height relative to a map panel
    hist_color: str = "#c2410c"
    hist_fill_alpha: float = 0.25
    hist_linewidth: float = 1.4
    gt_color: str = "black"
    gt_linewidth: float = 1.8
    gt_linestyle: str = "--"
    hist_headroom: float = 1.15
    grid_alpha: float = 0.25
    panel_size: float = 4.5
    title_size: float = 20.0
    caption_size: float = 15.0
    tick_size: float = 14.0
    legend_size: float = 15.0
    show_legend: bool = True
    dpi: int = 150


def get_style(**overrides: Any) -> DiscoveryStyle:
    return replace(DiscoveryStyle(), **overrides)


def _grid(resolution: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(-scale, scale, resolution)
    return np.meshgrid(axis, axis, indexing="xy")


def _draw_panel(ax, field: np.ndarray, points: np.ndarray, labels: np.ndarray, excluded_mode: int,
                scale: float, style: DiscoveryStyle) -> None:
    xx, yy = _grid(field.shape[0], scale)
    traced = smooth_field(np.asarray(field, dtype=np.float32), style.smooth_sigma)
    if traced.min() < 0.0:
        ax.contourf(xx, yy, traced, levels=[traced.min() - 1.0, 0.0], colors=[style.region_color],
                    alpha=style.region_alpha, zorder=1)

    excluded = labels == excluded_mode
    ax.scatter(points[~excluded, 0], points[~excluded, 1], s=style.point_size,
               c=style.target_color, alpha=style.point_alpha, linewidths=0, zorder=2,
               rasterized=True)
    ax.scatter(points[excluded, 0], points[excluded, 1], s=style.point_size,
               c=style.excluded_color, alpha=style.point_alpha, linewidths=0, zorder=2,
               rasterized=True)

    if traced.min() < 0.0 < traced.max():
        ax.contour(xx, yy, traced, levels=[0.0], colors=style.boundary_color,
                   linewidths=style.boundary_linewidth, zorder=4)

    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=style.tick_size)


def _legend_handles(style: DiscoveryStyle) -> list[Line2D]:
    return [
        Line2D([], [], marker="o", ls="", color=style.target_color, label="target modes"),
        Line2D([], [], marker="o", ls="", color=style.excluded_color, label="excluded mode"),
        Line2D([], [], color=style.boundary_color, lw=style.boundary_linewidth,
               label=r"decoded $f_\theta(x, z_c) = 0$"),
    ]


def _legend_above(fig: Figure, handles: list, ncol: int, style: DiscoveryStyle) -> None:
    """Legend outside the top edge; kept by ``savefig(bbox_inches="tight")``."""
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False,
               fontsize=style.legend_size, bbox_to_anchor=(0.5, 1.0))


def _density(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Histogram normalised to unit area; all zeros for an empty sample."""
    counts, _ = np.histogram(values, bins=edges)
    return counts / max(len(values), 1) / np.diff(edges)


def mode_caption(mode_inside: np.ndarray, excluded_mode: int) -> str:
    """``inside: m0 0.98, m1 0.99,`` / ``[m2 0.03], m3 0.97`` with the excluded mode bracketed."""
    parts = [f"[m{k} {v:.2f}]" if k == excluded_mode else f"m{k} {v:.2f}"
             for k, v in enumerate(mode_inside)]
    half = (len(parts) + 1) // 2
    return "inside: " + ", ".join(parts[:half]) + ",\n" + ", ".join(parts[half:])


def plot_discovery_frame(field: np.ndarray, points: np.ndarray, labels: np.ndarray,
                         excluded_mode: int, scale: float, title: str,
                         mode_inside: np.ndarray | None = None,
                         style: DiscoveryStyle | None = None) -> Figure:
    style = style or DiscoveryStyle()
    with plt.rc_context(SERIF_RC):
        fig, ax = plt.subplots(figsize=(style.panel_size, style.panel_size))
        _draw_panel(ax, field, points, labels, excluded_mode, scale, style)
        ax.set_title(title, fontsize=style.title_size)
        if mode_inside is not None:
            ax.set_xlabel(mode_caption(mode_inside, excluded_mode), fontsize=style.caption_size)
        fig.tight_layout()
        if style.show_legend:
            _legend_above(fig, _legend_handles(style), 1, style)
    return fig


def plot_discovery_strip(fields: np.ndarray, steps: Sequence[int], points: np.ndarray,
                         labels: np.ndarray, excluded_mode: int, scale: float,
                         inside_x: Sequence[np.ndarray], target_x: np.ndarray,
                         mode_inside: np.ndarray | None = None, bins: int = 120,
                         style: DiscoveryStyle | None = None) -> Figure:
    """Top: scatter, shaded region and decoded boundary; bottom: first-coordinate marginal.

    ``fields``: (K, R, R) decoded snapshots. ``inside_x[k]``: first coordinate of the labelled
    GMM points on the feasible side of snapshot k, shaded. ``target_x``: first coordinate of the
    3-mode optimisation target, dashed.
    """
    style = style or DiscoveryStyle()
    k = len(fields)

    edges = np.linspace(-scale, scale, bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    gt = _density(target_x, edges)
    hists = [_density(x, edges) for x in inside_x]
    y_max = style.hist_headroom * max([gt.max()] + [h.max() for h in hists])

    with plt.rc_context(SERIF_RC):
        fig = plt.figure(figsize=(style.panel_size * k,
                                  style.panel_size * (1.0 + style.hist_ratio) + 1.2),
                         layout="constrained")
        gs = fig.add_gridspec(2, k, height_ratios=[1.0, style.hist_ratio])
        top = [fig.add_subplot(gs[0, i]) for i in range(k)]
        bottom = [fig.add_subplot(gs[1, i]) for i in range(k)]
        for i in range(k):
            ax = top[i]
            _draw_panel(ax, fields[i], points, labels, excluded_mode, scale, style)
            ax.set_title(f"step {int(steps[i])}", fontsize=style.title_size)
            if mode_inside is not None:
                ax.set_xlabel(mode_caption(mode_inside[i], excluded_mode),
                              fontsize=style.caption_size)

            ax = bottom[i]
            ax.fill_between(centres, hists[i], step="mid", color=style.hist_color,
                            alpha=style.hist_fill_alpha, linewidth=0)
            ax.step(centres, hists[i], where="mid", color=style.hist_color,
                    linewidth=style.hist_linewidth)
            ax.step(centres, gt, where="mid", color=style.gt_color, linewidth=style.gt_linewidth,
                    linestyle=style.gt_linestyle)
            ax.set_xlim(-scale, scale)
            ax.set_ylim(0.0, y_max)
            ax.grid(True, alpha=style.grid_alpha)
            ax.tick_params(labelsize=style.tick_size)
            ax.set_xlabel(r"first coordinate $x$", fontsize=style.caption_size)
            if i > 0:
                top[i].tick_params(labelleft=False)
                ax.tick_params(labelleft=False)
        bottom[0].set_ylabel("density", fontsize=style.caption_size)
        if style.show_legend:
            handles = _legend_handles(style) + [
                Patch(facecolor=style.hist_color, alpha=style.hist_fill_alpha,
                      edgecolor=style.hist_color, label=r"GMM inside $f_\theta \leq 0$"),
                Line2D([], [], color=style.gt_color, lw=style.gt_linewidth,
                       ls=style.gt_linestyle, label="target (3 modes)"),
            ]
            _legend_above(fig, handles, len(handles), style)
    return fig


def plot_likelihood_map(likelihood: np.ndarray, scale: float, vmax: float,
                        style: DiscoveryStyle | None = None) -> Figure:
    """FM density on ``+-scale`` (row is y), laid out like the true-GMM likelihood figure.

    ``vmax`` is the target's peak density; exact-divergence blow-ups saturate instead of
    setting the colour scale.
    """
    style = style or DiscoveryStyle()
    norm = Normalize(vmin=0.0, vmax=vmax, clip=True)
    with plt.rc_context(SERIF_RC):
        fig, ax = plt.subplots(figsize=(6, 6))
        image = ax.imshow(np.nan_to_num(likelihood, nan=0.0, posinf=vmax), origin="lower",
                          extent=(-scale, scale, -scale, scale), cmap=style.likelihood_cmap,
                          norm=norm)
        ax.grid(False)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=style.tick_size)
        cbar = fig.colorbar(image, ax=ax, orientation="vertical")
        cbar.set_label("Model Density", fontsize=style.caption_size)
        cbar.ax.tick_params(labelsize=style.tick_size)
    return fig


def plot_discovery_history(losses: np.ndarray, snapshot_steps: np.ndarray,
                           eval_losses: np.ndarray, mode_inside: np.ndarray, excluded_mode: int,
                           ema: float = 0.98, style: DiscoveryStyle | None = None) -> Figure:
    """Left: per-step FM loss (+EMA) and fixed-batch loss; right: inside fraction per mode."""
    style = style or DiscoveryStyle()
    smoothed = np.empty_like(losses)
    acc = losses[0] if len(losses) else 0.0
    for i, v in enumerate(losses):
        acc = ema * acc + (1.0 - ema) * v
        smoothed[i] = acc

    with plt.rc_context(SERIF_RC):
        fig, (ax_l, ax_m) = plt.subplots(1, 2, figsize=(2 * style.panel_size + 1.5, style.panel_size))
        ax_l.plot(losses, color="0.75", lw=0.6, label="batch")
        ax_l.plot(smoothed, color="black", lw=1.4, label=f"EMA {ema}")
        ax_l.plot(snapshot_steps, eval_losses, "o-", color=style.target_color, ms=3,
                  label="fixed eval batch")
        ax_l.set_xlabel("step", fontsize=style.caption_size)
        ax_l.set_ylabel("FM loss", fontsize=style.caption_size)
        ax_l.tick_params(labelsize=style.tick_size)
        ax_l.legend(fontsize=style.legend_size)

        for m in range(mode_inside.shape[1]):
            is_excluded = m == excluded_mode
            ax_m.plot(snapshot_steps, mode_inside[:, m], ls="--" if is_excluded else "-",
                      color=style.excluded_color if is_excluded else None,
                      label=f"mode {m}" + (" (excluded)" if is_excluded else ""))
        ax_m.set_ylim(-0.02, 1.02)
        ax_m.set_xlabel("step", fontsize=style.caption_size)
        ax_m.set_ylabel(r"fraction with $f_\theta(x, z_c) \leq 0$", fontsize=style.caption_size)
        ax_m.tick_params(labelsize=style.tick_size)
        ax_m.legend(fontsize=style.legend_size)
        fig.tight_layout()
    return fig


__all__ = ["DiscoveryStyle", "get_style", "mode_caption", "plot_discovery_frame",
           "plot_discovery_strip", "plot_likelihood_map", "plot_discovery_history"]

# -*- coding: utf-8 -*-
"""Figures for test-time constraint discovery: a decoded boundary wrapping a subset of GMM modes.

Pure consumers of arrays. Fields follow ``meshgrid(axis, axis, indexing="ij")``:
``field[i, j]`` is the value at ``(x = axis[j], y = axis[i])``; the feasible side is ``field <= 0``.

* :func:`plot_discovery_frame` -- one snapshot: GMM scatter, shaded region, zero level set(s).
* :func:`plot_discovery_strip` -- 1xK strip of snapshots over the optimisation.
* :func:`plot_discovery_history` -- FM loss and per-mode inside fraction against step.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

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
    poly_color: str = "#ff7f0e"
    poly_linestyle: str = "--"
    poly_linewidth: float = 1.8
    smooth_sigma: float = 2.0      # blur before tracing; the w0=30 ripple fragments raw contours
    panel_size: float = 4.5
    title_size: float = 14.0
    caption_size: float = 10.0
    tick_size: float = 10.0
    legend_size: float = 10.0
    show_legend: bool = True
    dpi: int = 150


def get_style(**overrides: Any) -> DiscoveryStyle:
    return replace(DiscoveryStyle(), **overrides)


def _grid(resolution: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(-scale, scale, resolution)
    return np.meshgrid(axis, axis, indexing="xy")


def _draw_panel(ax, field: np.ndarray, points: np.ndarray, labels: np.ndarray, excluded_mode: int,
                scale: float, style: DiscoveryStyle, poly_field: np.ndarray | None = None) -> None:
    xx, yy = _grid(field.shape[0], scale)
    traced = smooth_field(np.asarray(field, dtype=np.float32), style.smooth_sigma)
    if traced.min() < 0.0:
        ax.contourf(xx, yy, traced, levels=[traced.min() - 1.0, 0.0], colors=[style.region_color],
                    alpha=style.region_alpha, zorder=1)

    excluded = labels == excluded_mode
    ax.scatter(points[~excluded, 0], points[~excluded, 1], s=style.point_size,
               c=style.target_color, alpha=style.point_alpha, linewidths=0, zorder=2)
    ax.scatter(points[excluded, 0], points[excluded, 1], s=style.point_size,
               c=style.excluded_color, alpha=style.point_alpha, linewidths=0, zorder=2)

    if traced.min() < 0.0 < traced.max():
        ax.contour(xx, yy, traced, levels=[0.0], colors=style.boundary_color,
                   linewidths=style.boundary_linewidth, zorder=4)
    if poly_field is not None and poly_field.min() < 0.0 < poly_field.max():
        ax.contour(xx, yy, poly_field, levels=[0.0], colors=style.poly_color,
                   linestyles=style.poly_linestyle, linewidths=style.poly_linewidth, zorder=3)

    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=style.tick_size)


def _legend_handles(style: DiscoveryStyle, with_poly: bool) -> list[Line2D]:
    handles = [
        Line2D([], [], marker="o", ls="", color=style.target_color, label="target modes"),
        Line2D([], [], marker="o", ls="", color=style.excluded_color, label="excluded mode"),
        Line2D([], [], color=style.boundary_color, lw=style.boundary_linewidth,
               label=r"decoded $f_\theta(x, z_c) = 0$"),
    ]
    if with_poly:
        handles.append(Line2D([], [], color=style.poly_color, ls=style.poly_linestyle,
                              lw=style.poly_linewidth, label=r"polynomial $P_C(x) = 0$"))
    return handles


def mode_caption(mode_inside: np.ndarray, excluded_mode: int) -> str:
    """``m0 0.98, m1 0.99, [m2 0.03]`` with the excluded mode bracketed."""
    parts = [f"[m{k} {v:.2f}]" if k == excluded_mode else f"m{k} {v:.2f}"
             for k, v in enumerate(mode_inside)]
    return "inside: " + ", ".join(parts)


def plot_discovery_frame(field: np.ndarray, points: np.ndarray, labels: np.ndarray,
                         excluded_mode: int, scale: float, title: str,
                         mode_inside: np.ndarray | None = None,
                         poly_field: np.ndarray | None = None,
                         style: DiscoveryStyle | None = None) -> Figure:
    style = style or DiscoveryStyle()
    with plt.rc_context(SERIF_RC):
        fig, ax = plt.subplots(figsize=(style.panel_size, style.panel_size))
        _draw_panel(ax, field, points, labels, excluded_mode, scale, style, poly_field)
        ax.set_title(title, fontsize=style.title_size)
        if mode_inside is not None:
            ax.set_xlabel(mode_caption(mode_inside, excluded_mode), fontsize=style.caption_size)
        if style.show_legend:
            ax.legend(handles=_legend_handles(style, poly_field is not None),
                      loc="upper left", fontsize=style.legend_size, framealpha=0.85)
        fig.tight_layout()
    return fig


def plot_discovery_strip(fields: np.ndarray, steps: Sequence[int], points: np.ndarray,
                         labels: np.ndarray, excluded_mode: int, scale: float,
                         mode_inside: np.ndarray | None = None,
                         poly_fields: np.ndarray | None = None,
                         style: DiscoveryStyle | None = None) -> Figure:
    """``fields``: (K, R, R) snapshots, one panel each, sharing axes."""
    style = style or DiscoveryStyle()
    k = len(fields)
    with plt.rc_context(SERIF_RC):
        fig, axes = plt.subplots(1, k, figsize=(style.panel_size * k, style.panel_size + 0.4),
                                 sharex=True, sharey=True, squeeze=False)
        for i, ax in enumerate(axes[0]):
            _draw_panel(ax, fields[i], points, labels, excluded_mode, scale, style,
                        None if poly_fields is None else poly_fields[i])
            ax.set_title(f"step {int(steps[i])}", fontsize=style.title_size)
            if mode_inside is not None:
                ax.set_xlabel(mode_caption(mode_inside[i], excluded_mode),
                              fontsize=style.caption_size)
            ax.label_outer()
        if style.show_legend:
            axes[0, 0].legend(handles=_legend_handles(style, poly_fields is not None),
                              loc="upper left", fontsize=style.legend_size, framealpha=0.85)
        fig.tight_layout()
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
        ax_l.set_xlabel("step")
        ax_l.set_ylabel("FM loss")
        ax_l.legend(fontsize=style.legend_size)

        for m in range(mode_inside.shape[1]):
            is_excluded = m == excluded_mode
            ax_m.plot(snapshot_steps, mode_inside[:, m], ls="--" if is_excluded else "-",
                      color=style.excluded_color if is_excluded else None,
                      label=f"mode {m}" + (" (excluded)" if is_excluded else ""))
        ax_m.set_ylim(-0.02, 1.02)
        ax_m.set_xlabel("step")
        ax_m.set_ylabel(r"fraction with $f_\theta(x, z_c) \leq 0$")
        ax_m.legend(fontsize=style.legend_size)
        fig.tight_layout()
    return fig


__all__ = ["DiscoveryStyle", "get_style", "mode_caption", "plot_discovery_frame",
           "plot_discovery_strip", "plot_discovery_history"]

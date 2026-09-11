# -*- coding: utf-8 -*-
"""Publication figures for the SIREN/CAVIA encoder: decoded fields and latent interpolations.

Two figure kinds, both pure consumers of already-decoded arrays:

* :func:`plot_encoder_panel` -- one constraint: the SIREN field as a heatmap, the true
  boundary dashed, the decoded zero level set solid.
* :func:`plot_interpolation_row` -- a 1xK strip of decoded fields along a latent geodesic,
  every panel sharing one colour scale so the drift is readable as colour, not just shape.

Fields follow the ``meshgrid(axis, axis, indexing="ij")`` convention used elsewhere:
``field[i, j]`` is the value at ``(x = axis[j], y = axis[i])``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from constrained_fm.src.consts import PLANE_SCALE
from constrained_fm.src.visualization.diagnostics import smooth_field

# Computer Modern, matching the LaTeX body text of the paper.
SERIF_RC: dict[str, Any] = {
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.grid": False,
}


@dataclass
class EncoderStyle:
    """Every visual knob of both figure kinds."""

    # --- decoded field ---
    cmap: str = "RdBu_r"
    field_render: str = "imshow"   # imshow keeps the PDF small and the gradient band-free
    levels: int = 60               # contourf only
    field_alpha: float = 1.0
    # Symmetric about 0 so the sign of the field, i.e. the inside/outside decision, maps to
    # the colour map's own midpoint.
    symmetric_scale: bool = True
    clip_to_unit: bool = False     # force vmin/vmax to -1/1, the range of tanh(P)

    # --- boundaries ---
    gt_color: str = "black"
    gt_linestyle: str = "--"
    gt_linewidth: float = 2.0
    pred_color: str = "#00c853"
    pred_linestyle: str = "-"
    pred_linewidth: float = 2.0
    # A w0=30 SIREN ripples around its own zero crossing; contour() on the raw field returns
    # fragments rather than a curve.
    smooth_sigma: float = 2.0

    # --- layout ---
    panel_size: float = 4.4
    tick_size: float = 11.0
    title_size: float = 15.0
    show_ticks: bool = True
    spine_color: str | None = None

    # --- colorbar ---
    show_colorbar: bool = True
    colorbar_label: str = r"$f_\theta(x, z)$"
    colorbar_fraction: float = 0.046
    colorbar_pad: float = 0.04
    label_size: float = 13.0

    # --- legend ---
    show_legend: bool = False
    gt_label: str = r"ground truth  $P(x) = 0$"
    pred_label: str = r"decoded  $f_\theta(x, z) = 0$"
    legend_loc: str = "upper right"
    legend_size: float = 10.0


STYLE_PRESETS: dict[str, EncoderStyle] = {
    "paper": EncoderStyle(),
    # Denser strip: no colorbar, no ticks, tighter panels.
    "strip": EncoderStyle(show_colorbar=False, show_ticks=False, panel_size=3.0),
}


def get_style(name: str = "paper", **overrides: Any) -> EncoderStyle:
    if name not in STYLE_PRESETS:
        raise ValueError(f"unknown style '{name}'; choose from {sorted(STYLE_PRESETS)}")
    return replace(STYLE_PRESETS[name], **overrides) if overrides else STYLE_PRESETS[name]


def field_limits(fields: Sequence[np.ndarray] | np.ndarray, style: EncoderStyle) -> tuple[float, float]:
    """One (vmin, vmax) for a whole sequence of panels, so colour is comparable across them."""
    if style.clip_to_unit:
        return -1.0, 1.0
    stacked = np.asarray(fields, dtype=np.float64)
    if style.symmetric_scale:
        bound = float(np.max(np.abs(stacked)))
        return -bound, bound
    return float(stacked.min()), float(stacked.max())


def _grid(resolution: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(-scale, scale, resolution)
    return np.meshgrid(axis, axis, indexing="xy")


def _draw_field(ax, field: np.ndarray, vmin: float, vmax: float, style: EncoderStyle,
                scale: float):
    if style.field_render == "imshow":
        return ax.imshow(field, origin="lower", extent=(-scale, scale, -scale, scale),
                         cmap=style.cmap, vmin=vmin, vmax=vmax, alpha=style.field_alpha,
                         interpolation="bilinear", aspect="equal", zorder=1)
    xx, yy = _grid(field.shape[0], scale)
    return ax.contourf(xx, yy, field, levels=np.linspace(vmin, vmax, style.levels),
                       cmap=style.cmap, alpha=style.field_alpha, extend="both", zorder=1)


def _draw_boundary(ax, field: np.ndarray, scale: float, color: str, linestyle: str,
                   linewidth: float, zorder: float, sigma: float = 0.0):
    """Zero level set as a contour line; ``sigma`` blurs the field first, never the geometry."""
    xx, yy = _grid(field.shape[0], scale)
    traced = smooth_field(np.asarray(field, dtype=np.float32), sigma) if sigma > 0 else field
    return ax.contour(xx, yy, traced, levels=[0.0], colors=color, linestyles=linestyle,
                      linewidths=linewidth, zorder=zorder)


def _finish_axes(ax, style: EncoderStyle, scale: float):
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    ax.set_aspect("equal")
    if style.show_ticks:
        ax.tick_params(labelsize=style.tick_size)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    if style.spine_color is not None:
        for spine in ax.spines.values():
            spine.set_color(style.spine_color)


def _add_colorbar(fig, mappable, ax, style: EncoderStyle):
    cbar = fig.colorbar(mappable, ax=ax, fraction=style.colorbar_fraction, pad=style.colorbar_pad)
    cbar.set_label(style.colorbar_label, fontsize=style.label_size, family="serif")
    cbar.ax.tick_params(labelsize=style.tick_size)
    for label in cbar.ax.get_yticklabels():
        label.set_family("serif")
    return cbar


def plot_encoder_panel(pred_field: np.ndarray, true_field: np.ndarray,
                       scale: float = PLANE_SCALE,
                       style: EncoderStyle | None = None) -> Figure:
    """SIREN field heatmap with the true and decoded zero level sets overlaid. No title.

    Args:
        pred_field: (R, R) decoded ``f_theta(x, z)`` over ``[-scale, scale]^2``.
        true_field: (R, R) the polynomial ``P(x)`` over the same lattice.
    """
    style = style or get_style()
    pred_field = np.asarray(pred_field)
    true_field = np.asarray(true_field)
    vmin, vmax = field_limits(pred_field[None], style)

    with plt.rc_context(SERIF_RC):
        fig, ax = plt.subplots(figsize=(style.panel_size, style.panel_size))
        mappable = _draw_field(ax, pred_field, vmin, vmax, style, scale)
        _draw_boundary(ax, true_field, scale, style.gt_color, style.gt_linestyle,
                       style.gt_linewidth, zorder=3)
        _draw_boundary(ax, pred_field, scale, style.pred_color, style.pred_linestyle,
                       style.pred_linewidth, zorder=4, sigma=style.smooth_sigma)
        _finish_axes(ax, style, scale)

        if style.show_colorbar:
            _add_colorbar(fig, mappable, ax, style)
        if style.show_legend:
            ax.legend(handles=[
                Line2D([0], [0], color=style.gt_color, lw=style.gt_linewidth,
                       linestyle=style.gt_linestyle, label=style.gt_label),
                Line2D([0], [0], color=style.pred_color, lw=style.pred_linewidth,
                       linestyle=style.pred_linestyle, label=style.pred_label),
            ], loc=style.legend_loc, fontsize=style.legend_size, framealpha=0.9)
        fig.tight_layout()
    return fig


def plot_interpolation_row(fields: np.ndarray, times: Sequence[float],
                           scale: float = PLANE_SCALE,
                           style: EncoderStyle | None = None) -> Figure:
    """1xK strip of decoded fields along ``z(t) = (1 - t) z_a + t z_b``.

    The colour scale is locked over the whole strip, so a panel's colour means the same
    thing in every panel and the level set is the only thing that moves.

    Args:
        fields: (K, R, R) decoded fields, one per entry of ``times``.
        times: interpolation coefficients, used verbatim in the panel titles.
    """
    style = style or get_style()
    fields = np.asarray(fields)
    if fields.shape[0] != len(times):
        raise ValueError(f"{fields.shape[0]} fields for {len(times)} interpolation times")
    vmin, vmax = field_limits(fields, style)

    with plt.rc_context(SERIF_RC):
        fig, axs = plt.subplots(1, len(times),
                                figsize=(style.panel_size * len(times), style.panel_size),
                                squeeze=False)
        mappable = None
        for ax, field, t in zip(axs[0], fields, times):
            mappable = _draw_field(ax, field, vmin, vmax, style, scale)
            _draw_boundary(ax, field, scale, style.gt_color, style.pred_linestyle,
                           style.gt_linewidth, zorder=3, sigma=style.smooth_sigma)
            _finish_axes(ax, style, scale)
            ax.set_title(rf"$t = {t:.2f}$", fontsize=style.title_size, family="serif")

        if style.show_colorbar and mappable is not None:
            _add_colorbar(fig, mappable, list(axs[0]), style)
        fig.tight_layout()
    return fig


def save_encoder_figure(fig: Figure, stem: str | Path, formats: Sequence[str] = ("png", "pdf"),
                        dpi: int = 300, close: bool = True) -> list[Path]:
    """Writes the same figure once per format under a shared stem, all tightly cropped."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in formats:
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(path)
    if close:
        plt.close(fig)
    return written


__all__ = ["SERIF_RC", "EncoderStyle", "STYLE_PRESETS", "get_style", "field_limits",
           "plot_encoder_panel", "plot_interpolation_row", "save_encoder_figure"]

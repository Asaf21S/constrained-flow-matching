# -*- coding: utf-8 -*-
"""Feasibility-vs-fidelity figure: what each sampler does to the constraint boundary.

Density is a raw 2D histogram drawn with ``imshow(..., interpolation="nearest")``, never a
KDE. The claim the figure has to carry is that projection- and guidance-based samplers leave
a delta-like pile-up of mass exactly on {P(x) = 0}; a Gaussian smoothing kernel of any
bandwidth spreads that spike back into the interior and erases the evidence.

Everything visual lives in :class:`FeasibilityStyle`, so the presentation can be retuned from
the command line without touching the sampling or scoring code.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm, Normalize, PowerNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from constrained_fm.src.consts import PLANE_SCALE, POLYNOMIAL_DEGREE
from constrained_fm.src.geometry.polynomials import compute_poly_features, evaluate_poly

# Short name and format string per metric, so panel captions stay compact enough to read.
METRIC_FORMATS: dict[str, tuple[str, str]] = {
    "success_rate": ("SR", "{:.1f}%"),
    "swd": ("SWD", "{:.3f}"),
    "mmd": ("MMD", "{:.4f}"),
    "jsd": ("JSD", "{:.4f}"),
    "nll": ("NLL", "{:.3f}"),
    "kld": ("KLD", "{:.4f}"),
}


@dataclass
class FeasibilityStyle:
    """Every knob the figure exposes. Construct one, or start from :data:`STYLE_PRESETS`."""

    # --- density field ---
    bins: int = 180
    cmap: str = "magma"
    norm: str = "power"          # linear | power | log
    gamma: float = 0.45          # PowerNorm exponent; < 1 lifts the low-density interior
    vmax_quantile: float = 0.995
    vmax_mode: str = "reference"  # reference (panel 0) | shared (all panels) | per_panel

    # --- constraint boundary overlay ---
    # Thin and widely dashed on purpose: an opaque line drawn on top of the boundary hides
    # the very pile-up the figure exists to show, so the wall stays visible in the gaps.
    boundary_color: str = "#22d3ee"
    boundary_linestyle: tuple = (0, (7, 7))
    boundary_linewidth: float = 1.3
    boundary_alpha: float = 0.95
    boundary_resolution: int = 400
    boundary_label: str = "constraint boundary  $P(x) = 0$"

    # --- layout / typography ---
    panel_size: float = 3.1
    title_size: float = 12.0
    metric_size: float = 9.0
    text_color: str = "black"
    highlight_color: str = "#1a7f37"
    highlight_linewidth: float = 2.2
    spine_color: str = "0.55"

    # --- captions ---
    show_metrics: bool = True
    metric_keys: Sequence[str] = ("success_rate", "swd", "jsd")
    metric_separator: str = "  "

    # --- optional boundary-distance profile row ---
    profile_bins: int = 121
    profile_span: float = 0.6     # plot |signed distance| <= this, in plane units
    profile_ratio: float = 0.42   # height of the profile row relative to a map panel
    profile_color: str = "#c2410c"
    profile_fill_alpha: float = 0.25
    profile_share_y: bool = True  # without this a 30x spike and a 1x bump look identical
    profile_headroom: float = 1.18
    profile_annotate_ratio: float = 3.0  # label the peak only this far above the shared limit
    profile_ylabel: str = "density"
    profile_xlabel: str = "signed distance to boundary"

    # --- wall-fraction annotation ---
    show_wall_fraction: bool = True
    wall_tolerance: float = 0.02  # |signed distance| below this counts as "on the wall"


STYLE_PRESETS: dict[str, FeasibilityStyle] = {
    # Perceptually-uniform map on its own black zero-level: the wall reads as a bright filament.
    "dark": FeasibilityStyle(),
    # Matches the existing repo figures (white background, sequential warm map, red boundary).
    "light": FeasibilityStyle(cmap="Oranges", boundary_color="#b91c1c", text_color="black",
                              norm="power", gamma=0.6),
    # Log density; use when the pile-up is several orders of magnitude above the interior.
    "log": FeasibilityStyle(norm="log", cmap="magma"),
}


def get_style(name: str = "dark", **overrides: Any) -> FeasibilityStyle:
    """Named preset with optional field overrides, e.g. ``get_style("light", bins=400)``."""
    if name not in STYLE_PRESETS:
        raise ValueError(f"unknown style '{name}'; choose from {sorted(STYLE_PRESETS)}")
    return replace(STYLE_PRESETS[name], **overrides) if overrides else STYLE_PRESETS[name]


@dataclass
class Panel:
    """One column of the figure."""

    label: str
    samples: Any                                   # (N, 2) array-like
    metrics: Mapping[str, float] | None = None
    highlight: bool = False                        # draws an accent frame, e.g. around "ours"
    caption: str | None = None                     # overrides the metric-derived caption


def to_numpy(data: Any) -> np.ndarray:
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def format_metrics(metrics: Mapping[str, float] | None, keys: Sequence[str],
                   separator: str = "   ") -> str:
    """Compact caption such as ``SR 100.0%   SWD 0.689   JSD 0.1450``."""
    if not metrics:
        return ""
    parts = []
    for key in keys:
        value = metrics.get(key)
        if value is None or not np.isfinite(value):
            continue
        short, fmt = METRIC_FORMATS.get(key, (key.upper(), "{:.3f}"))
        parts.append(f"{short} {fmt.format(value)}")
    return separator.join(parts)


def polynomial_grid(coeffs: torch.Tensor, resolution: int = 400,
                    degree: int = POLYNOMIAL_DEGREE,
                    scale: float = PLANE_SCALE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """P(x, y) on a square lattice over [-scale, scale]^2, for the zero-level contour."""
    device = coeffs.device
    axis = torch.linspace(-scale, scale, resolution, device=device)
    gx, gy = torch.meshgrid(axis, axis, indexing="xy")
    points = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)

    x_pow, y_pow = compute_poly_features(points, degree=degree, scale=scale)
    C = coeffs.unsqueeze(0).expand(points.shape[0], -1, -1)
    P = evaluate_poly(x_pow, y_pow, C).reshape(resolution, resolution)
    return gx.cpu().numpy(), gy.cpu().numpy(), P.detach().cpu().numpy()


def signed_boundary_distance(points: Any, coeffs: torch.Tensor,
                             degree: int = POLYNOMIAL_DEGREE,
                             scale: float = PLANE_SCALE) -> np.ndarray:
    """First-order signed distance P(x) / ||grad P(x)||, negative inside the feasible set.

    Raw P values are not comparable between constraints (the coefficients are only normalised
    in Frobenius norm), so the profile row is drawn in this rescaled coordinate instead.
    """
    pts = torch.as_tensor(to_numpy(points), dtype=torch.float32,
                          device=coeffs.device).clone().requires_grad_(True)
    x_pow, y_pow = compute_poly_features(pts, degree=degree, scale=scale)
    C = coeffs.unsqueeze(0).expand(pts.shape[0], -1, -1)
    P = evaluate_poly(x_pow, y_pow, C).squeeze(-1)

    grad = torch.autograd.grad(P.sum(), pts)[0]
    distance = P / grad.norm(dim=1).clamp(min=1e-6)
    return distance.detach().cpu().numpy()


def _histogram(samples: Any, bins: int, scale: float) -> np.ndarray:
    """Per-point density, not raw counts, so panels drawn from different sample counts share
    a colour scale."""
    data = to_numpy(samples)
    H, _, _ = np.histogram2d(data[:, 0], data[:, 1], bins=bins,
                             range=[[-scale, scale], [-scale, scale]])
    bin_area = (2.0 * scale / bins) ** 2
    return H / (max(data.shape[0], 1) * bin_area)


def _resolve_vmax(hists: Sequence[np.ndarray], style: FeasibilityStyle) -> list[float]:
    """One vmax per panel, following ``style.vmax_mode``.

    ``reference`` scales every panel by the ground-truth panel, which is what makes the
    pile-up legible: the wall saturates the colormap precisely because it carries far more
    mass per unit area than anything the true truncated density ever puts there.
    """
    def quantile(H: np.ndarray) -> float:
        nonzero = H[H > 0]
        if nonzero.size == 0:
            return 1.0
        return float(np.quantile(nonzero, style.vmax_quantile))

    if style.vmax_mode == "per_panel":
        return [quantile(H) for H in hists]
    if style.vmax_mode == "shared":
        return [max(quantile(H) for H in hists)] * len(hists)
    if style.vmax_mode == "reference":
        return [quantile(hists[0])] * len(hists)
    raise ValueError(f"unknown vmax_mode '{style.vmax_mode}'")


def _make_norm(vmax: float, style: FeasibilityStyle) -> Normalize:
    if style.norm == "linear":
        return Normalize(vmin=0.0, vmax=vmax)
    if style.norm == "power":
        return PowerNorm(gamma=style.gamma, vmin=0.0, vmax=vmax)
    if style.norm == "log":
        return LogNorm(vmin=vmax * 1e-3, vmax=vmax, clip=True)
    raise ValueError(f"unknown norm '{style.norm}'")


def _draw_map(ax, H: np.ndarray, vmax: float, contour, style: FeasibilityStyle,
              scale: float) -> None:
    field = np.clip(H, vmax * 1e-3, None) if style.norm == "log" else H
    # imshow over hist2d: nearest-neighbour resampling keeps the one-bin-wide wall one bin
    # wide, and empty bins take the colormap's zero colour instead of the figure background.
    ax.imshow(field.T, origin="lower", extent=(-scale, scale, -scale, scale),
              cmap=style.cmap, norm=_make_norm(vmax, style), interpolation="nearest")

    gx, gy, P = contour
    ax.contour(gx, gy, P, levels=[0.0], colors=style.boundary_color,
               linewidths=style.boundary_linewidth, linestyles=[style.boundary_linestyle],
               alpha=style.boundary_alpha)

    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _frame(ax, color: str, linewidth: float) -> None:
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(linewidth)


def plot_feasibility_row(panels: Sequence[Panel], coeffs: torch.Tensor,
                         style: FeasibilityStyle | None = None,
                         degree: int = POLYNOMIAL_DEGREE, scale: float = PLANE_SCALE,
                         show_profile: bool = False, legend: bool = True) -> Figure:
    """A 1 x len(panels) row of boundary-preserving density maps for one constraint.

    Args:
        panels: one :class:`Panel` per column, ground truth first when
            ``style.vmax_mode == "reference"``.
        coeffs: (degree+1, degree+1) coefficients of the shared constraint.
        show_profile: append a second row histogramming the signed distance to the
            boundary, which turns the visual pile-up into a readable spike.
    """
    style = style or STYLE_PRESETS["dark"]
    num = len(panels)
    contour = polynomial_grid(coeffs, resolution=style.boundary_resolution,
                              degree=degree, scale=scale)

    hists = [_histogram(p.samples, style.bins, scale) for p in panels]
    vmaxes = _resolve_vmax(hists, style)

    profiles, profile_top = [], None
    if show_profile:
        profiles = [_profile_data(p, coeffs, style, degree, scale) for p in panels]
        if style.profile_share_y:
            profile_top = profiles[0]["peak"] * style.profile_headroom

    rows = 2 if show_profile else 1
    height = style.panel_size * (1 + style.profile_ratio if show_profile else 1) + 0.6
    fig = plt.figure(figsize=(style.panel_size * num, height))
    gs = fig.add_gridspec(rows, num, hspace=0.45, wspace=0.06,
                          height_ratios=[1.0, style.profile_ratio] if show_profile else [1.0])

    for col, (panel, H, vmax) in enumerate(zip(panels, hists, vmaxes)):
        ax = fig.add_subplot(gs[0, col])
        _draw_map(ax, H, vmax, contour, style, scale)
        ax.set_title(panel.label, fontsize=style.title_size, color=style.text_color,
                     fontweight="bold" if panel.highlight else "normal")

        caption = panel.caption
        if caption is None and style.show_metrics:
            caption = format_metrics(panel.metrics, style.metric_keys, style.metric_separator)
        if caption:
            ax.set_xlabel(caption, fontsize=style.metric_size, color=style.text_color,
                          labelpad=4)

        if panel.highlight:
            _frame(ax, style.highlight_color, style.highlight_linewidth)
        else:
            _frame(ax, style.spine_color, 0.8)

        # Inside the first panel rather than under the figure: a figure-level legend lands on
        # top of the profile row's axis labels once that row is enabled.
        if legend and col == 0:
            ax.legend(handles=[Line2D([0], [0], color=style.boundary_color,
                                      lw=style.boundary_linewidth + 0.7,
                                      linestyle=style.boundary_linestyle,
                                      label=style.boundary_label)],
                      loc="lower left", fontsize=style.metric_size - 1.0, framealpha=0.75,
                      handlelength=2.6, borderpad=0.4)

        if show_profile:
            _draw_profile(fig.add_subplot(gs[1, col]), profiles[col], style, profile_top,
                          first=col == 0)

    fig.tight_layout()
    return fig


def _profile_data(panel: Panel, coeffs: torch.Tensor, style: FeasibilityStyle,
                  degree: int, scale: float) -> dict[str, Any]:
    distance = signed_boundary_distance(panel.samples, coeffs, degree=degree, scale=scale)
    edges = np.linspace(-style.profile_span, style.profile_span, style.profile_bins + 1)
    counts, _ = np.histogram(distance, bins=edges, density=True)
    return {
        "centers": 0.5 * (edges[:-1] + edges[1:]),
        "counts": counts,
        "peak": float(counts.max()) if counts.size else 0.0,
        "wall_fraction": float(np.mean(np.abs(distance) < style.wall_tolerance)),
    }


def _draw_profile(ax, data: dict[str, Any], style: FeasibilityStyle, top: float | None,
                  first: bool) -> None:
    """Histogram of the signed distance to {P = 0}, clipped to a band around the boundary.

    The true truncated density steps down to zero at 0; a projected or guided sampler instead
    shows a narrow spike immediately to its left, which is the wall effect stated numerically.
    All panels share a y-axis, otherwise a 30x spike and a mild bump render identically.
    """
    centers, counts = data["centers"], data["counts"]
    ax.fill_between(centers, counts, step="mid", color=style.profile_color,
                    alpha=style.profile_fill_alpha)
    ax.step(centers, counts, where="mid", color=style.profile_color, linewidth=1.4)
    ax.axvline(0.0, color=style.boundary_color, linestyle=style.boundary_linestyle,
               linewidth=style.boundary_linewidth + 0.5)

    ax.set_xlim(-style.profile_span, style.profile_span)
    if top is not None:
        ax.set_ylim(0.0, top)
        # Only call out a genuine spike; a panel that merely grazes the shared limit does not
        # need a label pointing at it.
        if data["peak"] > top * style.profile_annotate_ratio:
            ax.annotate(f"peak {data['peak']:.0f}", xy=(0.0, top),
                        xytext=(-0.10 * style.profile_span / 0.6, top * 0.78),
                        ha="right", va="center", fontsize=style.metric_size - 1.0,
                        color=style.profile_color, fontweight="bold",
                        arrowprops=dict(arrowstyle="-|>", color=style.profile_color, lw=1.2))
    else:
        ax.set_ylim(bottom=0.0)

    if style.show_wall_fraction:
        ax.text(0.97, 0.92, f"{100 * data['wall_fraction']:.1f}% on the wall",
                transform=ax.transAxes, fontsize=style.metric_size - 1.0,
                va="top", ha="right", color=style.text_color)

    ax.set_xlabel(style.profile_xlabel, fontsize=style.metric_size - 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    ax.tick_params(labelsize=style.metric_size - 1.5)
    if first:
        ax.set_ylabel(style.profile_ylabel, fontsize=style.metric_size - 0.5)
    else:
        ax.set_yticklabels([])
    ax.grid(True, alpha=0.25)


__all__ = ["FeasibilityStyle", "STYLE_PRESETS", "Panel", "get_style", "format_metrics",
           "polynomial_grid", "signed_boundary_distance", "plot_feasibility_row", "to_numpy"]

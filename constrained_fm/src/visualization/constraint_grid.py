# -*- coding: utf-8 -*-
"""A small-multiples grid of polynomial constraint boundaries from a validation set.

Standalone figure, separate from the GMM target heatmap in ``density.py``; the two are
combined manually in the paper's LaTeX source. Each panel shades the feasible region
``P(x) <= 0`` and draws its zero-level boundary; no axes, ticks, or model output involved.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch

from constrained_fm.src.consts import PLANE_SCALE, POLYNOMIAL_DEGREE
from constrained_fm.src.geometry.polynomials import compute_poly_features, evaluate_poly
from constrained_fm.src.visualization.style import PAPER_RC


@dataclass
class ConstraintGridStyle:
    """Every visual knob of the grid."""

    inside_color: str = "#3B6FA0"
    inside_alpha: float = 0.35
    boundary_color: str = "#0D1B2A"
    boundary_linewidth: float = 1.6
    panel_size: float = 2.1
    spine_color: str = "#999999"
    spine_linewidth: float = 0.8


STYLE_PRESETS: dict[str, ConstraintGridStyle] = {
    "paper": ConstraintGridStyle(),
}


def _render_panel(ax, coeffs: torch.Tensor, degree: int, scale: float, grid_size: int,
                  extent: tuple[float, float, float, float], style: ConstraintGridStyle) -> None:
    xx, yy = np.meshgrid(np.linspace(extent[0], extent[1], grid_size),
                         np.linspace(extent[2], extent[3], grid_size))
    grid_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    x_pow, y_pow = compute_poly_features(grid_points, degree=degree, scale=scale)
    C = coeffs.unsqueeze(0).expand(grid_points.shape[0], -1, -1).to(dtype=x_pow.dtype)
    P = evaluate_poly(x_pow, y_pow, C).squeeze(-1).cpu().numpy().reshape(grid_size, grid_size)

    mask = (P <= 0).astype(float)
    cmap = mcolors.ListedColormap(["white", style.inside_color])
    ax.imshow(mask, extent=extent, origin="lower", cmap=cmap, vmin=0, vmax=1,
             alpha=style.inside_alpha)
    ax.contour(xx, yy, P, levels=[0.0], colors=style.boundary_color,
              linewidths=style.boundary_linewidth)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(style.spine_color)
        spine.set_linewidth(style.spine_linewidth)


def plot_constraint_grid(polynomials: torch.Tensor, nrows: int = 2, ncols: int = 5,
                         seed: int = 0, grid_size: int = 200,
                         degree: int = POLYNOMIAL_DEGREE, scale: float = PLANE_SCALE,
                         extent: tuple[float, float, float, float] = (-4.5, 4.5, -4.5, 4.5),
                         style: ConstraintGridStyle | str = "paper", save_path=None, show: bool = True):
    """Renders ``nrows x ncols`` randomly chosen constraint boundaries from ``polynomials``.

    ``seed`` controls which polynomials are drawn, so different seeds give different variants
    of the same grid. ``save_path`` (with or without extension) writes both a ``.png`` and a
    ``.pdf``. Returns ``(fig, indices)`` so the caller can trace which validation polynomials
    were plotted.
    """
    if isinstance(style, str):
        style = STYLE_PRESETS[style]

    num_panels = nrows * ncols
    rng = np.random.default_rng(seed)
    indices = rng.choice(polynomials.shape[0], size=num_panels, replace=False)

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(style.panel_size * ncols, style.panel_size * nrows))
        for ax, idx in zip(axes.ravel(), indices):
            _render_panel(ax, polynomials[idx], degree, scale, grid_size, extent, style)
        fig.tight_layout(pad=0.6)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        stem = save_path.with_suffix("")
        fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, indices


def render_constraint_grid_variants(polynomials: torch.Tensor, out_dir, num_variants: int,
                                    nrows: int = 2, ncols: int = 5,
                                    seed_start: int = 0, **kwargs) -> list[Path]:
    """Renders ``num_variants`` grids, one seed apart, so the caller can pick one for the paper."""
    out_dir = Path(out_dir)
    paths = []
    for i in range(num_variants):
        seed = seed_start + i
        save_path = out_dir / f"constraint_grid_seed{seed}"
        plot_constraint_grid(polynomials, nrows=nrows, ncols=ncols, seed=seed,
                             save_path=save_path, show=False, **kwargs)
        paths.append(save_path)
    return paths


__all__ = ["ConstraintGridStyle", "STYLE_PRESETS", "plot_constraint_grid",
          "render_constraint_grid_variants"]

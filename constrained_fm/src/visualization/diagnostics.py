# -*- coding: utf-8 -*-
"""Figure-returning diagnostics for headless evaluation jobs.

Every function returns a Matplotlib Figure and never calls plt.show(), so the same code
renders to PNG under Agg in a batch job and displays inline in a notebook.
"""

from __future__ import annotations

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from pathlib import Path

from constrained_fm.src.consts import PLANE_SCALE, POLYNOMIAL_DEGREE
from constrained_fm.src.metrics.functa_fidelity import decode_region, uniform_grid_points
from constrained_fm.src.visualization.density import calculate_vmax
from constrained_fm.src.visualization.scatter import assign_gaussian_to_points, visualize_single_step


def save_figure(fig: Figure, path: str | Path, dpi: int = 110) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_loss_curve(losses, log_scale: bool = True) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    losses = np.asarray(losses, dtype=float)
    ax.plot(losses, color="indigo", alpha=0.6, linewidth=0.8)

    window = max(1, len(losses) // 200)
    if window > 1:
        smooth = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window - 1, len(losses)), smooth, color="crimson", linewidth=1.8,
                label=f"moving mean ({window})")
        ax.legend()

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("weighted MSE")
    ax.set_title("Functa-conditioned flow matching loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_sample_trajectory(trajectory, time_grid, coeffs: torch.Tensor | None = None,
                           degree: int = POLYNOMIAL_DEGREE, scale: float = PLANE_SCALE,
                           device=None) -> Figure:
    """Row of 2D histograms along the ODE integration path, with the GT boundary overlaid."""
    trajectory = np.asarray(trajectory)
    time_grid = np.asarray(time_grid)
    num_steps = trajectory.shape[0]

    labels = assign_gaussian_to_points(trajectory[-1], device=device)

    fig, axs = plt.subplots(1, num_steps, figsize=(2 * num_steps, 2.4))
    for i in range(num_steps):
        visualize_single_step(trajectory[i], title=f"t = {time_grid[i]:.2f}", ax=axs[i],
                              coeffs=coeffs, degree=degree, scale=scale, labels=labels)
    fig.tight_layout()
    return fig


def plot_final_samples(samples, coeffs: torch.Tensor | None = None, title: str = "Samples",
                       degree: int = POLYNOMIAL_DEGREE, scale: float = PLANE_SCALE) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    visualize_single_step(samples, title=title, ax=ax, cmap="Oranges",
                          coeffs=coeffs, degree=degree, scale=scale)
    fig.tight_layout()
    return fig


def plot_final_samples_gallery(samples_per_shape, coeffs_per_shape, titles,
                               degree: int = POLYNOMIAL_DEGREE,
                               scale: float = PLANE_SCALE) -> Figure:
    """Grid of final-sample panels across representative validation constraints."""
    num = len(titles)
    cols = min(3, max(1, num))
    rows = int(np.ceil(num / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 5.6 * rows), squeeze=False)

    for ax in axs.flatten():
        ax.set_visible(False)

    for i, (samples, coeffs, title) in enumerate(zip(samples_per_shape, coeffs_per_shape, titles)):
        ax = axs[i // cols][i % cols]
        ax.set_visible(True)
        visualize_single_step(samples, title=title, ax=ax, cmap="Oranges",
                              coeffs=coeffs, degree=degree, scale=scale)

    fig.tight_layout()
    return fig


def plot_believed_vs_true(siren, samples_per_shape, z_per_shape, coeffs_per_shape, titles,
                          grid_size: int = 200, degree: int = POLYNOMIAL_DEGREE,
                          scale: float = PLANE_SCALE, device=None) -> Figure:
    """Generated samples with the true P(x) = 0 curve (red) and the SIREN's decoded
    boundary (blue) overlaid, one panel per shape."""
    num_shapes = len(titles)
    points = uniform_grid_points(grid_size=grid_size, scale=scale, device=device)
    axis = torch.linspace(-scale, scale, grid_size).numpy()
    xx, yy = np.meshgrid(axis, axis, indexing="ij")

    fig, axs = plt.subplots(1, num_shapes, figsize=(5 * num_shapes, 5), squeeze=False)
    for ax, samples, z, C, title in zip(axs[0], samples_per_shape, z_per_shape,
                                        coeffs_per_shape, titles):
        visualize_single_step(samples, title="", ax=ax, cmap="Oranges",
                              coeffs=C, degree=degree, scale=scale)
        believed = decode_region(siren, z, points, scale=scale)
        believed = believed.reshape(grid_size, grid_size).cpu().numpy()
        ax.contour(xx, yy, believed, levels=[0.0], colors="blue", linewidths=2.0)
        ax.set_xlim(-scale, scale)
        ax.set_ylim(-scale, scale)
        ax.set_title(title)

    axs[0][0].legend(handles=[
        Line2D([0], [0], color="red", lw=2.5, linestyle="dashed", label="true P(x) = 0"),
        Line2D([0], [0], color="blue", lw=2.0, label="SIREN(x, z) = 0"),
    ], loc="upper right", fontsize="small")
    fig.tight_layout()
    return fig


def plot_likelihood(likelihood, coeffs: torch.Tensor | None = None,
                    degree: int = POLYNOMIAL_DEGREE, scale: float = PLANE_SCALE,
                    grid_size: int = 200, device=None) -> Figure:
    """Exact model likelihood heatmap, normalized against the truncated GMM's peak density."""
    from constrained_fm.src.geometry.polynomials import compute_poly_features, evaluate_poly

    vmax = calculate_vmax(coeffs=coeffs, degree=degree, scale=scale, device=device)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.grid(False)
    norm = cm.colors.Normalize(vmax=vmax, vmin=0.0)
    ax.imshow(likelihood, extent=(-scale, scale, -scale, scale), origin="lower",
              cmap="viridis", norm=norm)

    if coeffs is not None:
        xx, yy = np.meshgrid(np.linspace(-scale, scale, grid_size),
                             np.linspace(-scale, scale, grid_size))
        grid_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32,
                                   device=coeffs.device)
        x_pow, y_pow = compute_poly_features(grid_points, degree=degree, scale=scale)
        C_grid = coeffs.unsqueeze(0).expand(grid_points.shape[0], -1, -1)
        P_grid = evaluate_poly(x_pow, y_pow, C_grid).squeeze().cpu().numpy().reshape(grid_size, grid_size)
        ax.contour(xx, yy, P_grid, levels=[0.0], colors="red", linewidths=2.5, linestyles="dashed")
        ax.plot([], [], color="red", linewidth=2.5, linestyle="dashed", label="Constraint P(x) = 0")
        ax.legend(loc="upper right")

    ax.set_title("Model Likelihood")
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, label="density")
    fig.tight_layout()
    return fig


def plot_success_vs_fidelity(success_rate, mass, mass_iou) -> Figure:
    """Scatters the per-shape success rate against valid mass and against decoded-region IoU.

    A steep mass_iou trend attributes the failure tail to the conditioning rather than
    to the flow matcher.
    """
    success_rate = np.asarray(success_rate, dtype=float)
    series = [("valid GMM mass", np.asarray(mass, dtype=float)),
              ("mass-weighted region IoU", np.asarray(mass_iou, dtype=float))]

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (label, values) in zip(axs, series):
        ax.scatter(values, success_rate, s=22, alpha=0.75, color="darkslateblue")
        finite = np.isfinite(values) & np.isfinite(success_rate)
        if finite.sum() > 1:
            corr = np.corrcoef(values[finite], success_rate[finite])[0, 1]
            ax.set_title(f"{label}\ncorr = {corr:+.3f}")
        else:
            ax.set_title(label)
        ax.set_xlabel(label)
        ax.set_ylabel("success rate (%)")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


__all__ = ["save_figure", "plot_loss_curve", "plot_sample_trajectory", "plot_final_samples",
           "plot_final_samples_gallery", "plot_believed_vs_true", "plot_likelihood",
           "plot_success_vs_fidelity"]

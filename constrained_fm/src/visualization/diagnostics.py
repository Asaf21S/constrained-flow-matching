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
import torch.nn.functional as F
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter
from pathlib import Path

from constrained_fm.src.consts import PLANE_SCALE, POLYNOMIAL_DEGREE
from constrained_fm.src.metrics.functa_fidelity import decode_region
from constrained_fm.src.visualization.density import calculate_vmax
from constrained_fm.src.visualization.scatter import assign_gaussian_to_points, visualize_single_step


def save_figure(fig: Figure, path: str | Path, dpi: int = 110) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _true_field(C: torch.Tensor, points: torch.Tensor, degree: int = POLYNOMIAL_DEGREE,
                scale: float = PLANE_SCALE) -> np.ndarray:
    """P(x) for a single polynomial at the given points, as a host array of shape (M,)."""
    from constrained_fm.src.geometry.polynomials import compute_poly_features, evaluate_poly

    x_pow, y_pow = compute_poly_features(points, degree=degree, scale=scale)
    C_expanded = C.unsqueeze(0).expand(points.shape[0], -1, -1)
    return evaluate_poly(x_pow, y_pow, C_expanded).squeeze(-1).cpu().numpy()


def smooth_field(field: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur applied before tracing a zero level set.

    A w0=30 SIREN carries low-amplitude high-frequency ripple, so its raw sign flips many
    times inside a thin band around the boundary and contour() returns hundreds of disjoint
    fragments -- denser grids resolve more of the ripple and look worse, not better. The
    filter is symmetric, so it does not bias where the crossing sits, and it touches only
    the rendering: every reported IoU is computed on the raw field.
    """
    if sigma <= 0:
        return field

    radius = int(np.ceil(3 * sigma))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()

    out = torch.from_numpy(np.ascontiguousarray(field, dtype=np.float32)).view(1, 1, *field.shape)
    k = torch.from_numpy(kernel)
    out = F.conv2d(F.pad(out, (0, 0, radius, radius), mode="replicate"), k.view(1, 1, -1, 1))
    out = F.conv2d(F.pad(out, (radius, radius, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    return out.view(*field.shape).numpy()


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


def plot_believed_vs_true(samples_per_shape, believed_per_shape, coeffs_per_shape, titles,
                          degree: int = POLYNOMIAL_DEGREE, scale: float = PLANE_SCALE) -> Figure:
    """Generated samples with the true P(x) = 0 curve (red) and the SIREN's decoded
    boundary (blue) overlaid, one panel per shape.

    believed_per_shape holds the already-decoded SIREN(x, z) field on a square lattice
    spanning [-scale, scale]^2, so this never touches the encoder.
    """
    num_shapes = len(titles)

    fig, axs = plt.subplots(1, num_shapes, figsize=(5 * num_shapes, 5), squeeze=False)
    for ax, samples, believed, C, title in zip(axs[0], samples_per_shape, believed_per_shape,
                                               coeffs_per_shape, titles):
        believed = np.asarray(believed)
        axis = np.linspace(-scale, scale, believed.shape[0])
        xx, yy = np.meshgrid(axis, axis, indexing="ij")

        visualize_single_step(samples, title="", ax=ax, cmap="Oranges",
                              coeffs=C, degree=degree, scale=scale)
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


def plot_functa_extraction(siren, coeffs: torch.Tensor, z_batch: torch.Tensor,
                           degree: int = POLYNOMIAL_DEGREE, scale: float = PLANE_SCALE,
                           resolution: int = 500, smooth_sigma: float = 2.0) -> Figure:
    """Side-by-side ground-truth polynomial vs. the SIREN's decoded tanh(P) field.

    Left panel: the true region {P(x) <= 0} with its P(x) = 0 boundary.
    Right panel: SIREN(x, z) as a filled field, its zero level set, and the true
    boundary overlaid so the two curves can be compared directly.
    """
    from constrained_fm.src.geometry.polynomials import (compute_poly_features_batched,
                                                         evaluate_poly_batched)

    device = next(siren.parameters()).device
    num_shapes = coeffs.shape[0]

    axis = torch.linspace(-scale, scale, resolution)
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    xx, yy = grid_x.numpy(), grid_y.numpy()

    grid_raw = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 2).to(device)
    grid_raw = grid_raw.expand(num_shapes, -1, -1)

    x_pow, y_pow = compute_poly_features_batched(grid_raw, degree=degree, scale=scale)
    P_grid = evaluate_poly_batched(x_pow, y_pow, coeffs)
    P_grid = P_grid.view(num_shapes, resolution, resolution).cpu().numpy()

    with torch.no_grad():
        preds = siren(grid_raw / scale, z_batch).squeeze(-1)
    preds = preds.view(num_shapes, resolution, resolution).cpu().numpy()

    fig, axes = plt.subplots(num_shapes, 2, figsize=(12, 5 * num_shapes), squeeze=False)

    for i in range(num_shapes):
        ax_gt, ax_pred = axes[i]

        ax_gt.contourf(xx, yy, P_grid[i], levels=[-float("inf"), 0.0],
                       colors=["dodgerblue"], alpha=0.3)
        ax_gt.contour(xx, yy, P_grid[i], levels=[0.0], colors="black", linewidths=2.5)
        ax_gt.set_xlim(-scale, scale)
        ax_gt.set_ylim(-scale, scale)
        ax_gt.set_aspect("equal")
        ax_gt.set_title(f"GT Polynomial {i + 1}")
        ax_gt.legend(handles=[
            Line2D([0], [0], color="black", lw=2.5, label="GT Boundary (P=0)"),
            Patch(color="dodgerblue", alpha=0.3, label="Valid Region (P<=0)"),
        ], loc="upper right", fontsize="small")

        cf = ax_pred.contourf(xx, yy, preds[i], levels=50, cmap="RdBu_r", alpha=0.85,
                              vmin=-1, vmax=1)
        ax_pred.contour(xx, yy, P_grid[i], levels=[0.0], colors="black", linewidths=3.0,
                        linestyles="solid", zorder=3)
        ax_pred.contour(xx, yy, smooth_field(preds[i], smooth_sigma), levels=[0.0], colors="lime",
                        linewidths=1.8, linestyles="solid", zorder=4)
        ax_pred.set_xlim(-scale, scale)
        ax_pred.set_ylim(-scale, scale)
        ax_pred.set_aspect("equal")
        ax_pred.set_title(f"SIREN Prediction {i + 1}")
        ax_pred.legend(handles=[
            Line2D([0], [0], color="lime", lw=1.8, label="SIREN Boundary (pred=0)"),
            Line2D([0], [0], color="black", lw=3.0, label="GT Boundary (Overlay)"),
        ], loc="upper right", fontsize="small")

        cbar = fig.colorbar(cf, ax=ax_pred, fraction=0.046, pad=0.04)
        cbar.set_label("SIREN Prediction: tanh(P)", rotation=270, labelpad=15)

    fig.tight_layout()
    return fig


def plot_boundary_ablation_grid(siren, coeffs_list, z_grid, row_labels, col_labels,
                                cell_labels=None, degree: int = POLYNOMIAL_DEGREE,
                                scale: float = PLANE_SCALE, resolution: int = 400,
                                smooth_sigma: float = 2.0) -> Figure:
    """Rows = polynomials, columns = an ablated setting; each cell overlays the decoded
    SIREN zero level set on the true region.

    z_grid[r][c] is the latent for row r under setting c.
    """
    num_rows, num_cols = len(coeffs_list), len(col_labels)

    axis = torch.linspace(-scale, scale, resolution)
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    xx, yy = grid_x.numpy(), grid_y.numpy()
    device = next(siren.parameters()).device
    points = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2).to(device)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(3.1 * num_cols, 3.3 * num_rows),
                            squeeze=False)

    for r, C in enumerate(coeffs_list):
        P_grid = _true_field(C, points, degree=degree, scale=scale).reshape(resolution, resolution)
        for c in range(num_cols):
            ax = axs[r][c]
            ax.contourf(xx, yy, P_grid, levels=[-float("inf"), 0.0], colors=["dodgerblue"],
                        alpha=0.25)
            ax.contour(xx, yy, P_grid, levels=[0.0], colors="black", linewidths=2.2,
                       linestyles="solid", zorder=3)

            pred = decode_region(siren, z_grid[r][c], points, scale=scale)
            pred = pred.reshape(resolution, resolution).cpu().numpy()
            ax.contour(xx, yy, smooth_field(pred, smooth_sigma), levels=[0.0], colors="lime",
                       linewidths=1.6, linestyles="solid", zorder=4)

            ax.set_xlim(-scale, scale)
            ax.set_ylim(-scale, scale)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_labels[c], fontsize=11)
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=10)
            if cell_labels is not None:
                ax.set_xlabel(cell_labels[r][c], fontsize=9)

    fig.legend(handles=[
        Line2D([0], [0], color="black", lw=2.2, label="GT boundary P(x) = 0"),
        Line2D([0], [0], color="lime", lw=1.6, label="SIREN(x, z) = 0"),
    ], loc="lower center", ncol=2, fontsize="medium", bbox_to_anchor=(0.5, -0.015))
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


def plot_samples_ablation_grid(samples_grid, coeffs_list, row_labels, col_labels,
                               cell_labels=None, degree: int = POLYNOMIAL_DEGREE,
                               scale: float = PLANE_SCALE) -> Figure:
    """Same layout as plot_boundary_ablation_grid, but each cell is the generated particle
    density with the true constraint boundary overlaid."""
    num_rows, num_cols = len(coeffs_list), len(col_labels)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(3.1 * num_cols, 3.3 * num_rows),
                            squeeze=False)

    for r, C in enumerate(coeffs_list):
        for c in range(num_cols):
            ax = axs[r][c]
            visualize_single_step(samples_grid[r][c], title="", ax=ax, cmap="Oranges",
                                  coeffs=C, degree=degree, scale=scale)
            ax.set_xlim(-scale, scale)
            ax.set_ylim(-scale, scale)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if r == 0:
                ax.set_title(col_labels[c], fontsize=11)
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=10)
            if cell_labels is not None:
                ax.set_xlabel(cell_labels[r][c], fontsize=9)

    fig.legend(handles=[
        Line2D([0], [0], color="red", lw=2.5, linestyle="dashed", label="GT boundary P(x) = 0"),
    ], loc="lower center", fontsize="medium", bbox_to_anchor=(0.5, -0.015))
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


def plot_ablation_curves(x_values, series, xlabel: str, log_x: bool = True) -> Figure:
    """One panel per metric; each panel plots median with an inter-quartile band.

    series maps a metric name to a (len(x_values), num_shapes) array.
    """
    names = list(series)
    fig, axs = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4.0), squeeze=False)

    for ax, name in zip(axs[0], names):
        values = np.asarray(series[name], dtype=float)
        finite = np.where(np.isfinite(values), values, np.nan)
        median = np.nanmedian(finite, axis=1)
        q25, q75 = np.nanpercentile(finite, 25, axis=1), np.nanpercentile(finite, 75, axis=1)

        ax.plot(x_values, median, marker="o", color="darkslateblue", label="median")
        ax.fill_between(x_values, q25, q75, color="darkslateblue", alpha=0.2, label="IQR")
        if log_x:
            ax.set_xscale("log")
            ax.set_xticks(x_values)
            ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.set_xlabel(xlabel)
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="small")

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
           "plot_final_samples_gallery", "plot_believed_vs_true", "plot_functa_extraction",
           "plot_boundary_ablation_grid", "plot_samples_ablation_grid", "plot_ablation_curves",
           "plot_likelihood", "plot_success_vs_fidelity"]

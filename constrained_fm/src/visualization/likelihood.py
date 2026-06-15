import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm

from constrained_fm.src.utils.polynomials import compute_poly_features, evaluate_poly
from constrained_fm.src.consts import GMM_MEANS, GMM_COVS, GMM_WEIGHTS, POLYNOMIAL_DEGREE, PLANE_SCALE
from constrained_fm.src.data_handlers.gmm_2d import get_points, compute_gmm_density


def visualize_true_gmm_likelihood(means=GMM_MEANS, covs=GMM_COVS, weights=GMM_WEIGHTS, grid_size=200, device=None):
    """Render the ground-truth GMM likelihood heatmap."""
    density = compute_gmm_density(means=means, covs=covs, weights=weights, grid_size=grid_size, device=device)
    density_grid = density.reshape(grid_size, grid_size)
    true_vmax = torch.max(density_grid).item()

    fig, ax = plt.subplots(figsize=(6, 6))
    norm = cm.colors.Normalize(vmax=true_vmax, vmin=0.0)
    ax.imshow(density_grid.cpu().numpy(), extent=(-4.5, 4.5, -4.5, 4.5), origin='lower', cmap='viridis', norm=norm)
    ax.set_title(f'Ground Truth GMM Likelihood\nPeak Density: {true_vmax:.3f}')
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax, orientation='vertical', label='True Density')
    ax.set_aspect('equal')
    plt.show()


def calculate_vmax(bounds=None, coeffs=None, degree=POLYNOMIAL_DEGREE, scale=PLANE_SCALE, unconstrained_vmax=None,
                   num_samples=100000, device=None):
    """Calculate an appropriate vmax for visualizing model likelihoods.

    If ``bounds`` (a rectangular constraint) or ``coeffs`` (a polynomial constraint) are provided,
    the function estimates the fraction of the unconstrained space that satisfies the constraint and
    scales the vmax accordingly.
    """
    if unconstrained_vmax is None:
        density = compute_gmm_density(device=device)
        unconstrained_vmax = torch.max(density).item()

    if bounds is None and coeffs is None:
        return unconstrained_vmax

    points, _ = get_points(num_samples, device=device)

    if bounds is not None:
        x_min, y_min, x_max, y_max = bounds
        mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & \
               (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
    elif coeffs is not None:
        points = points.to(device)
        x_pow, y_pow = compute_poly_features(points, degree=degree, scale=scale)
        batch_C = coeffs.unsqueeze(0).expand(num_samples, -1, -1)
        p_vals = evaluate_poly(x_pow, y_pow, batch_C).squeeze()
        mask = (p_vals <= 0)

    valid_fraction = mask.float().mean().item()
    if valid_fraction == 0:
        print("Warning: Constraint contains no active area.")
        return unconstrained_vmax
    return unconstrained_vmax / valid_fraction


def visualize_likelihood(likelihood, bounds=None, coeffs=None, degree=POLYNOMIAL_DEGREE, scale=PLANE_SCALE,
                         grid_size=200, device=None):
    """Render a model's log‑likelihood heatmap with constraint overlays.

    ``likelihood`` is expected to be a 2‑D tensor matching the grid resolution used by the model.
    """
    auto_vmax = calculate_vmax(bounds=bounds, coeffs=coeffs, degree=degree, scale=scale, device=device)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.grid(False)
    norm = cm.colors.Normalize(vmax=auto_vmax, vmin=0.0)
    ax.imshow(likelihood, extent=(-4.5, 4.5, -4.5, 4.5), origin='lower', cmap='viridis', norm=norm)

    if bounds is not None:
        x_min, y_min, x_max, y_max = bounds
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width, height,
                                 linewidth=2.5, edgecolor='red', facecolor='none',
                                 linestyle='--', label='Constraint Boundaries')
        ax.add_patch(rect)
        ax.legend(loc='upper right')
    elif coeffs is not None:
        xx, yy = np.meshgrid(np.linspace(-4.5, 4.5, grid_size), np.linspace(-4.5, 4.5, grid_size))
        grid_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32, device=device)
        grid_x_pow, grid_y_pow = compute_poly_features(grid_points, degree=degree, scale=scale)
        C_grid = coeffs.unsqueeze(0).expand(grid_points.shape[0], -1, -1)
        P_grid = evaluate_poly(grid_x_pow, grid_y_pow, C_grid).squeeze().cpu().numpy()
        P_grid = P_grid.reshape(grid_size, grid_size)
        ax.contour(xx, yy, P_grid, levels=[0.0], colors='red', linewidths=2.5, linestyles='dashed')
        ax.plot([], [], color='red', linewidth=2.5, linestyle='dashed', label='Constraint P(x) = 0')
        ax.legend(loc='upper right')

    ax.set_title('Model Likelihood')
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax, orientation='vertical', label='density')
    plt.show()

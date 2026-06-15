import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm

from constrained_fm.src.utils.polynomials import compute_poly_features, evaluate_poly
from constrained_fm.src.consts import POLYNOMIAL_DEGREE, PLANE_SCALE


def visualize_single_step(data_slice, title, ax=None, cmap='Blues',
                          bbox=None, coeffs=None, degree=POLYNOMIAL_DEGREE, scale=PLANE_SCALE, labels=None):
    if ax is None:
        with_legend = True
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        with_legend = False

    if labels is not None:
        ax.scatter(data_slice[:, 0], data_slice[:, 1], c=labels, cmap='tab10', s=1, alpha=0.6)

        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
    else:
        H, _, _ = np.histogram2d(data_slice[:, 0], data_slice[:, 1], bins=300, range=[[-4.5, 4.5], [-4.5, 4.5]])
        cmin = 0.0
        cmax = np.quantile(H, 0.99)
        norm = cm.colors.Normalize(vmax=cmax, vmin=cmin)
        ax.hist2d(data_slice[:, 0], data_slice[:, 1], bins=300, range=[[-4.5, 4.5], [-4.5, 4.5]],
                  norm=norm, cmap=cmap, cmin=1e-5)

    # --- Bounding Box Constraint Overlay ---
    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min

        rect = patches.Rectangle((x_min, y_min), width, height,
                                 linewidth=2.5, edgecolor='red', facecolor='none',
                                 linestyle='--', label='Constraint Boundaries')
        ax.add_patch(rect)
        if with_legend:
            ax.legend(loc='upper right')

    # --- Polynomial Constraint Overlay ---
    elif coeffs is not None:
        device = coeffs.device
        grid_res = 200
        xx, yy = np.meshgrid(np.linspace(-4.5, 4.5, grid_res), np.linspace(-4.5, 4.5, grid_res))
        grid_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32, device=device)

        grid_x_pow, grid_y_pow = compute_poly_features(grid_points, degree=degree, scale=scale)
        C_grid = coeffs.unsqueeze(0).expand(grid_points.shape[0], -1, -1)
        P_grid = evaluate_poly(grid_x_pow, grid_y_pow, C_grid).squeeze().cpu().numpy()
        P_grid = P_grid.reshape(grid_res, grid_res)

        ax.contour(xx, yy, P_grid, levels=[0.0], colors='red', linewidths=2.5, linestyles='dashed')

        if with_legend:
            ax.plot([], [], color='red', linewidth=2.5, linestyle='dashed', label='Constraint P(x) = 0')
            ax.legend(loc='upper right')

    ax.set_aspect('equal')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return ax

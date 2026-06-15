import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from constrained_fm.src.geometry.bounding_boxes import sample_bbox_around_points
from constrained_fm.src.geometry.polynomials import compute_poly_features, evaluate_poly
from constrained_fm.src.consts import POLYNOMIAL_DEGREE, PLANE_SCALE
from constrained_fm.src.datasets.gmm_target import get_points


def visualize_sampled_polynomials(degree=POLYNOMIAL_DEGREE, num_samples=4,
                                   num_points=10000, scale=PLANE_SCALE, device=None):
    """Draw a few random polynomial constraints and their feasible regions.
    """
    x_1, _ = get_points(num_points, device=device)
    x1_pow, y1_pow = compute_poly_features(x_1, degree=degree, scale=scale)

    grid_res = 200
    xx, yy = np.meshgrid(np.linspace(-4, 4, grid_res), np.linspace(-4, 4, grid_res))
    grid_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32, device=device)
    grid_x_pow, grid_y_pow = compute_poly_features(grid_points, degree=degree, scale=scale)

    fig, axs = plt.subplots(1, num_samples, figsize=(5 * num_samples, 5))
    if num_samples == 1:
        axs = [axs]

    for i in range(num_samples):
        # Random coefficients ~ N(0, 1)
        C = torch.randn(1, degree + 1, degree + 1, device=device)
        C_norm = torch.linalg.matrix_norm(C, ord='fro', dim=(1, 2), keepdim=True)
        C = C / (C_norm + 1e-8)
        C_points = C.expand(num_points, -1, -1)

        P_vals = evaluate_poly(x1_pow, y1_pow, C_points).squeeze().cpu().numpy()
        valid_mask = P_vals <= 0

        C_grid = C.expand(grid_points.shape[0], -1, -1)
        P_grid = evaluate_poly(grid_x_pow, grid_y_pow, C_grid).squeeze().cpu().numpy()
        P_grid = P_grid.reshape(grid_res, grid_res)

        ax = axs[i]
        points_np = x_1.cpu().numpy()
        # Invalid points
        ax.scatter(points_np[~valid_mask, 0], points_np[~valid_mask, 1],
                   c='lightgray', s=1, alpha=0.5, label='Invalid (P > 0)')
        # Valid points
        ax.scatter(points_np[valid_mask, 0], points_np[valid_mask, 1],
                   c='darkorange', s=1, alpha=0.8, label='Valid (P <= 0)')
        # Boundary
        ax.contour(xx, yy, P_grid, levels=[0.0], colors='red', linewidths=2.5, linestyles='dashed')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        valid_rate = valid_mask.mean() * 100
        ax.set_title(f'Random Poly (Degree {degree})\nValid Area: {valid_rate:.2f}%')
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.show()


def visualize_predicted_bbox_mass(predictor, gmm_true_pool, num_samples=4, width_range=(0.1, 7.0), plot_points=40000,
                                  device=None):
    """
    Visualizes randomly generated bounding boxes, the GMM points inside/outside them,
    and compares the neural network's predicted mass to the exact true mass.
    """
    predictor.eval()

    indices = torch.randperm(gmm_true_pool.shape[0], device=device)[:num_samples]
    anchors = gmm_true_pool[indices]

    boxes = sample_bbox_around_points(anchors, width_range=width_range)

    with torch.no_grad():
        predicted_masses = predictor(boxes).squeeze(-1) * 100.0

    vis_indices = torch.randperm(gmm_true_pool.shape[0], device=device)[:plot_points]
    vis_pool = gmm_true_pool[vis_indices]
    vis_pool_np = vis_pool.cpu().numpy()


    fig, axs = plt.subplots(num_samples, 1, figsize=(5, 5 * num_samples))
    if num_samples == 1:
        axs = [axs]

    for i in range(num_samples):
        ax = axs[i]
        box = boxes[i]
        x_min, y_min, x_max, y_max = box.tolist()

        in_x_full = (gmm_true_pool[:, 0] >= x_min) & (gmm_true_pool[:, 0] <= x_max)
        in_y_full = (gmm_true_pool[:, 1] >= y_min) & (gmm_true_pool[:, 1] <= y_max)
        true_mask_full = in_x_full & in_y_full
        true_mass = true_mask_full.float().mean().item() * 100.0

        in_x_vis = (vis_pool[:, 0] >= x_min) & (vis_pool[:, 0] <= x_max)
        in_y_vis = (vis_pool[:, 1] >= y_min) & (vis_pool[:, 1] <= y_max)
        plot_mask = (in_x_vis & in_y_vis).cpu().numpy()

        pred_mass = predicted_masses[i].item()

        ax.scatter(vis_pool_np[~plot_mask, 0], vis_pool_np[~plot_mask, 1],
                   c='lightgray', s=1, alpha=0.5, label='Outside')
        ax.scatter(vis_pool_np[plot_mask, 0], vis_pool_np[plot_mask, 1],
                   c='darkorange', s=1, alpha=0.8, label='Inside')

        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                 linewidth=2.5, edgecolor='red', facecolor='none',
                                 linestyle='dashed', label='Constraint Boundary')
        ax.add_patch(rect)

        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
        ax.set_aspect('equal')

        title_color = 'black' if abs(pred_mass - true_mass) < 2.0 else 'red'
        ax.set_title(f"Predicted Mass: {pred_mass:.2f}%\nTrue Mass: {true_mass:.2f}%",
                     color=title_color, fontweight='bold')

        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.show()


def visualize_disjoint_bboxes(predictor, gmm_true_pool, disjoint_boxes, plot_points=40000, device=None):
    """
    Visualizes a set of disjoint bounding boxes on a single plot.
    Highlights the points falling inside ANY of the boxes and compares
    the total predicted mass to the exact true mass.
    """
    predictor.eval()

    with torch.no_grad():
        predicted_masses = predictor(disjoint_boxes).squeeze(-1) * 100.0
        total_pred_mass = predicted_masses.sum().item()

    vis_indices = torch.randperm(gmm_true_pool.shape[0], device=device)[:plot_points]
    vis_pool = gmm_true_pool[vis_indices]
    vis_pool_np = vis_pool.cpu().numpy()

    global_true_mask = torch.zeros(gmm_true_pool.shape[0], dtype=torch.bool, device=device)
    global_plot_mask = torch.zeros(vis_pool.shape[0], dtype=torch.bool, device=device)

    fig, ax = plt.subplots(figsize=(6, 6))

    for i in range(disjoint_boxes.shape[0]):
        box = disjoint_boxes[i]
        x_min, y_min, x_max, y_max = box.tolist()

        in_x_full = (gmm_true_pool[:, 0] >= x_min) & (gmm_true_pool[:, 0] <= x_max)
        in_y_full = (gmm_true_pool[:, 1] >= y_min) & (gmm_true_pool[:, 1] <= y_max)
        global_true_mask |= (in_x_full & in_y_full)

        in_x_vis = (vis_pool[:, 0] >= x_min) & (vis_pool[:, 0] <= x_max)
        in_y_vis = (vis_pool[:, 1] >= y_min) & (vis_pool[:, 1] <= y_max)
        global_plot_mask |= (in_x_vis & in_y_vis)

        # Only add the label to the first rectangle to avoid duplicate legend entries
        label = 'Constraint Boundary' if i == 0 else None
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                 linewidth=2.5, edgecolor='red', facecolor='none',
                                 linestyle='dashed', label=label)
        ax.add_patch(rect)

        text_x = max(x_min, -4.4)
        if y_max + 0.1 >= 4.5:
            text_y = 4.2
        else:
            text_y = y_max + 0.1

        pred_mass_single = predicted_masses[i].item()
        ax.text(text_x, text_y, f"{pred_mass_single:.1f}%",
                color='darkred', fontweight='bold', fontsize=9)

    total_true_mass = global_true_mask.float().mean().item() * 100.0
    plot_mask_np = global_plot_mask.cpu().numpy()

    ax.scatter(vis_pool_np[~plot_mask_np, 0], vis_pool_np[~plot_mask_np, 1],
               c='lightgray', s=1, alpha=0.5, label='Outside')
    ax.scatter(vis_pool_np[plot_mask_np, 0], vis_pool_np[plot_mask_np, 1],
               c='darkorange', s=1, alpha=0.8, label='Inside')

    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.set_aspect('equal')

    title_color = 'black' if abs(total_pred_mass - total_true_mass) < 3.0 else 'red'
    ax.set_title(f"Total Predicted Mass: {total_pred_mass:.2f}%\nTotal True Mass: {total_true_mass:.2f}%",
                 color=title_color, fontweight='bold')

    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.show()


def visualize_compound_generation(samples, disjoint_boxes, points_per_box, metrics: dict = None, device=None):
    """
    Visualizes the generated point cloud from a compound bounding box constraint.
    Annotates each box with the expected point count vs. the actual generated count.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    samples_np = samples.cpu().numpy()
    ax.scatter(samples_np[:, 0], samples_np[:, 1], c='darkturquoise', s=2, alpha=0.8, label='Generated Points')

    global_success_mask = torch.zeros(samples.shape[0], dtype=torch.bool, device=device)

    for i in range(disjoint_boxes.shape[0]):
        box = disjoint_boxes[i]
        x_min, y_min, x_max, y_max = box.tolist()
        expected_count = points_per_box[i].item()

        in_x = (samples[:, 0] >= x_min) & (samples[:, 0] <= x_max)
        in_y = (samples[:, 1] >= y_min) & (samples[:, 1] <= y_max)
        in_this_box = in_x & in_y

        global_success_mask |= in_this_box
        actual_count = in_this_box.sum().item()

        label = 'Constraint Boundary' if i == 0 else None
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                 linewidth=2.5, edgecolor='red', facecolor='none',
                                 linestyle='dashed', label=label)
        ax.add_patch(rect)

        text_x = max(x_min, -4.4)
        text_y = 4.2 if y_max + 0.1 >= 4.5 else y_max + 0.1

        stats_text = f"Exp: {expected_count}\nAct: {actual_count}"
        color = 'darkgreen' if abs(expected_count - actual_count) < (expected_count * 0.05) else 'darkred'

        ax.text(text_x, text_y, stats_text, color=color, fontweight='bold', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.0))

    total_success_rate = global_success_mask.float().mean().item() * 100.0
    title_str = f"Compound Generation\nGlobal Success Rate: {total_success_rate:.2f}%"

    if metrics is not None:
        metrics_str = f"SWD: {metrics.get('swd', 0):.4f}  |  MMD: {metrics.get('mmd', 0):.4f}  |  JSD: {metrics.get('jsd', 0):.4f}"
        title_str += f"\n{metrics_str}"

    ax.set_title(title_str, fontweight='bold')
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.show()

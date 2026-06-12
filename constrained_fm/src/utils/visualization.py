import torch
import numpy as np
from torch.distributions import Independent, Normal, MultivariateNormal
from flow_matching.solver import ODESolver
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from tqdm.auto import tqdm
import seaborn as sns

from constrained_fm.src.utils.polynomials import compute_poly_features, evaluate_poly
from constrained_fm.src.evaluation import compute_success_rate_bbox, compute_success_rate_polynomial
from constrained_fm.src.consts import GMM_MEANS, GMM_COVS, GMM_WEIGHTS, POLYNOMIAL_DEGREE, PLANE_SCALE
from constrained_fm.src.data_handlers.gmm_2d import get_points, compute_gmm_density
from constrained_fm.src.models.wrapper import WrappedModel


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


def visualize_true_gmm_likelihood(means=GMM_MEANS, covs=GMM_COVS, weights=GMM_WEIGHTS, grid_size=200, device=None):
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


def plot_loss_curve(losses):
    plt.figure(figsize=(8, 5))
    plt.plot(losses, color='indigo')

    plt.title("2D Flow Matching Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True, alpha=0.3)

    plt.gca().set_facecolor('#f8f9fa')

    plt.tight_layout()
    plt.show()


def assign_gaussian_to_points(points, means=GMM_MEANS, covs=GMM_COVS, weights=GMM_WEIGHTS, device=None):
    means = torch.tensor(means, device=device)
    covs = torch.tensor(covs, device=device)
    weights = torch.tensor(weights, device=device)

    pts_tensor = torch.tensor(points, dtype=torch.float32, device=device)
    num_components = means.shape[0]

    # Matrix to hold the log probability of each point for each cluster
    log_probs = torch.zeros(pts_tensor.shape[0], num_components, device=device)

    for k in range(num_components):
        comp_dist = MultivariateNormal(means[k], covs[k])

        # P(Class | X) ∝ P(X | Class) * P(Class)
        # In log space: log(P(X | Class)) + log(P(Class))
        log_probs[:, k] = comp_dist.log_prob(pts_tensor) + torch.log(weights[k])

    # The point belongs to the Gaussian that yields the highest probability
    labels = torch.argmax(log_probs, dim=1).cpu().numpy()

    return labels


def generate_and_visualize_samples(model, num_samples=50000, step_size=0.05,
                                   bounds: list | None = None, coeffs=None, degree=POLYNOMIAL_DEGREE, scale=PLANE_SCALE,
                                   cluster_points=True):
    model.eval()

    # Safely grab device from model to prevent crashes
    device = next(model.parameters()).device

    wrapped_vf = WrappedModel(model)
    solver = ODESolver(velocity_model=wrapped_vf)

    T = torch.linspace(0, 1, 10).to(device)
    x_init = torch.randn((num_samples, 2), dtype=torch.float32, device=device)

    kwargs = {}
    if bounds is not None:
        bounds_tensor = torch.tensor([bounds], dtype=torch.float32, device=device)
        kwargs['bounds'] = bounds_tensor.expand(num_samples, 4)
    elif coeffs is not None:
        coeffs_flat = coeffs.view(1, -1)
        kwargs['coeffs'] = coeffs_flat.expand(num_samples, -1)

    with torch.no_grad():
        samples = solver.sample(time_grid=T, x_init=x_init, method='midpoint',
                                step_size=step_size, return_intermediates=True,
                                **kwargs)

    samples_np = samples.cpu().numpy()
    T_np = T.cpu().numpy()

    # Labeling points into gaussian clusters:
    labels = None
    if cluster_points:
        labels = assign_gaussian_to_points(samples_np[-1], device=device)

    # Visualization 1: The 10-step trajectory
    fig, axs = plt.subplots(1, 10, figsize=(20, 2))
    for i in range(10):
        visualize_single_step(samples_np[i], title=f"t = {T_np[i]:.2f}", ax=axs[i],
                              bbox=bounds, coeffs=coeffs, degree=degree, scale=scale,
                              labels=labels)
    plt.tight_layout()
    plt.show()

    final_title = "Samples"
    final_samples = samples_np[-1]

    if bounds is not None:
        success_rate = compute_success_rate_bbox(final_samples, bounds)
        final_title = f"Samples\nSuccess Rate: {success_rate:.3f}%"

    elif coeffs is not None:
        success_rate = compute_success_rate_polynomial(final_samples, coeffs, degree, scale, device)
        final_title = f"Samples\nSuccess Rate: {success_rate:.3f}%"

    # Visualization 2: The final target distribution
    visualize_single_step(samples_np[-1], title=final_title, cmap='Oranges',
                          bbox=bounds, coeffs=coeffs, degree=degree, scale=scale,
                          labels=None)
    plt.show()


def calculate_vmax(bounds=None, coeffs=None, degree=POLYNOMIAL_DEGREE, scale=PLANE_SCALE, unconstrained_vmax=None,
                   num_samples=100000, device=None):
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


def compute_and_visualize_likelihood(model, bounds=None, coeffs=None, degree=POLYNOMIAL_DEGREE, scale=PLANE_SCALE,
                                     grid_size=200, step_size=0.05, eval_batch_size=4000, device=None):
    model.eval()

    wrapped_vf = WrappedModel(model)
    solver = ODESolver(velocity_model=wrapped_vf)

    x_grid = torch.meshgrid(torch.linspace(-4.5, 4.5, grid_size),
                            torch.linspace(-4.5, 4.5, grid_size),
                            indexing='ij')

    x_1 = torch.stack([x_grid[0].flatten(), x_grid[1].flatten()], dim=1).to(device)
    num_points = x_1.shape[0]

    gaussian_log_density = Independent(Normal(torch.zeros(2, device=device),
                                              torch.ones(2, device=device)), 1).log_prob

    full_bounds_tensor = None
    full_coeffs_tensor = None

    auto_vmax = calculate_vmax(bounds=bounds, coeffs=coeffs, degree=degree, scale=scale, device=device)

    if bounds is not None:
        absolute_bounds = torch.tensor([bounds], dtype=torch.float32, device=device)
        full_bounds_tensor = absolute_bounds.expand(num_points, 4)
    elif coeffs is not None:
        full_coeffs_tensor = coeffs.view(1, -1).expand(num_points, -1)

    exact_log_p_list = []
    print(f"Computing exact divergence for {num_points} points in chunks of {eval_batch_size}...")

    with torch.no_grad():
        for i in tqdm(range(0, num_points, eval_batch_size)):
            x_1_chunk = x_1[i:i + eval_batch_size]

            chunk_kwargs = {}
            if full_bounds_tensor is not None:
                chunk_kwargs['bounds'] = full_bounds_tensor[i:i + eval_batch_size]
            if full_coeffs_tensor is not None:
                chunk_kwargs['coeffs'] = full_coeffs_tensor[i:i + eval_batch_size]

            _, chunk_log_p = solver.compute_likelihood(
                x_1=x_1_chunk,
                method='midpoint',
                step_size=step_size,
                exact_divergence=True,
                log_p0=gaussian_log_density,
                **chunk_kwargs
            )

            exact_log_p_list.append(chunk_log_p)

    exact_log_p = torch.cat(exact_log_p_list, dim=0)

    likelihood = torch.exp(exact_log_p).cpu().reshape(grid_size, grid_size).numpy().T

    # Visualization
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.grid(False)
    norm = cm.colors.Normalize(vmax=auto_vmax, vmin=0.0)

    ax.imshow(likelihood, extent=(-4.5, 4.5, -4.5, 4.5), origin='lower', cmap='viridis', norm=norm)

    # --- Overlays ---
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


def visualize_sampled_polynomials(degree=POLYNOMIAL_DEGREE, num_samples=4, num_points=10000, scale=PLANE_SCALE, device=None):
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
        # Sample random coefficients ~ N(0, 1)
        C = torch.randn(1, degree + 1, degree + 1, device=device)

        C_norm = torch.linalg.matrix_norm(C, ord='fro', dim=(1,2), keepdim=True)
        C = C / (C_norm + 1e-8)
        C_points = C.expand(num_points, -1, -1)

        P_vals = evaluate_poly(x1_pow, y1_pow, C_points).squeeze().cpu().numpy()

        valid_mask = P_vals <= 0

        # Evaluate the polynomial on the dense grid to draw the boundary
        C_grid = C.expand(grid_points.shape[0], -1, -1)
        P_grid = evaluate_poly(grid_x_pow, grid_y_pow, C_grid).squeeze().cpu().numpy()
        P_grid = P_grid.reshape(grid_res, grid_res)


        ax = axs[i]
        points_np = x_1.cpu().numpy()

        # Plot Invalid points
        ax.scatter(points_np[~valid_mask, 0], points_np[~valid_mask, 1],
                   c='lightgray', s=1, alpha=0.5, label='Invalid (P > 0)')

        # Plot Valid points
        ax.scatter(points_np[valid_mask, 0], points_np[valid_mask, 1],
                   c='darkorange', s=1, alpha=0.8, label='Valid (P <= 0)')

        # Draw the actual polynomial boundary line P(x)=0
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

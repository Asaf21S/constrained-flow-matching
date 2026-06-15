import torch
import numpy as np
import matplotlib.pyplot as plt

from constrained_fm.src.utils.polynomials import compute_poly_features, evaluate_poly
from constrained_fm.src.evaluation import compute_success_rate_bbox, compute_success_rate_polynomial
from constrained_fm.src.consts import POLYNOMIAL_DEGREE, PLANE_SCALE, GMM_MEANS, GMM_COVS, GMM_WEIGHTS
from constrained_fm.src.data_handlers.gmm_2d import get_points


def assign_gaussian_to_points(points, means=GMM_MEANS, covs=GMM_COVS, weights=GMM_WEIGHTS, device=None):
    """Assign each point to the most likely Gaussian component.

    Returns an ``np.ndarray`` of integer labels.
    """
    means = torch.tensor(means, device=device)
    covs = torch.tensor(covs, device=device)
    weights = torch.tensor(weights, device=device)

    pts_tensor = torch.tensor(points, dtype=torch.float32, device=device)
    num_components = means.shape[0]

    log_probs = torch.zeros(pts_tensor.shape[0], num_components, device=device)
    for k in range(num_components):
        comp_dist = torch.distributions.MultivariateNormal(means[k], covs[k])
        log_probs[:, k] = comp_dist.log_prob(pts_tensor) + torch.log(weights[k])

    labels = torch.argmax(log_probs, dim=1).cpu().numpy()
    return labels


def plot_loss_curve(losses):
    """Simple helper to plot a training loss curve.

    ``losses`` should be an iterable of scalar loss values.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(losses, color='indigo')
    plt.title("2D Flow Matching Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True, alpha=0.3)
    plt.gca().set_facecolor('#f8f9fa')
    plt.tight_layout()
    plt.show()


def generate_and_visualize_samples(model, num_samples=50000, step_size=0.05,
                                   bounds: list | None = None, coeffs: torch.Tensor = None,
                                   degree=POLYNOMIAL_DEGREE, scale=PLANE_SCALE,
                                   cluster_points=True):
    """Generate samples from ``model`` and visualise intermediate steps.

    Parameters
    ----------
    model: torch.nn.Module
        The flow‑matching model exposing a ``sample`` method.
    num_samples: int
        Number of points to draw.
    step_size: float
        Integration step size for the sampler.
    bounds: list | None
        Optional rectangular constraint ``[x_min, y_min, x_max, y_max]``.
    coeffs: torch.Tensor | None
        Optional polynomial constraint.
    degree, scale: int, float
        Polynomial hyper‑parameters used for visualisation.
    cluster_points: bool
        Whether to colour points by their most likely Gaussian component.
    """
    device = next(model.parameters()).device
    kwargs = {"bounds": bounds} if bounds is not None else {"coeffs": coeffs} if coeffs is not None else {}
    samples, T = model.sample(num_points=num_samples, **kwargs, step_size=step_size,
                              return_intermediates=True, device=device)

    samples_np = samples.cpu().numpy()
    T_np = T.cpu().numpy()

    labels = None
    if cluster_points:
        labels = assign_gaussian_to_points(samples_np[-1], device=device)

    # Plot the 10‑step trajectory
    fig, axs = plt.subplots(1, 10, figsize=(20, 2))
    for i in range(10):
        from .single_step import visualize_single_step
        visualize_single_step(samples_np[i], title=f"t = {T_np[i]:.2f}", ax=axs[i],
                              bbox=bounds, coeffs=coeffs, degree=degree, scale=scale,
                              labels=labels)
    plt.tight_layout()
    plt.show()

    final_samples = samples_np[-1]
    final_title = "Samples"
    if bounds is not None:
        success_rate = compute_success_rate_bbox(final_samples, bounds)
        final_title = f"Samples\nSuccess Rate: {success_rate:.3f}%"
    elif coeffs is not None:
        success_rate = compute_success_rate_polynomial(final_samples, coeffs, degree, scale, device)
        final_title = f"Samples\nSuccess Rate: {success_rate:.3f}%"

    # Visualise the final distribution
    from .single_step import visualize_single_step
    visualize_single_step(samples_np[-1], title=final_title, cmap='Oranges',
                          bbox=bounds, coeffs=coeffs, degree=degree, scale=scale,
                          labels=None)
    plt.show()


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

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm

from constrained_fm.src.geometry.polynomials import compute_poly_features, evaluate_poly
from constrained_fm.src.metrics.success_rates import compute_success_rate_bbox, compute_success_rate_polynomial
from constrained_fm.src.consts import POLYNOMIAL_DEGREE, PLANE_SCALE, GMM_MEANS, GMM_COVS, GMM_WEIGHTS


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


def generate_and_visualize_samples(model, num_samples=50000, step_size=0.05, bounds=None, coeffs=None,
                                   degree=POLYNOMIAL_DEGREE, scale=PLANE_SCALE, cluster_points=True, metrics: dict = None):
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

    if metrics is not None:
        metrics_str = f"SWD: {metrics.get('swd', 0):.4f}  |  MMD: {metrics.get('mmd', 0):.4f}  |  JSD: {metrics.get('jsd', 0):.4f}"
        final_title += f"\n{metrics_str}"

    # Visualise the final distribution
    visualize_single_step(samples_np[-1], title=final_title, cmap='Oranges',
                          bbox=bounds, coeffs=coeffs, degree=degree, scale=scale,
                          labels=None)
    plt.show()

import torch
import numpy as np
import ot
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon


from constrained_fm.src.geometry.polynomials import compute_poly_features, evaluate_poly


def filter_true_samples(x_true: torch.Tensor, bounds=None, coeffs=None, x_pow=None, y_pow=None):
    """Filters the ground truth samples to only include those satisfying the constraint."""
    if bounds is not None:
        x_min, y_min, x_max, y_max = bounds
        mask = (x_true[:, 0] >= x_min) & (x_true[:, 0] <= x_max) & \
               (x_true[:, 1] >= y_min) & (x_true[:, 1] <= y_max)
        return x_true[mask]

    elif coeffs is not None:
        if x_pow is None or y_pow is None:
            x_pow, y_pow = compute_poly_features(x_true)
        batch_C = coeffs.unsqueeze(0).expand(x_true.shape[0], -1, -1)
        p_vals = evaluate_poly(x_pow, y_pow, batch_C).squeeze()
        mask = p_vals <= 0
        return x_true[mask]

    return x_true


def compute_swd(samples_gen: torch.Tensor, samples_true: torch.Tensor, num_projections=50) -> float:
    """Wrapper for Sliced Wasserstein Distance using the POT library."""
    if len(samples_gen) == 0 or len(samples_true) == 0:
        return float('inf')

    X = samples_gen.detach().cpu().numpy()
    Y = samples_true.detach().cpu().numpy()

    swd = ot.sliced_wasserstein_distance(X, Y, n_projections=num_projections)
    return float(swd)


def compute_mmd(samples_gen: torch.Tensor, samples_true: torch.Tensor, gamma=1.0) -> float:
    """Wrapper for Maximum Mean Discrepancy using Scikit-Learn RBF kernels."""
    if len(samples_gen) == 0 or len(samples_true) == 0:
        return float('inf')

    X = samples_gen.detach().cpu().numpy()
    Y = samples_true.detach().cpu().numpy()

    # Subsample to prevent RAM spikes on the NxN distance matrices
    max_pts = 5000
    if len(X) > max_pts: X = X[np.random.choice(len(X), max_pts, replace=False)]
    if len(Y) > max_pts: Y = Y[np.random.choice(len(Y), max_pts, replace=False)]

    K_XX = rbf_kernel(X, X, gamma=gamma)
    K_YY = rbf_kernel(Y, Y, gamma=gamma)
    K_XY = rbf_kernel(X, Y, gamma=gamma)

    mmd = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    return max(0.0, float(mmd))


def compute_jsd(samples_gen: torch.Tensor, samples_true: torch.Tensor, grid_size=100) -> float:
    """Wrapper for Jensen-Shannon Divergence using SciPy."""
    if len(samples_gen) < 10 or len(samples_true) < 10:
        return float('inf')

    X = samples_gen.detach().cpu().numpy()
    Y = samples_true.detach().cpu().numpy()

    # Define the evaluation grid bounding box
    x_min = min(X[:, 0].min(), Y[:, 0].min()) - 0.5
    x_max = max(X[:, 0].max(), Y[:, 0].max()) + 0.5
    y_min = min(X[:, 1].min(), Y[:, 1].min()) - 0.5
    y_max = max(X[:, 1].max(), Y[:, 1].max()) + 0.5

    grid_X, grid_Y = np.mgrid[x_min:x_max:complex(0, grid_size), y_min:y_max:complex(0, grid_size)]
    positions = np.vstack([grid_X.ravel(), grid_Y.ravel()])

    Z_gen = gaussian_kde(X.T)(positions)
    Z_true = gaussian_kde(Y.T)(positions)

    Z_gen /= Z_gen.sum()
    Z_true /= Z_true.sum()

    js_distance = jensenshannon(Z_gen, Z_true)
    return float(js_distance ** 2)

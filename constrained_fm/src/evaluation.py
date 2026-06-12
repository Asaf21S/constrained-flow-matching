"""Utility functions for evaluating sample constraints.

This module provides reusable helpers that compute the success rate of
samples with respect to a bounding‑box constraint or a polynomial constraint.
They are used by :func:`generate_and_visualize_samples` in
``visualization.py``.
"""

import torch
from flow_matching.solver import ODESolver
import numpy as np
from tqdm import tqdm
import ot
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
import os
import json
from datetime import datetime


from constrained_fm.src.utils.polynomials import compute_poly_features, evaluate_poly
from constrained_fm.src.models.wrapper import WrappedModel
from constrained_fm.src.consts import EVALUATION_RESULTS_PATH


def compute_success_rate_bbox(samples: torch.Tensor | list, bounds: list) -> float | list[float]:
    """Return the percentage of *samples* that lie inside *bounds*.

    Parameters
    ----------
    samples:
        Either a NumPy ``ndarray``/list or a ``torch.Tensor``.
        Shape can be ``(N, 2)`` for single constraints or ``(C, N, 2)`` for batched.
    bounds:
        ``[x_min, y_min, x_max, y_max]`` describing the axis-aligned rectangle,
        or a list of bounds ``[[x1, y1, x2, y2], ...]`` if samples are batched.

    Returns
    -------
    float | list[float]
        Success rate as a percentage (0-100). Returns a single float if input is 2D,
        or a list of floats if input is 3D batched.
    """
    if not isinstance(samples, torch.Tensor):
        samples = torch.tensor(samples, dtype=torch.float32)
    else:
        samples = samples.detach().cpu()

    if samples.ndim == 2:
        # --- Original Single-Constraint Logic ---
        x_min, y_min, x_max, y_max = bounds
        inside = (samples[:, 0] >= x_min) & (samples[:, 0] <= x_max) & \
                 (samples[:, 1] >= y_min) & (samples[:, 1] <= y_max)
        return inside.float().mean().item() * 100.0

    elif samples.ndim == 3:
        # --- New Batched Logic (C, N, 2) ---
        bounds_t = torch.tensor(bounds, dtype=torch.float32, device=samples.device)

        # Extract boundaries and reshape to (C, 1) for broadcasting against (C, N)
        x_min = bounds_t[:, 0].unsqueeze(1)
        y_min = bounds_t[:, 1].unsqueeze(1)
        x_max = bounds_t[:, 2].unsqueeze(1)
        y_max = bounds_t[:, 3].unsqueeze(1)

        inside = (samples[:, :, 0] >= x_min) & (samples[:, :, 0] <= x_max) & \
                 (samples[:, :, 1] >= y_min) & (samples[:, :, 1] <= y_max)

        success_rates = inside.float().mean(dim=1) * 100.0
        return success_rates.tolist()


def compute_success_rate_polynomial(
        samples: torch.Tensor | list,
        coeffs: torch.Tensor,
        degree: int,
        scale: float,
        device: torch.device | None = None,
) -> float | list[float]:
    """Return the percentage of *samples* that satisfy the polynomial constraint.

    A point is considered *valid* when the polynomial evaluated at that point is
    ``<= 0`` (i.e., it lies inside the feasible region defined by ``P(x) = 0``).

    Parameters
    ----------
    samples:
        Shape ``(N, 2)`` for single constraints or ``(C, N, 2)`` for batched.
    coeffs:
        Tensor of polynomial coefficients. Shape ``(1, D+1, D+1)`` or ``(D+1, D+1)``
        for single. Shape ``(C, D+1, D+1)`` for batched.
    degree:
        Polynomial degree.
    scale:
        Scaling factor applied inside ``compute_poly_features``.
    device:
        Optional torch device on which to perform the computation.

    Returns
    -------
    float | list[float]
        Success rate as a percentage (0-100). Returns a single float if input is 2D,
        or a list of floats if input is 3D batched.
    """
    if device is None:
        device = coeffs.device

    if not isinstance(samples, torch.Tensor):
        samples_t = torch.tensor(samples, dtype=torch.float32, device=device)
    else:
        samples_t = samples.to(device)

    if samples_t.ndim == 2:
        # --- Original Single-Constraint Logic ---
        x_pow, y_pow = compute_poly_features(samples_t, degree=degree, scale=scale)

        # Safely handle coeffs shape whether it's (D+1, D+1) or already batched (1, D+1, D+1)
        if coeffs.ndim == 2:
            batch_C = coeffs.unsqueeze(0).expand(samples_t.shape[0], -1, -1)
        else:
            batch_C = coeffs.expand(samples_t.shape[0], -1, -1)

        p_vals = evaluate_poly(x_pow, y_pow, batch_C).squeeze()
        return (p_vals <= 0).float().mean().item() * 100.0

    elif samples_t.ndim == 3:
        # --- New Batched Logic (C, N, 2) ---
        C_dim, N_dim, _ = samples_t.shape

        # Flatten to (C*N, 2) so compute_poly_features and evaluate_poly don't break
        samples_flat = samples_t.reshape(-1, 2)
        x_pow_flat, y_pow_flat = compute_poly_features(samples_flat, degree=degree, scale=scale)

        # Ensure coeffs is strictly (C, D+1, D+1)
        if coeffs.ndim == 4:  # In case it was passed as (C, 1, D+1, D+1)
            coeffs = coeffs.squeeze(1)

        # Expand coeffs to (C, N, D+1, D+1) and flatten to (C*N, D+1, D+1)
        coeffs_expanded = coeffs.unsqueeze(1).expand(C_dim, N_dim, -1, -1).reshape(-1, degree + 1, degree + 1)

        p_vals_flat = evaluate_poly(x_pow_flat, y_pow_flat, coeffs_expanded).squeeze()

        # Reshape back to (C, N) and calculate percentage per constraint
        p_vals = p_vals_flat.reshape(C_dim, N_dim)
        success_rates = (p_vals <= 0).float().mean(dim=1) * 100.0

        return success_rates.tolist()

def run_evaluation_inference(model, x0, bounds=None, coeffs=None, step_size=0.05, batch_size=100000):
    model.eval()
    device = next(model.parameters()).device

    wrapped_vf = WrappedModel(model)
    solver = ODESolver(velocity_model=wrapped_vf)

    num_samples = x0.shape[0]

    if bounds is not None:
        cond_t = torch.tensor(bounds, dtype=torch.float32, device=device)
        if cond_t.ndim == 1:
            cond_t = cond_t.unsqueeze(0)

        num_conditions = cond_t.shape[0]
        # Shape: [B, 4] -> [B, N, 4] -> [B * N, 4]
        cond_expanded = cond_t.unsqueeze(1).expand(num_conditions, num_samples, 4).reshape(-1, 4)
        kwarg_name = 'bounds'

    elif coeffs is not None:
        cond_t = torch.tensor(coeffs, dtype=torch.float32, device=device)
        if cond_t.ndim == 1:
            cond_t = cond_t.unsqueeze(0)
        elif cond_t.ndim == 3:  # Flatten [B, 4, 4] to [B, 16] if necessary
            cond_t = cond_t.reshape(cond_t.shape[0], -1)

        num_conditions = cond_t.shape[0]
        # Shape: [C, 16] -> [C, N, 16] -> [C * N, 16]
        cond_expanded = cond_t.unsqueeze(1).expand(num_conditions, num_samples, cond_t.shape[-1]).reshape(
            -1, cond_t.shape[-1])
        kwarg_name = 'coeffs'
    else:
        num_conditions = 1
        cond_expanded = None

    # Expand x0: [N, 2] -> [B, N, 2] -> [B * N, 2]
    x0_expanded = x0.unsqueeze(0).expand(num_conditions, num_samples, x0.shape[-1]).reshape(-1, x0.shape[-1])
    total_evaluations = x0_expanded.shape[0]

    final_samples_list = []

    print(f"Model device: {next(model.parameters()).device}")
    print(f"Data chunk device: {x0_expanded.device}")

    with torch.inference_mode():
        for i in tqdm(range(0, total_evaluations, batch_size), desc="Evaluating Batches"):
            x0_chunk = x0_expanded[i:i + batch_size]

            chunk_kwargs = {}
            if cond_expanded is not None:
                chunk_kwargs[kwarg_name] = cond_expanded[i:i + batch_size]

            samples_chunk = solver.sample(
                time_grid=torch.linspace(0, 1, int(1 / step_size)).to(device),
                x_init=x0_chunk,
                method='midpoint',
                step_size=step_size,
                return_intermediates=False,
                **chunk_kwargs
            )

            # Handle output shape depending on your specific ODESolver implementation
            if isinstance(samples_chunk, torch.Tensor) and samples_chunk.ndim == 3:
                final_samples_list.append(samples_chunk[-1].cpu().numpy())
            else:
                final_samples_list.append(samples_chunk.cpu().numpy())

    final_samples_np = np.concatenate(final_samples_list, axis=0)
    final_samples_reshaped = final_samples_np.reshape(num_conditions, num_samples, -1)

    if num_conditions == 1:
        return final_samples_reshaped[0]

    return final_samples_reshaped


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


def evaluate_distributional_metrics_batched(samples_gen_batched, x_true_pool, bounds=None, coeffs=None):
    """
    Evaluates SWD, MMD, and JSD for batched constraints.

    Parameters:
    samples_gen_batched: Tensor of shape (C, N, 2)
    x_true_pool: Tensor of shape (M, 2) containing all unconstrained ground truth points
    bounds: List of bounds [[x1,y1,x2,y2], ...] of length C
    coeffs: Tensor of shape (C, D+1, D+1)
    """
    return_single = False
    if samples_gen_batched.ndim == 2:
        return_single = True
        samples_gen_batched = samples_gen_batched.unsqueeze(0)

    C_dim = samples_gen_batched.shape[0]

    results = {
        "swd": [],
        "mmd": [],
        "jsd": []
    }

    if coeffs is not None:
        x_pow_true, y_pow_true = compute_poly_features(x_true_pool)
    else:
        x_pow_true, y_pow_true = None, None

    for i in tqdm(range(C_dim), desc="Distributional Metrics Evaluation"):
        samples_gen_single = samples_gen_batched[i]
        current_bounds = bounds[i] if bounds is not None else None
        current_coeffs = coeffs[i] if coeffs is not None else None
        x_true_filtered = filter_true_samples(x_true_pool, bounds=current_bounds, coeffs=current_coeffs,
                                              x_pow=x_pow_true, y_pow=y_pow_true)

        results["swd"].append(compute_swd(samples_gen_single, x_true_filtered))
        results["mmd"].append(compute_mmd(samples_gen_single, x_true_filtered))
        results["jsd"].append(compute_jsd(samples_gen_single, x_true_filtered))

    if return_single:
        return {k: v[0] for k, v in results.items()}

    return results


def log_evaluation_metrics(metrics_dict: dict, note: str, eval_type: str = "unconstrained",
                           path: str = EVALUATION_RESULTS_PATH):
    """
    Appends evaluation metrics to a JSON log file with a timestamp and tracking note.

    Parameters:
    -----------
    metrics_dict: dict
        The dictionary containing your metrics (e.g., {'success_rate': [...], 'swd': [...]})
    note: str
        A short description of what changed in this run (e.g., "Increased ResBlocks to 5")
    eval_type: str
        The type of evaluation (e.g., "unconstrained", "bbox", "polynomial")
    path: str
        Path to the JSON log file.
    """
    summary_stats = {}
    raw_data = {}

    for key, val in metrics_dict.items():
        if isinstance(val, (np.ndarray, torch.Tensor)):
            val = val.tolist()

        raw_data[key] = val

        if isinstance(val, list) and len(val) > 0:
            summary_stats[f"{key}_median"] = float(np.median(val))
            summary_stats[f"{key}_mean"] = float(np.mean(val))
        elif isinstance(val, (int, float)):
            summary_stats[key] = float(val)

    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "eval_type": eval_type,
        "note": note,
        "summary": summary_stats,
        "raw_metrics": raw_data
    }

    log_dir = os.path.dirname(path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                history = json.load(f)
        except json.JSONDecodeError:
            history = []
    else:
        history = []

    history.append(log_entry)

    with open(path, "w") as f:
        json.dump(history, f, indent=4)

    print(f"Logged {eval_type} metrics to '{path}'")
    print(f"   Note: {note}")
    for k, v in summary_stats.items():
        if "median" in k or not any(x in k for x in ["median", "mean"]):
            print(f"   - {k}: {v:.4f}")


def load_logged_metrics(path: str = EVALUATION_RESULTS_PATH, entry_index: int = -1):
    """
    Loads a specific run from the JSON log.

    Parameters:
    -----------
    path: str
        Path to the JSON log file.
    entry_index: int
        Which log entry to load. Defaults to -1 (the most recent run).
    """
    if not os.path.exists(path):
        print(f"Error: Log file not found at '{path}'")
        return None

    with open(path, "r") as f:
        try:
            history = json.load(f)
        except json.JSONDecodeError:
            print("Error: Could not decode JSON. File might be empty or corrupted.")
            return None

    if len(history) == 0:
        print("Log file is empty.")
        return None

    try:
        entry = history[entry_index]
    except IndexError:
        print(f"Error: Index {entry_index} out of bounds. The log only has {len(history)} entries.")
        return None

    print(f"Loaded Run: {entry.get('timestamp', 'Unknown Time')}")
    print(f"Evaluation Type: {entry.get('eval_type', 'N/A')}")
    print(f"Note: {entry.get('note', 'No note provided')}")

    metrics_dict = entry.get("raw_metrics", {})

    if not metrics_dict:
        print("Error: No raw metrics found in this log entry.")
        return None

    return metrics_dict


def print_readme_metrics_table(metrics_dict: dict):
    """
    Computes summary statistics from a metrics dictionary and prints
    a formatted Markdown table ready for a README file.
    Handles both batched lists (constrained) and single floats (unconstrained).
    """
    lines = [
        "| Metric | Median / Value | Mean | Worst 5% | Target |",
        "| :--- | :--- | :--- | :--- | :--- |"
    ]

    def clean_data(data_list):
        arr = np.array(data_list, dtype=float)
        return arr[~np.isinf(arr)]

    if 'success_rate' in metrics_dict:
        sr_data = metrics_dict['success_rate']

        if isinstance(sr_data, (list, np.ndarray)) and len(sr_data) > 0:
            clean_sr = clean_data(sr_data)
            if len(clean_sr) > 0:
                lines.append(
                    f"| **Success Rate (%)** | {np.median(clean_sr):.2f} | {np.mean(clean_sr):.2f} | {np.percentile(clean_sr, 5):.2f} | *Higher is better* |")

        elif isinstance(sr_data, (float, int)):
            lines.append(f"| **Success Rate (%)** | {float(sr_data):.2f} | - | - | *Higher is better* |")

    dist_metrics = [
        ('swd', 'Sliced Wasserstein (SWD)'),
        ('mmd', 'Mean Discrepancy (MMD)'),
        ('jsd', 'Jensen-Shannon (JSD)')
    ]

    for key, name in dist_metrics:
        if key in metrics_dict:
            raw_data = metrics_dict[key]

            if isinstance(raw_data, (list, np.ndarray)) and len(raw_data) > 0:
                clean_arr = clean_data(raw_data)
                if len(clean_arr) > 0:
                    lines.append(
                        f"| **{name}** | {np.median(clean_arr):.4f} | {np.mean(clean_arr):.4f} | {np.percentile(clean_arr, 95):.4f} | *Lower is better* |")

            elif isinstance(raw_data, (float, int)):
                if not np.isinf(raw_data):
                    lines.append(f"| **{name}** | {float(raw_data):.4f} | - | - | *Lower is better* |")

    markdown_table = "\n".join(lines)
    print(markdown_table)

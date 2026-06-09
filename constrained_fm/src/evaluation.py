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

from constrained_fm.src.utils.polynomials import compute_poly_features, evaluate_poly
from constrained_fm.src.models.wrapper import WrappedModel


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

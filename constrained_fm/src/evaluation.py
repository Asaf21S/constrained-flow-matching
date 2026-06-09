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


def compute_success_rate_bbox(samples: torch.Tensor | list, bounds: list) -> float:
    """Return the percentage of *samples* that lie inside *bounds*.

    Parameters
    ----------
    samples:
        Either a NumPy ``ndarray``/list or a ``torch.Tensor`` of shape ``(N, 2)``
        representing ``(x, y)`` coordinates.
    bounds:
        ``[x_min, y_min, x_max, y_max]`` describing the axis‑aligned rectangle.

    Returns
    -------
    float
        Success rate as a percentage (0‑100).
    """
    if not isinstance(samples, torch.Tensor):
        samples = torch.tensor(samples, dtype=torch.float32)
    else:
        samples = samples.detach().cpu()

    x_min, y_min, x_max, y_max = bounds
    inside = (samples[:, 0] >= x_min) & (samples[:, 0] <= x_max) & \
             (samples[:, 1] >= y_min) & (samples[:, 1] <= y_max)
    return inside.float().mean().item() * 100.0


def compute_success_rate_polynomial(
    samples: torch.Tensor | list,
    coeffs: torch.Tensor,
    degree: int,
    scale: float,
    device: torch.device | None = None,
) -> float:
    """Return the percentage of *samples* that satisfy the polynomial constraint.

    A point is considered *valid* when the polynomial evaluated at that point is
    ``<= 0`` (i.e., it lies inside the feasible region defined by ``P(x) = 0``).

    Parameters
    ----------
    samples:
        ``(N, 2)`` coordinates – can be a NumPy array, list, or torch tensor.
    coeffs:
        Tensor of polynomial coefficients with shape ``(1, D+1, D+1)`` where
        ``D = degree``.  The function expects the same layout used throughout the
        codebase.
    degree:
        Polynomial degree.
    scale:
        Scaling factor applied inside ``compute_poly_features``.
    device:
        Optional torch device on which to perform the computation.  If ``None``
        the device of ``coeffs`` is used.

    Returns
    -------
    float
        Success rate as a percentage (0‑100).
    """
    if device is None:
        device = coeffs.device

    if not isinstance(samples, torch.Tensor):
        samples_t = torch.tensor(samples, dtype=torch.float32, device=device)
    else:
        samples_t = samples.to(device)

    x_pow, y_pow = compute_poly_features(samples_t, degree=degree, scale=scale)
    batch_C = coeffs.unsqueeze(0).expand(samples_t.shape[0], -1, -1)
    p_vals = evaluate_poly(x_pow, y_pow, batch_C).squeeze().cpu().numpy()
    return (p_vals <= 0).mean() * 100.0


def run_evaluation_inference(model, x0, bounds=None, coeffs=None, step_size=0.05, batch_size=50000):
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

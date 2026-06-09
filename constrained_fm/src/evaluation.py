"""Utility functions for evaluating sample constraints.

This module provides reusable helpers that compute the success rate of
samples with respect to a bounding‑box constraint or a polynomial constraint.
They are used by :func:`generate_and_visualize_samples` in
``visualization.py``.
"""

import torch
from constrained_fm.src.utils.polynomials import compute_poly_features, evaluate_poly


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

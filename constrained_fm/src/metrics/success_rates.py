import torch


from constrained_fm.src.geometry.polynomials import compute_poly_features, evaluate_poly
from constrained_fm.src.metrics.distributional import compute_swd, compute_mmd, compute_jsd


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


def compute_success_rate_polynomial(samples: torch.Tensor | list, coeffs: torch.Tensor, degree: int, scale: float,
                                    device=None) -> float | list[float]:
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


def evaluate_compound_metrics(samples, disjoint_boxes, x_true_pool, device=None):
    """
    Evaluates Success Rate, SWD, MMD, and JSD for a compound bounding box constraint.
    """
    num_samples = samples.shape[0]

    sample_success_mask = torch.zeros(num_samples, dtype=torch.bool, device=device)
    true_pool_mask = torch.zeros(x_true_pool.shape[0], dtype=torch.bool, device=device)

    for i in range(disjoint_boxes.shape[0]):
        x_min, y_min, x_max, y_max = disjoint_boxes[i].tolist()

        s_in_x = (samples[:, 0] >= x_min) & (samples[:, 0] <= x_max)
        s_in_y = (samples[:, 1] >= y_min) & (samples[:, 1] <= y_max)
        sample_success_mask |= (s_in_x & s_in_y)

        t_in_x = (x_true_pool[:, 0] >= x_min) & (x_true_pool[:, 0] <= x_max)
        t_in_y = (x_true_pool[:, 1] >= y_min) & (x_true_pool[:, 1] <= y_max)
        true_pool_mask |= (t_in_x & t_in_y)

    success_rate = sample_success_mask.float().mean().item() * 100.0
    x_true_filtered = x_true_pool[true_pool_mask]

    # Compute Distributional Metrics
    metrics = {
        "success_rate": success_rate,
        "swd": float('inf'),
        "mmd": float('inf'),
        "jsd": float('inf')
    }

    if len(x_true_filtered) > 10 and len(samples) > 10:
        metrics["swd"] = compute_swd(samples, x_true_filtered)
        metrics["mmd"] = compute_mmd(samples, x_true_filtered)
        metrics["jsd"] = compute_jsd(samples, x_true_filtered)
    else:
        print("Warning: Not enough valid points in true pool or generated samples to compute SWD/MMD/JSD.")

    return metrics

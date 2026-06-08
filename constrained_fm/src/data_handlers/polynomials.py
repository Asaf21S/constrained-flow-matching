import torch

from constrained_fm.src.consts import POLYNOMIAL_DEGREE


def sample_valid_polynomials(batch_size, proxy_x_pow, proxy_y_pow, degree=POLYNOMIAL_DEGREE, min_area=0.05, max_area=0.95):
    """
    Sample polynomial coefficients that yield a valid region (area ratio between min_area and max_area).
    Args:
        batch_size: int
        degree: int
        proxy_x_pow: torch tensor [N, degree+1] (proxy points x powers)
        proxy_y_pow: torch tensor [N, degree+1] (proxy points y powers)
        min_area: float, minimum fraction of points that should satisfy P(x) <= 0
        max_area: float, maximum fraction of points that should satisfy P(x) <= 0
    Returns:
        C: torch tensor [batch_size, degree+1, degree+1] (normalized coefficients)
    """
    valid_C = torch.empty((batch_size, degree + 1, degree + 1), device=proxy_x_pow.device)
    needed = batch_size
    valid_idx = 0

    while needed > 0:
        # Over-sample slightly to minimize loop iterations
        C_cand = torch.randn(needed * 2, degree + 1, degree + 1, device=proxy_x_pow.device)

        # B = batch, N = proxy points, I/J = polynomial degrees
        P_vals = torch.einsum('ni, bij, nj -> bn', proxy_x_pow, C_cand, proxy_y_pow)

        area_ratios = (P_vals <= 0).float().mean(dim=1)
        safe_mask = (area_ratios >= min_area) & (area_ratios <= max_area)
        safe_C = C_cand[safe_mask]

        found = safe_C.shape[0]
        if found > 0:
            take = min(found, needed)
            valid_C[valid_idx : valid_idx + take] = safe_C[:take]
            valid_idx += take
            needed -= take

    # Normalize
    C_norm = torch.linalg.matrix_norm(valid_C, ord='fro', dim=(1,2), keepdim=True)
    return valid_C / (C_norm + 1e-8)

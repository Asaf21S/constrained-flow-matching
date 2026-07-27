import torch

from constrained_fm.src.consts import POLYNOMIAL_DEGREE, PLANE_SCALE


def compute_poly_features(x: torch.Tensor, degree: int = POLYNOMIAL_DEGREE,
                           scale: float = PLANE_SCALE) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes monomial feature vectors [1, u, u^2, ..., u^degree] for a single (unbatched) point set.

    Args:
        x: (N, 2) raw-scale coordinates.
        degree: highest monomial power to compute.
        scale: normalization factor mapping raw coordinates onto roughly [-1, 1]
            before evaluating the polynomial. Pass ``scale=1.0`` if ``x`` is already normalized.

    Returns:
        x_pow, y_pow: each (N, degree + 1).
    """
    x_scaled = x / scale

    x_pow = torch.stack([x_scaled[:, 0]**i for i in range(degree + 1)], dim=1)
    y_pow = torch.stack([x_scaled[:, 1]**i for i in range(degree + 1)], dim=1)

    return x_pow, y_pow


def evaluate_poly(x_pow: torch.Tensor, y_pow: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """
    Evaluates P(x, y) = X^T * C * Y
    x_pow, y_pow: [batch_size, d+1]
    C: [batch_size, d+1, d+1]
    """
    # (B, 1, d+1) @ (B, d+1, d+1) -> (B, 1, d+1)
    val = torch.bmm(x_pow.unsqueeze(1), C)

    # (B, 1, d+1) @ (B, d+1, 1) -> (B, 1, 1)
    val = torch.bmm(val, y_pow.unsqueeze(2)).squeeze(-1)

    return val


def compute_poly_features_batched(X: torch.Tensor, degree: int = POLYNOMIAL_DEGREE,
                                   scale: float = PLANE_SCALE) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched counterpart of :func:`compute_poly_features`.

    Args:
        X: (B, M, 2) raw-scale coordinates.
        degree: highest monomial power to compute. Defaults to the project-wide POLYNOMIAL_DEGREE.
        scale: normalization factor mapping raw coordinates onto roughly [-1, 1]
            before evaluating the polynomial. Pass ``scale=1.0`` if ``X`` is already normalized
            (e.g. sampled directly in the SIREN's canonical [-1, 1] input domain).

    Returns:
        x_pow, y_pow: each (B, M, degree + 1).
    """
    X_scaled = X / scale

    # Compute powers along the feature dimension (dim=-1)
    x_pow = torch.stack([X_scaled[:, :, 0]**i for i in range(degree + 1)], dim=-1)  # [B, M, d+1]
    y_pow = torch.stack([X_scaled[:, :, 1]**i for i in range(degree + 1)], dim=-1)  # [B, M, d+1]

    return x_pow, y_pow


def evaluate_poly_batched(x_pow: torch.Tensor, y_pow: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """
    x_pow, y_pow: [B, M, degree + 1]
    C: [B, degree + 1, degree + 1]
    Returns: [B, M] tensor containing P(x, y) values
    """
    # Contract polynomial degrees (i, j) with coefficient matrix C,
    # preserving Batch (b) and Points (m)
    return torch.einsum('bmi, bij, bmj -> bm', x_pow, C, y_pow)

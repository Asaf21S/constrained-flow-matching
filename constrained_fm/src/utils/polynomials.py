import torch

from constrained_fm.src.consts import POLYNOMIAL_DEGREE, PLANE_SCALE


def compute_poly_features(x, degree=POLYNOMIAL_DEGREE, scale=PLANE_SCALE):
    x_scaled = x / scale

    x_pow = torch.stack([x_scaled[:, 0]**i for i in range(degree + 1)], dim=1)
    y_pow = torch.stack([x_scaled[:, 1]**i for i in range(degree + 1)], dim=1)

    return x_pow, y_pow


def evaluate_poly(x_pow, y_pow, C):
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

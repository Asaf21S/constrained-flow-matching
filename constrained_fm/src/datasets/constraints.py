import torch
from tqdm import tqdm

from constrained_fm.src.consts import POLYNOMIAL_DEGREE, PLANE_SCALE
from constrained_fm.src.datasets.gmm_target import get_points
from constrained_fm.src.geometry.polynomials import compute_poly_features


def sample_valid_polynomials(batch_size, degree=POLYNOMIAL_DEGREE, scale=PLANE_SCALE, proxy_x_pow=None, proxy_y_pow=None,
                             min_area=0.05, max_area=0.95, device=None):
    """
    Sample polynomial coefficients that yield a valid region (area ratio between min_area and max_area).
    Args:
        batch_size: int
        degree: int
        scale: float, scaling factor for the proxy grid
        proxy_x_pow: precomputed x polynomial features of shape [N, degree+1]
        proxy_y_pow: precomputed y polynomial features of shape [N, degree+1]
        min_area: float, minimum fraction of points that should satisfy P(x) <= 0
        max_area: float, maximum fraction of points that should satisfy P(x) <= 0
        device: torch device
    Returns:
        C: torch tensor [batch_size, degree+1, degree+1] (normalized coefficients)
    """
    if proxy_x_pow is None or proxy_y_pow is None:
        proxy_x, _ = get_points(batch_size=10000, device=device)
        proxy_x = proxy_x.to(device)
        proxy_x_pow, proxy_y_pow = compute_poly_features(proxy_x, degree=degree, scale=scale)

    valid_C = torch.empty((batch_size, degree + 1, degree + 1), device=device)
    needed = batch_size
    valid_idx = 0

    while needed > 0:
        # Over-sample slightly to minimize loop iterations
        C_cand = torch.randn(needed * 2, degree + 1, degree + 1, device=device)

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


def sample_bbox_around_points(x_1: torch.Tensor, width_range, device=None):
    """Sample axis‑aligned bounding boxes around each point in ``x_1``."""
    if device is None:
        device = x_1.device
    N = x_1.shape[0]
    d = torch.empty((N, 2), device=device).uniform_(*width_range)
    o = torch.rand((N, 2), device=device) * d
    a = x_1 - o
    b = a + d
    return torch.cat([a, b], dim=-1)


def generate_mass_dataset_anchored(gmm_true_pool: torch.Tensor, num_boxes=50000, width_range=(0.1, 7.0), device=None):
    """Create a dataset of anchored boxes with their true probability mass.

    The function samples random points from ``gmm_true_pool`` to act as box anchors,
    constructs boxes around them and estimates the exact valid area (mass) of each box
    via Monte‑Carlo counting.
    """
    if device is None:
        device = gmm_true_pool.device

    indices = torch.randperm(gmm_true_pool.shape[0], device=device)[:num_boxes]
    x_1_anchors = gmm_true_pool[indices]
    boxes = sample_bbox_around_points(x_1_anchors, width_range=width_range, device=device)

    chunk_size = 5000
    masses = []
    print(f"Calculating true probability mass for {num_boxes} anchored boxes...")
    for i in tqdm(range(0, num_boxes, chunk_size)):
        box_chunk = boxes[i:i + chunk_size]
        gmm_exp = gmm_true_pool.unsqueeze(0)
        in_x = (gmm_exp[:, :, 0] >= box_chunk[:, 0:1]) & (gmm_exp[:, :, 0] <= box_chunk[:, 2:3])
        in_y = (gmm_exp[:, :, 1] >= box_chunk[:, 1:2]) & (gmm_exp[:, :, 1] <= box_chunk[:, 3:4])
        inside_mask = in_x & in_y
        chunk_mass = inside_mask.float().mean(dim=1)
        masses.append(chunk_mass)
    y_mass = torch.cat(masses, dim=0).unsqueeze(1)
    return boxes, y_mass

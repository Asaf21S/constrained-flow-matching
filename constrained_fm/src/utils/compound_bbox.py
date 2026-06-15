import torch
from tqdm import tqdm

from constrained_fm.src.data_handlers.gmm_2d import get_points


def sample_bbox_around_points(x_1: torch.Tensor, width_range, device=None):
    """Sample axis‑aligned bounding boxes around each point in ``x_1``.

    Parameters
    ----------
    x_1: torch.Tensor
        Tensor of shape (N, 2) containing point coordinates.
    width_range: tuple[float, float]
        Minimum and maximum width/height of the generated boxes.
    device: torch.device, optional
        Device on which to perform the computation. If ``None`` the device of ``x_1`` is used.

    Returns
    -------
    torch.Tensor
        Tensor of shape (N, 4) with ``[x_min, y_min, x_max, y_max]``.
    """
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


def generate_disjoint_bboxes(num_boxes: int, width_range=(0.5, 7.0), max_attempts=1000, device=None):
    """Greedy rejection sampling to obtain ``num_boxes`` non‑overlapping boxes.

    Parameters
    ----------
    num_boxes: int
        Desired number of disjoint boxes.
    width_range: tuple[float, float]
        Width/height range for each box.
    max_attempts: int
        Upper bound on how many candidate boxes to draw before giving up.
    device: torch.device, optional
        Device for tensor allocation.
    """
    if device is None:
        device = torch.device('cpu')
    target_points, _ = get_points(max_attempts, device=device)
    indices = torch.randperm(max_attempts, device=device)
    x_1_anchors = target_points[indices]
    boxes_pool = sample_bbox_around_points(x_1_anchors, width_range=width_range, device=device)

    accepted_boxes = []
    for candidate in boxes_pool:
        if len(accepted_boxes) == num_boxes:
            break
        c_xmin, c_ymin, c_xmax, c_ymax = candidate
        intersects = any(
            not (c_xmax <= b_xmin or c_xmin >= b_xmax or c_ymax <= b_ymin or c_ymin >= b_ymax)
            for b_xmin, b_ymin, b_xmax, b_ymax in accepted_boxes
        )
        if not intersects:
            accepted_boxes.append(candidate)
    if len(accepted_boxes) < num_boxes:
        print(f"Warning: Only found {len(accepted_boxes)} disjoint boxes after {max_attempts} attempts.")
    if not accepted_boxes:
        return torch.empty((0, 4), device=device)
    return torch.stack(accepted_boxes)


def generate_compound_constrained_points(num_points: int, boxes: torch.Tensor, predictor: torch.nn.Module,
                                         generator: torch.nn.Module, device=None):
    """
    Pipeline to generate points across multiple disjoint bounding boxes using
    amortized importance sampling (AreaPredictor).
    """
    K = boxes.shape[0]

    predictor.eval()
    with torch.no_grad():
        raw_masses = predictor(boxes).squeeze(-1)

    if raw_masses.sum() <= 1e-8:
        print("Warning: Predictor output 0 for all boxes. Falling back to uniform distribution.")
        probs = torch.ones_like(raw_masses) / K
    else:
        probs = raw_masses / raw_masses.sum()

    print("Predicted Normalized Probabilities:", probs.cpu().numpy())

    box_assignments = torch.multinomial(probs, num_samples=num_points, replacement=True)
    points_per_box = torch.bincount(box_assignments, minlength=K)


    all_generated_points = []

    print(f"Generating {num_points} total points across {K} boxes...")
    for i in range(K):
        n_i = points_per_box[i].item()
        if n_i == 0:
            continue

        box = boxes[i].tolist()

        points_i = generator.sample(num_points=n_i, bounds=box, device=device)

        all_generated_points.append(points_i)

    final_point_cloud = torch.cat(all_generated_points, dim=0)

    return final_point_cloud, points_per_box


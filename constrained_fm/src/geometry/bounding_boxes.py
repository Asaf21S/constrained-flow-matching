import torch

from constrained_fm.src.datasets.gmm_target import get_points
from constrained_fm.src.datasets.constraints import sample_bbox_around_points


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

import torch
import os

from constrained_fm.src.consts import POLYNOMIAL_DEGREE, PLANE_SCALE
from constrained_fm.src.data_handlers.gmm_2d import get_points
from constrained_fm.src.data_handlers.polynomials import sample_valid_polynomials
from constrained_fm.src.utils.polynomials import compute_poly_features


def generate_validation_set(num_bboxes=100, num_polys=100, degree=POLYNOMIAL_DEGREE, scale=PLANE_SCALE, device=None):
    val_set = {'bboxes': [], 'polynomials': []}

    print(f"Generating {num_bboxes} Random Bounding Boxes...")
    while len(val_set['bboxes']) < num_bboxes:
        xs = torch.sort(torch.rand(2) * 8.0 - 4.0)[0]
        ys = torch.sort(torch.rand(2) * 8.0 - 4.0)[0]

        if (1.0 <= xs[1] - xs[0] <= 6.5) and (1.0 <= ys[1] - ys[0] <= 6.5):
            val_set['bboxes'].append([xs[0].item(), ys[0].item(), xs[1].item(), ys[1].item()])

    print(f"Generating {num_polys} Valid Polynomials via Proxy-Grid...")
    proxy_x, _ = get_points(batch_size=10000, device=device)
    proxy_x = proxy_x.to(device)
    proxy_x_pow, proxy_y_pow = compute_poly_features(proxy_x, degree=degree, scale=scale)
    C_batch = sample_valid_polynomials(num_polys, proxy_x_pow, proxy_y_pow, degree=degree, min_area=0.1, max_area=0.9)
    val_set['polynomials'] = [C_batch[i] for i in range(num_polys)]

    return val_set


def get_validation_set(val_set_path='validation_set.pt', device=None):
    if os.path.exists(val_set_path):
        print(f"Found existing validation set at '{val_set_path}'. Loading...")
        val_set = torch.load(val_set_path, map_location=device)
        print(f"Loaded {len(val_set['bboxes'])} bboxes and {len(val_set['polynomials'])} polynomials.")
    else:
        print("Validation set not found. Generating a new static set...")
        val_set = generate_validation_set()
        torch.save(val_set, val_set_path)
        print(f"Saved generated validation set to '{val_set_path}'.")
    return val_set

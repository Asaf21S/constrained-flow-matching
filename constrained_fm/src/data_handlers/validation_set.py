import torch
import os

from constrained_fm.src.consts import POLYNOMIAL_DEGREE, PLANE_SCALE, VALIDATION_SET_PATH
from constrained_fm.src.data_handlers.polynomials import sample_valid_polynomials


def generate_validation_set(num_bboxes=100, num_polys=100, degree=POLYNOMIAL_DEGREE, scale=PLANE_SCALE, device=None):
    val_set = {'bboxes': [], 'polynomials': []}

    print(f"Generating {num_bboxes} Random Bounding Boxes...")
    while len(val_set['bboxes']) < num_bboxes:
        xs = torch.sort(torch.rand(2) * 8.0 - 4.0)[0]
        ys = torch.sort(torch.rand(2) * 8.0 - 4.0)[0]

        if (1.0 <= xs[1] - xs[0] <= 6.5) and (1.0 <= ys[1] - ys[0] <= 6.5):
            val_set['bboxes'].append([xs[0].item(), ys[0].item(), xs[1].item(), ys[1].item()])

    print(f"Generating {num_polys} Valid Polynomials via Proxy-Grid...")
    C_batch = sample_valid_polynomials(num_polys, degree=degree, scale=scale, min_area=0.1, max_area=0.9, device=device)
    val_set['polynomials'] = [C_batch[i] for i in range(num_polys)]

    return val_set


def get_validation_set(val_set_path=VALIDATION_SET_PATH, device=None):
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

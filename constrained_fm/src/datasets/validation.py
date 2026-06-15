import torch
import os

from constrained_fm.src.consts import POLYNOMIAL_DEGREE, PLANE_SCALE, VALIDATION_SET_PATH
from constrained_fm.src.datasets.constraints import sample_valid_polynomials
from constrained_fm.src.datasets.gmm_target import get_points


def generate_validation_set(
    num_bboxes=100,
    num_polys=100,
    n_train_samples_x0=10000,
    n_train_samples_x1=100000,
    degree=POLYNOMIAL_DEGREE,
    scale=PLANE_SCALE,
    device=None,
):
    """Generate a validation set containing constraints and a large sample batch.

    Returns a dict with:
    * ``bboxes`` – list of bounding‑box constraints.
    * ``polynomials`` – list of polynomial coefficient tensors.
    * ``x0`` – random Gaussian start points of shape ``(n_train_samples, 2)``.
    * ``x1`` – points drawn from the GMM via ``get_points`` of the same size.
    """
    val_set = {"bboxes": []}

    # ------------------------------------------------------------
    # 1. Bounding‑box constraints
    # ------------------------------------------------------------
    print(f"Generating {num_bboxes} Random Bounding Boxes...")
    while len(val_set["bboxes"]) < num_bboxes:
        xs = torch.sort(torch.rand(2) * 8.0 - 4.0)[0]
        ys = torch.sort(torch.rand(2) * 8.0 - 4.0)[0]

        if (1.0 <= xs[1] - xs[0] <= 6.5) and (1.0 <= ys[1] - ys[0] <= 6.5):
            val_set["bboxes"].append(
                [xs[0].item(), ys[0].item(), xs[1].item(), ys[1].item()]
            )
    val_set["bboxes"] = torch.tensor(val_set["bboxes"])

    # ------------------------------------------------------------
    # 2. Polynomial constraints via proxy‑grid
    # ------------------------------------------------------------
    print(f"Generating {num_polys} Valid Polynomials via Proxy-Grid...")
    C_batch = sample_valid_polynomials(
        num_polys, degree=degree, scale=scale, min_area=0.1, max_area=0.9, device=device
    )
    val_set["polynomials"] = C_batch

    # ------------------------------------------------------------
    # 3. Large sample batches for evaluation (stored as tensors)
    # ------------------------------------------------------------
    print(f"Generating {n_train_samples_x0}, {n_train_samples_x1} training samples (x0, x1)...")
    # x0 – standard normal samples.
    x0_tensor = torch.randn(n_train_samples_x0, 2, device=device)
    # x1 – samples from the GMM.
    x1_tensor, _ = get_points(batch_size=n_train_samples_x1, device=device)
    # Store tensors directly (retain device information). Users can move to CPU as needed.
    val_set["x0"] = x0_tensor
    val_set["x1"] = x1_tensor

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

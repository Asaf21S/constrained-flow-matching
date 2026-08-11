"""Separates three candidate causes of poor Functa boundary fidelity.

Runs on the frozen SIREN with no re-meta-training, comparing for each shape:

    CAVIA-15                 the deployed extraction budget
    SGD-{30..250}            same optimizer and lr, longer trajectory
    Adam-1000 (1k points)    optimizer no longer the constraint
    Adam-1000 (20k points)   query-sample size no longer the constraint

Boundary agreement is always scored on held-out GMM samples, so gains from merely
fitting the query points harder do not register.

Reading the output:
    CAVIA-15 << SGD-250              -> extraction budget is the limit
    SGD-250 << Adam-1000(1k)         -> plain SGD from zero-init is the limit
    Adam-1000(1k) << Adam-1000(20k)  -> 1000 query points is the limit
    Adam-1000(20k) still low         -> the SIREN cannot represent the shape at all

Usage:
    python -m constrained_fm.scripts.probe_extraction_budget
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from constrained_fm.src.consts import POLYNOMIAL_DEGREE, PLANE_SCALE
from constrained_fm.src.datasets.functa_conditioning import sample_query_points
from constrained_fm.src.datasets.gmm_target import get_points
from constrained_fm.src.datasets.validation import get_validation_set
from constrained_fm.src.geometry.polynomials import (compute_poly_features,
                                                     compute_poly_features_batched,
                                                     evaluate_poly,
                                                     evaluate_poly_batched)
from constrained_fm.src.models.functa_siren import build_modulated_siren

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim, hidden_dim, n_layers, w0 = 512, 512, 4, 30.0
inner_lr = 1e-2
sgd_checkpoints = (15, 30, 60, 120, 250)
adam_steps = 1000
adam_lr = 1e-2
small_points, large_points = 1000, 20000
iou_samples = 20000

# The validation set is static, so these indices are the shapes seen failing before.
shape_ids = [3, 90, 88, 24, 0, 1]


def load_siren() -> torch.nn.Module:
    from constrained_fm.src.consts import VALIDATION_SET_PATH
    from pathlib import Path

    ckpt = Path(VALIDATION_SET_PATH).parents[1] / "functa_dataset" / "siren_best.pt"
    siren = build_modulated_siren(latent_dim=latent_dim, hidden_dim=hidden_dim,
                                  n_layers=n_layers, w0=w0).to(device)
    siren.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    siren.eval()
    for p in siren.parameters():
        p.requires_grad_(False)
    print(f"loaded {ckpt}")
    return siren


def poly_targets(C: torch.Tensor, X_raw: torch.Tensor) -> torch.Tensor:
    """tanh(P(x)) for one shape at raw-scale query points."""
    x_pow, y_pow = compute_poly_features_batched(X_raw.unsqueeze(0), degree=POLYNOMIAL_DEGREE,
                                                 scale=PLANE_SCALE)
    return torch.tanh(evaluate_poly_batched(x_pow, y_pow, C.unsqueeze(0)))


def mass_iou(siren, z: torch.Tensor, C: torch.Tensor, points: torch.Tensor) -> float:
    """IoU of {SIREN<=0} against {P<=0}, weighted by GMM density via the point sample."""
    n = points.shape[0]
    x_pow, y_pow = compute_poly_features(points, degree=POLYNOMIAL_DEGREE, scale=PLANE_SCALE)
    true_in = evaluate_poly(x_pow, y_pow, C.unsqueeze(0).expand(n, -1, -1)).squeeze() <= 0
    with torch.no_grad():
        pred = siren((points / PLANE_SCALE).unsqueeze(0), z.view(1, -1)).squeeze()
    pred_in = pred <= 0
    inter = (true_in & pred_in).sum().float()
    union = (true_in | pred_in).sum().float()
    return float(inter / union.clamp(min=1.0))


def run_sgd(siren, X_scaled, Y, checkpoints):
    """Replicates extract_latents_batched, snapshotting z at each checkpoint."""
    z = torch.zeros(1, latent_dim, device=device, requires_grad=True)
    snapshots, losses = {}, {}
    for step in range(1, max(checkpoints) + 1):
        pred = siren(X_scaled, z).squeeze(-1)
        loss = F.mse_loss(pred, Y)
        z = z - inner_lr * torch.autograd.grad(loss, z)[0]
        if step in checkpoints:
            snapshots[step] = z.detach().clone()
            losses[step] = float(loss.detach())
    return snapshots, losses


def run_adam(siren, X_scaled, Y, steps):
    z = torch.zeros(1, latent_dim, device=device, requires_grad=True)
    opt = torch.optim.Adam([z], lr=adam_lr)
    final_loss = float("nan")
    for _ in range(steps):
        opt.zero_grad()
        loss = F.mse_loss(siren(X_scaled, z).squeeze(-1), Y)
        loss.backward()
        opt.step()
        final_loss = float(loss.detach())
    return z.detach(), final_loss


def main() -> None:
    siren = load_siren()

    val_set = get_validation_set(device=device)
    polys = val_set["polynomials"].to(device)

    gmm_pool, _ = get_points(100000, device=device)
    iou_points = gmm_pool[torch.randperm(gmm_pool.shape[0], device=device)[:iou_samples]]

    pool_x_pow, pool_y_pow = compute_poly_features(gmm_pool, degree=POLYNOMIAL_DEGREE,
                                                   scale=PLANE_SCALE)

    header = (f"{'shape':>5} {'mass':>6} | "
              + " ".join(f"{'sgd' + str(s):>8}" for s in sgd_checkpoints)
              + f" | {'adam1k':>8} {'adam20k':>8}")
    print("\nmass IoU on held-out GMM samples")
    print(header)
    print("-" * len(header))

    summary = {k: [] for k in ("cavia", "sgd_long", "adam_small", "adam_large")}

    for sid in shape_ids:
        C = polys[sid]
        n_pool = gmm_pool.shape[0]
        mass = float((evaluate_poly(pool_x_pow, pool_y_pow,
                                    C.unsqueeze(0).expand(n_pool, -1, -1)).squeeze() <= 0)
                     .float().mean())

        X_small = sample_query_points(1, small_points, scale=PLANE_SCALE, device=device)[0]
        Y_small = poly_targets(C, X_small).squeeze(0).unsqueeze(0)
        X_small_scaled = (X_small / PLANE_SCALE).unsqueeze(0)

        snapshots, _ = run_sgd(siren, X_small_scaled, Y_small, sgd_checkpoints)
        sgd_ious = {s: mass_iou(siren, z, C, iou_points) for s, z in snapshots.items()}

        z_adam_small, _ = run_adam(siren, X_small_scaled, Y_small, adam_steps)
        iou_adam_small = mass_iou(siren, z_adam_small, C, iou_points)

        X_large = sample_query_points(1, large_points, scale=PLANE_SCALE, device=device)[0]
        Y_large = poly_targets(C, X_large).squeeze(0).unsqueeze(0)
        z_adam_large, _ = run_adam(siren, (X_large / PLANE_SCALE).unsqueeze(0), Y_large, adam_steps)
        iou_adam_large = mass_iou(siren, z_adam_large, C, iou_points)

        print(f"{sid:>5} {mass:>6.3f} | "
              + " ".join(f"{sgd_ious[s]:>8.3f}" for s in sgd_checkpoints)
              + f" | {iou_adam_small:>8.3f} {iou_adam_large:>8.3f}")

        summary["cavia"].append(sgd_ious[min(sgd_checkpoints)])
        summary["sgd_long"].append(sgd_ious[max(sgd_checkpoints)])
        summary["adam_small"].append(iou_adam_small)
        summary["adam_large"].append(iou_adam_large)

    print("-" * len(header))
    means = {k: sum(v) / len(v) for k, v in summary.items()}
    print(f"mean  CAVIA-{min(sgd_checkpoints)} {means['cavia']:.3f} | "
          f"SGD-{max(sgd_checkpoints)} {means['sgd_long']:.3f} | "
          f"Adam@{small_points}pts {means['adam_small']:.3f} | "
          f"Adam@{large_points}pts {means['adam_large']:.3f}")

    print("\nverdict")
    if means["sgd_long"] - means["cavia"] > 0.03:
        print(f"  extraction budget matters: +{means['sgd_long'] - means['cavia']:.3f} "
              f"from more SGD steps alone")
    else:
        print(f"  more SGD steps changed little ({means['sgd_long'] - means['cavia']:+.3f}); "
              "the 15-step budget is NOT the binding constraint")

    if means["adam_small"] - means["sgd_long"] > 0.03:
        print(f"  optimizer matters: Adam adds {means['adam_small'] - means['sgd_long']:+.3f} "
              "over long SGD")
    if means["adam_large"] - means["adam_small"] > 0.03:
        print(f"  query-sample size matters: 20k points add "
              f"{means['adam_large'] - means['adam_small']:+.3f}")
    if means["adam_large"] < 0.9:
        print(f"  ceiling is low ({means['adam_large']:.3f}): the SIREN cannot represent these "
              "regions; capacity, not optimization, is the limit")


if __name__ == "__main__":
    main()


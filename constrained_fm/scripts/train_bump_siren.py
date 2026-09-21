# -*- coding: utf-8 -*-
"""Meta-trains a ModulatedSIREN to encode convex polygons as Functa latents (CAVIA).

The existing checkpoint only ever saw smooth cubic level sets, so it is retrained from
scratch here: a polygon boundary is piecewise linear with corners, and the latent has to
place several straight edges exactly rather than bend one smooth curve.

Validation reports the mass-IoU of the decoded region against the true polygon, which is the
quantity the downstream sampler actually depends on. The extraction MSE is only a proxy for
it -- a latent can carry a low MSE while misplacing a boundary through a region the target
never visits, and vice versa.

    python -m constrained_fm.scripts.train_bump_siren
    python -m constrained_fm.scripts.train_bump_siren --epochs 200 --steps-per-epoch 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from constrained_fm.src.consts import (BUMP_QUERY_TARGET_FRACTION, BUMP_SIREN_CHECKPOINT,
                                       BUMP_SIREN_TAU)
from constrained_fm.src.datasets.bump_conditioning import (mass_iou, polygon_batch,
                                                           regression_targets,
                                                           sample_query_points)
from constrained_fm.src.experiment.runtime import resolve_device, set_seed
from constrained_fm.src.inference.latent_extractor import extract_latents_batched
from constrained_fm.src.models.functa_siren import build_modulated_siren
from constrained_fm.src.problems.bump2d import BumpProblem

GATE_PERCENTILE = 5.0
GATE_IOU = 0.9
# The per-shape step extraction defaults to; meta-training must adapt at the same scale.
DEFAULT_INNER_LR = 6.25e-4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Meta-train the polygon SIREN encoder.")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--steps-per-epoch", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--points-per-shape", type=int, default=1000)

    parser.add_argument("--outer-lr", type=float, default=1e-4)
    parser.add_argument("--inner-lr", type=float, default=DEFAULT_INNER_LR)
    parser.add_argument("--inner-steps", type=int, default=15)
    parser.add_argument("--lambda-z", type=float, default=1e-4)

    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--w0", type=float, default=30.0)
    parser.add_argument("--tau", type=float, default=BUMP_SIREN_TAU)
    parser.add_argument("--target-fraction", type=float, default=BUMP_QUERY_TARGET_FRACTION)

    # Only decides which shapes are kept, so it needs far less resolution than a reported
    # mass; the filter runs on every training draw and dominates the step otherwise.
    parser.add_argument("--mass-pool-size", type=int, default=20000)
    parser.add_argument("--val-shapes", type=int, default=200)
    parser.add_argument("--iou-points", type=int, default=20000)
    parser.add_argument("--validate-every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default=BUMP_SIREN_CHECKPOINT)
    return parser


def adapt(siren: nn.Module, x: torch.Tensor, y: torch.Tensor, latent_dim: int,
          inner_lr: float, inner_steps: int, create_graph: bool) -> torch.Tensor:
    """CAVIA inner loop: SGD on a zero-initialised context vector, SIREN weights frozen.

    Averaged over points but *summed* over shapes, matching :func:`extract_latents_batched`.
    A mean over both would divide each z_i's gradient by the batch size, so the effective
    step would silently depend on how many shapes share the call and meta-training would not
    transfer to extraction at a different batch size.

    ``create_graph`` is what separates meta-training from extraction. With it the outer loss
    differentiates through the whole adaptation, so the weights are optimised to be a good
    *starting point* for exactly this budget; without it the same loop is plain inference.
    """
    z = torch.zeros(x.shape[0], latent_dim, device=x.device, requires_grad=True)
    for _ in range(inner_steps):
        loss = ((siren(x, z).squeeze(-1) - y) ** 2).mean(dim=-1).sum()
        (grad_z,) = torch.autograd.grad(loss, z, create_graph=create_graph)
        z = z - inner_lr * grad_z
    return z


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device()
    set_seed(args.seed)

    problem = BumpProblem()
    target = problem.target()
    mass_pool = target.sample(args.mass_pool_size, device=device)
    print(f"device {device} | tau {args.tau} | target fraction {args.target_fraction}")

    def draw(count: int) -> tuple[dict, torch.Tensor, torch.Tensor]:
        shapes = polygon_batch(target, mass_pool, count, domain=problem.domain,
                               min_mass=problem.min_mass, max_mass=problem.max_mass,
                               device=device)
        points = sample_query_points(target, count, args.points_per_shape, problem.domain,
                                     args.target_fraction, device)
        x, y = regression_targets(shapes, points, args.tau, problem.domain)
        return shapes, x, y

    val_shapes, val_x, val_y = draw(args.val_shapes)
    iou_points = target.sample(args.iou_points, device=device)
    print(f"holdout: {args.val_shapes} polygons | mass-IoU on {args.iou_points} target points")

    siren = build_modulated_siren(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim,
                                  n_layers=args.n_layers, w0=args.w0).to(device)
    optimizer = torch.optim.Adam(siren.parameters(), lr=args.outer_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history: list[float] = []
    validations: list[dict] = []
    best_p5 = -1.0
    stale = 0

    for epoch in tqdm(range(1, args.epochs + 1), desc="Meta-training polygon SIREN"):
        siren.train()
        epoch_loss = 0.0
        for _ in range(args.steps_per_epoch):
            _, x, y = draw(args.batch_size)
            z = adapt(siren, x, y, args.latent_dim, args.inner_lr, args.inner_steps, True)

            optimizer.zero_grad(set_to_none=True)
            loss = ((siren(x, z).squeeze(-1) - y) ** 2).mean() + args.lambda_z * (z ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(siren.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        history.append(epoch_loss / args.steps_per_epoch)
        scheduler.step(history[-1])

        if epoch % args.validate_every and epoch != 1:
            continue

        siren.eval()
        z_val, per_shape = extract_latents_batched(siren, val_x, val_y, lr=args.inner_lr,
                                                   steps=args.inner_steps)
        # extract_latents_batched freezes the SIREN for inference; meta-training needs it back.
        for p in siren.parameters():
            p.requires_grad_(True)
        val_mse = float(per_shape.mean())
        iou = mass_iou(siren, z_val, val_shapes, iou_points, problem.domain)
        p5 = float(np.percentile(iou.numpy(), GATE_PERCENTILE))

        validations.append({"epoch": epoch, "val_mse": val_mse, "iou_p5": p5,
                            "iou_median": float(iou.median()), "iou_mean": float(iou.mean())})
        print(f"\nepoch {epoch:4d} | train {history[-1]:.6f} | val mse {val_mse:.6f} "
              f"| mass-IoU p5 {p5:.4f} median {iou.median():.4f} min {iou.min():.4f} "
              f"| lr {optimizer.param_groups[0]['lr']:.2e}", flush=True)

        if p5 > best_p5 + args.min_delta:
            best_p5, stale = p5, 0
            torch.save(siren.state_dict(), checkpoint)
        else:
            stale += args.validate_every
            if stale >= args.patience:
                print(f"early stop at epoch {epoch}: mass-IoU p5 flat for {stale} epochs")
                break

    meta = {"best_iou_p5": best_p5, "gate": GATE_IOU, "tau": args.tau,
            "target_fraction": args.target_fraction, "inner_steps": args.inner_steps,
            "inner_lr": args.inner_lr, "latent_dim": args.latent_dim,
            "hidden_dim": args.hidden_dim, "n_layers": args.n_layers, "w0": args.w0,
            "seed": args.seed, "validations": validations}
    checkpoint.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    np.save(checkpoint.with_name(checkpoint.stem + "_loss.npy"), np.array(history))

    verdict = "PASSED" if best_p5 >= GATE_IOU else "FAILED"
    print(f"\n{verdict}: best mass-IoU p{GATE_PERCENTILE:.0f} = {best_p5:.4f} "
          f"(gate {GATE_IOU}) -> {checkpoint}")
    return 0 if best_p5 >= GATE_IOU else 1


if __name__ == "__main__":
    raise SystemExit(main())

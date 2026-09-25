# -*- coding: utf-8 -*-
"""Meta-trains one ModulatedSIREN (CAVIA) on a joint pool of polynomial and polygon constraints.

Every batch holds degree-3 polynomials ``tanh(P)`` and bump2d half-plane polygons
``tanh(C / tau)`` on the GMM plane in equal shares, each replaced by its exact complement with
probability 1/2. ``tau`` defaults to ``1 / median ||grad P||`` on polynomial zero sets, so both
families cross their boundary with the same slope. The kept checkpoint maximises the worse
family's mean mass-IoU on a fixed holdout.

    sbatch scripts/run_joint_siren.sh
    sbatch scripts/run_joint_siren.sh --smoke --outdir constrained_fm/functa_dataset/joint_siren_smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from constrained_fm.scripts.train_bump_siren import DEFAULT_INNER_LR, adapt
from constrained_fm.src.consts import (FUNCTA_QUERY_GMM_FRACTION, PLANE_SCALE,
                                       POLY_MAX_AREA_RATIO, POLY_MIN_AREA_RATIO,
                                       POLYNOMIAL_DEGREE)
from constrained_fm.src.datasets import joint_conditioning as jc
from constrained_fm.src.datasets.gmm_target import get_points
from constrained_fm.src.experiment.config import REPO_ROOT
from constrained_fm.src.experiment.registry import pin_baseline_run
from constrained_fm.src.experiment.runtime import resolve_device, set_seed
from constrained_fm.src.inference.latent_extractor import extract_latents_batched
from constrained_fm.src.models.functa_siren import build_modulated_siren

OUTDIR = "constrained_fm/functa_dataset/joint_siren"
SMOKE = {"epochs": 2, "steps_per_epoch": 5, "val_shapes": 16, "iou_points": 2000,
         "validate_every": 1}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--steps-per-epoch", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--points-per-shape", type=int, default=1000)
    parser.add_argument("--polygon-fraction", type=float, default=0.5)

    parser.add_argument("--outer-lr", type=float, default=1e-4)
    parser.add_argument("--inner-lr", type=float, default=DEFAULT_INNER_LR)
    parser.add_argument("--inner-steps", type=int, default=15)
    parser.add_argument("--lambda-z", type=float, default=1e-4)

    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--w0", type=float, default=30.0)
    parser.add_argument("--tau", type=float, default=None,
                        help="polygon target tanh(C / tau); default matches the polynomial boundary slope")
    parser.add_argument("--query-gmm-fraction", type=float, default=FUNCTA_QUERY_GMM_FRACTION)
    parser.add_argument("--min-mass", type=float, default=POLY_MIN_AREA_RATIO)
    parser.add_argument("--max-mass", type=float, default=POLY_MAX_AREA_RATIO)
    parser.add_argument("--proxy-points", type=int, default=10000,
                        help="GMM draws backing the mass filter of both families")

    parser.add_argument("--val-shapes", type=int, default=200)
    parser.add_argument("--iou-points", type=int, default=20000)
    parser.add_argument("--validate-every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=250, help="epochs without a better score")
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", default=OUTDIR)
    parser.add_argument("--smoke", action="store_true", help="a few steps end to end")
    return parser


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def validate(siren: nn.Module, shapes: jc.Shapes, x: torch.Tensor, y: torch.Tensor,
             iou_points: torch.Tensor, tau: float, args) -> dict[str, float]:
    """Holdout extraction MSE and mass-IoU per family and orientation, at the training budget."""
    z, mse = extract_latents_batched(siren, x, y, lr=args.inner_lr, steps=args.inner_steps)
    # extract_latents_batched freezes the SIREN for inference; meta-training needs it back.
    for p in siren.parameters():
        p.requires_grad_(True)
    iou = jc.mass_iou(siren, z, shapes, iou_points, tau, POLYNOMIAL_DEGREE, PLANE_SCALE)
    record = {f"mse_{k}": v for k, v in jc.summarize_by_group(mse, shapes).items()}
    record.update({f"iou_{k}": v for k, v in jc.summarize_by_group(iou, shapes).items()})
    record["score"] = min(record[f"iou_{name}_mean"] for name in jc.FAMILY_NAMES)
    return record


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        for key, value in SMOKE.items():
            setattr(args, key, value)
    device = resolve_device()
    set_seed(args.seed)

    slope = jc.polynomial_boundary_slope(POLYNOMIAL_DEGREE, PLANE_SCALE, args.min_mass,
                                         args.max_mass, device=device)
    tau = args.tau if args.tau is not None else 1.0 / slope
    outdir = resolve_path(args.outdir)
    run_id = pin_baseline_run(outdir, "joint_siren", args,
                              extra={"tau": tau, "boundary_slope": slope})
    print(f"run {run_id} | device {device} | boundary slope {slope:.4f} | tau {tau:.4f} | "
          f"polygon fraction {args.polygon_fraction}")

    proxy = jc.proxy_set(args.proxy_points, POLYNOMIAL_DEGREE, PLANE_SCALE, device)

    def draw(count: int) -> tuple[jc.Shapes, torch.Tensor, torch.Tensor]:
        return jc.draw_joint_batch(count, proxy, args.points_per_shape, tau,
                                   args.polygon_fraction, True, args.query_gmm_fraction,
                                   POLYNOMIAL_DEGREE, PLANE_SCALE, args.min_mass,
                                   args.max_mass, device)

    val_shapes, val_x, val_y = draw(args.val_shapes)
    iou_points, _ = get_points(args.iou_points, device=device)

    siren = build_modulated_siren(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim,
                                  n_layers=args.n_layers, w0=args.w0).to(device)
    optimizer = torch.optim.Adam(siren.parameters(), lr=args.outer_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=25)

    meta = {"run_id": run_id, "method": "joint_siren", "tau": tau, "boundary_slope": slope,
            "polygon_fraction": args.polygon_fraction, "degree": POLYNOMIAL_DEGREE,
            "scale": PLANE_SCALE, "min_mass": args.min_mass, "max_mass": args.max_mass,
            "latent_dim": args.latent_dim, "hidden_dim": args.hidden_dim,
            "n_layers": args.n_layers, "w0": args.w0, "inner_lr": args.inner_lr,
            "inner_steps": args.inner_steps, "points_per_shape": args.points_per_shape,
            "query_gmm_fraction": args.query_gmm_fraction, "seed": args.seed,
            "best_epoch": None, "best_score": -1.0, "finished": False, "validations": []}
    history: list[float] = []
    stale = 0

    def write_meta() -> None:
        (outdir / "metrics.json").write_text(json.dumps(meta, indent=2))
        np.save(outdir / "losses.npy", np.asarray(history, dtype=np.float32))

    epoch = 0
    for epoch in tqdm(range(1, args.epochs + 1), desc="Meta-training joint SIREN"):
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
        record = {"epoch": epoch, "train_loss": history[-1],
                  "lr": optimizer.param_groups[0]["lr"],
                  **validate(siren, val_shapes, val_x, val_y, iou_points, tau, args)}
        meta["validations"].append(record)
        print(f"\nepoch {epoch:4d} | train {history[-1]:.6f} | "
              + " | ".join(f"{name} mse {record[f'mse_{name}_mean']:.2e} "
                           f"IoU {record[f'iou_{name}_mean']:.4f} "
                           f"(in {record.get(f'iou_{name}_inside_mean', float('nan')):.4f} / "
                           f"out {record.get(f'iou_{name}_complement_mean', float('nan')):.4f})"
                           for name in jc.FAMILY_NAMES)
              + f" | lr {record['lr']:.2e}", flush=True)

        if record["score"] > meta["best_score"] + args.min_delta:
            meta["best_score"], meta["best_epoch"], stale = record["score"], epoch, 0
            torch.save(siren.state_dict(), outdir / "siren_best.pt")
        else:
            stale += args.validate_every
        write_meta()
        if stale >= args.patience:
            print(f"early stop at epoch {epoch}: score flat for {stale} epochs")
            break

    torch.save(siren.state_dict(), outdir / "siren_final.pt")
    meta["finished"] = True
    meta["final_epoch"] = epoch
    write_meta()
    print(f"best min-family mass-IoU {meta['best_score']:.4f} at epoch {meta['best_epoch']} "
          f"-> {outdir / 'siren_best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

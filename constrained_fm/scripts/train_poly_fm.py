# -*- coding: utf-8 -*-
"""Trains and evaluates the coefficient-conditioned baseline, the ablation the Functa model
is measured against.

This model receives the raw polynomial coefficients *and* the exact P(x_t) at every step, so
it is the upper reference: whatever it achieves is what a latent code has to match without
ever seeing the constraint.

No checkpoint for it survived -- it was originally trained inline in
notebooks/constrained_fm_2d_gmm.ipynb (cell 48) and never saved -- so this script rebuilds
it. Two deliberate departures from that notebook:

  * scale_factor defaults to PLANE_SCALE (4.5), not the class default of 4.0. PLANE_SCALE was
    4.0 when the notebook ran and later moved to 4.5, so the published numbers were measured
    on a scale-4.0 benchmark while the Functa model is measured on the current scale-4.5
    validation set. Training at 4.5 puts both on the same benchmark for the first time.
  * 15001 iterations rather than 5001, matching the Functa run's budget.

    python -m constrained_fm.scripts.train_poly_fm
    python -m constrained_fm.scripts.train_poly_fm --skip-train   # re-evaluate the checkpoint
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from tqdm import tqdm

from constrained_fm.src.consts import PLANE_SCALE, POLYNOMIAL_DEGREE
from constrained_fm.src.datasets.constraints import sample_valid_polynomials
from constrained_fm.src.datasets.gmm_target import get_points
from constrained_fm.src.datasets.validation import get_validation_set
from constrained_fm.src.experiment.registry import readme_table, summarize
from constrained_fm.src.experiment.runtime import resolve_device, set_seed
from constrained_fm.src.geometry.polynomials import compute_poly_features, evaluate_poly
from constrained_fm.src.inference.evaluator import (evaluate_validation_set_metrics,
                                                    run_evaluation_inference)
from constrained_fm.src.metrics.eval_points import load_nll_eval_set
from constrained_fm.src.metrics.functa_fidelity import constraint_masses
from constrained_fm.src.models.constrained_poly import PolynomialConstrainedFM

DEFAULT_OUTDIR = "constrained_fm/baselines/poly_fm"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train + evaluate the coefficient-conditioned baseline.")
    parser.add_argument("--iterations", type=int, default=15001)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--scale", type=float, default=PLANE_SCALE,
                        help="must match the scale the validation benchmark was generated at")
    parser.add_argument("--degree", type=int, default=POLYNOMIAL_DEGREE)
    parser.add_argument("--min-area", type=float, default=0.05)
    parser.add_argument("--max-area", type=float, default=0.95)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-train", action="store_true", help="evaluate an existing checkpoint")
    parser.add_argument("--num-polys", type=int, default=100)
    parser.add_argument("--num-x0", type=int, default=10000)
    parser.add_argument("--gmm-pool-size", type=int, default=100000)
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--nll-points", type=int, default=5000)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    return parser


def train(args, device: torch.device) -> tuple[PolynomialConstrainedFM, list[float]]:
    """Reproduces the notebook's loop: CondOT path, flip-trick orientation, plain MSE."""
    model = PolynomialConstrainedFM(degree=args.degree, hidden_dim=args.hidden_dim,
                                    scale_factor=args.scale).to(device)
    prob_path = AffineProbPath(scheduler=CondOTScheduler())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.iterations, eta_min=args.lr_min)

    proxy_x, _ = get_points(10000, device=device)
    proxy_x_pow, proxy_y_pow = compute_poly_features(proxy_x, degree=args.degree, scale=args.scale)

    losses = []
    for iteration in tqdm(range(args.iterations), desc="Training coefficient-conditioned FM"):
        optimizer.zero_grad(set_to_none=True)

        x_1, _ = get_points(args.batch_size, device=device)
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.shape[0], device=device)

        path_sample = prob_path.sample(t=t, x_0=x_0, x_1=x_1)
        x_1_aligned = path_sample.x_1

        C = sample_valid_polynomials(args.batch_size, degree=args.degree, scale=args.scale,
                                     proxy_x_pow=proxy_x_pow, proxy_y_pow=proxy_y_pow,
                                     min_area=args.min_area, max_area=args.max_area, device=device)

        # Flip trick: orient each polynomial so its paired target satisfies P(x_1) <= 0.
        x1_pow, y1_pow = compute_poly_features(x_1_aligned, degree=args.degree, scale=args.scale)
        P_x1 = evaluate_poly(x1_pow, y1_pow, C)
        C = C * (1.0 - 2.0 * (P_x1 > 0).float().unsqueeze(-1))

        pred_v = model(path_sample.x_t, path_sample.t, coeffs=C.view(args.batch_size, -1))
        loss = torch.pow(pred_v - path_sample.dx_t, 2).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if (iteration + 1) % args.log_every == 0:
            window = np.mean(losses[-args.log_every:])
            print(f"| iter {iteration + 1:6d} | loss {loss.item():.5f} | mean {window:.5f} "
                  f"| lr {optimizer.param_groups[0]['lr']:.2e}", flush=True)

    return model, losses


def save_plot_artifacts(args, out: Path, run_id: str, val_polys: torch.Tensor,
                        val_samples: np.ndarray, per_shape: dict[str, list[float]]) -> None:
    """Pins the scored sample tensors so the figures never need the checkpoint again."""
    order = np.argsort(np.asarray(per_shape["success_rate"]))
    gallery_ids = sorted({int(order[0]), int(order[len(order) // 4]), int(order[len(order) // 2]),
                          int(order[(3 * len(order)) // 4]), int(order[-1])})

    artifacts.save_arrays(
        out,
        samples=np.asarray(val_samples, dtype=np.float32),
        polynomials=val_polys,
        gallery_samples=np.stack([val_samples[i] for i in gallery_ids]).astype(np.float32),
        gallery_ids=np.asarray(gallery_ids, dtype=np.int32),
    )
    artifacts.write_manifest(out, run_id=run_id, method="poly_fm", degree=args.degree,
                             scale=args.scale, typical_id=int(order[len(order) // 2]))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "ckpt.pt"
    run_id = pin_baseline_run(out, "poly_fm", args)

    set_seed(args.seed)
    print(f"run_id {run_id} | device {device} | scale {args.scale} | degree {args.degree}")

    if args.skip_train:
        model = PolynomialConstrainedFM(degree=args.degree, hidden_dim=args.hidden_dim,
                                        scale_factor=args.scale).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        losses = []
        print(f"loaded {ckpt_path}")
    else:
        model, losses = train(args, device)
        torch.save(model.state_dict(), ckpt_path)
        np.save(out / "losses.npy", np.array(losses))
        print(f"saved {ckpt_path}")

    model.eval()

    gmm_pool, _ = get_points(args.gmm_pool_size, device=device)
    val_set = get_validation_set(device=device)
    val_polys = val_set["polynomials"][:args.num_polys].to(device)
    val_x0 = val_set["x0"][:args.num_x0].to(device)

    val_samples = run_evaluation_inference(model, val_x0, coeffs=val_polys,
                                           step_size=args.step_size, device=device)
    nll_set = load_nll_eval_set(num_points=args.nll_points, degree=args.degree, scale=args.scale,
                                device=device) if args.nll_points > 0 else None
    metrics = evaluate_validation_set_metrics(val_samples, x_true_pool=gmm_pool, coeffs=val_polys,
                                              degree=args.degree, scale=args.scale, model=model,
                                              nll_points=args.nll_points,
                                              nll_step_size=args.step_size,
                                              nll_eval_points=None if nll_set is None else nll_set["points"],
                                              nll_masses=None if nll_set is None else nll_set["mass"],
                                              device=device)

    per_shape = {k: [float(v) for v in metrics[k]] for k in metrics}
    per_shape["mass"] = constraint_masses(val_polys, gmm_pool, degree=args.degree,
                                          scale=args.scale).cpu().tolist()
    summary = summarize(per_shape)

    (out / "metrics.json").write_text(json.dumps({
        "run_id": run_id,
        "model": "PolynomialConstrainedFM",
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "train": {"iterations": args.iterations, "batch_size": args.batch_size, "lr": args.lr,
                  "hidden_dim": args.hidden_dim, "scale": args.scale, "seed": args.seed},
        "eval": {"num_polys": args.num_polys, "num_x0": args.num_x0,
                 "step_size": args.step_size, "nll_points": args.nll_points},
        "per_shape": per_shape,
        "summary": summary,
    }, indent=2))

    save_plot_artifacts(args, out, run_id, val_polys, val_samples, per_shape)
    render_run_figures(out, degree=args.degree, scale=args.scale)

    print()
    print(readme_table(summary))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

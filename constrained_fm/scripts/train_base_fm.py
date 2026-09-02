# -*- coding: utf-8 -*-
"""Trains the unconstrained base flow matcher that the ECI and HardFlow baselines steer.

The model transports N(0, I) to the *full* 2D GMM and never sees a constraint: all
constraint knowledge is injected at inference time by the samplers in
``src/inference/constrained_samplers.py``. Architecture matches the width and depth of
ConstrainedFlowMatcher (hidden 1024, 4 blocks, 128-dim time embedding) minus every
conditioning pathway, so the comparison isolates the conditioning mechanism rather than
capacity.

    python -m constrained_fm.scripts.train_base_fm
    python -m constrained_fm.scripts.train_base_fm --skip-train   # re-score the checkpoint
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

from constrained_fm.src.datasets.gmm_target import get_points
from constrained_fm.src.experiment import artifacts
from constrained_fm.src.experiment.registry import pin_baseline_run
from constrained_fm.src.experiment.runtime import resolve_device, set_seed
from constrained_fm.src.metrics.distributional import compute_jsd, compute_mmd, compute_swd
from constrained_fm.src.metrics.likelihood import constraint_nll
from constrained_fm.src.models.unconstrained import UnconstrainedFM
from constrained_fm.src.visualization import diagnostics as diag

DEFAULT_OUTDIR = "constrained_fm/baselines/base_fm"
# Enough points for a dense 2D histogram without storing the full 100k scoring tensor.
SAVED_SAMPLES = 50000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the unconstrained base flow matcher.")
    parser.add_argument("--iterations", type=int, default=15001)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=1024, help="matches ConstrainedFlowMatcher")
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--time-dim", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-train", action="store_true", help="score an existing checkpoint")

    parser.add_argument("--num-eval-samples", type=int, default=100000)
    parser.add_argument("--gmm-pool-size", type=int, default=100000)
    parser.add_argument("--nll-points", type=int, default=5000)
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    return parser


def build_model(args, device: torch.device) -> UnconstrainedFM:
    return UnconstrainedFM(time_dim=args.time_dim, hidden_dim=args.hidden_dim,
                           num_blocks=args.num_blocks).to(device)


def train(args, device: torch.device) -> tuple[UnconstrainedFM, list[float]]:
    """CondOT path, plain velocity MSE against the unconditional GMM."""
    model = build_model(args, device)
    prob_path = AffineProbPath(scheduler=CondOTScheduler())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.iterations, eta_min=args.lr_min)

    losses = []
    for iteration in tqdm(range(args.iterations), desc="Training unconstrained FM"):
        optimizer.zero_grad(set_to_none=True)

        x_1, _ = get_points(args.batch_size, device=device)
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.shape[0], device=device)

        path_sample = prob_path.sample(t=t, x_0=x_0, x_1=x_1)
        pred_v = model(path_sample.x_t, path_sample.t)
        loss = torch.pow(pred_v - path_sample.dx_t, 2).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if (iteration + 1) % args.log_every == 0:
            window = float(np.mean(losses[-args.log_every:]))
            print(f"| iter {iteration + 1:6d} | loss {loss.item():.5f} | mean {window:.5f} "
                  f"| lr {optimizer.param_groups[0]['lr']:.2e}", flush=True)

    return model, losses


def score_unconditional(model, gmm_pool: torch.Tensor, args, device: torch.device) -> dict:
    """Fidelity of the base model to the untruncated GMM; the ceiling both baselines start from."""
    samples = model.sample(num_points=args.num_eval_samples, step_size=args.step_size, device=device)
    if samples.ndim == 3:
        samples = samples[-1]

    metrics = {
        "swd": compute_swd(samples, gmm_pool),
        "mmd": compute_mmd(samples, gmm_pool),
        "jsd": compute_jsd(samples, gmm_pool),
    }
    # mass = 1.0: the target here is the whole GMM, so KLD is measured against its own entropy.
    metrics.update(constraint_nll(model, gmm_pool, mass=1.0, num_points=args.nll_points,
                                  step_size=args.step_size, device=device))
    return metrics, samples


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device()
    out = Path(args.outdir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "ckpt.pt"
    run_id = pin_baseline_run(out, "base_fm", args)

    set_seed(args.seed)
    print(f"run_id {run_id} | device {device} | hidden {args.hidden_dim} | {args.num_blocks} blocks")

    if args.skip_train:
        model = build_model(args, device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        losses = []
        print(f"loaded {ckpt_path}")
    else:
        model, losses = train(args, device)
        torch.save(model.state_dict(), ckpt_path)
        np.save(out / "losses.npy", np.array(losses))
        diag.save_figure(diag.plot_loss_curve(np.array(losses)), out / "figures" / "loss_curve.png")
        print(f"saved {ckpt_path}")

    model.eval()
    gmm_pool, _ = get_points(args.gmm_pool_size, device=device)
    metrics, samples = score_unconditional(model, gmm_pool, args, device)

    artifacts.save_arrays(out, samples=samples[:SAVED_SAMPLES])
    artifacts.write_manifest(out, run_id=run_id, method="base_fm")

    diag.save_figure(
        diag.plot_final_samples(artifacts.load_array(out, "samples")[:20000],
                                title="Unconstrained base model"),
        out / "figures" / "base_samples.png")

    (out / "metrics.json").write_text(json.dumps({
        "run_id": run_id,
        "model": "UnconstrainedFM",
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "train": {"iterations": args.iterations, "batch_size": args.batch_size, "lr": args.lr,
                  "hidden_dim": args.hidden_dim, "num_blocks": args.num_blocks,
                  "time_dim": args.time_dim, "seed": args.seed},
        "eval": {"num_eval_samples": args.num_eval_samples, "gmm_pool_size": args.gmm_pool_size,
                 "nll_points": args.nll_points, "step_size": args.step_size},
        "unconditional": {k: float(v) for k, v in metrics.items()},
        "final_loss": float(np.mean(losses[-500:])) if losses else None,
    }, indent=2))

    print("\nfidelity to the untruncated GMM")
    for key, value in metrics.items():
        print(f"  {key:>5}: {value:.5f}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

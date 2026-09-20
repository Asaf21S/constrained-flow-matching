# -*- coding: utf-8 -*-
"""Trains the unconstrained base flow matcher for the bump2d problem.

Same architecture and schedule as ``train_base_fm`` so the two datasets differ only in the
data, but trained in the normalised frame rather than in physical units. The bump2d box is
``[0, 10]^2`` with per-axis standard deviations near 1.9 and 1.5, which an ``N(0, I)`` prior
transports badly; the frame is the analytic one from ``BumpTarget.mean_std``, so it is
identical on every device and never has to be stored alongside the checkpoint.

    python -m constrained_fm.scripts.train_bump_fm
    python -m constrained_fm.scripts.train_bump_fm --skip-train   # re-score the checkpoint
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from tqdm import tqdm

from constrained_fm.src.experiment import artifacts
from constrained_fm.src.experiment.registry import pin_baseline_run
from constrained_fm.src.experiment.runtime import resolve_device, set_seed
from constrained_fm.src.metrics.distributional import compute_jsd, compute_mmd, compute_swd
from constrained_fm.src.models.unconstrained import UnconstrainedFM
from constrained_fm.src.problems.bump2d import BumpProblem

DEFAULT_OUTDIR = "constrained_fm/baselines/bump2d_base_fm"
SAVED_SAMPLES = 50000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the bump2d unconstrained flow matcher.")
    parser.add_argument("--iterations", type=int, default=15001)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--time-dim", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-train", action="store_true", help="score an existing checkpoint")

    parser.add_argument("--num-eval-samples", type=int, default=100000)
    parser.add_argument("--pool-size", type=int, default=100000)
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    return parser


def train(args, problem: BumpProblem, device: torch.device) -> tuple[UnconstrainedFM, list[float]]:
    target = problem.target()
    normalizer = problem.normalizer().to(device)

    model = UnconstrainedFM(input_dim=problem.dim, time_dim=args.time_dim,
                            hidden_dim=args.hidden_dim, num_blocks=args.num_blocks).to(device)
    prob_path = AffineProbPath(scheduler=CondOTScheduler())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.iterations, eta_min=args.lr_min)

    losses: list[float] = []
    for iteration in tqdm(range(args.iterations), desc="Training bump2d FM"):
        optimizer.zero_grad(set_to_none=True)

        x_1 = normalizer.forward(target.sample(args.batch_size, device=device))
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.shape[0], device=device)

        path_sample = prob_path.sample(t=t, x_0=x_0, x_1=x_1)
        loss = torch.pow(model(path_sample.x_t, path_sample.t) - path_sample.dx_t, 2).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if (iteration + 1) % args.log_every == 0:
            window = float(np.mean(losses[-args.log_every:]))
            print(f"| iter {iteration + 1:6d} | loss {loss.item():.5f} | mean {window:.5f} "
                  f"| lr {optimizer.param_groups[0]['lr']:.2e}", flush=True)

    return model, losses


def score(model, problem: BumpProblem, args, device) -> tuple[dict, np.ndarray]:
    """Distributional agreement in the normalised frame, plus physical-unit sanity checks."""
    target = problem.target()
    normalizer = problem.normalizer().to(device)

    samples = model.sample(args.num_eval_samples, step_size=args.step_size, device=device)
    reference = normalizer.forward(target.sample(args.pool_size, device=device))
    physical = normalizer.inverse(samples)

    in_domain = target.in_domain(physical).float().mean().item()
    mean, std = target.mean_std(device=device)
    metrics = {
        "swd": compute_swd(samples, reference),
        "mmd": compute_mmd(samples, reference),
        "jsd": compute_jsd(samples, reference),
        "in_domain_fraction": in_domain,
        "mean_abs_error": (physical.mean(dim=0) - mean).abs().max().item(),
        "std_rel_error": ((physical.std(dim=0) - std) / std).abs().max().item(),
    }
    return metrics, physical[:SAVED_SAMPLES].detach().cpu().numpy()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device()
    set_seed(args.seed)

    problem = BumpProblem()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "ckpt.pt"

    losses: list[float] = []
    if args.skip_train:
        model = UnconstrainedFM(input_dim=problem.dim, time_dim=args.time_dim,
                                hidden_dim=args.hidden_dim, num_blocks=args.num_blocks).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    else:
        model, losses = train(args, problem, device)
        torch.save(model.state_dict(), ckpt_path)
        np.save(out / "losses.npy", np.array(losses))

    model.eval()
    metrics, samples = score(model, problem, args, device)

    run_id = pin_baseline_run(out, "bump2d_base_fm", args)
    artifacts.save_arrays(out, samples=samples)
    artifacts.write_manifest(out, run_id=run_id, method="bump2d_base_fm")

    normalizer = problem.normalizer()
    (out / "metrics.json").write_text(json.dumps({
        "run_id": run_id,
        "problem": problem.name,
        "model": "UnconstrainedFM",
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "frame": {"mean": normalizer.mean.tolist(), "std": normalizer.std.tolist()},
        "train": {"iterations": args.iterations, "batch_size": args.batch_size, "lr": args.lr,
                  "hidden_dim": args.hidden_dim, "num_blocks": args.num_blocks,
                  "time_dim": args.time_dim, "seed": args.seed},
        "eval": {"num_eval_samples": args.num_eval_samples, "pool_size": args.pool_size,
                 "step_size": args.step_size},
        "unconditional": {k: float(v) for k, v in metrics.items()},
        "final_loss": float(np.mean(losses[-500:])) if losses else None,
    }, indent=2))

    print(f"\n### bump2d base FM ({run_id})")
    for key, value in metrics.items():
        print(f"  {key:20s} {value:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

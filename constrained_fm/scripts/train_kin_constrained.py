# -*- coding: utf-8 -*-
"""Trains the explicitly conditioned flow matcher on the kinematics6d mass shells.

This is the amortized counterpart to ECI and HardFlow: the constraint enters the network
rather than the integrator, so the probability-flow ODE stays intact and the sampler pays no
per-step projection cost. Each training step draws fresh shells, so there is no fixed
constraint set to overfit.

    python -m constrained_fm.scripts.train_kin_constrained
    python -m constrained_fm.scripts.train_kin_constrained --skip-train
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

from constrained_fm.src.consts import KIN_MMD_GAMMA
from constrained_fm.src.datasets.kinematics_conditioning import ShellPool
from constrained_fm.src.experiment import artifacts
from constrained_fm.src.experiment.registry import pin_baseline_run, summarize
from constrained_fm.src.experiment.runtime import resolve_device, set_seed
from constrained_fm.src.metrics.distributional import compute_mmd, compute_swd
from constrained_fm.src.models.constrained_mass import MassWindowConstrainedFM
from constrained_fm.src.problems.base import NormalizedConstraint
from constrained_fm.src.problems.kinematics6d import KinematicsProblem, sample_mass_constraints

DEFAULT_OUTDIR = "constrained_fm/baselines/kin6d_explicit"
SAVED_SAMPLES = 2000
SWD_PROJECTIONS = 200


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the conditioned kinematics6d FM.")
    parser.add_argument("--iterations", type=int, default=60001)
    parser.add_argument("--shells-per-batch", type=int, default=64)
    parser.add_argument("--points-per-shell", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--time-dim", type=int, default=128)
    parser.add_argument("--num-frequencies", type=int, default=16)
    parser.add_argument("--train-pool-size", type=int, default=2_000_000)
    parser.add_argument("--log-every", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-train", action="store_true")

    parser.add_argument("--num-shells", type=int, default=100)
    parser.add_argument("--num-eval-samples", type=int, default=10000)
    parser.add_argument("--eval-pool-size", type=int, default=500000)
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    return parser


def build_model(args, problem: KinematicsProblem, device) -> MassWindowConstrainedFM:
    normalizer = problem.normalizer()
    return MassWindowConstrainedFM(
        input_dim=problem.dim, time_dim=args.time_dim, hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks, num_frequencies=args.num_frequencies,
        frame_mean=normalizer.mean, frame_std=normalizer.std,
        mass_scale=problem.target().mass_scale()).to(device)


def train(args, problem: KinematicsProblem, device):
    normalizer = problem.normalizer().to(device)
    shells = ShellPool(problem.target(), args.train_pool_size, device)
    model = build_model(args, problem, device)

    prob_path = AffineProbPath(scheduler=CondOTScheduler())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.iterations, eta_min=args.lr_min)

    losses: list[float] = []
    for iteration in tqdm(range(args.iterations), desc="Training conditioned kinematics FM"):
        optimizer.zero_grad(set_to_none=True)

        physical, params = shells.draw(args.shells_per_batch, args.points_per_shell)
        x_1 = normalizer.forward(physical)
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.shape[0], device=device)

        path_sample = prob_path.sample(t=t, x_0=x_0, x_1=x_1)
        velocity = model(path_sample.x_t, path_sample.t, params)
        loss = torch.pow(velocity - path_sample.dx_t, 2).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if (iteration + 1) % args.log_every == 0:
            window = float(np.mean(losses[-args.log_every:]))
            print(f"| iter {iteration + 1:6d} | loss {loss.item():.5f} | mean {window:.5f} "
                  f"| lr {optimizer.param_groups[0]['lr']:.2e}", flush=True)

    return model, losses


def evaluate(model, problem: KinematicsProblem, args, device) -> tuple[dict, np.ndarray]:
    """Per-shell success rate and distributional agreement against the conditional truth."""
    normalizer = problem.normalizer().to(device)
    target = problem.target()

    set_seed(args.seed)
    pool = target.sample(args.eval_pool_size, device=device)
    constraints, mass = sample_mass_constraints(args.num_shells, target, pool)
    pool_u = normalizer.forward(pool)

    keys = ("swd", "mmd", "swd_noise_floor", "mmd_noise_floor", "success_rate")
    per_shape: dict[str, list[float]] = {k: [] for k in keys}
    saved = []

    for constraint in tqdm(constraints, desc="conditioned sampling"):
        wrapped = NormalizedConstraint(constraint, normalizer)
        params = constraint.params.to(device)
        samples = model.sample(args.num_eval_samples, params=params,
                               step_size=args.step_size, device=device)

        truth = pool_u[wrapped.is_feasible(pool_u)]
        half = truth.shape[0] // 2
        per_shape["swd"].append(compute_swd(samples, truth, num_projections=SWD_PROJECTIONS))
        per_shape["mmd"].append(compute_mmd(samples, truth, gamma=KIN_MMD_GAMMA))
        per_shape["swd_noise_floor"].append(
            compute_swd(truth[:half], truth[half:], num_projections=SWD_PROJECTIONS))
        per_shape["mmd_noise_floor"].append(
            compute_mmd(truth[:half], truth[half:], gamma=KIN_MMD_GAMMA))
        per_shape["success_rate"].append(wrapped.success_rate(samples))
        saved.append(normalizer.inverse(samples[:SAVED_SAMPLES]).cpu().numpy())

    per_shape["mass"] = mass.tolist()
    per_shape["epsilon"] = [c.epsilon for c in constraints]
    return per_shape, np.stack(saved)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device()
    set_seed(args.seed)

    problem = KinematicsProblem()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "ckpt.pt"

    losses: list[float] = []
    if args.skip_train:
        model = build_model(args, problem, device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    else:
        model, losses = train(args, problem, device)
        torch.save(model.state_dict(), ckpt_path)
        np.save(out / "losses.npy", np.array(losses))

    model.eval()
    per_shape, samples = evaluate(model, problem, args, device)

    run_id = pin_baseline_run(out, "kin6d_explicit", args)
    artifacts.save_arrays(out, samples=samples,
                          mass=np.array(per_shape["mass"], dtype=np.float64))
    artifacts.write_manifest(out, run_id=run_id, method="kin6d_explicit")

    summary = summarize(per_shape)
    (out / "metrics.json").write_text(json.dumps({
        "run_id": run_id,
        "problem": problem.name,
        "method": "kin6d_explicit",
        "model": "MassWindowConstrainedFM",
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "train": {"iterations": args.iterations, "shells_per_batch": args.shells_per_batch,
                  "points_per_shell": args.points_per_shell, "lr": args.lr,
                  "hidden_dim": args.hidden_dim, "num_blocks": args.num_blocks,
                  "num_frequencies": args.num_frequencies,
                  "train_pool_size": args.train_pool_size, "seed": args.seed},
        "eval": {"num_shells": args.num_shells, "num_eval_samples": args.num_eval_samples,
                 "eval_pool_size": args.eval_pool_size, "step_size": args.step_size,
                 "swd_projections": SWD_PROJECTIONS, "mmd_gamma": KIN_MMD_GAMMA},
        "per_shape": per_shape,
        "summary": summary,
        "final_loss": float(np.mean(losses[-1000:])) if losses else None,
    }, indent=2))

    print(f"\n### kinematics6d explicit ({run_id})")
    for key in ("success_rate", "swd", "swd_noise_floor", "mmd", "mmd_noise_floor"):
        print(f"  {key:16s} median {summary[f'{key}_median']:.4f} | "
              f"mean {summary[f'{key}_mean']:.4f} | p5 {summary[f'{key}_p5']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

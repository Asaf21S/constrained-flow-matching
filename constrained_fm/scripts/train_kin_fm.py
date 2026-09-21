# -*- coding: utf-8 -*-
"""Trains the unconstrained base flow matcher for the kinematics6d problem.

Same recipe as the 2D base models, with the differences the dimension forces. The Cartesian
components span two very different scales -- the transverse momenta sit near 45 while the
longitudinal one reaches into the hundreds -- so training happens in the analytic normalised
frame; an ``N(0, I)`` prior would otherwise have to transport most of its mass along ``p_z``.

Unlike ECI and HardFlow, this model leaves the probability-flow ODE intact, so its NLL is
meaningful and is reported against the exact analytic log-density.

    python -m constrained_fm.scripts.train_kin_fm
    python -m constrained_fm.scripts.train_kin_fm --skip-train
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
from constrained_fm.src.experiment import artifacts
from constrained_fm.src.experiment.registry import pin_baseline_run
from constrained_fm.src.experiment.runtime import resolve_device, set_seed
from constrained_fm.src.metrics.distributional import compute_mmd, compute_swd
from constrained_fm.src.models.unconstrained import UnconstrainedFM
from constrained_fm.src.problems.kinematics6d import KinematicsProblem

DEFAULT_OUTDIR = "constrained_fm/baselines/kin6d_base_fm"
SAVED_SAMPLES = 50000
# SWD needs far more directions in 6D than in 2D to cover the sphere at the same resolution.
SWD_PROJECTIONS = 200


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the kinematics6d flow matcher.")
    parser.add_argument("--iterations", type=int, default=30001)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--time-dim", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-train", action="store_true")

    parser.add_argument("--num-eval-samples", type=int, default=100000)
    parser.add_argument("--pool-size", type=int, default=100000)
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    return parser


def train(args, problem: KinematicsProblem,
          device: torch.device) -> tuple[UnconstrainedFM, list[float]]:
    target = problem.target()
    normalizer = problem.normalizer().to(device)

    model = UnconstrainedFM(input_dim=problem.dim, time_dim=args.time_dim,
                            hidden_dim=args.hidden_dim, num_blocks=args.num_blocks).to(device)
    prob_path = AffineProbPath(scheduler=CondOTScheduler())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.iterations, eta_min=args.lr_min)

    losses: list[float] = []
    for iteration in tqdm(range(args.iterations), desc="Training kinematics6d FM"):
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


def median_sq_distance(x: torch.Tensor, num_points: int = 4096) -> float:
    """Median pairwise squared distance, the usual RBF bandwidth scale."""
    subset = x[torch.randperm(x.shape[0], device=x.device)[:num_points]]
    return float(torch.cdist(subset, subset).pow(2).median())


def score(model, problem: KinematicsProblem, args, device) -> tuple[dict, np.ndarray]:
    """Distributional agreement in the normalised frame plus physical-unit sanity checks.

    The invariant mass is reported separately from the coordinate marginals because it is the
    only scalar the constraints act on: a model can match every per-axis marginal and still
    misplace the mass spectrum, which is exactly the failure that would invalidate M5.
    """
    target = problem.target()
    normalizer = problem.normalizer().to(device)

    samples = model.sample(args.num_eval_samples, step_size=args.step_size, device=device)
    reference = normalizer.forward(target.sample(args.pool_size, device=device))
    physical = normalizer.inverse(samples)

    # Two independent draws from the target bound what any sampler can achieve at this sample
    # count, so the raw SWD and MMD only mean something next to them.
    floor_a = normalizer.forward(target.sample(args.num_eval_samples, device=device))
    floor_b = normalizer.forward(target.sample(args.pool_size, device=device))

    mass_gen = target.invariant_mass(physical)
    mass_ref = target.invariant_mass(normalizer.inverse(reference))
    mean, std = target.mean_std(device=device)

    metrics = {
        "swd": compute_swd(samples, reference, num_projections=SWD_PROJECTIONS),
        "swd_noise_floor": compute_swd(floor_a, floor_b, num_projections=SWD_PROJECTIONS),
        "mmd": compute_mmd(samples, reference, gamma=KIN_MMD_GAMMA),
        "mmd_noise_floor": compute_mmd(floor_a, floor_b, gamma=KIN_MMD_GAMMA),
        "mmd_gamma_median_heuristic": 1.0 / median_sq_distance(reference),
        "in_support_fraction": target.in_support(physical).float().mean().item(),
        "mean_abs_error": (physical.mean(dim=0) - mean).abs().max().item(),
        "std_rel_error": ((physical.std(dim=0) - std) / std).abs().max().item(),
        "mass_median_rel_error": abs(float(mass_gen.median() / mass_ref.median()) - 1.0),
        "mass_swd": compute_swd(mass_gen.unsqueeze(-1), mass_ref.unsqueeze(-1)),
        "mass_swd_noise_floor": compute_swd(
            target.invariant_mass(normalizer.inverse(floor_a)).unsqueeze(-1),
            target.invariant_mass(normalizer.inverse(floor_b)).unsqueeze(-1)),
    }
    return metrics, physical[:SAVED_SAMPLES].detach().cpu().numpy()


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
        model = UnconstrainedFM(input_dim=problem.dim, time_dim=args.time_dim,
                                hidden_dim=args.hidden_dim, num_blocks=args.num_blocks).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    else:
        model, losses = train(args, problem, device)
        torch.save(model.state_dict(), ckpt_path)
        np.save(out / "losses.npy", np.array(losses))

    model.eval()
    metrics, samples = score(model, problem, args, device)

    run_id = pin_baseline_run(out, "kin6d_base_fm", args)
    artifacts.save_arrays(out, samples=samples)
    artifacts.write_manifest(out, run_id=run_id, method="kin6d_base_fm")

    normalizer = problem.normalizer()
    (out / "metrics.json").write_text(json.dumps({
        "run_id": run_id,
        "problem": problem.name,
        "model": "UnconstrainedFM",
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "frame": {"mean": normalizer.mean.tolist(), "std": normalizer.std.tolist()},
        "mass_scale": problem.target().mass_scale(),
        "train": {"iterations": args.iterations, "batch_size": args.batch_size, "lr": args.lr,
                  "hidden_dim": args.hidden_dim, "num_blocks": args.num_blocks,
                  "time_dim": args.time_dim, "seed": args.seed},
        "eval": {"num_eval_samples": args.num_eval_samples, "pool_size": args.pool_size,
                 "step_size": args.step_size, "swd_projections": SWD_PROJECTIONS,
                 "mmd_gamma": KIN_MMD_GAMMA},
        "unconditional": {k: float(v) for k, v in metrics.items()},
        "final_loss": float(np.mean(losses[-1000:])) if losses else None,
    }, indent=2))

    print(f"\n### kinematics6d base FM ({run_id})")
    for key, value in metrics.items():
        print(f"  {key:24s} {value:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

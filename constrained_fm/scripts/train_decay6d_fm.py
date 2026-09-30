# -*- coding: utf-8 -*-
"""Trains the decay6d flow matchers: the unconstrained ``p_uncon`` or the box-conditioned ``q``.

Both use the linear (CondOT, sigma_min = 0) path with independent coupling, identical ResBlock
backbones, and the same analytic normalizer, whose std is stored in each checkpoint so the
importance-sampling stage can refuse a mismatched pair. Data is drawn fresh from the simulator
every step. Box pairs come from the anchored sampler, so the conditional is exactly ``p(x | B)``.

    python -m constrained_fm.scripts.train_decay6d_fm --mode uncon
    python -m constrained_fm.scripts.train_decay6d_fm --mode box
    python -m constrained_fm.scripts.train_decay6d_fm --mode box --smoke
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from tqdm import tqdm

from constrained_fm.src.consts import (DECAY_BOX_HALF_WIDTH_RANGE, DECAY_BOX_MASS_RANGE,
                                       DECAY_EMA_DECAY, DECAY_GRAD_CLIP, DECAY_ODE_ATOL,
                                       DECAY_ODE_RTOL)
from constrained_fm.src.experiment.registry import pin_baseline_run
from constrained_fm.src.experiment.runtime import resolve_device, set_seed
from constrained_fm.src.models.constrained_box6d import BoxConstrainedFM6D
from constrained_fm.src.models.unconstrained import UnconstrainedFM
from constrained_fm.src.problems.decay6d import (DecayProblem, conditioning_to_box,
                                                 sample_conditioned_batch)
from constrained_fm.src.solvers import cnf

DEFAULT_ROOT = "constrained_fm/baselines/decay6d_is"
SMOKE_ROOT = "constrained_fm/baselines/decay6d_is/smoke"
DIAG_CHUNK = 5000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a decay6d flow matcher.")
    parser.add_argument("--mode", choices=("uncon", "box"), required=True)
    parser.add_argument("--iterations", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=DECAY_GRAD_CLIP,
                        help="max global gradient norm; <= 0 disables")
    parser.add_argument("--ema-decay", type=float, default=DECAY_EMA_DECAY,
                        help="EMA of the weights, saved and evaluated; 0 disables")
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--time-dim", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--diag-samples", type=int, default=5000,
                        help="uncon: KL(p || p_uncon) points; box: samples per diagnostic box")
    parser.add_argument("--diag-boxes", type=int, default=16)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.smoke:
        args.iterations, args.log_every = 300, 100
        args.diag_samples, args.diag_boxes = 500, 4
    if args.outdir is None:
        args.outdir = f"{SMOKE_ROOT if args.smoke else DEFAULT_ROOT}/{args.mode}"
    return args


def model_kwargs(args: argparse.Namespace) -> dict[str, int]:
    return {"input_dim": 6, "time_dim": args.time_dim, "hidden_dim": args.hidden_dim,
            "num_blocks": args.num_blocks}


def build_model(mode: str, kwargs: dict[str, int]) -> torch.nn.Module:
    return BoxConstrainedFM6D(**kwargs) if mode == "box" else UnconstrainedFM(**kwargs)


def train(args, problem: DecayProblem, device: torch.device
          ) -> tuple[torch.nn.Module, list[float], list[float]]:
    target = problem.target()
    normalizer = problem.normalizer(torch.float64).to(device)
    table = problem.mass_table(device) if args.mode == "box" else None

    model = build_model(args.mode, model_kwargs(args)).to(device)
    ema = (AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay))
           if args.ema_decay > 0 else None)
    path = AffineProbPath(scheduler=CondOTScheduler())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iterations,
                                                           eta_min=args.lr_min)

    losses: list[float] = []
    acceptance: list[float] = []
    for iteration in tqdm(range(args.iterations), desc=f"Training decay6d {args.mode}"):
        optimizer.zero_grad(set_to_none=True)

        extras = {}
        if table is None:
            x_1 = normalizer.forward(target.sample(args.batch_size, device)).float()
        else:
            x_1, box, rate = sample_conditioned_batch(target, normalizer, table, args.batch_size,
                                                      problem.half_width_range,
                                                      problem.mass_range, device)
            x_1, extras["box"] = x_1.float(), box.float()
            acceptance.append(rate)

        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.shape[0], device=device)
        path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
        loss = (model(path_sample.x_t, path_sample.t, **extras) - path_sample.dx_t).pow(2).mean()
        loss.backward()
        grad_norm = (torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
                     if args.grad_clip > 0 else float("nan"))
        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update_parameters(model)
        losses.append(loss.item())

        if (iteration + 1) % args.log_every == 0:
            window = float(np.mean(losses[-args.log_every:]))
            print(f"| iter {iteration + 1:6d} | loss {loss.item():.5f} | mean {window:.5f} "
                  f"| lr {optimizer.param_groups[0]['lr']:.2e} | gnorm {grad_norm:.3e}", flush=True)

    return (model if ema is None else ema.module), losses, acceptance


def freeze_fp64(model: torch.nn.Module) -> torch.nn.Module:
    model.eval().double()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def diagnose_uncon(model, problem: DecayProblem, args, device) -> dict[str, float]:
    """``KL(p || p_uncon)`` on simulator draws, both densities in physical units."""
    target = problem.target()
    normalizer = problem.normalizer(torch.float64).to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    x = target.sample(args.diag_samples, device, generator)

    log_p = target.log_prob(x)
    log_model = torch.cat([cnf.log_prob(model, chunk, atol=DECAY_ODE_ATOL, rtol=DECAY_ODE_RTOL)[0]
                           for chunk in normalizer.forward(x).split(DIAG_CHUNK)])
    gap = log_p - (log_model + normalizer.log_det_forward)
    return {"kl_p_uncon": gap.mean().item(),
            "kl_p_uncon_se": (gap.std() / math.sqrt(gap.numel())).item(),
            "exact_nll": -log_p.mean().item()}


def diagnose_box(model, problem: DecayProblem, args, device) -> dict[str, float]:
    """In-box rate of ``q`` on fresh boxes from the training distribution."""
    normalizer = problem.normalizer(torch.float64).to(device)
    torch.manual_seed(args.seed + 1)
    _, boxes, _ = sample_conditioned_batch(problem.target(), normalizer,
                                           problem.mass_table(device), args.diag_boxes,
                                           problem.half_width_range, problem.mass_range, device)
    rates = []
    for box in boxes:
        x0 = torch.randn(args.diag_samples, 6, device=device, dtype=torch.float64)
        x1, _ = cnf.sample(model, x0, {"box": box[None]}, DECAY_ODE_ATOL, DECAY_ODE_RTOL)
        constraint = conditioning_to_box(box, normalizer)
        rates.append(constraint.contains(normalizer.inverse(x1)).double().mean().item())
    return {"in_box_rate_mean": float(np.mean(rates)), "in_box_rate_min": float(np.min(rates)),
            "in_box_rate_max": float(np.max(rates))}


def main(argv: list[str] | None = None) -> int:
    args = resolve_args(build_parser().parse_args(argv))
    device = resolve_device()
    out = Path(args.outdir)
    run_id = pin_baseline_run(out, f"decay6d_{args.mode}", args, extra={
        "half_width_range": list(DECAY_BOX_HALF_WIDTH_RANGE),
        "mass_range": list(DECAY_BOX_MASS_RANGE)} if args.mode == "box" else None)
    set_seed(args.seed)

    problem = DecayProblem(DECAY_BOX_HALF_WIDTH_RANGE, DECAY_BOX_MASS_RANGE)
    normalizer = problem.normalizer(torch.float64)
    ckpt_path = out / "ckpt.pt"

    losses: list[float] = []
    acceptance: list[float] = []
    if args.skip_train:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model = build_model(args.mode, ckpt["model_kwargs"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
    else:
        model, losses, acceptance = train(args, problem, device)
        torch.save({"state_dict": model.state_dict(), "model_kwargs": model_kwargs(args),
                    "mode": args.mode, "run_id": run_id,
                    "normalizer_mean": normalizer.mean.tolist(),
                    "normalizer_std": normalizer.std.tolist()}, ckpt_path)
        np.save(out / "losses.npy", np.array(losses))

    model = freeze_fp64(model)
    diag = (diagnose_box if args.mode == "box" else diagnose_uncon)(model, problem, args, device)

    (out / "metrics.json").write_text(json.dumps({
        "run_id": run_id,
        "problem": problem.name,
        "mode": args.mode,
        "model": type(model).__name__,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "normalizer_std": normalizer.std.tolist(),
        "train": {k: getattr(args, k) for k in ("iterations", "batch_size", "lr", "lr_min",
                                                 "grad_clip", "ema_decay", "hidden_dim",
                                                 "num_blocks", "time_dim", "seed")},
        "box_filter_acceptance": float(np.mean(acceptance)) if acceptance else None,
        "final_loss": float(np.mean(losses[-args.log_every:])) if losses else None,
        "diagnostics": diag,
    }, indent=2))

    print(f"\n### decay6d {args.mode} ({run_id})")
    for key, value in diag.items():
        print(f"  {key:24s} {value:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

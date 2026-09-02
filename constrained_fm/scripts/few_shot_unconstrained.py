# -*- coding: utf-8 -*-
"""Few-shot unconstrained baseline: what if you had N valid points instead of a constraint?

For each (polynomial, N) pair this rejection-samples exactly N GMM points satisfying
P(x) <= 0, trains an *unconditional* flow matcher from scratch on just those points, and
scores it against the truncated target. It is the data-driven alternative to conditioning:
no coefficients, no latent, no knowledge of P beyond the N samples it induced.

Two things to keep in mind when reading the numbers:

  * N is not the same quantity as the Functa ablation's N. There it counts CAVIA *query*
    points labelled with tanh(P), spread across the plane and mostly outside the region.
    Here it counts valid *samples from the target*. Same axis, different information.
  * Early stopping uses a held-out set of 10k constraint-satisfying points, far more data
    than the model is allowed to train on. That is deliberate -- it removes training length
    as a confound so each N gets its best-case model -- but it makes this an optimistic
    upper bound on the baseline, not a realistic few-shot result.

Work is sharded so the full benchmark can run as concurrent jobs:

    python -m constrained_fm.scripts.few_shot_unconstrained --shard 0 --num-shards 8
    python -m constrained_fm.scripts.few_shot_unconstrained --plot-only
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler

from constrained_fm.src.consts import PLANE_SCALE, POLYNOMIAL_DEGREE
from constrained_fm.src.datasets.gmm_target import get_points
from constrained_fm.src.datasets.validation import get_validation_set
from constrained_fm.src.experiment.runtime import resolve_device, set_seed
from constrained_fm.src.geometry.polynomials import compute_poly_features, evaluate_poly
from constrained_fm.src.inference.evaluator import evaluate_single_configuration
from constrained_fm.src.metrics.functa_fidelity import constraint_masses
from constrained_fm.src.metrics.likelihood import constraint_nll
from constrained_fm.src.models.unconstrained import UnconstrainedFM
from constrained_fm.src.visualization import diagnostics as diag

DEFAULT_N_VALUES = [50, 100, 300, 500, 1000, 2000]
DEFAULT_OUTDIR = "constrained_fm/baselines/few_shot"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Few-shot unconstrained flow matching baseline.")
    parser.add_argument("--shapes", type=int, nargs="+",
                        help="validation-set indices; default picks --num-shapes spanning mass")
    parser.add_argument("--num-shapes", type=int, default=4)
    parser.add_argument("--all-shapes", action="store_true",
                        help="run the full validation benchmark instead of a subset")
    parser.add_argument("--num-points", type=int, nargs="+", default=DEFAULT_N_VALUES)

    parser.add_argument("--hidden-dim", type=int, default=1024, help="matches ConstrainedFlowMatcher")
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--time-dim", type=int, default=128)

    parser.add_argument("--iterations", type=int, default=20000, help="upper bound; early stopping decides")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-points", type=int, default=10000,
                        help="held-out constraint-satisfying points driving early stopping")
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--patience", type=int, default=12, help="evaluations without improvement")

    parser.add_argument("--num-x0", type=int, default=10000)
    parser.add_argument("--nll-points", type=int, default=5000)
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--gmm-pool-size", type=int, default=100000)

    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--figure-dir", default="constrained_fm/images/functa/few_shot",
                        help="figures live with the README that embeds them, not with the results")
    parser.add_argument("--plot-only", action="store_true",
                        help="assemble figures and summary from existing per-item results")
    parser.add_argument("--degree", type=int, default=POLYNOMIAL_DEGREE)
    parser.add_argument("--scale", type=float, default=PLANE_SCALE)
    return parser


def poly_values(C: torch.Tensor, x: torch.Tensor, degree: int, scale: float) -> torch.Tensor:
    x_pow, y_pow = compute_poly_features(x, degree=degree, scale=scale)
    return evaluate_poly(x_pow, y_pow, C.unsqueeze(0).expand(x.shape[0], -1, -1)).squeeze(-1)


def rejection_sample(C: torch.Tensor, count: int, degree: int, scale: float,
                     device, chunk: int = 200000, max_draws: int = 50_000_000) -> torch.Tensor:
    """Exactly `count` GMM draws satisfying P(x) <= 0."""
    collected, total = [], 0
    drawn = 0
    while total < count:
        pts, _ = get_points(chunk, device=device)
        keep = pts[poly_values(C, pts, degree, scale) <= 0]
        if keep.shape[0]:
            collected.append(keep)
            total += keep.shape[0]
        drawn += chunk
        if drawn > max_draws:
            raise RuntimeError(f"constraint too small: {total}/{count} valid points in {drawn} draws")
    return torch.cat(collected, dim=0)[:count]


def train_few_shot(x_train: torch.Tensor, x_val: torch.Tensor, args, device) -> tuple[UnconstrainedFM, dict]:
    """Trains an unconditional flow matcher on x_train, early-stopping on x_val.

    The validation loss is evaluated on a *fixed* (t, x_0) draw so the stopping signal is
    deterministic; resampling it every check would make the comparison across N noisier
    than the effect being measured.
    """
    model = UnconstrainedFM(time_dim=args.time_dim, hidden_dim=args.hidden_dim,
                            num_blocks=args.num_blocks).to(device)
    prob_path = AffineProbPath(scheduler=CondOTScheduler())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    val_x0 = torch.randn(x_val.shape[0], 2, generator=generator).to(device)
    val_t = torch.rand(x_val.shape[0], generator=generator).to(device)
    val_sample = prob_path.sample(t=val_t, x_0=val_x0, x_1=x_val)

    best_loss, best_state, best_iter, stale = float("inf"), None, 0, 0
    history = []

    for iteration in range(1, args.iterations + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        idx = torch.randint(0, x_train.shape[0], (args.batch_size,), device=device)
        x_1 = x_train[idx]
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.shape[0], device=device)

        path_sample = prob_path.sample(t=t, x_0=x_0, x_1=x_1)
        pred_v = model(path_sample.x_t, path_sample.t)
        loss = torch.pow(pred_v - path_sample.dx_t, 2).mean()
        loss.backward()
        optimizer.step()

        if iteration % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(val_sample.x_t, val_sample.t)
                val_loss = float(torch.pow(val_pred - val_sample.dx_t, 2).mean())
            history.append((iteration, float(loss), val_loss))

            if val_loss < best_loss - 1e-5:
                best_loss, best_iter, stale = val_loss, iteration, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= args.patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"best_val_loss": best_loss, "best_iteration": best_iter,
                   "stopped_at": iteration, "history": history}


def run_item(shape_id: int, n_points: int, C: torch.Tensor, gmm_pool: torch.Tensor,
             mass: float, args, device) -> tuple[dict, np.ndarray]:
    set_seed(args.seed + 1000 * shape_id + n_points)
    started = time.time()

    x_train = rejection_sample(C, n_points, args.degree, args.scale, device)
    x_val = rejection_sample(C, args.val_points, args.degree, args.scale, device)

    model, train_info = train_few_shot(x_train, x_val, args, device)
    train_seconds = time.time() - started

    samples = model.sample(num_points=args.num_x0, step_size=args.step_size, device=device)
    if samples.ndim == 3:
        samples = samples[-1]

    metrics = evaluate_single_configuration(samples, x_true_pool=gmm_pool, coeffs=C,
                                            degree=args.degree, scale=args.scale, device=device)

    # Unconditional model: pass no conditioning to the backward ODE.
    x_true_valid = gmm_pool[poly_values(C, gmm_pool, args.degree, args.scale) <= 0]
    metrics.update(constraint_nll(model, x_true_valid, mass, num_points=args.nll_points,
                                  step_size=args.step_size, device=device))

    record = {
        "shape_id": shape_id, "n_points": n_points, "mass": mass,
        "train_seconds": train_seconds, "total_seconds": time.time() - started,
        **{k: float(v) for k, v in metrics.items()},
        **{k: train_info[k] for k in ("best_val_loss", "best_iteration", "stopped_at")},
    }
    return record, samples.detach().cpu().numpy().astype(np.float32)


def select_shapes(mass: torch.Tensor, count: int) -> list[int]:
    """Shapes spanning the constraint-mass range; low-mass ones are where few-shot hurts most."""
    order = torch.argsort(mass).tolist()
    picks = np.linspace(0, len(order) - 1, min(count, len(order))).round().astype(int)
    return sorted(int(order[p]) for p in picks)


def assemble(args, out: Path, polys: torch.Tensor, mass: torch.Tensor) -> int:
    figures = Path(args.figure_dir)
    figures.mkdir(parents=True, exist_ok=True)
    records = [json.loads(p.read_text()) for p in sorted((out / "results").glob("*.json"))]
    if not records:
        print("no results to assemble")
        return 1

    n_values = sorted({r["n_points"] for r in records})
    shape_ids = sorted({r["shape_id"] for r in records})
    by_key = {(r["shape_id"], r["n_points"]): r for r in records}
    print(f"{len(records)} results | {len(shape_ids)} shapes | N {n_values}")

    summary = {}
    for n in n_values:
        rows = [r for r in records if r["n_points"] == n]
        summary[str(n)] = {
            key: float(np.nanmedian([r[key] for r in rows if np.isfinite(r.get(key, np.nan))]))
            for key in ("success_rate", "swd", "mmd", "jsd", "nll", "kld")
        }
        summary[str(n)]["count"] = len(rows)
        summary[str(n)]["median_train_seconds"] = float(np.median([r["train_seconds"] for r in rows]))

    (out / "summary.json").write_text(json.dumps(
        {"n_values": n_values, "shape_ids": shape_ids, "per_n_median": summary,
         "records": records}, indent=2))

    header = f"{'N':>6}{'SR':>9}{'SWD':>9}{'MMD':>10}{'JSD':>9}{'NLL':>9}{'KLD':>9}{'train_s':>10}{'n':>5}"
    print("\nmedian over shapes")
    print(header)
    print("-" * len(header))
    for n in n_values:
        s = summary[str(n)]
        print(f"{n:>6}{s['success_rate']:>9.2f}{s['swd']:>9.4f}{s['mmd']:>10.5f}"
              f"{s['jsd']:>9.4f}{s['nll']:>9.3f}{s['kld']:>9.4f}"
              f"{s['median_train_seconds']:>10.1f}{s['count']:>5}")

    plot_ids = shape_ids if len(shape_ids) <= 6 else select_shapes(mass[shape_ids], 4)
    sample_grid, cell_labels = [], []
    for sid in plot_ids:
        row, labels = [], []
        for n in n_values:
            path = out / "samples" / f"shape{sid}_N{n}.npy"
            if not path.exists():
                row, labels = None, None
                break
            row.append(np.load(path))
            rec = by_key.get((sid, n), {})
            labels.append(f"SR {rec.get('success_rate', float('nan')):.1f}%\n"
                          f"KLD {rec.get('kld', float('nan')):.2f}")
        if row is not None:
            sample_grid.append(row)
            cell_labels.append(labels)

    if sample_grid:
        rows_used = plot_ids[:len(sample_grid)]
        diag.save_figure(
            diag.plot_samples_ablation_grid(
                sample_grid, [polys[i] for i in rows_used],
                [f"shape {i}\nmass {mass[i]:.2f}" for i in rows_used],
                [f"N = {n}" for n in n_values], cell_labels=cell_labels,
                degree=args.degree, scale=args.scale),
            figures / "few_shot_grid.png")

    # Indexed explicitly by (N, shape) so rows stay aligned even if shards finished unevenly.
    complete = [sid for sid in shape_ids if all((sid, n) in by_key for n in n_values)]
    if complete:
        series = {}
        for name, key in (("success rate (%)", "success_rate"), ("SWD", "swd"), ("KLD", "kld")):
            series[name] = np.array([[by_key[(sid, n)][key] for sid in complete] for n in n_values],
                                    dtype=float)
        diag.save_figure(
            diag.plot_ablation_curves(n_values, series, xlabel="training points N"),
            figures / "few_shot_curves.png")
        print(f"curves over {len(complete)} shapes with every N present")

    print(f"\nwrote {out} (figures in {figures})")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device()
    out = Path(args.outdir)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    gmm_pool, _ = get_points(args.gmm_pool_size, device=device)
    val_set = get_validation_set(device=device)
    polys = val_set["polynomials"].to(device)
    mass = constraint_masses(polys, gmm_pool, degree=args.degree, scale=args.scale)

    if args.plot_only:
        return assemble(args, out, polys, mass)

    if args.all_shapes:
        shape_ids = list(range(polys.shape[0]))
    elif args.shapes:
        shape_ids = sorted(args.shapes)
    else:
        shape_ids = select_shapes(mass, args.num_shapes)

    items = [(sid, n) for sid in shape_ids for n in sorted(args.num_points)]
    mine = items[args.shard::args.num_shards]
    print(f"device {device} | shard {args.shard}/{args.num_shards} | {len(mine)}/{len(items)} items")
    print(f"model: hidden {args.hidden_dim}, {args.num_blocks} blocks, time_dim {args.time_dim}")
    print(f"shapes {shape_ids}", flush=True)

    for position, (sid, n) in enumerate(mine, start=1):
        record_path = out / "results" / f"shape{sid}_N{n}.json"
        if record_path.exists():
            print(f"[{position}/{len(mine)}] shape {sid} N {n}: already done, skipping", flush=True)
            continue

        # One pathological constraint must not cost the whole shard; it is retried on rerun.
        try:
            record, samples = run_item(sid, n, polys[sid], gmm_pool, float(mass[sid]), args, device)
        except Exception as exc:
            print(f"[{position}/{len(mine)}] shape {sid} N {n}: FAILED ({type(exc).__name__}: {exc})",
                  flush=True)
            continue

        record_path.write_text(json.dumps(record, indent=2))
        np.save(out / "samples" / f"shape{sid}_N{n}.npy", samples)

        print(f"[{position}/{len(mine)}] shape {sid} N {n:>4} | "
              f"SR {record['success_rate']:6.2f}% | SWD {record['swd']:.4f} | "
              f"KLD {record['kld']:.4f} | stopped {record['stopped_at']} "
              f"(best {record['best_iteration']}) | {record['train_seconds']:.0f}s", flush=True)

    print("\nshard complete. Run with --plot-only once every shard has finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

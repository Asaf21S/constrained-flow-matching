# -*- coding: utf-8 -*-
"""Test-time discovery of a constraint that isolates 3 of the 4 GMM modes.

Both the Functa FM and its SIREN are frozen. The only trainable tensor is the latent
``z_c in R^512``, fitted by the standard conditional FM loss on a target batch drawn from the
3 kept modes only, so gradients reach it exclusively through the frozen vector field:

    min_{z_c}  E_{t, x0, x1 ~ p_target} || v_phi(x_t, t, z_c) - (x1 - x0) ||^2
               + lambda * (z_c - mu)^T Sigma^{-1} (z_c - mu) / d

By default z_c = mu + U_k Lambda_k^{1/2} w is confined to the top-k principal subspace of the
pool latents (k from --subspace-var) and w is what Adam steps. In the full 512-d space the FM
loss is flat along the ~500 directions the pool never visits, so Adam's per-coordinate noise
walks z_c off-manifold and the decoded boundary drifts while the loss stays put.
``--subspace-var 0`` restores the unconstrained z_c (initialised at 0, the CAVIA meta-init).

The optional Mahalanobis term uses the pool latents' Gaussian (mu, Sigma); its value is logged
at every snapshot regardless of lambda, as an off-manifold indicator for the frozen FM.

    frames/step_<k>.png   decoded boundary over the 4-mode scatter, every --viz-every steps
    strip.{png,pdf}       evenly spaced steps: scatter + boundary, x-marginal below
    likelihood.{png,pdf}  FM density of the final z_c over the whole domain, as a histogram of
                          forward samples: the exact-divergence trace through the w0=30 SIREN
                          feature blows up (total mass ~1e33 at step 0.01), sampling does not
    history.{png,pdf}     FM loss and per-mode inside fraction against step

    sbatch scripts/run_discover_constraint.sh
    sbatch scripts/run_discover_constraint.sh --exclude-mode 1 --z-reg-weight 1e-2
    sbatch scripts/run_discover_constraint.sh --plot-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

from constrained_fm.src.consts import GMM_COVS, GMM_MEANS, GMM_WEIGHTS, PLANE_SCALE
from constrained_fm.src.datasets.gmm_target import compute_gmm_density, get_points
from constrained_fm.src.experiment import artifacts
from constrained_fm.src.experiment.config import REPO_ROOT
from constrained_fm.src.experiment.registry import load_config, pin_baseline_run
from constrained_fm.src.experiment.runtime import (build_flow_matcher, load_checkpoint, load_pool,
                                                   load_siren, resolve_device, set_seed)
from constrained_fm.src.visualization import constraint_discovery as cd
from constrained_fm.src.visualization.siren_encoder import save_encoder_figure

FUNCTA_RUN = "siren-uniform-8d6375ab"
OUTDIR = "constrained_fm/baselines/constraint_discovery"
FIGURE_DIR = "constrained_fm/images/thesis_pool/constraint_discovery"
NUM_MODES = len(GMM_MEANS)
DEFAULT_LR = 1e-5   # CAVIA latents have ||z|| ~ 0.013
DEFAULT_SUBSPACE_LR = 1e-2   # whitened coordinates, pool latents have |w_i| ~ 1
PRIOR_REFERENCE_LATENTS = 10000
EVAL_SEED_OFFSET = 7


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-id", default=FUNCTA_RUN,
                        help="Functa FM run providing the frozen SIREN, FM checkpoint and pool")
    parser.add_argument("--exclude-mode", type=int, default=2, choices=range(NUM_MODES),
                        help="GMM component removed from the target batch")

    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=None,
                        help=f"Adam step; default {DEFAULT_SUBSPACE_LR:g} on w, {DEFAULT_LR:g} on a full z_c")
    parser.add_argument("--subspace-var", type=float, default=0.999,
                        help="explained pool-latent variance kept by the z_c subspace; 0 = full 512-d")
    parser.add_argument("--z-reg-weight", type=float, default=0.0,
                        help="lambda on the per-dimension Mahalanobis distance to the pool latents")
    parser.add_argument("--cov-shrinkage", type=float, default=1e-3,
                        help="ridge added to Sigma, relative to its mean diagonal")

    parser.add_argument("--target-pool-size", type=int, default=1000000,
                        help="GMM draws filtered to the kept modes, then resampled every step")
    parser.add_argument("--eval-batch-size", type=int, default=16384,
                        help="fixed (x0, t, x1) batch scored at every snapshot")
    parser.add_argument("--metric-points", type=int, default=200000,
                        help="labelled GMM draws backing the per-mode inside fractions")
    parser.add_argument("--scatter-points", type=int, default=6000)
    parser.add_argument("--resolution", type=int, default=300)
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--viz-every", type=int, default=100)
    parser.add_argument("--strip-panels", type=int, default=6)
    parser.add_argument("--density-grid", type=int, default=200,
                        help="bins per side of the full-domain FM density of the final z_c")
    parser.add_argument("--density-samples", type=int, default=4000000)
    parser.add_argument("--density-step", type=float, default=0.01,
                        help="midpoint step of the forward sampling ODE")
    parser.add_argument("--hist-bins", type=int, default=120)
    parser.add_argument("--smooth-sigma", type=float, default=2.0)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"],
                        choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=150)

    parser.add_argument("--outdir", default=None,
                        help=f"default {OUTDIR}/latent_exclude<mode>")
    parser.add_argument("--figure-dir", default=None,
                        help=f"default {FIGURE_DIR}/latent_exclude<mode>")
    parser.add_argument("--smoke", action="store_true", help="shrink every knob for a fast test")
    parser.add_argument("--plot-only", action="store_true",
                        help="redraw from saved arrays; no checkpoint, no optimisation")
    return parser


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def variant_name(args) -> str:
    suffix = "_full512" if args.subspace_var <= 0 else ""
    return f"latent_exclude{args.exclude_mode}{suffix}" + ("_smoke" if args.smoke else "")


def make_lattice(scale: float, resolution: int, device: torch.device) -> torch.Tensor:
    """(R*R, 2) raw coordinates in ``meshgrid(..., indexing="ij")`` order."""
    axis = torch.linspace(-scale, scale, resolution)
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1).view(-1, 2).to(device)


def filtered_target_pool(num_draws: int, exclude_mode: int, device: torch.device) -> torch.Tensor:
    """GMM draws with every sample of ``exclude_mode`` removed, i.e. the 3-mode target."""
    x, labels = get_points(num_draws, device=device)
    return x[labels != exclude_mode]


def latent_prior(pool: dict[str, torch.Tensor], shrinkage: float) -> dict[str, torch.Tensor]:
    """Gaussian (mu, Sigma^{-1}) and PCA over both pool orientations, in float64 to dodge TF32."""
    Z = torch.cat([pool["z_pos"], pool["z_neg"]]).double()
    mu = Z.mean(dim=0)
    cov = torch.cov(Z.T)
    evals, evecs = torch.linalg.eigh(cov)
    cov += shrinkage * cov.diagonal().mean() * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    reference = Z[torch.randperm(Z.shape[0], device=Z.device)[:PRIOR_REFERENCE_LATENTS]]
    return {"mu": mu.float(), "precision": torch.linalg.inv(cov).float(), "reference": reference.float(),
            "evals": evals.flip(0).clamp_min(0.0), "evecs": evecs.flip(1)}


def latent_basis(prior: dict[str, torch.Tensor], explained: float) -> tuple[torch.Tensor, float]:
    """(d, k) basis U_k Lambda_k^{1/2} of the fewest PCs reaching ``explained`` variance."""
    cumulative = prior["evals"].cumsum(0) / prior["evals"].sum()
    k = int((cumulative < explained).sum()) + 1
    basis = prior["evecs"][:, :k] * prior["evals"][:k].sqrt()
    return basis.float(), float(cumulative[k - 1])


def mahalanobis(z: torch.Tensor, prior: dict[str, torch.Tensor]) -> torch.Tensor:
    """Per-dimension squared Mahalanobis distance; ~1 for a typical pool latent."""
    diff = z - prior["mu"]
    return ((diff @ prior["precision"]) * diff).sum(dim=-1) / diff.shape[-1]


def decode(siren, points: torch.Tensor, z: torch.Tensor, scale: float, chunk_size: int) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat([siren(points[s:s + chunk_size] / scale, z).squeeze(-1)
                          for s in range(0, points.shape[0], chunk_size)])


def mode_inside(values: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    """(NUM_MODES,) fraction of each component's samples on the feasible side (value <= 0)."""
    inside = (values <= 0).float()
    return np.asarray([float(inside[labels == m].mean()) for m in range(NUM_MODES)])


def fm_loss(model, z: torch.Tensor, x1: torch.Tensor, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Conditional FM loss with x_t = t x1 + (1 - t) x0 and target velocity x1 - x0."""
    x_t = t.unsqueeze(-1) * x1 + (1.0 - t.unsqueeze(-1)) * x0
    v = model(x_t, t, z.expand(x1.shape[0], -1))
    return ((v - (x1 - x0)) ** 2).mean(dim=-1).mean()


def sample_density(model, z: torch.Tensor, args, device: torch.device) -> np.ndarray:
    """(G, G) row-is-y density on +-PLANE_SCALE from ``--density-samples`` forward FM samples."""
    edges = np.linspace(-PLANE_SCALE, PLANE_SCALE, args.density_grid + 1)
    counts = np.zeros((args.density_grid, args.density_grid))
    for start in range(0, args.density_samples, args.chunk_size):
        n = min(args.chunk_size, args.density_samples - start)
        x = model.sample(n, z=z, step_size=args.density_step, device=device).cpu().numpy()
        counts += np.histogram2d(x[:, 1], x[:, 0], bins=[edges, edges])[0]
    return (counts / args.density_samples / (edges[1] - edges[0]) ** 2).astype(np.float32)


def strip_indices(num_snapshots: int, num_panels: int) -> np.ndarray:
    """Evenly spaced snapshot indices, always including the first and the last."""
    return np.unique(np.linspace(0, num_snapshots - 1, min(num_panels, num_snapshots)).round().astype(int))


def render_frame(root_fig: Path, step: int, field: np.ndarray, points: np.ndarray,
                 labels: np.ndarray, inside: np.ndarray, scale: float, args,
                 style: cd.DiscoveryStyle) -> Path:
    fig = cd.plot_discovery_frame(field, points, labels, args.exclude_mode, scale,
                                  f"$z_c$ optimisation, step {step}", mode_inside=inside, style=style)
    return save_encoder_figure(fig, root_fig / "frames" / f"step_{step:05d}",
                               formats=["png"], dpi=args.dpi)[0]


def render_summary(root_fig: Path, arrays: dict[str, np.ndarray], scale: float, args,
                   style: cd.DiscoveryStyle) -> list[Path]:
    steps, idx = arrays["snapshot_steps"], arrays["strip_indices"]
    inside = arrays["strip_inside"].astype(bool)
    fig = cd.plot_discovery_strip(
        arrays["fields"][idx], steps[idx], arrays["scatter_points"], arrays["scatter_labels"],
        args.exclude_mode, scale, inside_x=[arrays["metric_x"][row] for row in inside],
        target_x=arrays["target_x"], mode_inside=arrays["mode_inside"][idx],
        bins=args.hist_bins, style=style)
    written = save_encoder_figure(fig, root_fig / "strip", formats=args.formats, dpi=args.dpi)
    fig = cd.plot_likelihood_map(arrays["density"], PLANE_SCALE,
                                 float(arrays["likelihood_vmax"][0]))
    written += save_encoder_figure(fig, root_fig / "likelihood", formats=args.formats, dpi=300)
    fig = cd.plot_discovery_history(arrays["losses"], steps, arrays["eval_losses"],
                                    arrays["mode_inside"], args.exclude_mode, style=style)
    written += save_encoder_figure(fig, root_fig / "history", formats=args.formats, dpi=args.dpi)
    return written


def replot(args, root: Path, root_fig: Path, style: cd.DiscoveryStyle) -> int:
    record = json.loads((root / "metrics.json").read_text())
    names = ["snapshot_steps", "fields", "mode_inside", "eval_losses", "losses", "scatter_points",
             "scatter_labels", "strip_indices", "density", "strip_inside", "metric_x",
             "target_x", "likelihood_vmax"]
    arrays = {name: artifacts.load_array(root, name) for name in names}
    args.exclude_mode = record["exclude_mode"]
    written = [render_frame(root_fig, int(step), arrays["fields"][i], arrays["scatter_points"],
                            arrays["scatter_labels"], arrays["mode_inside"][i],
                            float(record["scale"]), args, style)
               for i, step in enumerate(arrays["snapshot_steps"])]
    written += render_summary(root_fig, arrays, float(record["scale"]), args, style)
    print(f"wrote {len(written)} figures under {root_fig}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        args.steps, args.viz_every, args.batch_size = 40, 10, 512
        args.target_pool_size, args.metric_points, args.eval_batch_size = 50000, 20000, 1024
        args.resolution, args.formats, args.density_grid = 120, ["png"], 100
        args.density_samples = 200000
    root = resolve_path(args.outdir or f"{OUTDIR}/{variant_name(args)}")
    root_fig = resolve_path(args.figure_dir or f"{FIGURE_DIR}/{variant_name(args)}")
    style = cd.get_style(smooth_sigma=args.smooth_sigma)
    if args.plot_only:
        return replot(args, root, root_fig, style)

    cfg = load_config(args.run_id)
    if cfg.scale != PLANE_SCALE:
        raise ValueError(f"the density figure spans +-{PLANE_SCALE}, run scale is {cfg.scale}")
    device = resolve_device()
    siren = load_siren(cfg, device)
    model = build_flow_matcher(cfg, siren, device)
    iteration = load_checkpoint(cfg, model, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if any(p.requires_grad for p in list(model.parameters()) + list(siren.parameters())):
        raise RuntimeError("FM or SIREN still has trainable parameters")

    prior = latent_prior(load_pool(cfg, device), args.cov_shrinkage)
    ref_norm = float(prior["reference"].norm(dim=-1).median())
    ref_maha = float(mahalanobis(prior["reference"], prior).median())

    if args.subspace_var > 0:
        basis, kept_var = latent_basis(prior, args.subspace_var)
        origin = prior["mu"]
    else:
        basis, kept_var = torch.eye(cfg.siren.latent_dim, device=device), 1.0
        origin = torch.zeros(cfg.siren.latent_dim, device=device)
    args.lr = args.lr or (DEFAULT_SUBSPACE_LR if args.subspace_var > 0 else DEFAULT_LR)
    subspace_dim = int(basis.shape[1])
    w = torch.zeros(subspace_dim, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([w], lr=args.lr)

    def latent() -> torch.Tensor:
        return origin + basis @ w

    set_seed(args.seed)
    target = filtered_target_pool(args.target_pool_size, args.exclude_mode, device)
    metric_x, metric_labels = get_points(args.metric_points, device=device)
    scatter_points = metric_x[:args.scatter_points].cpu().numpy()
    scatter_labels = metric_labels[:args.scatter_points].cpu().numpy()
    lattice = make_lattice(cfg.scale, args.resolution, device)

    gen = torch.Generator(device=device).manual_seed(args.seed + EVAL_SEED_OFFSET)
    eval_x1 = target[torch.randint(target.shape[0], (args.eval_batch_size,), device=device, generator=gen)]
    eval_x0 = torch.randn(eval_x1.shape, device=device, generator=gen)
    eval_t = torch.rand(eval_x1.shape[0], device=device, generator=gen)

    kept = [m for m in range(NUM_MODES) if m != args.exclude_mode]
    print(f"run {cfg.run_id} | iteration {iteration} | device {device} | lr {args.lr:g} | "
          f"z_c subspace {subspace_dim}-d ({kept_var:.4f} of pool variance)\n"
          f"target modes {kept} ({target.shape[0]} of {args.target_pool_size} draws) | "
          f"excluded mode {args.exclude_mode} | pool ||z|| median {ref_norm:.4f} | "
          f"pool Mahalanobis/d median {ref_maha:.3f}")

    snaps: dict[str, list] = {k: [] for k in ["snapshot_steps", "latents", "fields", "mode_inside",
                                              "eval_losses", "z_norm", "mahalanobis"]}
    losses: list[float] = []

    def snapshot(step: int) -> None:
        z = latent().detach()
        with torch.no_grad():
            eval_loss = float(fm_loss(model, z, eval_x1, eval_x0, eval_t))
            maha = float(mahalanobis(z, prior))
        field = decode(siren, lattice, z, cfg.scale, args.chunk_size).view(
            args.resolution, args.resolution).cpu().numpy()
        inside = mode_inside(decode(siren, metric_x, z, cfg.scale, args.chunk_size), metric_labels)
        for key, value in [("snapshot_steps", step), ("latents", z.cpu().numpy()), ("fields", field),
                           ("mode_inside", inside), ("eval_losses", eval_loss),
                           ("z_norm", float(z.norm())), ("mahalanobis", maha)]:
            snaps[key].append(value)
        ema = float(np.mean(losses[-args.viz_every:])) if losses else float("nan")
        print(f"step {step:5d} | train {ema:.4f} | eval {eval_loss:.4f} | inside "
              f"{np.round(inside, 3).tolist()} | ||z|| {float(z.norm()):.4f} | "
              f"maha/d {maha:.3f}", flush=True)
        render_frame(root_fig, step, field, scatter_points, scatter_labels, inside, cfg.scale,
                     args, style)

    for step in range(args.steps + 1):
        if step % args.viz_every == 0 or step == args.steps:
            snapshot(step)
        if step == args.steps:
            break
        x1 = target[torch.randint(target.shape[0], (args.batch_size,), device=device)]
        x0 = torch.randn_like(x1)
        t = torch.rand(args.batch_size, device=device)
        z_c = latent()
        fm = fm_loss(model, z_c, x1, x0, t)
        loss = fm + args.z_reg_weight * mahalanobis(z_c, prior) if args.z_reg_weight > 0 else fm
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(fm))

    idx = strip_indices(len(snaps["snapshot_steps"]), args.strip_panels)
    strip_inside = [(decode(siren, metric_x, torch.from_numpy(snaps["latents"][i]).to(device),
                            cfg.scale, args.chunk_size) <= 0).cpu().numpy() for i in idx]

    kept_weights = torch.tensor(GMM_WEIGHTS)[kept]
    target_peak = float(compute_gmm_density(
        means=torch.tensor(GMM_MEANS)[kept], covs=torch.tensor(GMM_COVS)[kept],
        weights=kept_weights / kept_weights.sum(), grid_size=args.density_grid).max())

    run_id = pin_baseline_run(root, "constraint_discovery", args, extra={"config_run_id": cfg.run_id})
    arrays = {k: np.asarray(v, dtype=np.float32) for k, v in snaps.items()}
    arrays["snapshot_steps"] = np.asarray(snaps["snapshot_steps"], dtype=np.int64)
    arrays.update(losses=np.asarray(losses, dtype=np.float32), scatter_points=scatter_points,
                  scatter_labels=scatter_labels, strip_indices=idx,
                  density=sample_density(model, latent().detach(), args, device),
                  strip_inside=np.stack(strip_inside).astype(np.uint8),
                  likelihood_vmax=np.asarray([target_peak], dtype=np.float32),
                  metric_x=metric_x[:, 0].cpu().numpy(), target_x=target[:, 0].cpu().numpy())
    artifacts.save_arrays(root, **arrays)
    artifacts.write_manifest(root, run_id=run_id, config_run_id=cfg.run_id, iteration=iteration,
                             exclude_mode=args.exclude_mode)
    final = {"mode_inside": snaps["mode_inside"][-1].tolist(), "eval_loss": snaps["eval_losses"][-1],
             "z_norm": snaps["z_norm"][-1], "mahalanobis": snaps["mahalanobis"][-1]}
    (root / "metrics.json").write_text(json.dumps({
        "run_id": run_id, "config_run_id": cfg.run_id, "iteration": iteration,
        "exclude_mode": args.exclude_mode, "target_modes": kept,
        "scale": cfg.scale, "lr": args.lr, "steps": args.steps, "batch_size": args.batch_size,
        "subspace_dim": subspace_dim, "subspace_explained_var": kept_var,
        "z_reg_weight": args.z_reg_weight, "reference_z_norm": ref_norm,
        "reference_mahalanobis": ref_maha, "initial_eval_loss": snaps["eval_losses"][0],
        "final": final}, indent=2))

    written = render_summary(root_fig, arrays, cfg.scale, args, style)
    print("\n".join(f"wrote {p}" for p in written))
    print(f"frames -> {root_fig / 'frames'} | arrays -> {artifacts.artifacts_dir(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

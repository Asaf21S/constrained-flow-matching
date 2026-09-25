# -*- coding: utf-8 -*-
"""Test-time discovery of a constraint that isolates 3 of the 4 GMM modes.

Both the Functa FM and its SIREN are frozen. The only trainable tensor is the constraint
itself, fitted by the standard conditional FM loss on a target batch drawn from the 3 kept
modes only, so gradients reach it exclusively through the frozen vector field:

    min_theta  E_{t, x0, x1 ~ p_target} || v_phi(x_t, t, z(theta)) - (x1 - x0) ||^2
               + lambda * (z - mu)^T Sigma^{-1} (z - mu) / d

``--param latent``  theta = z_c in R^512, initialised at 0 (the CAVIA meta-init), z(theta) = z_c.
``--param coeffs``  theta = C in R^{4x4}, z(theta) = CAVIA_15(tanh(P_{C/||C||_F})), i.e. the
                    FM loss is backpropagated through the unrolled SGD inner loop into C.

The optional Mahalanobis term uses the pool latents' Gaussian (mu, Sigma); its value is logged
at every snapshot regardless of lambda, as an off-manifold indicator for the frozen FM.

    frames/step_<k>.png   decoded boundary over the 4-mode scatter, every --viz-every steps
    strip.{png,pdf}       evenly spaced frames in one row
    history.{png,pdf}     FM loss and per-mode inside fraction against step

    sbatch scripts/run_discover_constraint.sh
    sbatch scripts/run_discover_constraint.sh --param coeffs
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

from constrained_fm.src.consts import GMM_MEANS
from constrained_fm.src.datasets.constraints import sample_valid_polynomials
from constrained_fm.src.datasets.functa_conditioning import sample_query_points
from constrained_fm.src.datasets.gmm_target import get_points
from constrained_fm.src.experiment import artifacts
from constrained_fm.src.experiment.config import REPO_ROOT
from constrained_fm.src.experiment.registry import load_config, pin_baseline_run
from constrained_fm.src.experiment.runtime import (build_flow_matcher, load_checkpoint, load_pool,
                                                   load_siren, resolve_device, set_seed)
from constrained_fm.src.geometry.polynomials import (compute_poly_features,
                                                     compute_poly_features_batched,
                                                     evaluate_poly_batched)
from constrained_fm.src.visualization import constraint_discovery as cd
from constrained_fm.src.visualization.siren_encoder import save_encoder_figure

FUNCTA_RUN = "siren-uniform-8d6375ab"
OUTDIR = "constrained_fm/baselines/constraint_discovery"
FIGURE_DIR = "constrained_fm/images/thesis_pool/constraint_discovery"
NUM_MODES = len(GMM_MEANS)
DEFAULT_LR = {"latent": 1e-5, "coeffs": 3e-3}   # CAVIA latents have ||z|| ~ 0.013; ||C||_F = 1
PRIOR_REFERENCE_LATENTS = 10000
EVAL_SEED_OFFSET = 7


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-id", default=FUNCTA_RUN,
                        help="Functa FM run providing the frozen SIREN, FM checkpoint and pool")
    parser.add_argument("--param", choices=["latent", "coeffs"], default="latent",
                        help="optimise z_c directly, or polynomial coefficients through CAVIA")
    parser.add_argument("--exclude-mode", type=int, default=2, choices=range(NUM_MODES),
                        help="GMM component removed from the target batch")

    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=None,
                        help=f"Adam step; default {DEFAULT_LR}")
    parser.add_argument("--z-reg-weight", type=float, default=0.0,
                        help="lambda on the per-dimension Mahalanobis distance to the pool latents")
    parser.add_argument("--cov-shrinkage", type=float, default=1e-3,
                        help="ridge added to Sigma, relative to its mean diagonal")
    parser.add_argument("--init-seed", type=int, default=0,
                        help="--param coeffs: seed of the valid cubic used as initial C")
    parser.add_argument("--query-points", type=int, default=None,
                        help="--param coeffs: CAVIA query points per step; default the run's extraction value")

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
    parser.add_argument("--smooth-sigma", type=float, default=2.0)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"],
                        choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=150)

    parser.add_argument("--outdir", default=None,
                        help=f"default {OUTDIR}/<param>_exclude<mode>")
    parser.add_argument("--figure-dir", default=None,
                        help=f"default {FIGURE_DIR}/<param>_exclude<mode>")
    parser.add_argument("--smoke", action="store_true", help="shrink every knob for a fast test")
    parser.add_argument("--plot-only", action="store_true",
                        help="redraw from saved arrays; no checkpoint, no optimisation")
    return parser


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def variant_name(args) -> str:
    return f"{args.param}_exclude{args.exclude_mode}" + ("_smoke" if args.smoke else "")


def make_lattice(scale: float, resolution: int, device: torch.device) -> torch.Tensor:
    """(R*R, 2) raw coordinates in ``meshgrid(..., indexing="ij")`` order."""
    axis = torch.linspace(-scale, scale, resolution)
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1).view(-1, 2).to(device)


def filtered_target_pool(num_draws: int, exclude_mode: int, device: torch.device) -> torch.Tensor:
    """GMM draws with every sample of ``exclude_mode`` removed, i.e. the 3-mode target."""
    x, labels = get_points(num_draws, device=device)
    return x[labels != exclude_mode]


def cavia_encode(siren, X: torch.Tensor, Y: torch.Tensor, lr: float, steps: int) -> torch.Tensor:
    """Differentiable ``extract_latents_batched``: same zero init, loss and step, with ``create_graph``.

    ``z_K = -lr * sum_k grad_z L(z_k; Y)``, so ``dz_K/dY`` exists and carries the FM gradient back to Y.
    """
    z = torch.zeros(X.shape[0], siren.latent_dim, device=X.device, requires_grad=True)
    for _ in range(steps):
        loss = ((siren(X, z).squeeze(-1) - Y) ** 2).mean(dim=1).sum()
        z = z - lr * torch.autograd.grad(loss, z, create_graph=True)[0]
    return z


class LatentParam:
    """theta = z_c, the decoded latent itself."""

    def __init__(self, latent_dim: int, device: torch.device):
        self.z = torch.zeros(latent_dim, device=device, requires_grad=True)

    def parameters(self) -> list[torch.Tensor]:
        return [self.z]

    def latent(self, queries: torch.Tensor | None = None) -> torch.Tensor:
        return self.z

    def coefficients(self) -> torch.Tensor | None:
        return None


class CoefficientParam:
    """theta = C_raw; the constraint is P_C with C = C_raw / ||C_raw||_F, encoded by unrolled CAVIA."""

    def __init__(self, C_init: torch.Tensor, siren, cfg, num_query: int):
        self.C_raw = C_init.clone().requires_grad_(True)
        self.siren, self.cfg, self.num_query = siren, cfg, num_query

    def parameters(self) -> list[torch.Tensor]:
        return [self.C_raw]

    def coefficients(self) -> torch.Tensor:
        return self.C_raw / torch.linalg.matrix_norm(self.C_raw)

    def fresh_queries(self) -> torch.Tensor:
        return sample_query_points(1, self.num_query, scale=self.cfg.scale,
                                   gmm_fraction=self.cfg.extraction.query_gmm_fraction,
                                   device=self.C_raw.device)

    def latent(self, queries: torch.Tensor | None = None) -> torch.Tensor:
        X = self.fresh_queries() if queries is None else queries
        x_pow, y_pow = compute_poly_features_batched(X, degree=self.cfg.degree, scale=self.cfg.scale)
        Y = torch.tanh(evaluate_poly_batched(x_pow, y_pow, self.coefficients().unsqueeze(0)))
        return cavia_encode(self.siren, X / self.cfg.scale, Y, lr=self.cfg.extraction.lr,
                            steps=self.cfg.extraction.steps)[0]


def latent_prior(pool: dict[str, torch.Tensor], shrinkage: float) -> dict[str, torch.Tensor]:
    """Gaussian (mu, Sigma^{-1}) over both pool orientations, in float64 to dodge TF32."""
    Z = torch.cat([pool["z_pos"], pool["z_neg"]]).double()
    mu = Z.mean(dim=0)
    cov = torch.cov(Z.T)
    cov += shrinkage * cov.diagonal().mean() * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    reference = Z[torch.randperm(Z.shape[0], device=Z.device)[:PRIOR_REFERENCE_LATENTS]]
    return {"mu": mu.float(), "precision": torch.linalg.inv(cov).float(), "reference": reference.float()}


def mahalanobis(z: torch.Tensor, prior: dict[str, torch.Tensor]) -> torch.Tensor:
    """Per-dimension squared Mahalanobis distance; ~1 for a typical pool latent."""
    diff = z - prior["mu"]
    return ((diff @ prior["precision"]) * diff).sum(dim=-1) / diff.shape[-1]


def decode(siren, points: torch.Tensor, z: torch.Tensor, scale: float, chunk_size: int) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat([siren(points[s:s + chunk_size] / scale, z).squeeze(-1)
                          for s in range(0, points.shape[0], chunk_size)])


def poly_values(points: torch.Tensor, C: torch.Tensor, degree: int, scale: float) -> torch.Tensor:
    """P_C(x) by elementwise contraction; TF32 matmuls misclassify points near the zero set."""
    x_pow, y_pow = compute_poly_features(points, degree=degree, scale=scale)
    return (x_pow.unsqueeze(-1) * C * y_pow.unsqueeze(-2)).sum(dim=(-1, -2))


def mode_inside(values: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    """(NUM_MODES,) fraction of each component's samples on the feasible side (value <= 0)."""
    inside = (values <= 0).float()
    return np.asarray([float(inside[labels == m].mean()) for m in range(NUM_MODES)])


def fm_loss(model, z: torch.Tensor, x1: torch.Tensor, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Conditional FM loss with x_t = t x1 + (1 - t) x0 and target velocity x1 - x0."""
    x_t = t.unsqueeze(-1) * x1 + (1.0 - t.unsqueeze(-1)) * x0
    v = model(x_t, t, z.expand(x1.shape[0], -1))
    return ((v - (x1 - x0)) ** 2).mean(dim=-1).mean()


def frame_title(step: int, param: str) -> str:
    return f"{'$z_c$' if param == 'latent' else '$C$'} optimisation, step {step}"


def render_frame(root_fig: Path, step: int, field: np.ndarray, points: np.ndarray,
                 labels: np.ndarray, inside: np.ndarray, poly_field: np.ndarray | None,
                 scale: float, args, style: cd.DiscoveryStyle) -> Path:
    fig = cd.plot_discovery_frame(field, points, labels, args.exclude_mode, scale,
                                  frame_title(step, args.param), mode_inside=inside,
                                  poly_field=poly_field, style=style)
    return save_encoder_figure(fig, root_fig / "frames" / f"step_{step:05d}",
                               formats=["png"], dpi=args.dpi)[0]


def render_summary(root_fig: Path, arrays: dict[str, np.ndarray], scale: float, args,
                   style: cd.DiscoveryStyle) -> list[Path]:
    steps = arrays["snapshot_steps"]
    idx = np.unique(np.linspace(0, len(steps) - 1, min(args.strip_panels, len(steps))).round().astype(int))
    poly = arrays.get("poly_fields")
    fig = cd.plot_discovery_strip(arrays["fields"][idx], steps[idx], arrays["scatter_points"],
                                  arrays["scatter_labels"], args.exclude_mode, scale,
                                  mode_inside=arrays["mode_inside"][idx],
                                  poly_fields=None if poly is None else poly[idx], style=style)
    written = save_encoder_figure(fig, root_fig / "strip", formats=args.formats, dpi=args.dpi)
    fig = cd.plot_discovery_history(arrays["losses"], steps, arrays["eval_losses"],
                                    arrays["mode_inside"], args.exclude_mode, style=style)
    written += save_encoder_figure(fig, root_fig / "history", formats=args.formats, dpi=args.dpi)
    return written


def replot(args, root: Path, root_fig: Path, style: cd.DiscoveryStyle) -> int:
    record = json.loads((root / "metrics.json").read_text())
    names = ["snapshot_steps", "fields", "mode_inside", "eval_losses", "losses",
             "scatter_points", "scatter_labels"]
    if record["param"] == "coeffs":
        names.append("poly_fields")
    arrays = {name: artifacts.load_array(root, name) for name in names}
    args.param, args.exclude_mode = record["param"], record["exclude_mode"]
    written = []
    for i, step in enumerate(arrays["snapshot_steps"]):
        written.append(render_frame(root_fig, int(step), arrays["fields"][i],
                                    arrays["scatter_points"], arrays["scatter_labels"],
                                    arrays["mode_inside"][i],
                                    arrays["poly_fields"][i] if "poly_fields" in arrays else None,
                                    float(record["scale"]), args, style))
    written += render_summary(root_fig, arrays, float(record["scale"]), args, style)
    print(f"wrote {len(written)} figures under {root_fig}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        args.steps, args.viz_every, args.batch_size = 40, 10, 512
        args.target_pool_size, args.metric_points, args.eval_batch_size = 50000, 20000, 1024
        args.resolution, args.formats = 120, ["png"]
    root = resolve_path(args.outdir or f"{OUTDIR}/{variant_name(args)}")
    root_fig = resolve_path(args.figure_dir or f"{FIGURE_DIR}/{variant_name(args)}")
    style = cd.get_style(smooth_sigma=args.smooth_sigma)
    if args.plot_only:
        return replot(args, root, root_fig, style)

    lr = args.lr if args.lr is not None else DEFAULT_LR[args.param]
    cfg = load_config(args.run_id)
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

    if args.param == "latent":
        param = LatentParam(cfg.siren.latent_dim, device)
    else:
        set_seed(args.init_seed)
        C_init = sample_valid_polynomials(1, degree=cfg.degree, scale=cfg.scale,
                                          min_area=cfg.pool.min_area, max_area=cfg.pool.max_area,
                                          device=device)[0]
        param = CoefficientParam(C_init, siren, cfg, args.query_points or cfg.extraction.points_per_shape)
    optimizer = torch.optim.Adam(param.parameters(), lr=lr)

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
    viz_queries = param.fresh_queries() if isinstance(param, CoefficientParam) else None

    kept = [m for m in range(NUM_MODES) if m != args.exclude_mode]
    print(f"run {cfg.run_id} | iteration {iteration} | device {device} | param {args.param} | lr {lr:g}\n"
          f"target modes {kept} ({target.shape[0]} of {args.target_pool_size} draws) | "
          f"excluded mode {args.exclude_mode} | pool ||z|| median {ref_norm:.4f} | "
          f"pool Mahalanobis/d median {ref_maha:.3f}")

    snaps: dict[str, list] = {k: [] for k in ["snapshot_steps", "latents", "fields", "mode_inside",
                                              "eval_losses", "z_norm", "mahalanobis",
                                              "coefficients", "poly_fields", "poly_mode_inside"]}
    losses: list[float] = []

    def snapshot(step: int) -> None:
        z = param.latent(viz_queries).detach()
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
        poly_field, poly_msg = None, ""
        C = param.coefficients()
        if C is not None:
            C = C.detach()
            with torch.no_grad():
                poly_field = poly_values(lattice, C, cfg.degree, cfg.scale).view(
                    args.resolution, args.resolution).cpu().numpy()
                poly_inside = mode_inside(poly_values(metric_x, C, cfg.degree, cfg.scale), metric_labels)
            snaps["coefficients"].append(C.cpu().numpy())
            snaps["poly_fields"].append(poly_field)
            snaps["poly_mode_inside"].append(poly_inside)
            poly_msg = f" | poly {np.round(poly_inside, 3).tolist()}"
        ema = float(np.mean(losses[-args.viz_every:])) if losses else float("nan")
        print(f"step {step:5d} | train {ema:.4f} | eval {eval_loss:.4f} | inside "
              f"{np.round(inside, 3).tolist()}{poly_msg} | ||z|| {float(z.norm()):.4f} | "
              f"maha/d {maha:.3f}", flush=True)
        render_frame(root_fig, step, field, scatter_points, scatter_labels, inside, poly_field,
                     cfg.scale, args, style)

    for step in range(args.steps + 1):
        if step % args.viz_every == 0 or step == args.steps:
            snapshot(step)
        if step == args.steps:
            break
        x1 = target[torch.randint(target.shape[0], (args.batch_size,), device=device)]
        x0 = torch.randn_like(x1)
        t = torch.rand(args.batch_size, device=device)
        z = param.latent()
        fm = fm_loss(model, z, x1, x0, t)
        loss = fm + args.z_reg_weight * mahalanobis(z, prior) if args.z_reg_weight > 0 else fm
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(fm))

    run_id = pin_baseline_run(root, "constraint_discovery", args, extra={
        "config_run_id": cfg.run_id, "lr_resolved": lr})
    arrays = {k: np.asarray(v, dtype=np.float32) for k, v in snaps.items() if v}
    arrays["snapshot_steps"] = np.asarray(snaps["snapshot_steps"], dtype=np.int64)
    arrays.update(losses=np.asarray(losses, dtype=np.float32), scatter_points=scatter_points,
                  scatter_labels=scatter_labels)
    artifacts.save_arrays(root, **arrays)
    artifacts.write_manifest(root, run_id=run_id, config_run_id=cfg.run_id, iteration=iteration,
                             param=args.param, exclude_mode=args.exclude_mode)
    final = {"mode_inside": snaps["mode_inside"][-1].tolist(), "eval_loss": snaps["eval_losses"][-1],
             "z_norm": snaps["z_norm"][-1], "mahalanobis": snaps["mahalanobis"][-1]}
    if snaps["poly_mode_inside"]:
        final["poly_mode_inside"] = snaps["poly_mode_inside"][-1].tolist()
        final["coefficients"] = snaps["coefficients"][-1].tolist()
    (root / "metrics.json").write_text(json.dumps({
        "run_id": run_id, "config_run_id": cfg.run_id, "iteration": iteration,
        "param": args.param, "exclude_mode": args.exclude_mode, "target_modes": kept,
        "scale": cfg.scale, "lr": lr, "steps": args.steps, "batch_size": args.batch_size,
        "z_reg_weight": args.z_reg_weight, "reference_z_norm": ref_norm,
        "reference_mahalanobis": ref_maha, "initial_eval_loss": snaps["eval_losses"][0],
        "final": final}, indent=2))

    written = render_summary(root_fig, arrays, cfg.scale, args, style)
    print("\n".join(f"wrote {p}" for p in written))
    print(f"frames -> {root_fig / 'frames'} | arrays -> {artifacts.artifacts_dir(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""Builds the feasibility-vs-fidelity figure for a single 2D constraint.

Five distributions over the same truncated GMM target:

    Ground Truth   rejection sampling, the distribution every method is trying to match
    ECI            inference-time projection onto {P(x) <= 0}
    HardFlow       inference-time gradient guidance towards {P(x) <= 0}
    Functa         ours; the constraint enters through the conditioning, not the trajectory
    Coefficients   conditions on the raw (4, 4) coefficient matrix instead of a functa latent

ECI and HardFlow are re-sampled here rather than loaded, because the benchmark run in
``eci_hardflow.py`` predates the artifact store and left no per-shape sample arrays behind.
Functa is re-sampled from ``runs/<run_id>/ckpt.pt`` at the same point count, so no panel is
visibly noisier than its neighbours; ``--functa-source cache`` instead reads the 10k samples
already in that run's artifact store. Either way the constraint is cross-checked against the
validation polynomial, so every panel is guaranteed to describe the same region.

All panels are drawn from the same number of points (``--num-samples``) with the same
``np.histogram2d`` -> ``imshow`` path, and scored on that same number against an independent
rejection-sampled ground-truth set of equal size. The ground-truth panel is therefore scored
GT-set-1 against GT-set-2, which is what a distributional metric means, and its SWD/MMD is
the finite-sample noise floor the others should be read against.

``--variants`` selects which panel compositions to draw; each lands in its own subfolder of
``--figure-dir``. Everything drawn is also written to ``<outdir>/poly<id>/artifacts/``, so
``--plot-only`` recomposes and restyles the figures with no GPU and no sampling.

    sbatch scripts/run_feasibility_fidelity.sh
    sbatch scripts/run_feasibility_fidelity.sh --poly-id 13
    python -m constrained_fm.scripts.plot_feasibility_fidelity --plot-only --variants 5panel_all
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from constrained_fm.src.consts import PLANE_SCALE, POLYNOMIAL_DEGREE
from constrained_fm.src.datasets.gmm_target import get_points
from constrained_fm.src.datasets.validation import get_validation_set
from constrained_fm.src.experiment import artifacts
from constrained_fm.src.experiment.config import REPO_ROOT
from constrained_fm.src.experiment.registry import load_config, pin_baseline_run
from constrained_fm.src.experiment.runtime import (build_flow_matcher, load_checkpoint,
                                                   load_siren, resolve_device, set_seed)
from constrained_fm.src.inference.constrained_samplers import (DEFAULT_CHUNK, DEFAULT_STEPS,
                                                               sample_eci, sample_hardflow)
from constrained_fm.src.inference.constraint_projection import DEFAULT_MARGIN
from constrained_fm.src.inference.evaluator import (evaluate_single_configuration,
                                                    run_evaluation_inference)
from constrained_fm.src.metrics.functa_fidelity import true_region_mask
from constrained_fm.src.models.constrained_poly import PolynomialConstrainedFM
from constrained_fm.src.models.unconstrained import UnconstrainedFM
from constrained_fm.src.visualization import feasibility as feas

BASE_CKPT = "constrained_fm/baselines/base_fm/ckpt.pt"
POLY_CKPT = "constrained_fm/baselines/poly_fm/ckpt.pt"
FUNCTA_RUN = "runs/siren-uniform-8d6375ab"
OUTDIR = "constrained_fm/baselines/feasibility_fidelity"
FIGURE_DIR = "constrained_fm/images/thesis_pool/feasibility_fidelity"

# Shape 86 of the validation set: mass 0.48, and the boundary cuts straight through a GMM
# mode, so both baselines pile a visible wall onto it while Functa keeps the interior intact.
DEFAULT_POLY_ID = 86

METHODS = ("gt", "eci", "hardflow", "coeff", "functa")
METHOD_LABELS = {
    "gt": "Ground Truth\n(rejection sampling)",
    "eci": "ECI\n(inference projection)",
    "hardflow": "HardFlow\n(inference guidance)",
    "coeff": "Constraint Params (ours)",
    "functa": "Functa (ours)",
}

# One figure per entry, each in its own subfolder. Ground truth must stay first: it sets the
# colour scale for every other panel and the shared y-limit of the profile row.
PANEL_VARIANTS: dict[str, tuple[str, ...]] = {
    "3panel_baselines": ("gt", "eci", "hardflow"),
    "4panel_functa": ("gt", "eci", "hardflow", "functa"),
    "4panel_coeff": ("gt", "eci", "hardflow", "coeff"),
    "5panel_all": ("gt", "eci", "hardflow", "coeff", "functa"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--poly-id", type=int, default=DEFAULT_POLY_ID,
                        help="index into the validation polynomial set")
    parser.add_argument("--num-samples", type=int, default=100000,
                        help="points per panel; identical for all four methods")
    parser.add_argument("--metric-samples", type=int, default=None,
                        help="points scored per method, and the size of the independent "
                             "ground-truth reference set; defaults to --num-samples")

    parser.add_argument("--ckpt", default=BASE_CKPT)
    parser.add_argument("--poly-ckpt", default=POLY_CKPT,
                        help="coefficient-conditioned flow matcher (train_poly_fm.py)")
    parser.add_argument("--coeff-step-size", type=float, default=0.05)
    parser.add_argument("--functa-run", default=FUNCTA_RUN,
                        help="run directory holding ckpt.pt and artifacts/")
    parser.add_argument("--functa-source", default="resample", choices=["resample", "cache"],
                        help="re-run the Functa sampler, or read its 10k cached samples")
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--time-dim", type=int, default=128)

    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--correction-loops", type=int, default=1)
    parser.add_argument("--projection-iters", type=int, default=16)
    parser.add_argument("--guidance-scale", type=float, default=100.0)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK)

    parser.add_argument("--gmm-pool-size", type=int, default=100000,
                        help="oversampling pool size for each rejection-sampling batch")
    parser.add_argument("--mmd-chunk", type=int, default=2048,
                        help="rows per block of the MMD kernel matrix")
    parser.add_argument("--degree", type=int, default=POLYNOMIAL_DEGREE)
    parser.add_argument("--scale", type=float, default=PLANE_SCALE)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--variants", nargs="+", default=list(PANEL_VARIANTS),
                        choices=list(PANEL_VARIANTS),
                        help="which panel compositions to render; one subfolder each")
    parser.add_argument("--highlight", nargs="*", default=["coeff", "functa"], choices=METHODS,
                        help="methods drawn with the accent frame")
    parser.add_argument("--style", nargs="+", default=["light"], choices=sorted(feas.STYLE_PRESETS),
                        help="one figure set per style")
    parser.add_argument("--bins", type=int, default=None, help="override histogram resolution")
    parser.add_argument("--cmap", default=None)
    parser.add_argument("--norm", default=None, choices=["linear", "power", "log"])
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--vmax-mode", default=None,
                        choices=["reference", "shared", "per_panel"])
    parser.add_argument("--metric-keys", nargs="*", default=None,
                        help="metrics printed under each panel; pass none to hide them")
    parser.add_argument("--boundary-profile", action="store_true",
                        help="add a signed-distance histogram row under the maps")
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=300)

    parser.add_argument("--outdir", default=OUTDIR)
    parser.add_argument("--figure-dir", default=FIGURE_DIR)
    parser.add_argument("--plot-only", action="store_true",
                        help="redraw from saved arrays; no checkpoint, no sampling")
    return parser


def resolve_style(args, name: str) -> feas.FeasibilityStyle:
    overrides = {key: getattr(args, key) for key in ("bins", "cmap", "norm", "gamma")
                 if getattr(args, key) is not None}
    if args.vmax_mode is not None:
        overrides["vmax_mode"] = args.vmax_mode
    if args.metric_keys is not None:
        overrides["metric_keys"] = tuple(args.metric_keys)
        overrides["show_metrics"] = bool(args.metric_keys)
    return feas.get_style(name, **overrides)


def artifact_root(args) -> Path:
    """One directory per shape, so runs on different polynomials do not overwrite each other."""
    root = Path(args.outdir)
    if not root.is_absolute():
        root = REPO_ROOT / root
    return root / f"poly{args.poly_id}"


def load_base_model(args, device: torch.device) -> UnconstrainedFM:
    path = REPO_ROOT / args.ckpt if not Path(args.ckpt).is_absolute() else Path(args.ckpt)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run scripts/run_base_fm.sh first")
    model = UnconstrainedFM(time_dim=args.time_dim, hidden_dim=args.hidden_dim,
                            num_blocks=args.num_blocks).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def rejection_sample(coeffs: torch.Tensor, num_samples: int, args,
                     device: torch.device) -> torch.Tensor:
    """Draws from the GMM and keeps only {P(x) <= 0}: the exact truncated target.

    Successive calls consume fresh draws, so calling it twice yields two independent sets.
    """
    batch = max(args.gmm_pool_size, num_samples * 2)
    kept, collected = [], 0
    while collected < num_samples:
        pool, _ = get_points(batch, device=device)
        inside = pool[true_region_mask(coeffs, pool, degree=args.degree, scale=args.scale)]
        kept.append(inside)
        collected += inside.shape[0]
    return torch.cat(kept, dim=0)[:num_samples]


def source_noise(val_set, num_samples: int, device: torch.device) -> torch.Tensor:
    """Flow-matching start points, extended past the validation set's stored 10k if needed.

    ``val_set["x0"]`` holds only ``n_train_samples_x0`` rows, so slicing it for a larger
    request silently returns fewer points and leaves the generated panels sparser than the
    rejection-sampled one. The stored prefix is reused so the first 10k match the benchmark.
    """
    x0 = val_set["x0"].to(device)
    if x0.shape[0] < num_samples:
        x0 = torch.cat([x0, torch.randn(num_samples - x0.shape[0], 2, device=device)], dim=0)
    return x0[:num_samples]


def load_functa_samples(args, coeffs: torch.Tensor) -> np.ndarray:
    """Cached Functa samples for this constraint, verified to be the same polynomial."""
    root = REPO_ROOT / args.functa_run
    cached_poly = torch.from_numpy(artifacts.load_array(root, "polynomials")[args.poly_id])
    if not torch.allclose(cached_poly.to(coeffs.device), coeffs, atol=1e-5):
        raise ValueError(
            f"{root} artifact polynomial {args.poly_id} does not match the validation set; "
            f"the cached samples describe a different constraint")
    return artifacts.load_array(root, "samples")[args.poly_id][:args.num_samples]


def sample_functa(args, val_polys: torch.Tensor, x0: torch.Tensor,
                  device: torch.device) -> np.ndarray:
    """Re-runs the trained Functa flow matcher for one constraint at the plotting point count.

    Latents are extracted for the whole validation set in one CAVIA call before indexing,
    which is exactly what ``eval_fm.py`` does, so this reproduces the cached samples rather
    than a differently-conditioned near-miss.
    """
    from constrained_fm.scripts.eval_fm import extract_validation_latents

    run_id = Path(args.functa_run).name
    cfg = load_config(run_id)
    siren = load_siren(cfg, device)
    model = build_flow_matcher(cfg, siren, device)
    iteration = load_checkpoint(cfg, model, device)
    model.eval()
    print(f"functa {run_id} at iteration {iteration}")

    z_val, _ = extract_validation_latents(siren, cfg, val_polys, device)
    z = z_val[args.poly_id:args.poly_id + 1]
    return run_evaluation_inference(model, x0, z=z, step_size=cfg.evaluation.step_size,
                                    device=device)


def rbf_mmd(x: torch.Tensor, y: torch.Tensor, gamma: float = 1.0,
            chunk: int = 2048, k_yy: float | None = None) -> tuple[float, float]:
    """Biased RBF-kernel MMD^2 over *all* points of both sets.

    ``metrics.distributional.compute_mmd`` caps both sets at 5000 points with an unseeded
    draw, which neither uses the N the figure claims nor returns the same number twice. The
    kernel means are accumulated blockwise here instead, so 100k x 100k fits in memory.
    Returns the MMD and the reusable E[k(y, y')] term.
    """
    def mean_kernel(a: torch.Tensor, b: torch.Tensor) -> float:
        total = 0.0
        for i in range(0, a.shape[0], chunk):
            d2 = torch.cdist(a[i:i + chunk], b).pow_(2)
            total += float(d2.mul_(-gamma).exp_().sum(dtype=torch.float64))
        return total / (a.shape[0] * b.shape[0])

    if k_yy is None:
        k_yy = mean_kernel(y, y)
    mmd = mean_kernel(x, x) + k_yy - 2.0 * mean_kernel(x, y)
    return max(0.0, mmd), k_yy


def sample_coeff(args, coeffs: torch.Tensor, x0: torch.Tensor,
                 device: torch.device) -> np.ndarray:
    """Flow matcher conditioned on the raw (4, 4) coefficient matrix rather than a latent.

    Built from ``train_poly_fm.py``'s defaults, which are the values the checkpoint was
    trained with; the model takes no config file of its own.
    """
    path = REPO_ROOT / args.poly_ckpt if not Path(args.poly_ckpt).is_absolute() \
        else Path(args.poly_ckpt)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run scripts/run_poly_fm.sh first")

    model = PolynomialConstrainedFM(degree=args.degree, hidden_dim=args.hidden_dim,
                                    scale_factor=args.scale).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    samples = run_evaluation_inference(model, x0, coeffs=coeffs.unsqueeze(0).detach().clone(),
                                       step_size=args.coeff_step_size, device=device)
    del model
    return np.asarray(samples, dtype=np.float32)


def score(samples, gt_reference: torch.Tensor, coeffs: torch.Tensor, args,
          device: torch.device, k_yy: float | None = None) -> tuple[dict[str, float], float]:
    """Success rate plus SWD / MMD against an independent rejection-sampled reference.

    ``gt_reference`` is a second, independently drawn set of exactly ``--metric-samples``
    points from the same truncated target. The ground-truth panel is therefore scored the
    same way as every other panel -- GT set 1 against GT set 2 -- which is the only reading
    of a distributional metric that makes sense, and gives the finite-sample noise floor the
    other three should be compared against.

    Scored on exactly the points the figure draws, so a caption can never disagree with its
    own panel -- the stored benchmark numbers came from a different seed and sample count.
    """
    subset = feas.to_numpy(samples)[:args.metric_samples]
    tensor = torch.as_tensor(subset, dtype=torch.float32, device=device)
    if tensor.shape[0] != gt_reference.shape[0]:
        raise ValueError(f"scoring {tensor.shape[0]} points against {gt_reference.shape[0]} "
                         f"reference points; both sets must hold --metric-samples")

    metrics = evaluate_single_configuration(tensor, x_true_pool=gt_reference, coeffs=coeffs,
                                            degree=args.degree, scale=args.scale,
                                            device=device)
    mmd, k_yy = rbf_mmd(tensor, gt_reference, chunk=args.mmd_chunk, k_yy=k_yy)
    return {"success_rate": float(metrics["success_rate"]),
            "swd": float(metrics["swd"]),
            "mmd": mmd}, k_yy


def render(samples_by_method: dict[str, np.ndarray], metrics_by_method: dict[str, dict],
           coeffs: torch.Tensor, args) -> list[Path]:
    figure_dir = Path(args.figure_dir)
    if not figure_dir.is_absolute():
        figure_dir = REPO_ROOT / figure_dir

    written = []
    for style_name in args.style:
        style = resolve_style(args, style_name)
        for variant in args.variants:
            methods = PANEL_VARIANTS[variant]
            missing = [m for m in methods if m not in samples_by_method]
            if missing:
                print(f"[skip] {variant}: no samples for {missing}")
                continue

            panels = [feas.Panel(label=METHOD_LABELS[m], samples=samples_by_method[m],
                                 metrics=metrics_by_method.get(m),
                                 highlight=(m in args.highlight))
                      for m in methods]
            fig = feas.plot_feasibility_row(panels, coeffs, style=style, degree=args.degree,
                                            scale=args.scale, show_profile=args.boundary_profile)

            stem = f"feasibility_fidelity_{variant}_poly{args.poly_id}"
            if style_name != "light":
                stem = f"{stem}_{style_name}"
            target = figure_dir / variant
            target.mkdir(parents=True, exist_ok=True)
            for suffix in args.formats:
                path = target / f"{stem}.{suffix}"
                fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
                written.append(path)
            plt.close(fig)

    return written


def replot(args) -> int:
    """Redraws from the artifact store. Loads no checkpoint and integrates nothing."""
    root = artifact_root(args)
    record = json.loads((root / "metrics.json").read_text())
    if record["poly_id"] != args.poly_id:
        raise ValueError(f"saved arrays are for poly {record['poly_id']}, not {args.poly_id}; "
                         f"re-run without --plot-only")

    coeffs = torch.from_numpy(artifacts.load_array(root, "polynomial"))
    samples = {m: artifacts.load_array(root, f"samples_{m}")
               for m in METHODS if artifacts.has_array(root, f"samples_{m}")}

    written = render(samples, record["metrics"], coeffs, args)
    print("\n".join(f"wrote {p}" for p in written))
    return 0 if written else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.metric_samples is None:
        args.metric_samples = args.num_samples
    if args.plot_only:
        return replot(args)

    device = resolve_device()
    set_seed(args.seed)

    val_set = get_validation_set(device=device)
    val_polys = val_set["polynomials"][:100].to(device)
    coeffs = val_polys[args.poly_id]
    x0 = source_noise(val_set, args.num_samples, device)

    model = load_base_model(args, device)
    print(f"device {device} | poly {args.poly_id} | {args.num_samples} samples per panel")

    samples: dict[str, np.ndarray] = {}
    samples["gt"] = rejection_sample(coeffs, args.num_samples, args, device).cpu().numpy()
    samples["eci"] = sample_eci(model, x0, coeffs, degree=args.degree, scale=args.scale,
                                steps=args.steps, correction_loops=args.correction_loops,
                                margin=args.margin, projection_iters=args.projection_iters,
                                chunk_size=args.chunk_size).detach().cpu().numpy()
    samples["hardflow"] = sample_hardflow(model, x0, coeffs, degree=args.degree,
                                          scale=args.scale, steps=args.steps,
                                          guidance_scale=args.guidance_scale,
                                          margin=args.margin,
                                          chunk_size=args.chunk_size).detach().cpu().numpy()
    del model

    if args.functa_source == "resample":
        samples["functa"] = np.asarray(sample_functa(args, val_polys, x0, device),
                                       dtype=np.float32)
    else:
        samples["functa"] = load_functa_samples(args, coeffs)

    samples["coeff"] = sample_coeff(args, coeffs, x0, device)

    counts = {m: int(s.shape[0]) for m, s in samples.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"panels must be drawn from equal sample counts, got {counts}")

    # Second, independent draw from the same truncated target: this is what the GT panel is
    # scored against, so its SWD/MMD measure sampling noise rather than a set against itself.
    gt_reference = rejection_sample(coeffs, args.metric_samples, args, device)
    print(f"scoring {args.metric_samples} points per method against an independent "
          f"{gt_reference.shape[0]}-point ground-truth set")

    metrics, k_yy = {}, None
    for method, points in samples.items():
        metrics[method], k_yy = score(points, gt_reference, coeffs, args, device, k_yy=k_yy)

    root = artifact_root(args)
    run_id = pin_baseline_run(root, "feasibility_fidelity", args,
                              extra={"base_ckpt": args.ckpt, "poly_ckpt": args.poly_ckpt,
                                     "functa_run": args.functa_run})
    artifacts.save_arrays(root, polynomial=coeffs,
                          **{f"samples_{m}": s for m, s in samples.items()})
    artifacts.write_manifest(root, run_id=run_id, poly_id=args.poly_id,
                             degree=args.degree, scale=args.scale, methods=list(samples))
    (root / "metrics.json").write_text(json.dumps({
        "run_id": run_id,
        "poly_id": args.poly_id,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "num_samples": args.num_samples,
        "metric_samples": args.metric_samples,
        "functa_source": args.functa_source,
        "scored_against": (f"an independent {args.metric_samples}-point rejection-sampled draw "
                           f"from the same truncated GMM; MMD uses every point of both sets"),
        "metrics": metrics,
    }, indent=2))

    print(f"\n### {run_id}")
    for method in METHODS:
        print(f"{METHOD_LABELS[method].splitlines()[0]:<16} "
              f"{feas.format_metrics(metrics[method], ('success_rate', 'swd', 'mmd'))}")

    written = render(samples, metrics, coeffs, args)
    print("\n".join(f"wrote {p}" for p in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

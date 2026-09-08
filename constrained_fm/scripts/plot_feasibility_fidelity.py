# -*- coding: utf-8 -*-
"""Builds the feasibility-vs-fidelity figure for a single 2D constraint.

Four distributions over the same truncated GMM target:

    Ground Truth   rejection sampling, the distribution every method is trying to match
    ECI            inference-time projection onto {P(x) <= 0}
    HardFlow       inference-time gradient guidance towards {P(x) <= 0}
    Functa         ours; the constraint enters through the conditioning, not the trajectory

ECI and HardFlow are re-sampled here rather than loaded, because the benchmark run in
``eci_hardflow.py`` predates the artifact store and left no per-shape sample arrays behind.
Functa is re-sampled from ``runs/<run_id>/ckpt.pt`` at the same point count, so no panel is
visibly noisier than its neighbours; ``--functa-source cache`` instead reads the 10k samples
already in that run's artifact store. Either way the constraint is cross-checked against the
validation polynomial, so the four panels are guaranteed to describe the same region.

Panels are drawn from ``--num-samples`` points but scored on a ``--metric-samples`` prefix, so
the captions stay comparable to the numbers in the benchmark tables while the density maps
get enough points to look like densities rather than confetti.

Everything drawn is also written to ``<outdir>/poly<id>/artifacts/``, so ``--plot-only`` restyles
the figure with no GPU and no sampling. ``--style`` takes several names at once and drops each
figure set into its own subfolder of ``--figure-dir``.

    sbatch scripts/run_feasibility_fidelity.sh
    sbatch scripts/run_feasibility_fidelity.sh --poly-id 13 --style light dark
    python -m constrained_fm.scripts.plot_feasibility_fidelity --plot-only --style log
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
from constrained_fm.src.models.unconstrained import UnconstrainedFM
from constrained_fm.src.visualization import feasibility as feas

BASE_CKPT = "constrained_fm/baselines/base_fm/ckpt.pt"
FUNCTA_RUN = "runs/siren-uniform-8d6375ab"
OUTDIR = "constrained_fm/baselines/feasibility_fidelity"
FIGURE_DIR = "constrained_fm/images/thesis_pool/feasibility_fidelity"

# Shape 86 of the validation set: mass 0.48, and the boundary cuts straight through a GMM
# mode, so both baselines pile a visible wall onto it while Functa keeps the interior intact.
DEFAULT_POLY_ID = 86

METHODS = ("gt", "eci", "hardflow", "functa")
METHOD_LABELS = {
    "gt": "Ground Truth\n(rejection sampling)",
    "eci": "ECI\n(inference projection)",
    "hardflow": "HardFlow\n(inference guidance)",
    "functa": "Functa (ours)\n(constrained conditioning)",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--poly-id", type=int, default=DEFAULT_POLY_ID,
                        help="index into the validation polynomial set")
    parser.add_argument("--num-samples", type=int, default=100000,
                        help="points per panel; the density maps need far more than the metrics")
    parser.add_argument("--metric-samples", type=int, default=10000,
                        help="prefix of each panel actually scored, matching the benchmark tables")

    parser.add_argument("--ckpt", default=BASE_CKPT)
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
                        help="reference pool the distributional metrics are scored against")
    parser.add_argument("--degree", type=int, default=POLYNOMIAL_DEGREE)
    parser.add_argument("--scale", type=float, default=PLANE_SCALE)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--panels-3", nargs=3, default=["gt", "eci", "hardflow"], choices=METHODS,
                        help="columns of the 1x3 figure")
    parser.add_argument("--panels-4", nargs=4, default=list(METHODS), choices=METHODS,
                        help="columns of the 1x4 figure")
    parser.add_argument("--style", nargs="+", default=["dark"], choices=sorted(feas.STYLE_PRESETS),
                        help="one figure set per style, each in its own subfolder")
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
    """Draws from the GMM and keeps only {P(x) <= 0}: the exact truncated target."""
    kept, collected = [], 0
    while collected < num_samples:
        pool, _ = get_points(max(num_samples * 4, 50000), device=device)
        inside = pool[true_region_mask(coeffs, pool, degree=args.degree, scale=args.scale)]
        kept.append(inside)
        collected += inside.shape[0]
    return torch.cat(kept, dim=0)[:num_samples]


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


def score(samples, gmm_pool: torch.Tensor, coeffs: torch.Tensor, args,
          device: torch.device) -> dict[str, float]:
    """Success rate plus SWD / MMD / JSD against the rejection-sampled truncated target.

    Scored on exactly the points the figure draws, so a caption can never disagree with its
    own panel -- the stored benchmark numbers came from a different seed and sample count.
    """
    subset = feas.to_numpy(samples)[:args.metric_samples]
    tensor = torch.as_tensor(subset, dtype=torch.float32, device=device)
    metrics = evaluate_single_configuration(tensor, x_true_pool=gmm_pool, coeffs=coeffs,
                                            degree=args.degree, scale=args.scale,
                                            device=device)
    return {key: float(value) for key, value in metrics.items()}


def render(samples_by_method: dict[str, np.ndarray], metrics_by_method: dict[str, dict],
           coeffs: torch.Tensor, args) -> list[Path]:
    figure_dir = Path(args.figure_dir)
    if not figure_dir.is_absolute():
        figure_dir = REPO_ROOT / figure_dir

    written = []
    for style_name in args.style:
        style = resolve_style(args, style_name)
        target = figure_dir / style_name
        for name, methods in (("3panel", args.panels_3), ("4panel", args.panels_4)):
            missing = [m for m in methods if m not in samples_by_method]
            if missing:
                print(f"[skip] {name}: no samples for {missing}")
                continue

            panels = [feas.Panel(label=METHOD_LABELS[m], samples=samples_by_method[m],
                                 metrics=metrics_by_method.get(m), highlight=(m == "functa"))
                      for m in methods]
            fig = feas.plot_feasibility_row(panels, coeffs, style=style, degree=args.degree,
                                            scale=args.scale, show_profile=args.boundary_profile)

            stem = f"feasibility_fidelity_{name}_poly{args.poly_id}"
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
    if args.plot_only:
        return replot(args)

    device = resolve_device()
    set_seed(args.seed)

    val_set = get_validation_set(device=device)
    val_polys = val_set["polynomials"][:100].to(device)
    coeffs = val_polys[args.poly_id]
    x0 = val_set["x0"][:args.num_samples].to(device)
    gmm_pool, _ = get_points(args.gmm_pool_size, device=device)

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

    metrics = {m: score(s, gmm_pool, coeffs, args, device) for m, s in samples.items()}

    root = artifact_root(args)
    run_id = pin_baseline_run(root, "feasibility_fidelity", args,
                              extra={"base_ckpt": args.ckpt, "functa_run": args.functa_run})
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
        "scored_against": "rejection-sampled truncated GMM from the shared reference pool",
        "metrics": metrics,
    }, indent=2))

    print(f"\n### {run_id}")
    for method in METHODS:
        print(f"{METHOD_LABELS[method].splitlines()[0]:<16} "
              f"{feas.format_metrics(metrics[method], ('success_rate', 'swd', 'mmd', 'jsd'))}")

    written = render(samples, metrics, coeffs, args)
    print("\n".join(f"wrote {p}" for p in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

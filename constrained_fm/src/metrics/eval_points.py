# -*- coding: utf-8 -*-
"""Frozen ground-truth points shared by every exact-likelihood (NLL / KLD) evaluation.

NLL and KLD are means over a finite sample of constraint-satisfying GT points, so unless
every baseline scores the *same* sample, part of the gap between two models is sampling
noise rather than model quality. Before this module the points were subsampled with a bare
``torch.randperm`` off the global RNG: its state at that call depends on how many random
numbers the surrounding script happened to consume beforehand (model init, CAVIA
extraction, ODE sampling), so no two scripts -- and no two runs of the same script with a
different architecture -- ever scored the same points.

The sample is therefore built once, cached next to the validation set, and handed to every
baseline. The build is itself fully deterministic (fixed pool seed plus a per-polynomial
generator, all on the CPU so the result is device-independent), so a deleted cache is
reproduced bit-for-bit.

The per-constraint mass is cached alongside the points because KLD subtracts the truncated
reference entropy ``log p_gmm(x) - log(mass)``; sharing the points but letting each script
re-estimate ``mass`` from its own GMM pool would leave a per-model offset in the KLD.

Build (optional -- callers build on cache miss automatically)::

    python -m constrained_fm.src.metrics.eval_points --rebuild
"""

from __future__ import annotations

import hashlib
import os

import torch

from constrained_fm.src.consts import POLYNOMIAL_DEGREE, PLANE_SCALE
from constrained_fm.src.datasets.gmm_target import get_points
from constrained_fm.src.datasets.validation import get_validation_set
from constrained_fm.src.geometry.polynomials import compute_poly_features

NLL_EVAL_SET_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "benchmark", "nll_eval_points.pt"))

# Frozen by definition: changing any of these invalidates every previously reported NLL/KLD,
# so they are constants here rather than per-run config knobs.
NLL_EVAL_SEED = 42
NLL_EVAL_POOL_SIZE = 100_000
NLL_EVAL_MAX_POINTS = 5000
NLL_EVAL_VERSION = 1


def deterministic_subset(x: torch.Tensor, num_points: int, seed: int) -> torch.Tensor:
    """A reproducible ``num_points`` subset of x, independent of the global RNG state."""
    if num_points <= 0 or x.shape[0] <= num_points:
        return x
    generator = torch.Generator().manual_seed(seed)
    idx = torch.randperm(x.shape[0], generator=generator)[:num_points]
    return x[idx.to(x.device)]


def _digest(polys: torch.Tensor) -> str:
    flat = polys.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return hashlib.sha256(flat.numpy().tobytes()).hexdigest()[:16]


def _seeded_gmm_pool(pool_size: int, seed: int) -> torch.Tensor:
    """GMM pool drawn on the CPU under a forked RNG, so it never depends on the caller."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        pool, _ = get_points(batch_size=pool_size, device="cpu")
    return pool


def build_nll_eval_set(polys: torch.Tensor, degree: int = POLYNOMIAL_DEGREE,
                       scale: float = PLANE_SCALE, pool_size: int = NLL_EVAL_POOL_SIZE,
                       max_points: int = NLL_EVAL_MAX_POINTS,
                       seed: int = NLL_EVAL_SEED) -> dict:
    """Rejection-samples the shared evaluation points for every polynomial in ``polys``."""
    pool = _seeded_gmm_pool(pool_size, seed)
    x_pow, y_pow = compute_poly_features(pool, degree=degree, scale=scale)

    points, mass, available = [], [], []
    for i in range(polys.shape[0]):
        C = polys[i].detach().to(device="cpu", dtype=torch.float32)
        valid = pool[torch.einsum("ni,ij,nj->n", x_pow, C, y_pow) <= 0]
        mass.append(valid.shape[0] / pool.shape[0])
        available.append(valid.shape[0])
        # Per-polynomial seed: adding a polynomial never perturbs the points of the others.
        points.append(deterministic_subset(valid, max_points, seed + i).contiguous())

    return {
        "version": NLL_EVAL_VERSION,
        "seed": seed,
        "pool_size": pool_size,
        "max_points": max_points,
        "degree": degree,
        "scale": scale,
        "poly_digest": _digest(polys),
        "points": points,
        "mass": torch.tensor(mass, dtype=torch.float64),
        "available": torch.tensor(available, dtype=torch.long),
    }


def load_nll_eval_set(num_points: int = NLL_EVAL_MAX_POINTS, degree: int = POLYNOMIAL_DEGREE,
                      scale: float = PLANE_SCALE, device=None, path: str = NLL_EVAL_SET_PATH,
                      rebuild: bool = False) -> dict:
    """The shared NLL points and masses for the static validation polynomials.

    Indices line up with ``get_validation_set()["polynomials"]``, so a script evaluating the
    first ``k`` polynomials just slices the returned lists. Asking for fewer than the cached
    ``max_points`` takes a prefix, which stays a shared set across baselines.
    """
    polys = get_validation_set(device="cpu")["polynomials"].to(dtype=torch.float32)
    digest = _digest(polys)

    cached = None
    if os.path.exists(path) and not rebuild:
        cached = torch.load(path, map_location="cpu", weights_only=False)
        stale = (cached.get("version") != NLL_EVAL_VERSION
                 or cached.get("poly_digest") != digest
                 or cached.get("degree") != degree
                 or cached.get("scale") != scale)
        if stale:
            raise RuntimeError(
                f"cached NLL evaluation set at {path} does not match the current benchmark "
                f"(version/digest/degree/scale mismatch). Delete it or rerun with --rebuild "
                f"to regenerate; note that this changes every reported NLL/KLD.")

    if cached is None:
        print(f"Building shared NLL evaluation set ({polys.shape[0]} polynomials, "
              f"{NLL_EVAL_MAX_POINTS} points each)...")
        cached = build_nll_eval_set(polys, degree=degree, scale=scale)
        tmp = f"{path}.tmp"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(cached, tmp)
        os.replace(tmp, path)
        print(f"Saved shared NLL evaluation set to '{path}'.")

    if num_points > cached["max_points"]:
        raise ValueError(
            f"requested {num_points} NLL points but the shared set caches "
            f"{cached['max_points']}; raise NLL_EVAL_MAX_POINTS and rebuild.")

    return {
        "points": [p[:num_points].to(device) for p in cached["points"]],
        "mass": cached["mass"].to(device=device, dtype=torch.float64),
        "available": cached["available"],
        "num_points": num_points,
        "seed": cached["seed"],
        "pool_size": cached["pool_size"],
        "poly_digest": cached["poly_digest"],
        "path": path,
    }


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="regenerate even if cached")
    args = parser.parse_args()

    eval_set = load_nll_eval_set(rebuild=args.rebuild)
    short = eval_set["available"].min().item()
    print(f"path        {eval_set['path']}")
    print(f"digest      {eval_set['poly_digest']} | pool {eval_set['pool_size']} | "
          f"seed {eval_set['seed']}")
    print(f"points      {eval_set['num_points']} per polynomial "
          f"({len(eval_set['points'])} polynomials)")
    print(f"mass        min {eval_set['mass'].min():.4f} | "
          f"median {eval_set['mass'].median():.4f}")
    if short < eval_set["num_points"]:
        print(f"warning: the smallest constraint only has {short} valid pool points")
    return 0


__all__ = ["NLL_EVAL_SET_PATH", "NLL_EVAL_SEED", "NLL_EVAL_POOL_SIZE", "NLL_EVAL_MAX_POINTS",
           "deterministic_subset", "build_nll_eval_set", "load_nll_eval_set"]


if __name__ == "__main__":
    raise SystemExit(_main())

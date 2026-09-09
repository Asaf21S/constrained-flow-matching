# -*- coding: utf-8 -*-
"""The v1k benchmark: 1000 polynomial constraints stratified uniformly over GMM mass.

The 100-polynomial set draws constraints straight from the area-ratio rejection sampler, so
its mass histogram inherits the sampler's prior and thins out at both ends. Every figure in
this pipeline reads a metric *as a function of* the constraint's mass, and a trend line is
only as trustworthy as the number of constraints in its bin, so this set instead fills a
fixed number of equal-width mass bins to the same depth.

Mass here is the exact Monte Carlo estimate against one fixed million-point GMM pool rather
than the sampler's 10k proxy: shared across all 1000 constraints, so masses are mutually
consistent, and drawn under a forked RNG, so the whole set is rebuildable bit-for-bit.

Build (heavy, GPU)::

    sbatch scripts/run_val1k_build.sh
"""

from __future__ import annotations

import hashlib
import os

import torch

from constrained_fm.src.consts import (PLANE_SCALE, POLYNOMIAL_DEGREE, VAL1K_MASS_BINS,
                                       VAL1K_MAX_MASS, VAL1K_MC_POOL_SIZE, VAL1K_MIN_MASS,
                                       VAL1K_NUM_POLYS, VAL1K_NUM_X0, VAL1K_SEED, VAL1K_SET_PATH,
                                       VAL1K_VERSION, VALIDATION_SET_PATH)
from constrained_fm.src.datasets.constraints import sample_valid_polynomials
from constrained_fm.src.datasets.gmm_target import get_points
from constrained_fm.src.geometry.polynomials import compute_poly_features

# Poly-major chunking for the mass einsum; 32 x 1e6 floats is ~128 MB of intermediates.
MASS_CHUNK = 32
PROXY_POOL_SIZE = 20000


def seeded_gmm_pool(pool_size: int, seed: int, device=None) -> torch.Tensor:
    """A GMM pool drawn under a forked RNG, so it never depends on the caller's RNG state."""
    with torch.random.fork_rng(devices=[] if device in (None, "cpu") else [device]):
        torch.manual_seed(seed)
        pool, _ = get_points(batch_size=pool_size, device=device)
    return pool


def poly_digest(polys: torch.Tensor) -> str:
    flat = polys.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return hashlib.sha256(flat.numpy().tobytes()).hexdigest()[:16]


def monte_carlo_masses(polys: torch.Tensor, pool: torch.Tensor | None = None,
                       degree: int = POLYNOMIAL_DEGREE, scale: float = PLANE_SCALE,
                       chunk_size: int = MASS_CHUNK,
                       features: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
    """Fraction of ``pool`` satisfying P(x) <= 0, per polynomial, in float64.

    Returns (B,) on the CPU. The binomial standard error at a million points is ~0.03% at
    mass 0.1, i.e. below the width of any plotted mass bin.
    """
    x_pow, y_pow = features if features is not None else compute_poly_features(
        pool, degree=degree, scale=scale)
    masses = []
    for start in range(0, polys.shape[0], chunk_size):
        C = polys[start:start + chunk_size]
        # b = polynomial, n = pool point, i/j = polynomial degrees
        inside = torch.einsum("ni,bij,nj->bn", x_pow, C, y_pow) <= 0
        masses.append(inside.to(torch.float64).mean(dim=1).cpu())
    return torch.cat(masses, dim=0)


def _bin_quota(num_polys: int, num_bins: int) -> list[int]:
    """Equal depth per bin, with the remainder spread over the leading bins."""
    base, remainder = divmod(num_polys, num_bins)
    return [base + (1 if i < remainder else 0) for i in range(num_bins)]


def generate_stratified_polynomials(pool: torch.Tensor, num_polys: int = VAL1K_NUM_POLYS,
                                    min_mass: float = VAL1K_MIN_MASS,
                                    max_mass: float = VAL1K_MAX_MASS,
                                    num_bins: int = VAL1K_MASS_BINS,
                                    degree: int = POLYNOMIAL_DEGREE, scale: float = PLANE_SCALE,
                                    candidate_batch: int = 512, max_rounds: int = 2000,
                                    device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Fills ``num_bins`` equal-width mass bins to equal depth by rejection sampling.

    Candidates are pre-filtered on the cheap proxy area ratio before paying for the exact
    million-point mass, and are then binned by that exact mass. Returns the accepted
    coefficients (N, degree+1, degree+1) and their masses (N,), ordered bin by bin.
    """
    edges = torch.linspace(min_mass, max_mass, num_bins + 1, dtype=torch.float64)
    quota = _bin_quota(num_polys, num_bins)
    accepted: list[list[torch.Tensor]] = [[] for _ in range(num_bins)]
    accepted_mass: list[list[float]] = [[] for _ in range(num_bins)]

    pool_features = compute_poly_features(pool, degree=degree, scale=scale)
    proxy_x_pow = pool_features[0][:PROXY_POOL_SIZE]
    proxy_y_pow = pool_features[1][:PROXY_POOL_SIZE]

    for round_idx in range(max_rounds):
        remaining = sum(q - len(a) for q, a in zip(quota, accepted))
        if remaining <= 0:
            break

        candidates = sample_valid_polynomials(candidate_batch, degree=degree, scale=scale,
                                              proxy_x_pow=proxy_x_pow, proxy_y_pow=proxy_y_pow,
                                              min_area=min_mass, max_area=max_mass, device=device)
        masses = monte_carlo_masses(candidates, degree=degree, scale=scale, features=pool_features)
        # right=True keeps the closed upper edge from spilling into a non-existent bin.
        bins = torch.bucketize(masses, edges, right=True) - 1
        bins = bins.clamp(0, num_bins - 1)

        for i in range(candidates.shape[0]):
            b = int(bins[i])
            m = float(masses[i])
            if not (min_mass <= m <= max_mass) or len(accepted[b]) >= quota[b]:
                continue
            accepted[b].append(candidates[i].detach().cpu())
            accepted_mass[b].append(m)

        if round_idx % 20 == 0:
            filled = sum(len(a) for a in accepted)
            print(f"round {round_idx:4d} | accepted {filled}/{num_polys} | "
                  f"emptiest bin {min(len(a) for a in accepted)}", flush=True)
    else:
        shortfall = {i: quota[i] - len(accepted[i]) for i in range(num_bins) if len(accepted[i]) < quota[i]}
        raise RuntimeError(f"stratification did not converge in {max_rounds} rounds; "
                           f"bins still short: {shortfall}")

    polys = torch.stack([c for bin_polys in accepted for c in bin_polys], dim=0)
    mass = torch.tensor([m for bin_mass in accepted_mass for m in bin_mass], dtype=torch.float64)
    return polys, mass


def build_validation_set_v1k(num_polys: int = VAL1K_NUM_POLYS, min_mass: float = VAL1K_MIN_MASS,
                             max_mass: float = VAL1K_MAX_MASS, num_bins: int = VAL1K_MASS_BINS,
                             pool_size: int = VAL1K_MC_POOL_SIZE, num_x0: int = VAL1K_NUM_X0,
                             degree: int = POLYNOMIAL_DEGREE, scale: float = PLANE_SCALE,
                             seed: int = VAL1K_SEED, device=None) -> dict:
    """Generates the whole benchmark under a forked RNG; everything here is a pure function of ``seed``."""
    with torch.random.fork_rng(devices=[] if device in (None, "cpu") else [device]):
        torch.manual_seed(seed)

        print(f"Drawing the {pool_size} point Monte Carlo mass pool (seed {seed})...", flush=True)
        pool = seeded_gmm_pool(pool_size, seed, device=device)

        print(f"Stratifying {num_polys} polynomials over {num_bins} mass bins "
              f"in [{min_mass}, {max_mass}]...", flush=True)
        polys, mass = generate_stratified_polynomials(
            pool, num_polys=num_polys, min_mass=min_mass, max_mass=max_mass, num_bins=num_bins,
            degree=degree, scale=scale, device=device)

        # Bin-major order would make every shard a single mass regime; interleave so each
        # shard sees the whole range and the shards cost roughly the same.
        order = torch.randperm(polys.shape[0])
        polys, mass = polys[order], mass[order]

        x0 = torch.randn(num_x0, 2)

    return {
        "version": VAL1K_VERSION,
        "name": "v1k",
        "seed": seed,
        "polynomials": polys.cpu(),
        "mass": mass,
        "x0": x0,
        "mass_pool_size": pool_size,
        "mass_bins": num_bins,
        "mass_range": (min_mass, max_mass),
        "degree": degree,
        "scale": scale,
        "poly_digest": poly_digest(polys),
    }


def save_validation_set_v1k(val_set: dict, path: str = VAL1K_SET_PATH) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    torch.save(val_set, tmp)
    os.replace(tmp, path)
    return path


def get_validation_set_v1k(path: str = VAL1K_SET_PATH, device=None) -> dict:
    """Loads the frozen v1k benchmark; never generates it, since the build costs a GPU job."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"v1k validation set missing: {path}\n"
            f"Build it first: sbatch scripts/run_val1k_build.sh")

    val_set = torch.load(path, map_location="cpu", weights_only=False)
    if val_set.get("version") != VAL1K_VERSION:
        raise RuntimeError(f"{path} was written by version {val_set.get('version')}, "
                           f"but this code expects {VAL1K_VERSION}; rebuild it.")

    val_set["polynomials"] = val_set["polynomials"].to(device)
    val_set["x0"] = val_set["x0"].to(device)
    return val_set


def resolve_validation_set(name: str = "v1k", device=None) -> dict:
    """Uniform view over both benchmarks: polynomials, start points, and mass when known.

    ``legacy100`` carries no cached mass, so callers that need one estimate it themselves.
    """
    if name == "v1k":
        val_set = get_validation_set_v1k(device=device)
        return {"name": "v1k", "polynomials": val_set["polynomials"], "x0": val_set["x0"],
                "mass": val_set["mass"], "poly_digest": val_set["poly_digest"],
                "path": VAL1K_SET_PATH}

    if name == "legacy100":
        from constrained_fm.src.datasets.validation import get_validation_set

        val_set = get_validation_set(device=device)
        polys = val_set["polynomials"]
        return {"name": "legacy100", "polynomials": polys, "x0": val_set["x0"], "mass": None,
                "poly_digest": poly_digest(polys), "path": VALIDATION_SET_PATH}

    raise ValueError(f"unknown validation set '{name}'; expected 'v1k' or 'legacy100'")


__all__ = ["seeded_gmm_pool", "poly_digest", "monte_carlo_masses",
           "generate_stratified_polynomials", "build_validation_set_v1k",
           "save_validation_set_v1k", "get_validation_set_v1k", "resolve_validation_set"]

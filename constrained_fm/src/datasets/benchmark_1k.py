# -*- coding: utf-8 -*-
"""The 1000-constraint benchmarks for bump2d and kinematics6d.

Both problems report every metric *as a function of* how much probability mass the constraint
encloses, and a trend line is only as trustworthy as the number of constraints in its bin.
The constraint samplers each inherit a prior that thins out at one or both ends of that
range, so this module instead fills a fixed number of mass bins to equal depth.

A constraint is an object, not a tensor, so what is frozen to disk is the family's defining
parameters -- half-planes for a polygon, centre and half-width for a shell -- and
:func:`constraints_from` rebuilds the objects. The whole file is a pure function of its seed,
so a deleted cache is reproduced bit-for-bit.

Build (heavy, GPU)::

    sbatch scripts/run_bench1k_build.sh
"""

from __future__ import annotations

import hashlib
import math
import os

import torch

from constrained_fm.src.consts import (BENCH1K_DIR, BENCH1K_MASS_BINS, BENCH1K_MC_POOL_SIZE,
                                       BENCH1K_NUM_CONSTRAINTS, BENCH1K_NUM_X0, BENCH1K_SEED,
                                       BENCH1K_VERSION, BUMP_POLY_MAX_MASS,
                                       BUMP_POLY_MAX_VERTICES, BUMP_POLY_MIN_MASS,
                                       BUMP_POLY_MIN_VERTICES, BUMP_POLY_RADIUS_RANGE,
                                       KIN_SHELL_MAX_MASS, KIN_SHELL_MIN_MASS)
from constrained_fm.src.problems.bump2d import (BumpProblem, PolygonConstraint, polygon_mass,
                                                propose_polygons)
from constrained_fm.src.problems.kinematics6d import (KinematicsProblem, MassWindowConstraint,
                                                      sample_mass_constraints)

PROBLEM_NAMES = ("bump2d", "kinematics6d")
# Every half-plane a polygon can carry: its own faces plus the four faces of the domain box.
MAX_HALF_PLANES = BUMP_POLY_MAX_VERTICES + 4
POLYGON_BATCH = 512
POLYGON_MAX_ROUNDS = 2000


def benchmark_path(problem: str, split: str = "1k") -> str:
    return os.path.join(BENCH1K_DIR, f"benchmark_{split}_{problem}.pt")


def _digest(*tensors: torch.Tensor) -> str:
    """Content hash over the constraint-defining tensors, in a device-independent dtype."""
    sha = hashlib.sha256()
    for tensor in tensors:
        sha.update(tensor.detach().to(device="cpu", dtype=torch.float64).contiguous()
                   .numpy().tobytes())
    return sha.hexdigest()[:16]


def _bin_quota(num_constraints: int, num_bins: int) -> list[int]:
    """Equal depth per bin, with the remainder spread over the leading bins."""
    base, remainder = divmod(num_constraints, num_bins)
    return [base + (1 if i < remainder else 0) for i in range(num_bins)]


# --- bump2d -------------------------------------------------------------------------------


def stratified_polygons(pool: torch.Tensor, num_constraints: int = BENCH1K_NUM_CONSTRAINTS,
                        min_mass: float = BUMP_POLY_MIN_MASS,
                        max_mass: float = BUMP_POLY_MAX_MASS,
                        num_bins: int = BENCH1K_MASS_BINS,
                        domain: float = 10.0,
                        radius_range: tuple[float, float] = BUMP_POLY_RADIUS_RANGE,
                        batch_size: int = POLYGON_BATCH,
                        max_rounds: int = POLYGON_MAX_ROUNDS) -> tuple[list, torch.Tensor]:
    """Fills ``num_bins`` equal-width mass bins to equal depth by rejection on the mass.

    The bins are equal-width in mass rather than log-spaced so the polygon benchmark matches
    the protocol the polynomial one already uses; the kinematics shells are log-spaced
    instead, because their mass range spans two decades rather than one.
    """
    edges = torch.linspace(min_mass, max_mass, num_bins + 1, dtype=torch.float64)
    quota = _bin_quota(num_constraints, num_bins)
    accepted: list[list[PolygonConstraint]] = [[] for _ in range(num_bins)]
    accepted_mass: list[list[float]] = [[] for _ in range(num_bins)]

    for round_idx in range(max_rounds):
        if sum(len(a) for a in accepted) >= num_constraints:
            break

        normals, offsets, active = propose_polygons(
            batch_size, domain=domain, min_vertices=BUMP_POLY_MIN_VERTICES,
            max_vertices=BUMP_POLY_MAX_VERTICES, radius_range=radius_range, device=pool.device)
        normals, offsets = normals.to(pool.dtype), offsets.to(pool.dtype)
        mass = polygon_mass(normals, offsets, active, pool).to(torch.float64).cpu()

        # right=True keeps the closed upper edge from spilling into a non-existent bin.
        bins = (torch.bucketize(mass, edges, right=True) - 1).clamp(0, num_bins - 1)

        for i in range(batch_size):
            b, m = int(bins[i]), float(mass[i])
            if not (min_mass <= m <= max_mass) or len(accepted[b]) >= quota[b]:
                continue
            rows = active[i]
            faces, shifts = normals[i][rows].clone(), offsets[i][rows].clone()
            deepest = (pool @ faces.T - shifts).amax(dim=-1).argmin()
            accepted[b].append(PolygonConstraint(faces, shifts, pool[deepest].clone()))
            accepted_mass[b].append(m)

        if round_idx % 20 == 0:
            filled = sum(len(a) for a in accepted)
            print(f"round {round_idx:4d} | accepted {filled}/{num_constraints} | "
                  f"emptiest bin {min(len(a) for a in accepted)}", flush=True)
    else:
        shortfall = {i: quota[i] - len(accepted[i]) for i in range(num_bins)
                     if len(accepted[i]) < quota[i]}
        raise RuntimeError(f"stratification did not converge in {max_rounds} rounds; "
                           f"bins still short: {shortfall}")

    constraints = [c for bin_polys in accepted for c in bin_polys]
    mass = torch.tensor([m for bin_mass in accepted_mass for m in bin_mass],
                        dtype=torch.float64)
    return constraints, mass


def _serialize_polygons(constraints: list[PolygonConstraint]) -> dict[str, torch.Tensor]:
    """Ragged half-plane sets into one padded tensor plus the active row count.

    Padding repeats the first face, so a consumer that forgets to slice by ``counts`` still
    gets the same ``amax`` rather than a silently different polygon.
    """
    counts = torch.tensor([c.normals.shape[0] for c in constraints], dtype=torch.long)
    normals = torch.zeros(len(constraints), MAX_HALF_PLANES, 2)
    offsets = torch.zeros(len(constraints), MAX_HALF_PLANES)
    interior = torch.stack([c.interior.cpu() for c in constraints])

    for i, c in enumerate(constraints):
        k = int(counts[i])
        normals[i, :k], offsets[i, :k] = c.normals.cpu(), c.offsets.cpu()
        normals[i, k:], offsets[i, k:] = c.normals[0].cpu(), c.offsets[0].cpu()

    return {"normals": normals, "offsets": offsets, "counts": counts, "interior": interior}


def _deserialize_polygons(payload: dict, device) -> list[PolygonConstraint]:
    normals, offsets = payload["normals"].to(device), payload["offsets"].to(device)
    interior, counts = payload["interior"].to(device), payload["counts"]
    return [PolygonConstraint(normals[i, :int(counts[i])], offsets[i, :int(counts[i])],
                              interior[i])
            for i in range(counts.numel())]


# --- kinematics6d -------------------------------------------------------------------------


def _serialize_shells(constraints: list[MassWindowConstraint]) -> dict[str, torch.Tensor]:
    return {"mass_target": torch.tensor([c.mass_target for c in constraints],
                                        dtype=torch.float64),
            "epsilon": torch.tensor([c.epsilon for c in constraints], dtype=torch.float64)}


def _deserialize_shells(payload: dict, target) -> list[MassWindowConstraint]:
    return [MassWindowConstraint(target, float(m), float(e))
            for m, e in zip(payload["mass_target"], payload["epsilon"])]


# --- build / load -------------------------------------------------------------------------


def build_benchmark_1k(problem_name: str, num_constraints: int = BENCH1K_NUM_CONSTRAINTS,
                       num_bins: int = BENCH1K_MASS_BINS,
                       pool_size: int = BENCH1K_MC_POOL_SIZE, num_x0: int = BENCH1K_NUM_X0,
                       seed: int = BENCH1K_SEED, device=None) -> dict:
    """Stratifies ``num_constraints`` constraints and freezes them with shared start points.

    The accepted constraints come out bin-major, which would make every shard a single mass
    regime; a permutation interleaves them so each shard sees the whole range and the shards
    cost roughly the same wall time.
    """
    if problem_name not in PROBLEM_NAMES:
        raise ValueError(f"unknown problem '{problem_name}'; expected one of {PROBLEM_NAMES}")

    with torch.random.fork_rng(devices=[] if device in (None, "cpu") else [device]):
        torch.manual_seed(seed)

        problem = BumpProblem() if problem_name == "bump2d" else KinematicsProblem()
        target = problem.target()
        print(f"Drawing the {pool_size} point mass pool (seed {seed})...", flush=True)
        pool = target.sample(pool_size, device=device)

        print(f"Stratifying {num_constraints} constraints over {num_bins} mass bins...",
              flush=True)
        if problem_name == "bump2d":
            constraints, mass = stratified_polygons(
                pool, num_constraints=num_constraints, min_mass=problem.min_mass,
                max_mass=problem.max_mass, num_bins=num_bins, domain=problem.domain)
            payload = _serialize_polygons(constraints)
            mass_range = (problem.min_mass, problem.max_mass)
        else:
            constraints, mass = sample_mass_constraints(
                num_constraints, target, pool, KIN_SHELL_MIN_MASS, KIN_SHELL_MAX_MASS,
                num_bins)
            payload = _serialize_shells(constraints)
            mass_range = (KIN_SHELL_MIN_MASS, KIN_SHELL_MAX_MASS)

        order = torch.randperm(num_constraints)
        payload = {k: v[order] for k, v in payload.items()}
        mass = mass[order]

        x0 = torch.randn(num_x0, problem.dim)

    return {
        "version": BENCH1K_VERSION,
        "problem": problem_name,
        "seed": seed,
        "dim": problem.dim,
        "constraints": payload,
        "mass": mass,
        "x0": x0,
        "mass_pool_size": pool_size,
        "mass_bins": num_bins,
        "mass_range": mass_range,
        "digest": _digest(*payload.values()),
    }


def save_benchmark_1k(benchmark: dict, split: str = "1k") -> str:
    path = benchmark_path(benchmark["problem"], split)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    torch.save(benchmark, tmp)
    os.replace(tmp, path)
    return path


def load_benchmark_1k(problem_name: str, device=None, split: str = "1k") -> dict:
    """Loads a frozen benchmark; never generates it, since the build costs a GPU job."""
    path = benchmark_path(problem_name, split)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{problem_name} benchmark missing: {path}\n"
            f"Build it first: sbatch scripts/run_bench1k_build.sh --problem {problem_name}")

    benchmark = torch.load(path, map_location="cpu", weights_only=False)
    if benchmark.get("version") != BENCH1K_VERSION:
        raise RuntimeError(f"{path} was written by version {benchmark.get('version')}, "
                           f"but this code expects {BENCH1K_VERSION}; rebuild it.")

    recomputed = _digest(*benchmark["constraints"].values())
    if recomputed != benchmark["digest"]:
        raise RuntimeError(f"{path} is corrupt: digest {recomputed} != {benchmark['digest']}")

    benchmark["x0"] = benchmark["x0"].to(device)
    return benchmark


def constraints_from(benchmark: dict, problem, device=None) -> list:
    """Rebuilds the constraint objects a frozen benchmark describes."""
    if benchmark["problem"] == "bump2d":
        return _deserialize_polygons(benchmark["constraints"], device)
    return _deserialize_shells(benchmark["constraints"], problem.target())


def mass_bin_histogram(mass: torch.Tensor, num_bins: int, lo: float, hi: float,
                       log_spaced: bool) -> list[tuple[float, float, int]]:
    """``(lo, hi, count)`` per bin, on the spacing the problem was stratified with.

    The shells are stratified onto ``num_bins`` discrete log-spaced *levels* rather than into
    intervals, so the bin edges are placed at the geometric midpoints between them. Edges
    taken as ``logspace(lo, hi, num_bins + 1)`` would straddle the levels instead, and an
    exactly balanced benchmark would print as an alternating occupancy of 100 and 0.
    """
    if log_spaced:
        levels = torch.logspace(math.log10(lo), math.log10(hi), num_bins, dtype=torch.float64)
        half_step = (levels[1] / levels[0]).sqrt()
        edges = torch.cat([levels[:1] / half_step, levels * half_step])
    else:
        edges = torch.linspace(lo, hi, num_bins + 1, dtype=torch.float64)
    bins = (torch.bucketize(mass, edges, right=True) - 1).clamp(0, num_bins - 1)
    return [(float(edges[b]), float(edges[b + 1]), int((bins == b).sum()))
            for b in range(num_bins)]


__all__ = ["PROBLEM_NAMES", "benchmark_path", "build_benchmark_1k", "save_benchmark_1k",
           "load_benchmark_1k", "constraints_from", "stratified_polygons",
           "mass_bin_histogram"]

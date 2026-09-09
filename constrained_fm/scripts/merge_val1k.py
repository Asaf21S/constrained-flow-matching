# -*- coding: utf-8 -*-
"""Stage 3 of the v1k pipeline: stitch the per-shard scores into one metrics.json.

Pure stdlib, so it runs on the login node without the container. Refuses to merge a method
whose shards do not cover every constraint exactly once, since a silently truncated
benchmark would still produce a plausible-looking plot.

    python3 -m constrained_fm.scripts.merge_val1k
    python3 -m constrained_fm.scripts.merge_val1k --allow-partial
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

from constrained_fm.src.experiment.registry import summarize

DEFAULT_OUTDIR = "constrained_fm/baselines/val1k"
METRIC_KEYS = ("success_rate", "swd", "mmd", "jsd", "nll", "kld")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge v1k evaluation shards.")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--expected", type=int, default=None,
                        help="constraint count to require (default: inferred from the shards)")
    parser.add_argument("--allow-partial", action="store_true",
                        help="merge anyway and leave the gaps as NaN")
    return parser


def load_shards(shard_dir: Path) -> dict[str, list[dict]]:
    if not shard_dir.exists():
        raise FileNotFoundError(f"no shards at {shard_dir} -- run sbatch scripts/run_val1k_eval.sh")

    by_method: dict[str, list[dict]] = {}
    for path in sorted(shard_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        by_method.setdefault(payload["method"], []).append(payload)
    if not by_method:
        raise FileNotFoundError(f"{shard_dir} holds no shard files")
    return by_method


def merge_method(shards: list[dict], expected: int, allow_partial: bool) -> dict:
    """Scatters every shard's rows into one array per metric, indexed by constraint."""
    per_shape = {key: [float("nan")] * expected for key in (*METRIC_KEYS, "mass")}
    seen: dict[int, str] = {}

    for shard in shards:
        label = f"[{shard['start_idx']}, {shard['end_idx']})"
        for position, index in enumerate(shard["indices"]):
            if index in seen:
                raise RuntimeError(f"constraint {index} appears in both {seen[index]} and {label}")
            seen[index] = label
            for key, values in shard["per_shape"].items():
                if key in per_shape:
                    per_shape[key][index] = float(values[position])

    missing = [i for i in range(expected) if i not in seen]
    if missing and not allow_partial:
        raise RuntimeError(
            f"shards cover {len(seen)}/{expected} constraints; {len(missing)} missing "
            f"(first few: {missing[:10]}). Re-run the failed array tasks, or pass "
            f"--allow-partial to merge the gaps as NaN.")

    per_shape = {key: values for key, values in per_shape.items()
                 if any(math.isfinite(v) for v in values)}
    reference = shards[0]
    return {
        "method": reference["method"],
        "run_id": reference["run_id"],
        "eval": reference["eval"],
        "num_constraints": expected,
        "num_scored": len(seen),
        "missing": missing,
        "per_shape": per_shape,
        "summary": summarize(per_shape),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.outdir)
    by_method = load_shards(out / "shards")

    expected = args.expected
    if expected is None:
        expected = max(shard["end_idx"] for shards in by_method.values() for shard in shards)

    digests = {shard["poly_digest"] for shards in by_method.values() for shard in shards}
    if len(digests) > 1:
        raise RuntimeError(f"shards were scored against different validation sets: {digests}")

    methods = {name: merge_method(shards, expected, args.allow_partial)
               for name, shards in sorted(by_method.items())}

    mass = next((m["per_shape"]["mass"] for m in methods.values() if "mass" in m["per_shape"]), None)
    payload = {
        "validation_set": "v1k",
        "poly_digest": digests.pop(),
        "num_constraints": expected,
        "merged_at": datetime.now().isoformat(timespec="seconds"),
        "mass": mass,
        "methods": methods,
    }

    path = out / "metrics.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {path}")
    for name, merged in methods.items():
        summary = merged["summary"]
        print(f"  {name:9s} {merged['num_scored']:4d}/{expected} constraints | "
              f"SR {summary.get('success_rate_median', float('nan')):6.2f} | "
              f"SWD {summary.get('swd_median', float('nan')):.4f} | "
              f"JSD {summary.get('jsd_median', float('nan')):.4f} | "
              f"KLD {summary.get('kld_median', float('nan')):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

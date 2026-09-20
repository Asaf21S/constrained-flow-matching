# -*- coding: utf-8 -*-
"""Regression gate: every pinned run must still resolve to the fingerprint it was written with.

Any change to the identity payload in ``ExperimentConfig`` silently re-keys every historical
run, orphaning the numbers already reported. Pure stdlib plus yaml, so it runs on the login
node.

    python3 -m constrained_fm.scripts.check_fingerprints
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from constrained_fm.src.experiment.config import ExperimentConfig
from constrained_fm.src.experiment.registry import (CONFIG_NAME, PROVENANCE_NAME, RUNS_ROOT,
                                                    read_json)


def check_run(path: Path) -> tuple[bool | None, str]:
    """Returns (ok, message); ok is None when the run predates provenance tracking."""
    provenance = read_json(path / PROVENANCE_NAME)
    if provenance is None or "fingerprint" not in provenance:
        return None, f"{path.name}: skipped, no pinned fingerprint"

    try:
        cfg = ExperimentConfig.from_yaml(path / CONFIG_NAME)
    except Exception as exc:
        return False, f"{path.name}: config no longer loads -- {exc}"

    stored = provenance["fingerprint"]
    try:
        recomputed = cfg.fingerprint()
    except FileNotFoundError as exc:
        # The fingerprint hashes the SIREN weights, so a deleted checkpoint is unverifiable.
        return None, f"{path.name}: skipped, {exc}"

    if recomputed != stored:
        return False, (f"{path.name}: FINGERPRINT DRIFT\n"
                       f"        stored     {stored}\n"
                       f"        recomputed {recomputed}")
    if cfg.run_id != path.name:
        return False, f"{path.name}: run id drift -> {cfg.run_id}"

    pool = cfg.pool_path()
    suffix = "" if pool.exists() else f"  (pool missing: {pool.name})"
    return True, f"{path.name}: ok{suffix}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default=str(RUNS_ROOT))
    args = parser.parse_args()

    root = Path(args.runs_root)
    if not root.exists():
        print(f"no runs directory at {root}")
        return 0

    failures, checked, skipped = [], 0, 0
    for path in sorted(root.iterdir()):
        if not (path / CONFIG_NAME).exists():
            continue
        ok, message = check_run(path)
        print(f"  {message}")
        if ok is None:
            skipped += 1
        elif ok:
            checked += 1
        else:
            failures.append(message)

    print(f"\n{checked} verified, {skipped} skipped, {len(failures)} drifted")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

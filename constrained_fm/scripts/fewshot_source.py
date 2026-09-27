# -*- coding: utf-8 -*-
"""Which few-shot run the v1k figures and tables report.

The from-scratch sweep is merged into ``val1k/metrics.json``; the run fine-tuned from the
base_fm checkpoint is merged into its own metrics.json. Consumers keep reading the main
payload and ask here for the few-shot entry, so every other method's numbers are untouched
and the ``fewshot`` key, colour and legend logic keep working for either run.

Pure stdlib, so the table scripts still run on the login node.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FEWSHOT = "fewshot"
SCRATCH = "scratch"
FINETUNED = "finetuned"
DEFAULT_SOURCE = FINETUNED
FINETUNED_METRICS = "constrained_fm/baselines/few_shot_finetuned_v1k/metrics.json"
# The headline budget: what the main table and the trend plots show.
DEFAULT_BUDGET = 2000

# From scratch, N=2000 kept the bare name; fine-tuned budgets all carry the suffix.
METHOD_PATTERNS = {
    SCRATCH: re.compile(r"^fewshot(?:_N(\d+))?$"),
    FINETUNED: re.compile(r"^fewshot_ft_N(\d+)$"),
}
LABELS = {SCRATCH: "Few-Shot", FINETUNED: "Few-Shot FT"}


def add_arguments(parser: argparse.ArgumentParser, budget: bool = True) -> None:
    parser.add_argument("--fewshot-source", choices=(FINETUNED, SCRATCH), default=DEFAULT_SOURCE,
                        help="few-shot run to report: fine-tuned from base_fm, or from scratch")
    parser.add_argument("--fewshot-metrics", default=FINETUNED_METRICS,
                        help="merged metrics.json of the fine-tuned run")
    if budget:
        parser.add_argument("--fewshot-budget", type=int, default=DEFAULT_BUDGET,
                            help="shot budget N reported wherever a single budget is shown")


def budgets(source: str, main_payload: dict, finetuned_path: str | Path) -> tuple[dict, dict]:
    """({N: merged method entry}, payload they came from) for one few-shot run."""
    if source == SCRATCH:
        payload = main_payload
    else:
        path = Path(finetuned_path)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing -- run `python3 -m constrained_fm.scripts.merge_val1k "
                f"--outdir {path.parent}`, or pass --fewshot-source {SCRATCH}")
        payload = json.loads(path.read_text())
        if payload["poly_digest"] != main_payload["poly_digest"]:
            raise RuntimeError(f"{path} was scored on a different validation set "
                               f"({payload['poly_digest']} vs {main_payload['poly_digest']})")

    by_n: dict[int, dict] = {}
    for method, merged in payload.get("methods", {}).items():
        match = METHOD_PATTERNS[source].match(method)
        if match is None:
            continue
        n_points = merged.get("eval", {}).get("n_points") or match.group(1)
        if n_points is not None:
            by_n[int(n_points)] = merged
    return by_n, payload


def with_fewshot(main_payload: dict, args: argparse.Namespace) -> dict:
    """The main payload with its ``fewshot`` entry replaced by the selected run and budget."""
    by_n, _ = budgets(args.fewshot_source, main_payload, args.fewshot_metrics)
    if args.fewshot_budget not in by_n:
        raise KeyError(f"no {args.fewshot_source} few-shot run at N={args.fewshot_budget}; "
                       f"available budgets: {sorted(by_n)}")
    methods = {name: merged for name, merged in main_payload["methods"].items()
               if not METHOD_PATTERNS[SCRATCH].match(name)}
    methods[FEWSHOT] = {**by_n[args.fewshot_budget], "source": args.fewshot_source}
    return {**main_payload, "methods": methods}


def source_of(payload: dict) -> str:
    return payload["methods"].get(FEWSHOT, {}).get("source", SCRATCH)


__all__ = ["FEWSHOT", "SCRATCH", "FINETUNED", "LABELS", "add_arguments", "budgets",
           "with_fewshot", "source_of"]

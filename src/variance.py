"""Estimate sigma_within: reward variance under the SAME bundle, repeated.

Without this baseline, perturbation effects are not interpretable: we cannot
tell whether a 0.08 reward delta is a real composition effect or just
LLM-decoding noise. Preregistration requires effect >= 2 * sigma_within to
claim sensitivity.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


def load_default_runs(results_dir: Path) -> dict[str, list[float]]:
    """Group rewards by query_id over all default-condition runs."""
    by_query: dict[str, list[float]] = defaultdict(list)
    runs = (results_dir / "runs").glob("*__default__*.json")
    for f in runs:
        record = json.loads(f.read_text())
        if record.get("reward") is None:
            continue
        by_query[record["query_id"]].append(float(record["reward"]))
    return by_query


def estimate_sigma_within(per_query_rewards: dict[str, list[float]]) -> dict[str, float | None]:
    """Per-query sample stddev. None when fewer than 2 valid runs (cannot estimate)."""
    return {
        qid: pstdev(rs) if len(rs) >= 2 else None
        for qid, rs in per_query_rewards.items()
    }


def pooled_sigma_within(per_query_sigmas: dict[str, float | None]) -> float | None:
    """Mean across queries with >=2 runs. Used as the project-wide noise floor."""
    valid = [s for s in per_query_sigmas.values() if s is not None]
    if not valid:
        return None
    return mean(valid)


def report(per_query_rewards: dict[str, list[float]]) -> dict:
    sigmas = estimate_sigma_within(per_query_rewards)
    return {
        "per_query_sigma": sigmas,
        "pooled_sigma": pooled_sigma_within(sigmas),
        "queries_with_enough_runs": sum(1 for s in sigmas.values() if s is not None),
        "total_queries": len(sigmas),
    }

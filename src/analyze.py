"""Apply the pre-registered decision rule to collected runs.

Decision logic mirrors configs/preregistration.yaml. Any change to the rule
must go through that file's amendments_log, NOT through code edits here.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .variance import load_default_runs, pooled_sigma_within, estimate_sigma_within


PERTURBATION_FAMILIES = ("loo", "unrelated_swap", "eps_swap")


def _family_of(condition: str) -> str | None:
    for fam in PERTURBATION_FAMILIES:
        if condition.startswith(fam):
            return fam
    return None


def load_perturbation_runs(results_dir: Path) -> dict[tuple[str, str], list[float]]:
    """Group rewards by (query_id, perturbation_family). Specific swap targets
    are pooled inside a family — the decision rule is at the family level.
    """
    out: dict[tuple[str, str], list[float]] = defaultdict(list)
    for f in (results_dir / "runs").glob("*.json"):
        record = json.loads(f.read_text())
        if record.get("reward") is None:
            continue
        fam = _family_of(record["condition"])
        if fam is None:
            continue
        out[(record["query_id"], fam)].append(float(record["reward"]))
    return out


def per_query_default_means(default_runs: dict[str, list[float]]) -> dict[str, float]:
    return {qid: mean(rs) for qid, rs in default_runs.items() if rs}


def family_effect_sizes(
    default_means: dict[str, float],
    perturbed: dict[tuple[str, str], list[float]],
) -> dict[str, float]:
    """For each perturbation family, mean across queries of |R_pert - R_default|.

    Pairing on query_id matters because different queries have very different
    base reward levels; absolute deltas only make sense within-query.
    """
    deltas: dict[str, list[float]] = defaultdict(list)
    for (qid, fam), rewards in perturbed.items():
        if qid not in default_means:
            continue
        for r in rewards:
            deltas[fam].append(abs(r - default_means[qid]))
    return {fam: mean(ds) if ds else float("nan") for fam, ds in deltas.items()}


def decide(
    results_dir: Path,
    delta_threshold: float,
    variance_ratio_required: float,
) -> dict:
    """Run the full decision pipeline. Returns a structured verdict.

    Verdict shape:
        {
          "verdict": "proceed" | "terminate" | "narrow_scope",
          "winning_families": [...],          # families that cleared both bars
          "family_effects": {...},
          "sigma_pooled": float | None,
          "thresholds": {"delta": ..., "ratio": ...},
        }
    """
    default_runs = load_default_runs(results_dir)
    sigma_pooled = pooled_sigma_within(estimate_sigma_within(default_runs))
    default_means = per_query_default_means(default_runs)
    perturbed = load_perturbation_runs(results_dir)
    effects = family_effect_sizes(default_means, perturbed)

    cleared = []
    for fam, eff in effects.items():
        if eff != eff:                                     # NaN guard
            continue
        if eff < delta_threshold:
            continue
        if sigma_pooled is None:
            continue
        if eff < variance_ratio_required * sigma_pooled:
            continue
        cleared.append(fam)

    if len(cleared) >= 2:
        verdict = "proceed"
    elif len(cleared) == 1:
        verdict = "narrow_scope"
    else:
        verdict = "terminate"

    return {
        "verdict": verdict,
        "winning_families": cleared,
        "family_effects": effects,
        "sigma_pooled": sigma_pooled,
        "thresholds": {
            "delta": delta_threshold,
            "ratio": variance_ratio_required,
        },
    }

"""Three perturbation generators for the GoS bundle-sensitivity study.

A "bundle" is a list of skill IDs (strings). The skill library is a dict
mapping skill_id -> {embedding, domain_tag, ...metadata}. Rho scores come
from the upstream GoS retrieval and are passed in as a dict skill_id -> float.

Each generator returns a list of (label, perturbed_bundle) pairs so the
caller can name the result for logging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SkillRecord:
    skill_id: str
    embedding: np.ndarray            # unit-normalized
    domain_tag: str


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))       # assumes unit-normalized inputs


def leave_one_out(bundle: list[str]) -> list[tuple[str, list[str]]]:
    """Drop each skill in turn. Returns |B| perturbed bundles."""
    return [
        (f"loo:{dropped}", [s for s in bundle if s != dropped])
        for dropped in bundle
    ]


def unrelated_swap(
    bundle: list[str],
    library: dict[str, SkillRecord],
    query_embedding: np.ndarray,
    bundle_skill_to_replace: str,
    cosine_max: float = 0.25,
    require_different_domain_tag: bool = True,
    rng: np.random.Generator | None = None,
) -> tuple[str, list[str]] | None:
    """Replace one bundle skill with a skill that is unrelated to the query.

    "Unrelated" = cosine(query, skill) <= cosine_max AND optionally a domain
    tag different from every skill currently in the bundle. Returns None if
    no candidate exists (caller should log and skip).
    """
    rng = rng or np.random.default_rng()
    bundle_tags = {library[s].domain_tag for s in bundle if s in library}

    candidates = []
    for sid, rec in library.items():
        if sid in bundle:
            continue
        if _cosine(query_embedding, rec.embedding) > cosine_max:
            continue
        if require_different_domain_tag and rec.domain_tag in bundle_tags:
            continue
        candidates.append(sid)

    if not candidates:
        return None

    chosen = str(rng.choice(candidates))
    new_bundle = [chosen if s == bundle_skill_to_replace else s for s in bundle]
    return (f"unrelated_swap:{bundle_skill_to_replace}->{chosen}", new_bundle)


def epsilon_neighbor_swap(
    bundle: list[str],
    rho_scores: dict[str, float],
    bundle_skill_to_replace: str,
    rho_epsilon: float = 0.05,
    must_be_outside_topk: bool = True,
    rng: np.random.Generator | None = None,
) -> tuple[str, list[str]] | None:
    """Replace one bundle skill with a skill whose rho score is within epsilon
    of the replaced skill but was NOT chosen by GoS.

    Tests whether GoS's ranking carries fine-grained signal beyond a coarse
    above-threshold cut. If reward is invariant to this swap, the surrogate
    has nothing to learn.
    """
    rng = rng or np.random.default_rng()
    target_rho = rho_scores.get(bundle_skill_to_replace)
    if target_rho is None:
        return None

    bundle_set = set(bundle)
    candidates = []
    for sid, rho in rho_scores.items():
        if must_be_outside_topk and sid in bundle_set:
            continue
        if abs(rho - target_rho) > rho_epsilon:
            continue
        candidates.append(sid)

    if not candidates:
        return None

    chosen = str(rng.choice(candidates))
    new_bundle = [chosen if s == bundle_skill_to_replace else s for s in bundle]
    return (f"eps_swap:{bundle_skill_to_replace}->{chosen}", new_bundle)


def generate_all(
    bundle: list[str],
    library: dict[str, SkillRecord],
    rho_scores: dict[str, float],
    query_embedding: np.ndarray,
    cosine_max: float = 0.25,
    rho_epsilon: float = 0.05,
    rng: np.random.Generator | None = None,
) -> list[tuple[str, list[str]]]:
    """Convenience: generate one perturbation of each non-LOO type per bundle
    skill, plus full leave-one-out. Caller decides which to actually run.
    """
    rng = rng or np.random.default_rng()
    out: list[tuple[str, list[str]]] = list(leave_one_out(bundle))

    for sid in bundle:
        u = unrelated_swap(bundle, library, query_embedding, sid, cosine_max, rng=rng)
        if u is not None:
            out.append(u)
        e = epsilon_neighbor_swap(bundle, rho_scores, sid, rho_epsilon, rng=rng)
        if e is not None:
            out.append(e)

    return out

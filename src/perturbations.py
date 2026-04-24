"""Three perturbations of a GoS-retrieved skill bundle.

Each function returns ONE perturbed bundle (or None if no candidate exists).
Caller pairs them up with the original bundle to form the 4 conditions per
query: gos_original, delete_top, add_irrelevant, replace_similar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SkillRecord:
    skill_id: str
    name: str
    embedding: np.ndarray            # unit-normalized
    domain_tag: str


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def delete_top(bundle: list[str], ppr_scores: dict[str, float]) -> list[str]:
    """Drop the highest-PPR skill in the bundle.

    Hypothesis: if GoS's top pick is load-bearing, removing it should hurt
    reward. Invariant when the agent doesnt actually rely on the top skill.
    """
    if not bundle:
        return []
    top = max(bundle, key=lambda s: ppr_scores.get(s, float("-inf")))
    return [s for s in bundle if s != top]


def add_irrelevant(
    bundle: list[str],
    library: dict[str, SkillRecord],
    query_embedding: np.ndarray,
    cosine_max: float = 0.20,
    rng: np.random.Generator | None = None,
) -> list[str] | None:
    """Append one randomly-chosen skill that is semantically far from the query.

    Tests "is the bundle hurt by adding noise?" — if reward is invariant the
    agent is robust to irrelevant context (or context just doesnt matter).
    """
    rng = rng or np.random.default_rng()
    bundle_set = set(bundle)
    candidates = [
        sid for sid, rec in library.items()
        if sid not in bundle_set and _cosine(query_embedding, rec.embedding) <= cosine_max
    ]
    if not candidates:
        return None
    chosen = str(rng.choice(candidates))
    return list(bundle) + [chosen]


def replace_similar(
    bundle: list[str],
    ppr_scores: dict[str, float],
    rng: np.random.Generator | None = None,
    rho_epsilon: float = 0.001,
) -> tuple[list[str], str, str] | None:
    """Pick a random bundle skill and swap it with its nearest non-bundle PPR
    neighbor (within rho_epsilon).

    The diagnostic perturbation: if reward is invariant to swapping in a
    PPR-equivalent skill, GoSs ranking has no fine-grained signal above the
    bundle-cutoff threshold, and surrogate modeling has nothing to learn from.

    Returns (new_bundle, swapped_out, swapped_in) or None if no neighbor found.
    """
    rng = rng or np.random.default_rng()
    if not bundle:
        return None

    # Pick which bundle skill to replace, then find its PPR-nearest non-bundle peer.
    bundle_set = set(bundle)
    target = str(rng.choice(bundle))
    target_rho = ppr_scores.get(target)
    if target_rho is None:
        return None

    candidates = [
        (sid, rho) for sid, rho in ppr_scores.items()
        if sid not in bundle_set and abs(rho - target_rho) <= rho_epsilon
    ]
    if not candidates:
        return None

    # Take the absolute-closest, with random tie break.
    candidates.sort(key=lambda x: (abs(x[1] - target_rho), rng.random()))
    chosen = candidates[0][0]
    new_bundle = [chosen if s == target else s for s in bundle]
    return new_bundle, target, chosen

"""Thin wrapper around the GoS retrieval pipeline.

GoS lives in a separate repo (https://github.com/davidliuk/graph-of-skills).
We do NOT vendor it. Instead, point `gos_repo` in configs/experiment.yaml at
a local clone and import its modules through this wrapper.

Three calls are needed:
  1. load_library(path)          -> {skill_id: SkillRecord, ...}
  2. retrieve_default_bundle(query, library, retrieval_cfg)
                                  -> (bundle: list[str], rho: dict[str,float])
  3. embed(text)                 -> np.ndarray (unit-normalized)

Each is left as TODO until the GoS repo is cloned and its module layout
inspected. Keep this module the ONLY place that knows about GoS internals;
everything else in the project takes plain Python types.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .perturbations import SkillRecord


def _ensure_gos_on_path(gos_repo: str | Path) -> None:
    p = Path(gos_repo).resolve()
    if not p.exists():
        raise FileNotFoundError(
            f"GoS repo not found at {p}. Clone it first: "
            "git clone https://github.com/davidliuk/graph-of-skills"
        )
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def load_library(gos_repo: str | Path, library_path: str | Path) -> dict[str, SkillRecord]:
    """Load skills_200 (or whichever release) into normalized SkillRecords.

    TODO(integration): inspect GoSs skill record schema and map fields here.
    Expected: a directory of JSON/YAML skill specs with at least name,
    capability summary, domain tag, and either a precomputed embedding or
    enough text to compute one via embed().
    """
    _ensure_gos_on_path(gos_repo)
    raise NotImplementedError(
        "load_library: wire to GoS data loader once repo layout is known."
    )


def retrieve_default_bundle(
    query: str,
    library: dict[str, SkillRecord],
    retrieval_cfg: dict,
    gos_repo: str | Path,
) -> tuple[list[str], dict[str, float]]:
    """Run GoS end-to-end on query, return (bundle, per-skill rho scores).

    Returned rho dict must cover the FULL library (not just the bundle), so
    epsilon_neighbor_swap can find candidates outside top-k.

    TODO(integration): import GoSs retriever module, build the graph (or
    load a cached one), run hybrid seed + reverse-aware PPR + rerank, and
    return both the selected bundle and the dense rho vector.
    """
    _ensure_gos_on_path(gos_repo)
    raise NotImplementedError(
        "retrieve_default_bundle: wire to GoS retrieval pipeline."
    )


def embed(text: str, model_name: str = "text-embedding-3-large") -> np.ndarray:
    """Embed a string. Default model matches GoS paper sec 4.1.

    For the unrelated-swap perturbation we only need cosine similarities
    inside our own pipeline, so this can be backed by either OpenAI or a
    local SentenceTransformer; choose at the script layer.

    TODO(integration): swap to whichever embedder GoS itself uses for the
    skill index, so query-skill cosines are comparable to GoS's seed scores.
    """
    raise NotImplementedError("embed: choose a backend in scripts/.")

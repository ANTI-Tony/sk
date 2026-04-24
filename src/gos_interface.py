"""Wrapper around the GoS retrieval pipeline at github.com/davidliuk/graph-of-skills.

Three calls expose what the rest of the project needs as plain Python types:
    load_library         -> {skill_id: SkillRecord}
    retrieve_default_bundle -> (bundle: list[str], rho: dict[str, float])
    embed                -> np.ndarray (unit-normalized)

Field mapping matches gos.core.schema.SkillNode (raw_content + domain_tags
newline-separated string, with domain_tags_list for the parsed view).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np

from .perturbations import SkillRecord


_FULL_LIBRARY_TOP_N = 10_000          # large enough to score every node in the 200/2000 libraries
_FULL_LIBRARY_CONTEXT_CHARS = 10**9   # disable context-budget pruning so we get the dense PPR vector


def _ensure_gos_on_path(gos_repo: str | Path) -> None:
    p = Path(gos_repo).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(
            f"GoS repo not found at {p}. Run: "
            "git clone https://github.com/davidliuk/graph-of-skills"
        )
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _build_rag(gos_repo: str | Path, workspace: str | Path) -> object:
    """Construct a SkillGraphRAG against a prebuilt or freshly-indexed workspace.

    Embedding model and LLM service come from env (see gos.core.engine
    build_default_*). enable_query_rewrite=False keeps retrieval deterministic
    on the raw query, matching the experiment described in the GoS paper sec 4.1.
    """
    _ensure_gos_on_path(gos_repo)
    from gos import SkillGraphRAG                                          # type: ignore
    from gos.core.engine import (                                          # type: ignore
        build_default_embedding_service,
        build_default_llm_service,
    )

    workspace = str(Path(workspace).expanduser().resolve())
    return SkillGraphRAG(
        working_dir=workspace,
        config=SkillGraphRAG.Config(
            working_dir=workspace,
            prebuilt_working_dir=workspace,
            llm_service=build_default_llm_service(),
            embedding_service=build_default_embedding_service(),
            enable_query_rewrite=False,
        ),
    )


def _run(coro):
    """Run an async coro from sync code without leaking event loops."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return asyncio.run_coroutine_threadsafe(coro, loop).result()
    except RuntimeError:
        pass
    return asyncio.run(coro)


def load_library(gos_repo: str | Path, library_path: str | Path) -> dict[str, SkillRecord]:
    """Parse skill packages on disk into SkillRecords.

    library_path points to data/skillsets/skills_200 (a directory of skill
    package subdirs). Embeddings come from build_default_embedding_service so
    cosine values are comparable to GoS's own seed scoring.
    """
    _ensure_gos_on_path(gos_repo)
    from gos.core.engine import build_default_embedding_service             # type: ignore
    from gos.core.parsing import parse_skill_document                       # type: ignore

    library_path = Path(library_path).expanduser().resolve()
    if not library_path.exists():
        raise FileNotFoundError(
            f"Skill library not found at {library_path}. Run "
            "scripts/download_data.sh --skillsets in the GoS repo."
        )

    embedder = build_default_embedding_service()
    records: dict[str, SkillRecord] = {}

    skill_dirs = [p for p in library_path.iterdir() if p.is_dir()]
    skill_dirs.sort()
    texts_for_embedding: list[str] = []
    pending: list[tuple[str, str]] = []   # (skill_id, primary_tag)

    for sd in skill_dirs:
        spec = sd / "SKILL.md"
        if not spec.exists():
            continue
        parsed = parse_skill_document(spec.read_text())
        if parsed is None:
            continue
        # parse_skill_document returns a SkillNode-like dict; field names
        # follow gos.core.schema.SkillNode. Be defensive about missing fields.
        name = getattr(parsed, "name", None) or sd.name
        domain_raw = getattr(parsed, "domain_tags", "") or ""
        primary_tag = domain_raw.split("\n", 1)[0].strip() or "_unknown"
        text_for_embedding = (
            getattr(parsed, "one_line_capability", "") or ""
        ) + "\n" + (getattr(parsed, "description", "") or "")
        texts_for_embedding.append(text_for_embedding.strip() or name)
        pending.append((name, primary_tag))

    # Batch embed for throughput; embedder.encode returns a list-like of vectors.
    vectors = _run(embedder.encode(texts_for_embedding))
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms

    for (skill_id, tag), vec in zip(pending, arr):
        records[skill_id] = SkillRecord(skill_id=skill_id, embedding=vec, domain_tag=tag)

    return records


def retrieve_default_bundle(
    query: str,
    library: dict[str, SkillRecord],
    retrieval_cfg: dict,
    gos_repo: str | Path,
    workspace: str | Path,
) -> tuple[list[str], dict[str, float]]:
    """Run GoS twice: once under production budget for the default bundle, once
    with effectively unbounded budget so we get PPR scores for the full
    library (needed by epsilon_neighbor_swap).

    The personalization vector and PPR computation depend only on (query,
    nodes, edges) so per-skill scores are identical between the two calls.
    """
    rag = _build_rag(gos_repo, workspace)

    bundle_result = _run(rag.async_retrieve(
        query,
        top_n=retrieval_cfg.get("top_n", 8),
        max_context_chars=retrieval_cfg.get("context_budget_tokens", 8000) * 4,  # rough char proxy
    ))
    bundle = [s.name for s in bundle_result.skills]

    full_result = _run(rag.async_retrieve(
        query,
        top_n=_FULL_LIBRARY_TOP_N,
        max_context_chars=_FULL_LIBRARY_CONTEXT_CHARS,
    ))
    rho = {s.name: float(s.score) for s in full_result.skills}

    # Sanity: every default-bundle skill should also appear in the full result.
    missing = set(bundle) - set(rho)
    if missing:
        raise RuntimeError(
            f"PPR coverage gap: {len(missing)} default-bundle skills missing from full retrieval. "
            "Bump _FULL_LIBRARY_TOP_N or _FULL_LIBRARY_CONTEXT_CHARS."
        )

    return bundle, rho


def embed(text: str, gos_repo: str | Path) -> np.ndarray:
    """Embed a query using the same service as the skill index. Unit-normalized."""
    _ensure_gos_on_path(gos_repo)
    from gos.core.engine import build_default_embedding_service             # type: ignore

    embedder = build_default_embedding_service()
    vec = _run(embedder.encode([text]))
    arr = np.asarray(vec, dtype=np.float32)[0]
    n = float(np.linalg.norm(arr))
    return arr if n == 0 else arr / n

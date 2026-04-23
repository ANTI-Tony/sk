"""Run an agent on (query, bundle) and collect a numeric reward.

Kept deliberately thin: the agent backend, prompt format, and reward function
are all controlled by configs/experiment.yaml. This module only enforces the
contract that a single run produces one RunRecord, and that records are
written to disk after each run (preregistration requires crash-safe logging).
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunRecord:
    run_id: str
    query_id: str
    condition: str                       # "default" | "loo:..." | "unrelated_swap:..." | "eps_swap:..."
    repeat: int                          # 0..K-1
    bundle: list[str]
    bundle_size: int
    reward: float | None
    raw_score: dict[str, Any] = field(default_factory=dict)   # benchmark-specific breakdown
    tokens_in: int | None = None
    tokens_out: int | None = None
    runtime_s: float | None = None
    agent_model: str | None = None
    agent_temperature: float | None = None
    error: str | None = None
    started_at: float = field(default_factory=time.time)


def _new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def hydrate_bundle_into_prompt(
    query: str,
    bundle: list[str],
    library: dict,
) -> str:
    """Materialize selected skills into the agent prompt.

    GoS calls this "hydration" (sec 3.3): inline the canonical name, summary,
    artifacts, and entry point for each selected skill so the agent can call
    them without further lookup.

    TODO(integration): match GoS's hydration template exactly so we are not
    confounding bundle composition with prompt formatting differences.
    """
    raise NotImplementedError("hydrate_bundle_into_prompt: align with GoS template.")


def run_agent_once(
    query: str,
    query_id: str,
    bundle: list[str],
    condition: str,
    repeat: int,
    library: dict,
    agent_cfg: dict,
    results_dir: Path,
) -> RunRecord:
    """Execute one agent run and persist a RunRecord. Always returns; on
    failure writes a record with error set and reward=None so analysis can
    distinguish substantive failure from infrastructure failure.
    """
    record = RunRecord(
        run_id=_new_run_id(),
        query_id=query_id,
        condition=condition,
        repeat=repeat,
        bundle=list(bundle),
        bundle_size=len(bundle),
        reward=None,
        agent_model=agent_cfg.get("model"),
        agent_temperature=agent_cfg.get("temperature"),
    )
    t0 = time.time()
    try:
        # TODO(integration): build prompt, call backend, parse output, score.
        # The scoring function depends on benchmark — for SkillsBench, reuse
        # GoS's reward computation to stay comparable to the published numbers.
        raise NotImplementedError(
            "run_agent_once: wire backend + benchmark scorer once GoS is cloned."
        )
    except NotImplementedError:
        raise
    except Exception as exc:                      # noqa: BLE001
        record.error = repr(exc)
    finally:
        record.runtime_s = time.time() - t0
        _flush(record, results_dir)

    return record


def _flush(record: RunRecord, results_dir: Path) -> None:
    runs_dir = results_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    out = runs_dir / f"{record.query_id}__{record.condition}__r{record.repeat}__{record.run_id}.json"
    out.write_text(json.dumps(asdict(record), indent=2))

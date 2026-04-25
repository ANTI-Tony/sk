"""Run one (query, bundle) trial via Harbor and persist a JSONL line.

The bundle is materialized by symlinking each chosen skill package into a
temp directory, then setting SKILLSBENCH_SKILLS_HOST_DIR so the SkillsBench
docker-compose mounts it at /opt/skillsbench/skills. The agent therefore sees
exactly the bundle we picked -- no GoS runtime retrieval, no full library.

Output schema is fixed by the experiment design:
  query_id, query, bundle_type, skill_ids, skill_names, ppr_scores,
  token_count, agent_output, reward, success, execution_time, error_type
One JSON object per line, appended to results/runs.jsonl.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunRecord:
    query_id: str
    query: str
    bundle_type: str                 # "gos_original" | "delete_top" | "add_irrelevant" | "replace_similar"
    skill_ids: list[str]
    skill_names: list[str]
    ppr_scores: list[float]
    token_count: int | None
    agent_output: str
    reward: float | None
    success: bool
    execution_time: float
    error_type: str | None
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


def _stage_bundle(bundle: list[str], full_library_dir: Path) -> Path:
    """Build a temp dir containing only the bundle's skill packages (symlinks)."""
    stage = Path(tempfile.mkdtemp(prefix="gos_sanity_"))
    for skill_id in bundle:
        src = full_library_dir / skill_id
        if not src.exists():
            shutil.rmtree(stage, ignore_errors=True)
            raise FileNotFoundError(
                f"Skill {skill_id!r} not found in {full_library_dir}"
            )
        (stage / skill_id).symlink_to(src.resolve())
    return stage


def _harbor_run(
    task_dir: Path,
    out_dir: Path,
    agent_cfg: dict,
    bundle_skills_dir: Path,
    timeout_s: int,
) -> subprocess.CompletedProcess:
    backend_to_agent = {"openai": "codex", "anthropic": "claude-code", "gemini": "gemini-cli"}
    cmd = [
        "harbor", "run",
        "--agent", backend_to_agent.get(agent_cfg["backend"], agent_cfg["backend"]),
        "--model", agent_cfg["model"],
        "--force-build",
        "--timeout-multiplier", str(agent_cfg.get("harbor_timeout_multiplier", 5)),
        "-p", str(task_dir),
        "-o", str(out_dir),
    ]
    env = {**os.environ, "SKILLSBENCH_SKILLS_HOST_DIR": str(bundle_skills_dir)}
    return subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=timeout_s, check=False
    )


def _read_harbor_result(out_dir: Path) -> dict:
    """Return verifier_result + agent_result blob from harbor's output.json."""
    candidates = list(out_dir.glob("**/result.json"))
    if not candidates:
        return {}
    trial_results = [c for c in candidates if c.parent.parent == out_dir]
    chosen = trial_results[0] if trial_results else candidates[0]
    return json.loads(chosen.read_text())


def _classify_error(payload: dict, harbor_stderr: str) -> str | None:
    """Coarse error bucket. None when the run completed and verifier returned a reward."""
    if payload.get("verifier_result", {}).get("rewards", {}).get("reward") is not None:
        return None
    if "Timeout" in harbor_stderr or "timeout" in harbor_stderr.lower():
        return "agent_timeout"
    if not payload:
        return "harbor_no_result"
    if payload.get("agent_result") is None:
        return "agent_failed"
    if payload.get("verifier_result") is None:
        return "verifier_failed"
    return "unknown"


def run_once(
    *,
    query_id: str,
    query: str,
    bundle_type: str,
    bundle: list[str],
    library: dict,                       # skill_id -> SkillRecord (for skill_names)
    ppr_scores: dict[str, float],
    agent_cfg: dict,
    paths_cfg: dict,
    results_path: Path,
) -> RunRecord:
    skill_names = [getattr(library.get(s), "name", s) for s in bundle]
    bundle_ppr = [float(ppr_scores.get(s, 0.0)) for s in bundle]

    record = RunRecord(
        query_id=query_id,
        query=query,
        bundle_type=bundle_type,
        skill_ids=list(bundle),
        skill_names=skill_names,
        ppr_scores=bundle_ppr,
        token_count=None,
        agent_output="",
        reward=None,
        success=False,
        execution_time=0.0,
        error_type=None,
    )

    t0 = time.time()
    stage_dir: Path | None = None
    try:
        stage_dir = _stage_bundle(
            bundle, Path(paths_cfg["skills_library"]).expanduser().resolve()
        )
        task_dir = (
            Path(paths_cfg["skillsbench_tasks"]).expanduser().resolve() / query_id
        )
        if not task_dir.exists():
            raise FileNotFoundError(f"SkillsBench task missing: {task_dir}")

        out_dir = Path(paths_cfg["results_dir"]) / "harbor_jobs" / f"{query_id}__{bundle_type}__{record.run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        proc = _harbor_run(
            task_dir, out_dir, agent_cfg, stage_dir,
            timeout_s=int(agent_cfg.get("timeout_s", 1800)),
        )
        payload = _read_harbor_result(out_dir)
        rewards = payload.get("verifier_result", {}).get("rewards", {})
        agent_result = payload.get("agent_result", {}) or {}

        record.reward = (
            float(rewards["reward"]) if rewards.get("reward") is not None else None
        )
        record.success = bool(record.reward and record.reward > 0)
        record.token_count = (
            (agent_result.get("n_input_tokens") or 0) + (agent_result.get("n_output_tokens") or 0)
        ) or None
        record.agent_output = (agent_result.get("final_output") or "")[:8000]
        record.error_type = _classify_error(payload, proc.stderr)

    except subprocess.TimeoutExpired:
        record.error_type = "agent_timeout"
    except FileNotFoundError as exc:
        record.error_type = f"missing:{exc}"
    except Exception as exc:                           # noqa: BLE001
        record.error_type = f"exception:{type(exc).__name__}:{exc}"
    finally:
        record.execution_time = time.time() - t0
        if stage_dir is not None:
            shutil.rmtree(stage_dir, ignore_errors=True)
        _append_jsonl(record, results_path)

    return record


def _append_jsonl(record: RunRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

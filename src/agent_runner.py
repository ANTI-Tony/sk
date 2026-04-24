"""Run a SkillsBench task with a chosen skill bundle and collect reward.

Mechanism (mirrors evaluation/skillsbench/_allskills_template/docker-compose.yaml):
the docker-compose file mounts ${SKILLSBENCH_SKILLS_HOST_DIR} into the agent
container at /opt/skillsbench/skills. By staging a temp directory containing
ONLY the bundle's skill packages and pointing that env var at it, the agent
sees exactly the bundle we choose -- no GoS runtime retrieval, no full library
exposure. This isolates the single experimental knob we care about.

The agent itself is invoked via Harbor (https://github.com/harbor-ai/harbor),
which is what GoS's own SkillsBench runner uses. Reward is the binary
verifier_result.rewards.reward field in result.json.
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
    run_id: str
    query_id: str                    # SkillsBench task name
    condition: str                   # "default" | "loo:..." | "unrelated_swap:..." | "eps_swap:..."
    repeat: int
    bundle: list[str]
    bundle_size: int
    reward: float | None
    raw_score: dict[str, Any] = field(default_factory=dict)
    tokens_in: int | None = None
    tokens_out: int | None = None
    runtime_s: float | None = None
    agent_model: str | None = None
    agent_temperature: float | None = None
    error: str | None = None
    started_at: float = field(default_factory=time.time)


def _new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def _stage_bundle_skills(
    bundle: list[str],
    full_library_dir: Path,
    stage_root: Path,
) -> Path:
    """Materialize a temp skills dir containing only the bundle's packages.

    Uses symlinks to avoid copying large skill assets. Harbor mounts the
    resulting dir read-only into the agent container.
    """
    stage_root.mkdir(parents=True, exist_ok=True)
    bundle_dir = stage_root / f"bundle_{_new_run_id()}"
    bundle_dir.mkdir()
    for skill_id in bundle:
        src = full_library_dir / skill_id
        if not src.exists():
            raise FileNotFoundError(
                f"Skill {skill_id!r} not in {full_library_dir}. "
                "Bundle contains a name that doesn't match a package directory."
            )
        (bundle_dir / skill_id).symlink_to(src.resolve())
    return bundle_dir


def _harbor_run(
    task_dir: Path,
    out_dir: Path,
    agent_cfg: dict,
    bundle_skills_dir: Path,
    timeout_s: int,
) -> subprocess.CompletedProcess:
    """Invoke Harbor against a single SkillsBench task with a custom skills mount.

    agent_cfg keys consumed:
        backend  -> "openai" | "anthropic" -> harbor agent name
        model    -> e.g. "openai/gpt-5.2-codex"
    """
    backend_to_agent = {
        "openai": "codex",
        "anthropic": "claude-code",
        "gemini": "gemini-cli",
    }
    agent_name = backend_to_agent.get(agent_cfg["backend"], agent_cfg["backend"])
    cmd = [
        "harbor", "run",
        "--agent", agent_name,
        "--model", agent_cfg["model"],
        "--force-build",
        "-p", str(task_dir),
        "-o", str(out_dir),
    ]
    env = {**os.environ, "SKILLSBENCH_SKILLS_HOST_DIR": str(bundle_skills_dir)}
    return subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=timeout_s, check=False
    )


def _read_reward(out_dir: Path) -> tuple[float | None, dict]:
    """Pull verifier_result.rewards.reward and a small diagnostic blob from
    Harbor's output. Returns (None, {}) if not found.
    """
    candidates = list(out_dir.glob("**/result.json"))
    if not candidates:
        return None, {}
    # Per-trial result.json sits one level deep; pick that one if present.
    trial_results = [c for c in candidates if c.parent.parent == out_dir]
    chosen = trial_results[0] if trial_results else candidates[0]
    payload = json.loads(chosen.read_text())
    reward = (payload.get("verifier_result") or {}).get("rewards", {}).get("reward")
    diag = {
        "result_path": str(chosen),
        "n_input_tokens": (payload.get("agent_result") or {}).get("n_input_tokens"),
        "n_output_tokens": (payload.get("agent_result") or {}).get("n_output_tokens"),
    }
    return (float(reward) if reward is not None else None), diag


def run_agent_once(
    query: str,                         # unused here; kept for interface symmetry
    query_id: str,                      # SkillsBench task directory name
    bundle: list[str],
    condition: str,
    repeat: int,
    library: dict,                      # unused at runtime; kept for symmetry
    agent_cfg: dict,
    paths_cfg: dict,
    results_dir: Path,
) -> RunRecord:
    """Stage the bundle, run the task, persist a RunRecord."""
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
    stage_root = Path(tempfile.mkdtemp(prefix="gos_sanity_stage_"))
    bundle_dir = None
    try:
        bundle_dir = _stage_bundle_skills(
            bundle,
            full_library_dir=Path(paths_cfg["skills_library"]).expanduser().resolve(),
            stage_root=stage_root,
        )
        # The base task directory lives under evaluation/skillsbench/tasks/<name>/.
        # We deliberately use the allskills-style template path so the agent has
        # no runtime retrieval -- the mounted skills dir IS the bundle.
        task_dir = (
            Path(paths_cfg["skillsbench_tasks"]).expanduser().resolve() / query_id
        )
        if not task_dir.exists():
            raise FileNotFoundError(f"SkillsBench task not found: {task_dir}")

        out_dir = results_dir / "harbor_jobs" / f"{query_id}__{condition}__r{repeat}__{record.run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        proc = _harbor_run(
            task_dir=task_dir,
            out_dir=out_dir,
            agent_cfg=agent_cfg,
            bundle_skills_dir=bundle_dir,
            timeout_s=int(agent_cfg.get("timeout_s", 600)),
        )
        if proc.returncode != 0:
            record.error = f"harbor exit {proc.returncode}: {proc.stderr[-2000:]}"

        reward, diag = _read_reward(out_dir)
        record.reward = reward
        record.raw_score = diag
        record.tokens_in = diag.get("n_input_tokens")
        record.tokens_out = diag.get("n_output_tokens")
    except subprocess.TimeoutExpired:
        record.error = "harbor timed out"
    except Exception as exc:                       # noqa: BLE001
        record.error = repr(exc)
    finally:
        record.runtime_s = time.time() - t0
        if bundle_dir is not None:
            shutil.rmtree(stage_root, ignore_errors=True)
        _flush(record, results_dir)

    return record


def _flush(record: RunRecord, results_dir: Path) -> None:
    runs_dir = results_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    out = runs_dir / f"{record.query_id}__{record.condition}__r{record.repeat}__{record.run_id}.json"
    out.write_text(json.dumps(asdict(record), indent=2))

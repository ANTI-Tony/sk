"""Pick 20 SkillsBench tasks for the sensitivity study.

A SkillsBench task is a directory under evaluation/skillsbench/tasks/<name>/.
Each task has an environment/ subdir and a description (instruction.md or
README.md depending on benchflow vintage). We treat the directory name as
the query_id and the description as the natural-language query.

Stratification: tasks aren't tagged with a single domain in the benchflow
metadata, so we stratify by the FIRST domain tag of the task's "primary"
skill -- a heuristic, not authoritative. If a smarter grouping becomes
available (e.g. SkillsBench publishes domain labels), swap in here.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


def _read_task_description(task_dir: Path) -> str:
    """Return the task's natural-language prompt. Tries common file names."""
    for candidate in ("instruction.md", "INSTRUCTION.md", "README.md", "task.md"):
        f = task_dir / candidate
        if f.exists():
            return f.read_text().strip()
    # Fallback: docker-compose label or just the dir name.
    return task_dir.name


def _infer_stratum(task_dir: Path) -> str:
    """Coarse domain bucket. Looks for a domain hint in the task description's
    first line; falls back to '_unknown'. Good enough for stratified sampling.
    """
    desc = _read_task_description(task_dir).lower()
    keywords = (
        "video", "audio", "image", "pdf", "spreadsheet", "csv", "json", "sql",
        "finance", "seismic", "macroeconomic", "power-grid", "scan",
    )
    for kw in keywords:
        if kw in desc:
            return kw
    return "_unknown"


def stratified_sample(
    items: list[tuple[str, str, str]],   # (task_id, prompt, stratum)
    n: int,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    by_stratum: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for it in items:
        by_stratum[it[2]].append(it)
    for bucket in by_stratum.values():
        rng.shuffle(bucket)

    picked: list[tuple[str, str, str]] = []
    strata = list(by_stratum.keys())
    rng.shuffle(strata)
    while len(picked) < n:
        progress = False
        for s in strata:
            if len(picked) >= n:
                break
            if by_stratum[s]:
                picked.append(by_stratum[s].pop())
                progress = True
        if not progress:
            break

    return [
        {"id": tid, "prompt": prompt, "stratum": stratum}
        for tid, prompt, stratum in picked
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--out", default="data/queries.json")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    tasks_root = Path(cfg["paths"]["skillsbench_tasks"]).expanduser().resolve()
    if not tasks_root.exists():
        raise SystemExit(
            f"SkillsBench tasks not found at {tasks_root}. In the GoS repo run: "
            "./scripts/download_data.sh --tasks"
        )

    items: list[tuple[str, str, str]] = []
    for td in sorted(tasks_root.iterdir()):
        if not td.is_dir():
            continue
        if not (td / "environment").exists():
            continue
        items.append((td.name, _read_task_description(td), _infer_stratum(td)))

    if not items:
        raise SystemExit(f"No tasks found under {tasks_root}.")

    picked = stratified_sample(
        items,
        n=cfg["queries"]["count"],
        seed=cfg["queries"]["selection_seed"],
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(picked, indent=2))
    print(f"Wrote {len(picked)} queries to {out}")
    by_stratum = defaultdict(int)
    for q in picked:
        by_stratum[q["stratum"]] += 1
    for s, n in sorted(by_stratum.items()):
        print(f"  {s:20s} {n}")


if __name__ == "__main__":
    main()

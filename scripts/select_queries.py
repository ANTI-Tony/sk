"""Pick the 20 SkillsBench queries used throughout the sanity check.

Stratified by domain_tag to cover SkillsBench's 11 domains. Selection is
deterministic via configs/experiment.yaml: queries.selection_seed.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


def stratified_sample(
    tasks: list[dict],
    n: int,
    seed: int,
) -> list[dict]:
    """Round-robin across domain_tag buckets until n picked. Falls back to
    random fill from any bucket if some domains are short.
    """
    rng = np.random.default_rng(seed)
    by_tag: dict[str, list[dict]] = defaultdict(list)
    for t in tasks:
        by_tag[t.get("domain_tag", "_unknown")].append(t)
    for bucket in by_tag.values():
        rng.shuffle(bucket)

    picked: list[dict] = []
    tags = list(by_tag.keys())
    rng.shuffle(tags)
    while len(picked) < n:
        progress = False
        for tag in tags:
            if len(picked) >= n:
                break
            if by_tag[tag]:
                picked.append(by_tag[tag].pop())
                progress = True
        if not progress:
            break
    return picked


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--out", default="data/queries.json")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    tasks_path = Path(cfg["paths"]["skillsbench_tasks"])

    if not tasks_path.exists():
        raise SystemExit(
            f"SkillsBench task set not found at {tasks_path}. "
            "Clone the GoS repo and update configs/experiment.yaml::paths.skillsbench_tasks."
        )

    # TODO(integration): replace with the real loader once the SkillsBench
    # task layout is known. Expected per-task fields: id, prompt, domain_tag,
    # reference / scoring metadata.
    tasks = json.loads((tasks_path / "tasks.json").read_text())

    picked = stratified_sample(
        tasks,
        n=cfg["queries"]["count"],
        seed=cfg["queries"]["selection_seed"],
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(picked, indent=2))
    print(f"Wrote {len(picked)} queries to {out}")


if __name__ == "__main__":
    main()

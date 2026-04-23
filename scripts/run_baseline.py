"""Run the default GoS bundle K times per query to estimate sigma_within.

This MUST complete (and yield non-trivial sigma) before run_perturbations.py.
If sigma is large (say >= 0.5 * delta_threshold), we cannot detect the
preregistered effect with K=3; bump K=5 in configs/experiment.yaml and rerun.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from tqdm import tqdm

from src.agent_runner import run_agent_once
from src.gos_interface import load_library, retrieve_default_bundle


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--prereg", default="configs/preregistration.yaml")
    p.add_argument("--queries", default="data/queries.json")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    prereg = yaml.safe_load(Path(args.prereg).read_text())
    queries = json.loads(Path(args.queries).read_text())

    library = load_library(cfg["paths"]["gos_repo"], cfg["paths"]["skills_library"])
    K = prereg["sample_size"]["repeats_default"]
    results_dir = Path(cfg["paths"]["results_dir"])

    for q in tqdm(queries, desc="queries"):
        bundle, _rho = retrieve_default_bundle(
            q["prompt"], library, cfg["retrieval"], cfg["paths"]["gos_repo"]
        )
        for k in range(K):
            run_agent_once(
                query=q["prompt"],
                query_id=q["id"],
                bundle=bundle,
                condition="default",
                repeat=k,
                library=library,
                agent_cfg=cfg["agent"],
                results_dir=results_dir,
            )

    print(f"Baseline complete. Inspect with: python scripts/analyze_results.py --variance-only")


if __name__ == "__main__":
    main()

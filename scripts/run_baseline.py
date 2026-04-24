"""Run the default GoS bundle K times per query to estimate sigma_within.

Must complete -- and yield a non-trivial sigma -- before run_perturbations.py.
If pooled sigma >= 0.5 * delta_threshold, bump K in preregistration.yaml or
chase down sources of stochasticity (temperature != 0, timestamps in prompts,
seed mismatches in the agent backend).
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
    results_dir = Path(cfg["paths"]["results_dir"]).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    bundle_cache_path = results_dir / "default_bundles.json"
    bundle_cache: dict[str, dict] = {}
    if bundle_cache_path.exists():
        bundle_cache = json.loads(bundle_cache_path.read_text())

    for q in tqdm(queries, desc="queries"):
        if q["id"] in bundle_cache:
            bundle = bundle_cache[q["id"]]["bundle"]
        else:
            bundle, rho = retrieve_default_bundle(
                q["prompt"],
                library,
                cfg["retrieval"],
                cfg["paths"]["gos_repo"],
                cfg["paths"]["gos_workspace"],
            )
            bundle_cache[q["id"]] = {"bundle": bundle, "rho": rho}
            bundle_cache_path.write_text(json.dumps(bundle_cache, indent=2))

        for k in range(K):
            run_agent_once(
                query=q["prompt"],
                query_id=q["id"],
                bundle=bundle,
                condition="default",
                repeat=k,
                library=library,
                agent_cfg=cfg["agent"],
                paths_cfg=cfg["paths"],
                results_dir=results_dir,
            )

    print(
        "Baseline complete. Inspect with: "
        "python scripts/analyze_results.py --variance-only"
    )


if __name__ == "__main__":
    main()

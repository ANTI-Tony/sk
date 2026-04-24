"""Run the full sanity check: 20 queries x 4 bundles = 80 agent rollouts.

For each query:
  1. Retrieve GoS default bundle + full-library PPR scores.
  2. Build 3 perturbed bundles: delete_top, add_irrelevant, replace_similar.
  3. Run all 4 bundles through harbor, append a JSONL line per run.

Crash-safe: results/runs.jsonl is appended after every run. Re-running this
script with the same queries will produce duplicate JSONL lines -- analyze.py
groups by (query_id, bundle_type) so the latest still drives the verdict, but
clean state is wiping results/runs.jsonl beforehand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from src.agent_runner import run_once
from src.gos_interface import embed_query, load_library, retrieve
from src.perturbations import add_irrelevant, delete_top, replace_similar


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--queries", default="data/queries.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    queries = json.loads(Path(args.queries).read_text())
    library = load_library(cfg["paths"]["gos_repo"], cfg["paths"]["skills_library"])

    results_dir = Path(cfg["paths"]["results_dir"]).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = results_dir / "runs.jsonl"
    rng = np.random.default_rng(cfg["queries"]["selection_seed"])

    bundles_cache = results_dir / "default_bundles.json"
    cache: dict = json.loads(bundles_cache.read_text()) if bundles_cache.exists() else {}

    for q in tqdm(queries, desc="queries"):
        qid, query = q["id"], q["query"]

        # 1. Default bundle + full-library PPR (cache so reruns dont repay GoS)
        if qid in cache:
            bundle = cache[qid]["bundle"]
            ppr = cache[qid]["ppr"]
        else:
            bundle, ppr = retrieve(
                query,
                gos_repo=cfg["paths"]["gos_repo"],
                workspace=cfg["paths"]["gos_workspace"],
                top_n=cfg["retrieval"]["top_n"],
                max_context_chars=cfg["retrieval"]["max_context_chars"],
            )
            cache[qid] = {"bundle": bundle, "ppr": ppr}
            bundles_cache.write_text(json.dumps(cache, indent=2))

        q_emb = embed_query(query, cfg["paths"]["gos_repo"])

        # 2. Build the 3 perturbations
        b_del = delete_top(bundle, ppr)
        b_add = add_irrelevant(
            bundle, library, q_emb,
            cosine_max=cfg["perturbations"]["add_irrelevant_cosine_max"],
            rng=rng,
        )
        rep = replace_similar(
            bundle, ppr,
            rng=rng,
            rho_epsilon=cfg["perturbations"]["replace_similar_rho_epsilon"],
        )

        plan = [("gos_original", bundle), ("delete_top", b_del)]
        if b_add is not None:
            plan.append(("add_irrelevant", b_add))
        else:
            print(f"[skip] {qid} add_irrelevant: no candidate below cosine_max")
        if rep is not None:
            new_bundle, _, _ = rep
            plan.append(("replace_similar", new_bundle))
        else:
            print(f"[skip] {qid} replace_similar: no PPR-neighbor within epsilon")

        # 3. Run each
        for bundle_type, b in plan:
            run_once(
                query_id=qid,
                query=query,
                bundle_type=bundle_type,
                bundle=b,
                library=library,
                ppr_scores=ppr,
                agent_cfg=cfg["agent"],
                paths_cfg=cfg["paths"],
                results_path=jsonl_path,
            )

    print(f"\nDone. Inspect with: python scripts/analyze.py")


if __name__ == "__main__":
    main()

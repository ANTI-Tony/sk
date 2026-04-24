"""Run the three perturbation families on each query's default bundle.

Run AFTER scripts/run_baseline.py has produced sigma_within. The default
bundle and rho scores come from results/default_bundles.json (written by the
baseline script) so we don't pay for a second GoS retrieval per query.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from src.agent_runner import run_agent_once
from src.gos_interface import embed, load_library
from src.perturbations import (
    epsilon_neighbor_swap,
    leave_one_out,
    unrelated_swap,
)


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
    K = prereg["sample_size"]["repeats_perturbation"]
    results_dir = Path(cfg["paths"]["results_dir"]).expanduser().resolve()
    bundle_cache_path = results_dir / "default_bundles.json"
    if not bundle_cache_path.exists():
        raise SystemExit(
            "Missing results/default_bundles.json. Run scripts/run_baseline.py first."
        )
    bundle_cache = json.loads(bundle_cache_path.read_text())
    rng = np.random.default_rng(abs(hash(prereg.get("study_id", "seed"))) & 0xFFFFFFFF)

    pcfg = cfg["perturbations"]
    for q in tqdm(queries, desc="queries"):
        cached = bundle_cache.get(q["id"])
        if cached is None:
            print(f"WARN: no cached bundle for {q['id']}, skipping")
            continue
        bundle: list[str] = cached["bundle"]
        rho: dict[str, float] = cached["rho"]
        q_emb = embed(q["prompt"], cfg["paths"]["gos_repo"])

        loo_pairs = leave_one_out(bundle)

        unrel_pairs = []
        for sid in bundle:
            res = unrelated_swap(
                bundle, library, q_emb, sid,
                cosine_max=pcfg["unrelated_swap"]["cosine_max"],
                require_different_domain_tag=pcfg["unrelated_swap"]["require_different_domain_tag"],
                rng=rng,
            )
            if res is not None:
                unrel_pairs.append(res)

        eps_pairs = []
        for sid in bundle:
            res = epsilon_neighbor_swap(
                bundle, rho, sid,
                rho_epsilon=pcfg["epsilon_neighbor_swap"]["rho_epsilon"],
                must_be_outside_topk=pcfg["epsilon_neighbor_swap"]["must_be_outside_topk"],
                rng=rng,
            )
            if res is not None:
                eps_pairs.append(res)

        for label, pert_bundle in loo_pairs + unrel_pairs + eps_pairs:
            for k in range(K):
                run_agent_once(
                    query=q["prompt"],
                    query_id=q["id"],
                    bundle=pert_bundle,
                    condition=label,
                    repeat=k,
                    library=library,
                    agent_cfg=cfg["agent"],
                    paths_cfg=cfg["paths"],
                    results_dir=results_dir,
                )


if __name__ == "__main__":
    main()

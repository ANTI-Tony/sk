"""Run the three perturbation families on each query's default bundle.

Run AFTER scripts/run_baseline.py has produced sigma_within. If sigma is too
large, fix that first — perturbation runs without a noise floor are wasted
compute.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from src.agent_runner import run_agent_once
from src.gos_interface import embed, load_library, retrieve_default_bundle
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
    results_dir = Path(cfg["paths"]["results_dir"])
    rng = np.random.default_rng(prereg.get("study_id", "seed").__hash__() & 0xFFFFFFFF)

    pcfg = cfg["perturbations"]
    for q in tqdm(queries, desc="queries"):
        bundle, rho = retrieve_default_bundle(
            q["prompt"], library, cfg["retrieval"], cfg["paths"]["gos_repo"]
        )
        q_emb = embed(q["prompt"])

        # 1) leave-one-out: |B| perturbations
        loo_pairs = leave_one_out(bundle)

        # 2) unrelated swap: one per bundle skill (skip if no candidate)
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

        # 3) epsilon-neighbor swap: one per bundle skill (skip if no candidate)
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
                    results_dir=results_dir,
                )


if __name__ == "__main__":
    main()

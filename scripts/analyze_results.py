"""Apply the pre-registered decision rule and print a verdict.

--variance-only: print sigma_within report after baseline runs (decision gate
                 before launching perturbations).
default mode:    full verdict including family-level effect sizes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.analyze import decide
from src.variance import estimate_sigma_within, load_default_runs, pooled_sigma_within


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--prereg", default="configs/preregistration.yaml")
    p.add_argument("--variance-only", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    prereg = yaml.safe_load(Path(args.prereg).read_text())
    results_dir = Path(cfg["paths"]["results_dir"])

    if args.variance_only:
        runs = load_default_runs(results_dir)
        sigmas = estimate_sigma_within(runs)
        report = {
            "per_query_sigma": sigmas,
            "pooled_sigma": pooled_sigma_within(sigmas),
            "queries_with_enough_runs": sum(1 for s in sigmas.values() if s is not None),
            "delta_threshold": prereg["delta_threshold"],
            "advisory": (
                "Bump K in configs if pooled_sigma >= 0.5 * delta_threshold."
            ),
        }
        print(json.dumps(report, indent=2, default=str))
        return

    verdict = decide(
        results_dir=results_dir,
        delta_threshold=prereg["delta_threshold"],
        variance_ratio_required=prereg["variance_ratio_required"],
    )
    print(json.dumps(verdict, indent=2, default=str))


if __name__ == "__main__":
    main()

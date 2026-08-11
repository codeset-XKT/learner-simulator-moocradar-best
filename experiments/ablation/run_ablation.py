from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common import (  # noqa: E402
    add_shared_arguments,
    load_fixed_cohort,
    metric_view,
    run_experiment,
    save_report,
)


ABLATIONS = {
    "full": {},
    "no-learner-state-profile": {
        "profile": False,
        "cognitive_profile": False,
        "ability_profile": False,
    },
    "no-item-conditioned-integration": {"item_conditioned_ability": False},
    "no-dynamic-state-evolution": {"dynamic_state_evolution": False},
    "no-ncdm": {"ncdm_evidence": False},
    "no-irt": {"irt_evidence": False},
    "no-ncdm-irt": {
        "ncdm_evidence": False,
        "irt_evidence": False,
    },
    "no-four-tier": {"four_tier": False},
}

PAPER_ABLATIONS = {
    "no-learner-state-profile",
    "no-item-conditioned-integration",
    "no-dynamic-state-evolution",
    "no-ncdm",
    "no-irt",
    "no-ncdm-irt",
    "no-four-tier",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run key-module ablations.")
    add_shared_arguments(parser)
    parser.add_argument(
        "--variants",
        default=(
            "full,no-learner-state-profile,no-item-conditioned-integration,"
            "no-four-tier,no-dynamic-state-evolution"
        ),
        help="Comma-separated ablation variants.",
    )
    parser.add_argument(
        "--parallel-variants",
        type=int,
        default=1,
        help="Number of ablation variants to run concurrently.",
    )
    parser.add_argument(
        "--variant-start-stagger-seconds",
        type=float,
        default=0.0,
        help=(
            "Delay each parallel variant's first request by this many seconds "
            "times its variant index. This avoids simultaneous streaming-request "
            "handshakes without changing experiment semantics."
        ),
    )
    parser.add_argument(
        "--simulator",
        choices=["full", "multi-role"],
        default="multi-role",
        help=(
            "Simulator used for ablations. The paper-facing suite targets "
            "the current multi-role method."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="outputs/ablation/checkpoints",
        help="Save each completed variant immediately in this directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = set(selected) - set(ABLATIONS)
    if unknown:
        raise ValueError(f"Unknown ablations: {sorted(unknown)}")
    incompatible = set(selected) & PAPER_ABLATIONS
    if args.simulator != "multi-role" and incompatible:
        raise ValueError(
            "Paper-facing ablations require --simulator multi-role: "
            f"{sorted(incompatible)}"
        )

    questions, history_rows, target_rows = load_fixed_cohort(args)
    variant_index = {variant: index for index, variant in enumerate(selected)}

    def run_variant(variant: str):
        stagger = max(0.0, float(args.variant_start_stagger_seconds))
        if stagger:
            time.sleep(stagger * variant_index[variant])
        modules = {
            "profile": True,
            "memory": True,
            "cognitive_profile": True,
            "ability_profile": True,
            "item_conditioned_ability": True,
            "four_tier": True,
            "historical_reflection": True,
            "irt_evidence": True,
            "ncdm_evidence": True,
            "dynamic_state_evolution": True,
            **ABLATIONS[variant],
        }
        report = run_experiment(
            args.simulator,
            args,
            questions,
            history_rows,
            target_rows,
            modules=modules,
            progress_name=variant,
        )
        report["ablation_variant"] = variant
        checkpoint = Path(args.checkpoint_dir) / f"{variant}.json"
        save_report(report, checkpoint)
        return variant, report

    reports = {}
    workers = max(1, min(args.parallel_variants, len(selected)))
    if workers == 1:
        for variant in selected:
            name, report = run_variant(variant)
            reports[name] = report
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(run_variant, variant): variant
                for variant in selected
            }
            for future in as_completed(futures):
                name, report = future.result()
                reports[name] = report

    reports = {variant: reports[variant] for variant in selected}

    combined = {
        "study": "ablation",
        "variants": selected,
        "same_fixed_cohort": True,
        "summary": {name: metric_view(report) for name, report in reports.items()},
        "reports": reports,
    }
    output = args.output or "outputs/ablation/ablation.json"
    path = save_report(combined, output)
    print(json.dumps({"ok": True, "output": str(path), **combined["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

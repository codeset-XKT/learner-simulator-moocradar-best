from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common import (
    add_shared_arguments,
    load_fixed_cohort,
    metric_view,
    run_experiment,
    save_report,
)


ABLATIONS = {
    "full": {},
    "no-profile": {"profile": False},
    "no-memory": {"memory": False},
    "no-proficiency": {"proficiency": False},
    "no-four-tier": {"four_tier": False},
    "no-cognitive-selection": {"cognitive_strategy": False},
    "no-cognitive-profile": {"cognitive_profile": False},
    "no-ability-profile": {"ability_profile": False},
    "no-irt-evidence": {"irt_evidence": False},
    "no-learning-tool-state": {"learning_tool_state": False},
    "no-item-conditioned-ability": {"item_conditioned_ability": False},
    "no-historical-reflection": {"historical_reflection": False},
    "with-dkt-predictor": {"dkt_predictor": True},
    "best-full": {"item_conditioned_ability": False},
    "best-no-ability-profile": {
        "ability_profile": False,
        "item_conditioned_ability": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run key-module ablations.")
    add_shared_arguments(parser)
    parser.add_argument(
        "--variants",
        default="full,no-profile,no-memory,no-proficiency,no-four-tier",
        help="Comma-separated ablation variants.",
    )
    parser.add_argument(
        "--parallel-variants",
        type=int,
        default=1,
        help="Number of ablation variants to run concurrently.",
    )
    parser.add_argument(
        "--simulator",
        choices=["full", "multi-role"],
        default="full",
        help=(
            "Simulator used for ablations. Use multi-role for "
            "no-item-conditioned-ability."
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

    questions, history_rows, target_rows = load_fixed_cohort(args)
    def run_variant(variant: str):
        modules = {
            "profile": True,
            "memory": True,
            "proficiency": True,
            "behavior": True,
            "four_tier": True,
            "historical_reflection": True,
            "irt_evidence": True,
            "learning_tool_state": True,
            "dkt_predictor": False,
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

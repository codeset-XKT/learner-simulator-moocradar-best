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

from learner_simulator.formal_metrics import (  # noqa: E402
    build_ability_difficulty_partition,
    evaluate_formal_response_metrics,
)

from experiments.common import (  # noqa: E402
    add_shared_arguments,
    load_fixed_cohort,
    metric_view,
    run_experiment,
    save_report,
)


ABLATIONS = {
    "full": {},
    "no-evidence-representation": {"evidence_representation": False},
    "no-state-item-alignment": {"state_item_alignment": False},
    "no-structured-response-process": {"structured_response": False},
    "no-dynamic-state-evolution": {"dynamic_state_evolution": False},
    "direct-response-generation": {
        "evidence_representation": False,
        "state_item_alignment": False,
        "structured_response": False,
        "dynamic_state_evolution": False,
    },
}

PAPER_ABLATIONS = {
    "no-evidence-representation",
    "no-state-item-alignment",
    "no-structured-response-process",
    "no-dynamic-state-evolution",
    "direct-response-generation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run key-module ablations.")
    add_shared_arguments(parser)
    parser.add_argument(
        "--variants",
        default=(
            "full,no-evidence-representation,no-state-item-alignment,"
            "no-structured-response-process,no-dynamic-state-evolution"
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
            "evidence_representation": True,
            "state_item_alignment": True,
            "structured_response": True,
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
        report["method_version"] = "formal-2026-08-26-v6-primaryroute-auditable-formalmetrics"
        report["method_architecture"] = {
            "conceptual_modules": [
                "traceable-learner-evidence-representation",
                "cognitive-state-item-alignment",
                "structured-response-generation",
                "auditable-state-evolution",
            ],
            "sequential_protocol": "sequential-feedback-and-state-update",
            "response_contract": "single_pass_process_verifiable_v6_label_conditioned_answer_realization",
            "information_flow": "module1_to_module2_to_module3_to_module4_no_bypass",
            "evidence_selection": "fixed_role_hierarchy_without_manual_weight_fusion",
            "multi_concept_policy": "primary_kc_route_only; preserve_primary_route_hierarchy_for_evidence",
            "process_diagnostics": "evidence_grounded_process_audit_v1_noncausal",
            "api_calls_per_item": 1,
            "irt_calibration_scope": "cohort-history-transductive-no-target-labels",
            "confidence_contract": "discrete_numeric_anchors_0.20_0.50_0.80",
            "historical_context_policy": {
                "external_repository": "all_observed_history",
                "prompt_max_evidence": 4,
                "prompt_max_full_questions": 2,
                "prompt_evidence_char_budget": 2400,
                "agent_memory": False,
                "reflection": False,
            },
        }
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

    formal_metrics = _formal_metrics_on_common_population(reports, selected)
    combined = {
        "study": "ablation",
        "variants": selected,
        "same_fixed_cohort": True,
        "summary": {name: metric_view(report) for name, report in reports.items()},
        "formal_metrics_v6": formal_metrics,
        "reports": reports,
    }
    output = args.output or "outputs/ablation/ablation.json"
    path = save_report(combined, output)
    print(json.dumps({"ok": True, "output": str(path), **combined["summary"]}, ensure_ascii=False, indent=2))


def _formal_metrics_on_common_population(
    reports: dict[str, dict],
    selected: list[str],
) -> dict:
    """Evaluate every selected variant on exactly the same valid learners."""
    if "full" not in reports:
        return {
            "available": False,
            "reason": "the fixed ADCDE partition requires a full-model report",
        }
    per_variant_steps: dict[str, list[dict]] = {}
    uid_sets: list[set[str]] = []
    for variant in selected:
        report = reports[variant]
        excluded = {str(uid) for uid in report.get("metric_population", {}).get("excluded_uids", [])}
        steps = [
            step for step in (report.get("all_steps") or [])
            if str(step.get("uid")) not in excluded
            and step.get("real_response") in {0, 1}
            and step.get("simulated_response") in {0, 1}
        ]
        per_variant_steps[variant] = steps
        uid_sets.append({str(step.get("uid")) for step in steps})
    common_uids = set.intersection(*uid_sets) if uid_sets else set()
    fair_steps = {
        variant: [step for step in steps if str(step.get("uid")) in common_uids]
        for variant, steps in per_variant_steps.items()
    }
    partition = build_ability_difficulty_partition(fair_steps["full"])
    return {
        "available": True,
        "metric_version": "formal_response_metrics_v6",
        "fairness_policy": "intersection_of_valid_learner_uids_across_selected_variants",
        "common_uid_count": len(common_uids),
        "common_step_count": len(fair_steps["full"]),
        "fixed_full_irt_2x2_partition": partition,
        "methods": {
            variant: evaluate_formal_response_metrics(steps, partition)
            for variant, steps in fair_steps.items()
        },
    }


if __name__ == "__main__":
    main()

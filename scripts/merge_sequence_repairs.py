from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.common import (  # noqa: E402
    metric_view,
    runtime_summary,
    save_report,
    valid_metric_steps,
    validity_summary,
)
from learner_simulator.evaluation import evaluate_steps  # noqa: E402
from learner_simulator.evaluation_views import layered_metric_view  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace failed learner sequences with fit-consistent repair runs."
    )
    parser.add_argument("--base", required=True)
    parser.add_argument("--repair", action="append", required=True)
    parser.add_argument("--variant", default="full")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--allow-repair-token-budget-increase",
        action="store_true",
        help=(
            "Allow a repair config whose only LLM differences are larger "
            "max_tokens and/or timeout_seconds values. The operational exception "
            "is recorded in the merged report."
        ),
    )
    return parser.parse_args()


def load_payload(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def report_for(payload: dict, variant: str) -> dict:
    reports = payload.get("reports")
    if not isinstance(reports, dict) or variant not in reports:
        raise ValueError(f"Missing reports.{variant}")
    return reports[variant]


def checkpoint_hash(report: dict) -> str | None:
    grounding = report.get("simulator_summary", {}).get(
        "ncdm_checkpoint_grounding", {}
    )
    return grounding.get("checkpoint_sha256")


def compatibility_signature(report: dict) -> dict:
    archive = report.get("archive", {})
    arguments = archive.get("arguments", {})
    return {
        "ncdm_checkpoint_sha256": checkpoint_hash(report),
        "cohort_file": arguments.get("cohort_file"),
        "feedback_mode": arguments.get("feedback_mode"),
        "llm_config_path": arguments.get("llm_config"),
        "llm_config": archive.get("llm_config"),
        "implementation_fingerprint": archive.get("implementation_fingerprint"),
        "threshold": arguments.get("threshold"),
        "protocol": report.get("protocol"),
        "modules": report.get("modules"),
        "fit_cohort_summary": report.get("fit_cohort_summary"),
    }


def token_budget_only_compatibility(
    expected_signature: dict, repair_signature: dict
) -> dict | None:
    expected = copy.deepcopy(expected_signature)
    repair = copy.deepcopy(repair_signature)
    expected_path = expected.pop("llm_config_path", None)
    repair_path = repair.pop("llm_config_path", None)
    expected.pop("llm_config", None)
    repair.pop("llm_config", None)
    if expected != repair:
        return None

    def load_config(path_value: str | None) -> dict:
        if not path_value:
            return {}
        path = Path(path_value)
        if not path.is_absolute():
            path = ROOT / path
        return load_payload(str(path))

    expected_config = load_config(expected_path)
    repair_config = load_config(repair_path)
    expected_tokens = expected_config.pop("max_tokens", None)
    repair_tokens = repair_config.pop("max_tokens", None)
    expected_timeout = expected_config.pop("timeout_seconds", None)
    repair_timeout = repair_config.pop("timeout_seconds", None)
    if expected_config != repair_config:
        return None
    if not isinstance(expected_tokens, (int, float)) or not isinstance(
        repair_tokens, (int, float)
    ):
        return None
    if repair_tokens < expected_tokens:
        return None
    if not isinstance(expected_timeout, (int, float)) or not isinstance(
        repair_timeout, (int, float)
    ):
        return None
    if repair_timeout < expected_timeout:
        return None
    return {
        "type": "operational_budget_increase_only",
        "base_config_path": expected_path,
        "repair_config_path": repair_path,
        "base_max_tokens": expected_tokens,
        "repair_max_tokens": repair_tokens,
        "base_timeout_seconds": expected_timeout,
        "repair_timeout_seconds": repair_timeout,
    }


def complete_sequences(report: dict) -> dict[str, list[dict]]:
    expected = int(report.get("protocol", {}).get("target_steps", 0))
    grouped: dict[str, list[dict]] = {}
    for step in report.get("all_steps", []):
        grouped.setdefault(str(step["uid"]), []).append(step)
    return {
        uid: steps
        for uid, steps in grouped.items()
        if len(steps) == expected
        and all(
            not step.get("llm_error")
            and step.get("prediction_valid", True)
            and isinstance(step.get("llm_parsed_action"), dict)
            for step in steps
        )
    }


def replace_sequences(
    base_steps: list[dict], replacements: dict[str, list[dict]]
) -> list[dict]:
    output: list[dict] = []
    emitted: set[str] = set()
    base_uids = {str(step["uid"]) for step in base_steps}
    unknown = sorted(set(replacements) - base_uids)
    if unknown:
        raise ValueError(f"Repair UIDs are absent from the base report: {unknown}")
    for step in base_steps:
        uid = str(step["uid"])
        if uid not in replacements:
            output.append(step)
        elif uid not in emitted:
            output.extend(replacements[uid])
            emitted.add(uid)
    return output


def main() -> None:
    args = parse_args()
    base_payload = load_payload(args.base)
    base_report = report_for(base_payload, args.variant)
    expected_signature = compatibility_signature(base_report)
    steps = list(base_report.get("all_steps", []))
    if not steps:
        raise ValueError("Base report does not contain all_steps")

    prior_repairs = base_report.get("sequence_repairs", {})
    repair_records = copy.deepcopy(prior_repairs.get("sources", []))
    attempted_steps = int(
        base_report.get("runtime", {}).get(
            "actual_attempted_llm_steps_including_repairs",
            len(steps),
        )
    )
    total_seconds = float(base_report.get("runtime", {}).get("total_seconds", 0.0))
    replaced_uids: set[str] = set(prior_repairs.get("replaced_uids", []))
    prior_source_run_count = int(
        base_report.get("runtime", {}).get("source_run_count", 1)
    )
    new_repair_count = 0
    for repair_path in args.repair:
        payload = load_payload(repair_path)
        report = report_for(payload, args.variant)
        repair_signature = compatibility_signature(report)
        compatibility_exception = None
        if repair_signature != expected_signature:
            mismatches = sorted(
                key
                for key in expected_signature
                if expected_signature[key] != repair_signature.get(key)
            )
            if args.allow_repair_token_budget_increase:
                compatibility_exception = token_budget_only_compatibility(
                    expected_signature, repair_signature
                )
            if compatibility_exception is None:
                raise ValueError(
                    f"Incompatible repair report {repair_path}; mismatched fields: "
                    f"{mismatches}"
                )
        replacements = complete_sequences(report)
        if not replacements:
            raise ValueError(f"No complete valid learner sequence in {repair_path}")
        steps = replace_sequences(steps, replacements)
        replaced_uids.update(replacements)
        runtime = report.get("runtime", {})
        attempted_steps += int(runtime.get("steps", 0))
        total_seconds += float(runtime.get("total_seconds", 0.0))
        repair_records.append(
            {
                "path": str(Path(repair_path)),
                "accepted_uids": sorted(replacements),
                "source_validity": report.get("validity"),
                "source_runtime": runtime,
                "compatibility_exception": compatibility_exception,
            }
        )
        new_repair_count += 1

    metric_steps, excluded_uids = valid_metric_steps(
        str(base_report.get("experiment", "multi-role")), steps
    )
    report = copy.deepcopy(base_report)
    threshold = float(base_report.get("metrics", {}).get("threshold", 0.5))
    report["metrics"] = evaluate_steps(metric_steps, threshold=threshold)
    report["validity"] = validity_summary(
        str(report.get("experiment", "multi-role")), steps
    )
    report["metric_population"] = {
        "included_steps": len(metric_steps),
        "excluded_steps": len(steps) - len(metric_steps),
        "excluded_uids": excluded_uids,
        "policy": "exclude_entire_learner_sequence_after_any_llm_failure",
    }
    report["runtime"] = runtime_summary(steps, total_seconds)
    report["runtime"].update(
        {
            "actual_attempted_llm_steps_including_repairs": attempted_steps,
            "source_run_count": prior_source_run_count + new_repair_count,
        }
    )
    report["sample_steps"] = steps[:20]
    report["all_steps"] = steps
    report["sequence_repairs"] = {
        "policy": "replace_complete_learner_sequence_only",
        "replaced_uids": sorted(replaced_uids),
        "sources": repair_records,
    }
    report["metric_layers"] = layered_metric_view(
        {
            "name": report.get("experiment", "multi-role"),
            "source": "sequence_repaired_run",
            "report": report,
            "metrics": report["metrics"],
            "validity": report["validity"],
            "runtime": report["runtime"],
        }
    )

    combined = copy.deepcopy(base_payload)
    combined["reports"][args.variant] = report
    combined["summary"][args.variant] = metric_view(report)
    combined["sequence_repair_summary"] = report["sequence_repairs"]
    output = save_report(combined, args.output)
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(output),
                "replaced_uids": sorted(replaced_uids),
                args.variant: combined["summary"][args.variant],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

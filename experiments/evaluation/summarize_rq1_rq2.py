from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.common import save_report, valid_metric_steps  # noqa: E402
from learner_simulator.evaluation import (  # noqa: E402
    accuracy,
    balanced_accuracy,
    evaluate_steps,
    f1_score,
    matthews_corrcoef,
    specificity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate saved full traces for RQ1 and RQ2 without rerunning an LLM."
    )
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument(
        "--dataset-name",
        default="MoocRadar",
        help="Dataset label used in the JSON payload and Markdown title.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = [_load_full_report(_resolve(path)) for path in args.inputs]
    steps: list[dict[str, Any]] = []
    batch_uids: list[set[str]] = []
    excluded_uids: list[str] = []
    for report in reports:
        current_steps = list(report.get("all_steps") or [])
        if not current_steps:
            raise ValueError("Every input must contain all_steps; summary-only files are insufficient")
        current_steps, current_excluded = valid_metric_steps(
            "multi-role",
            current_steps,
        )
        steps.extend(current_steps)
        excluded_uids.extend(current_excluded)
        batch_uids.append({str(step["uid"]) for step in current_steps})
    _validate_disjoint_batches(batch_uids)

    metrics = evaluate_steps(steps)
    intervals = learner_cluster_bootstrap(
        steps,
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    rq1 = {
        "target_count": metrics["count"],
        "learner_count": len({str(step["uid"]) for step in steps}),
        "real_correct_rate": metrics["real_correct_rate"],
        "simulated_correct_rate": metrics["sampled_correct_rate"],
        "acc": metrics["sample_match_acc"],
        "f1": metrics["sample_f1"],
        "balanced_acc": metrics["response_balanced_accuracy"],
        "specificity": metrics["response_specificity"],
        "mcc": metrics["response_mcc"],
        "confusion": metrics["sample_confusion"],
        "learner_distribution_error": metrics["learner_distribution_error"],
        "concept_distribution_error": metrics["concept_distribution_error"],
        "learner_cluster_bootstrap_95ci": intervals,
    }
    rq2 = {
        "concept_accuracy": metrics.get("task2_concept_accuracy"),
        "concept_count": metrics.get("task2_concept_perception_count"),
        "decision_answer_consistency": metrics.get(
            "four_tier_decision_answer_consistency"
        ),
        "decision_answer_consistency_count": metrics.get(
            "four_tier_decision_answer_consistency_count"
        ),
        "answer_confidence_ece": metrics.get("four_tier_answer_confidence_ece"),
        "high_confidence_answer_accuracy": metrics.get(
            "four_tier_high_confidence_accuracy"
        ),
        "low_confidence_answer_accuracy": metrics.get(
            "four_tier_low_confidence_accuracy"
        ),
        "conditional_confidence_gap": metrics.get(
            "four_tier_conditional_confidence_gap"
        ),
        "conditional_confidence_comparable_groups": metrics.get(
            "four_tier_conditional_confidence_comparable_groups"
        ),
        "conditional_confidence_details": metrics.get(
            "four_tier_conditional_confidence_details"
        ),
        "answer_reasoning_confidence_gap": metrics.get(
            "four_tier_answer_reasoning_confidence_gap"
        ),
        "reasoning_ground_truth_count": metrics.get(
            "four_tier_fully_scored_count"
        ),
        "mastery_response_monotonicity": metrics.get(
            "mastery_response_monotonicity"
        ),
        "mastery_confidence_monotonicity": metrics.get(
            "mastery_confidence_monotonicity"
        ),
    }
    payload = {
        "study": "rq1_response_fidelity_and_rq2_process_consistency",
        "dataset_name": args.dataset_name,
        "inputs": [str(_resolve(path)) for path in args.inputs],
        "batch_count": len(reports),
        "batches_disjoint": True,
        "rq1": rq1,
        "rq2": rq2,
        "trace_coverage": {
            "all_steps": len(steps),
            "prompt_count": sum("response_agent_prompt" in step for step in steps),
            "raw_llm_response_count": sum(bool(step.get("llm_result")) for step in steps),
            "llm_error_count": sum(bool(step.get("llm_error")) for step in steps),
            "excluded_invalid_uids": sorted(set(excluded_uids)),
        },
        "interpretation_boundary": (
            "RQ2 evaluates observable process consistency and confidence behavior. "
            "It does not establish ground-truth learner cognition because the dataset "
            "contains no observed reasoning or confidence labels."
        ),
    }
    json_path = save_report(payload, args.output)
    markdown_path = _write_new_text(_resolve(args.markdown_output), render_markdown(payload))
    print(
        json.dumps(
            {
                "ok": True,
                "json": str(json_path),
                "markdown": str(markdown_path),
                "targets": len(steps),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def learner_cluster_bootstrap(
    steps: list[dict[str, Any]],
    samples: int,
    seed: int,
) -> dict[str, list[float] | None]:
    by_uid: dict[str, list[dict[str, Any]]] = {}
    for step in steps:
        by_uid.setdefault(str(step["uid"]), []).append(step)
    uids = list(by_uid)
    rng = random.Random(seed)
    values: dict[str, list[float]] = {
        "acc": [],
        "f1": [],
        "balanced_acc": [],
        "specificity": [],
        "mcc": [],
        "learner_distribution_error": [],
    }
    for _ in range(samples):
        selected = [rng.choice(uids) for _ in uids]
        selected_steps = [step for uid in selected for step in by_uid[uid]]
        y_true = [int(step["real_response"]) for step in selected_steps]
        y_pred = [int(step["simulated_response"]) for step in selected_steps]
        rates = []
        for uid in selected:
            learner_steps = by_uid[uid]
            real_rate = sum(int(step["real_response"]) for step in learner_steps) / len(
                learner_steps
            )
            simulated_rate = sum(
                int(step["simulated_response"]) for step in learner_steps
            ) / len(learner_steps)
            rates.append(abs(real_rate - simulated_rate))
        measures: dict[str, float | None] = {
            "acc": accuracy(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "balanced_acc": balanced_accuracy(y_true, y_pred),
            "specificity": specificity(y_true, y_pred),
            "mcc": matthews_corrcoef(y_true, y_pred),
            "learner_distribution_error": sum(rates) / len(rates),
        }
        for name, value in measures.items():
            if value is not None:
                values[name].append(float(value))
    return {name: _percentile_interval(items) for name, items in values.items()}


def _percentile_interval(values: list[float]) -> list[float] | None:
    if not values:
        return None
    ordered = sorted(values)
    low = ordered[int(0.025 * len(ordered))]
    high = ordered[max(0, int(0.975 * len(ordered)) - 1)]
    return [round(low, 6), round(high, 6)]


def render_markdown(payload: dict[str, Any]) -> str:
    rq1 = payload["rq1"]
    rq2 = payload["rq2"]
    ci = rq1["learner_cluster_bootstrap_95ci"]
    return "\n".join(
        [
            f"# {payload.get('dataset_name', 'Dataset')} RQ1-RQ2 Evaluation",
            "",
            f"- Batches: {payload['batch_count']}",
            f"- Learners: {rq1['learner_count']}",
            f"- Target responses: {rq1['target_count']}",
            "",
            "## RQ1: Response Fidelity",
            "",
            "| Metric | Value | Learner-bootstrap 95% CI |",
            "|---|---:|---:|",
            _row("ACC", rq1["acc"], ci["acc"]),
            _row("F1", rq1["f1"], ci["f1"]),
            _row("Balanced ACC", rq1["balanced_acc"], ci["balanced_acc"]),
            _row("Specificity", rq1["specificity"], ci["specificity"]),
            _row("MCC", rq1["mcc"], ci["mcc"]),
            _row(
                "Learner Distribution Error",
                rq1["learner_distribution_error"],
                ci["learner_distribution_error"],
            ),
            f"| Concept Distribution Error | {rq1['concept_distribution_error']:.4f} | - |",
            "",
            "## RQ2: Process Consistency and Credibility",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Concept accuracy | {_pct(rq2['concept_accuracy'])} |",
            f"| Decision-answer consistency | {_pct(rq2['decision_answer_consistency'])} |",
            f"| Answer-confidence ECE (lower is better) | {rq2['answer_confidence_ece']:.4f} |",
            f"| High-confidence answer accuracy | {_pct(rq2['high_confidence_answer_accuracy'])} |",
            f"| Low-confidence answer accuracy | {_pct(rq2['low_confidence_answer_accuracy'])} |",
            f"| Conditional confidence gap | {_pct(rq2['conditional_confidence_gap'])} |",
            f"| Comparable conditional groups | {rq2['conditional_confidence_comparable_groups']} |",
            f"| Answer-reasoning confidence gap | {rq2['answer_reasoning_confidence_gap']:.4f} |",
            f"| Ground-truth reasoning labels | {rq2['reasoning_ground_truth_count']} |",
            "",
            f"> {payload['interpretation_boundary']}",
            "",
        ]
    )


def _row(name: str, value: float, interval: list[float] | None) -> str:
    rendered = "-" if interval is None else f"[{interval[0]:.4f}, {interval[1]:.4f}]"
    return f"| {name} | {value:.4f} | {rendered} |"


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{100 * value:.2f}%"


def _load_full_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    reports = payload.get("reports")
    if isinstance(reports, dict):
        if "full" in reports:
            return reports["full"]
        if len(reports) == 1:
            return next(iter(reports.values()))
    return payload


def _validate_disjoint_batches(batch_uids: list[set[str]]) -> None:
    seen: set[str] = set()
    for index, uids in enumerate(batch_uids, start=1):
        overlap = seen & uids
        if overlap:
            raise ValueError(f"Batch {index} repeats learner UIDs: {sorted(overlap)[:5]}")
        seen.update(uids)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _write_new_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path
    counter = 1
    while True:
        try:
            with candidate.open("x", encoding="utf-8") as file:
                file.write(text)
            return candidate
        except FileExistsError:
            candidate = path.with_name(f"{path.stem}__{counter}{path.suffix}")
            counter += 1


if __name__ == "__main__":
    main()

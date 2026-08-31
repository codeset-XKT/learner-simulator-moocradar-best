from __future__ import annotations

from pathlib import Path
from typing import Any


RESPONSE_METRICS = [
    "acc",
    "f1",
    "balanced_acc",
    "specificity",
    "mcc",
    "rouge_3",
    "auc",
    "confusion",
    "real_correct_rate",
    "predicted_correct_rate",
]

DISTRIBUTION_METRICS = [
    "rouge_3",
    "rouge_3_precision",
    "rouge_3_recall",
    "rouge_3_user_count",
    "learner_distribution_error",
    "concept_distribution_error",
    "real_correct_rate",
    "predicted_correct_rate",
]

TASK_METRICS = [
    "task1_state_inference_count",
    "task2_concept_accuracy",
    "task3_response_acc",
    "task3_response_balanced_accuracy",
    "task4_mean_abs_mastery_delta",
]

DIAGNOSTIC_METRICS = [
    "mastery_response_monotonicity",
    "mastery_confidence_monotonicity",
    "four_tier_answer_confidence_ece",
    "four_tier_mean_answer_confidence",
    "four_tier_mean_reasoning_confidence",
    "four_tier_high_confidence_accuracy",
    "four_tier_low_confidence_accuracy",
]


def extract_named_reports(
    payload: dict[str, Any],
    source: str | Path = "",
) -> list[dict[str, Any]]:
    """Return report records from all current and historical JSON wrappers."""

    source_text = str(source)
    records: list[dict[str, Any]] = []
    if isinstance(payload.get("reports"), dict):
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        for name, report in payload["reports"].items():
            if not isinstance(report, dict):
                continue
            metrics = _report_metrics(report)
            summary_metrics = summary.get(name) if isinstance(summary.get(name), dict) else {}
            metrics = {**_summary_to_metrics(summary_metrics), **metrics}
            records.append(
                {
                    "name": str(name),
                    "source": source_text,
                    "report": report,
                    "metrics": metrics,
                    "validity": _validity(report, summary_metrics),
                    "runtime": _runtime(report, summary_metrics),
                }
            )
        return records

    name = _single_report_name(payload, source_text)
    metrics = _report_metrics(payload)
    records.append(
        {
            "name": name,
            "source": source_text,
            "report": payload,
            "metrics": metrics,
            "validity": _validity(payload, {}),
            "runtime": _runtime(payload, {}),
        }
    )
    return records


def layered_metric_view(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics") or {}
    response = {key: metrics.get(key) for key in RESPONSE_METRICS}
    distribution = {key: metrics.get(key) for key in DISTRIBUTION_METRICS}
    task = {key: metrics.get(key) for key in TASK_METRICS}
    diagnostic = {key: metrics.get(key) for key in DIAGNOSTIC_METRICS}
    return {
        "name": record.get("name"),
        "source": record.get("source"),
        "validity": record.get("validity"),
        "runtime": record.get("runtime"),
        "response_consistency": response,
        "distribution_consistency": distribution,
        "task_consistency": task,
        "diagnostic_consistency": diagnostic,
    }


def compact_table_row(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics") or {}
    validity = record.get("validity") or {}
    runtime = record.get("runtime") or {}
    return {
        "method": _display_method_name(record.get("name")),
        "valid": validity.get("valid"),
        "n": metrics.get("count"),
        "acc": metrics.get("acc"),
        "f1": metrics.get("f1"),
        "balanced_acc": metrics.get("balanced_acc"),
        "specificity": metrics.get("specificity"),
        "mcc": metrics.get("mcc"),
        "rouge_3": metrics.get("rouge_3"),
        "lde": metrics.get("learner_distribution_error"),
        "cde": metrics.get("concept_distribution_error"),
        "auc": metrics.get("auc"),
        "task2_concept_accuracy": metrics.get("task2_concept_accuracy"),
        "task3_response_acc": metrics.get("task3_response_acc"),
        "four_tier_ece": metrics.get("four_tier_answer_confidence_ece"),
        "predicted_correct_rate": metrics.get("predicted_correct_rate"),
        "confusion": metrics.get("confusion"),
        "runtime": runtime.get("total_human"),
        "source": record.get("source"),
    }


def _display_method_name(name: Any) -> str:
    text = str(name or "")
    if text.lower() == "random":
        return "Probability-sampling Baseline"
    return text


def _report_metrics(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    if metrics:
        return _normalize_metrics(metrics)
    return _normalize_metrics(report)


def _summary_to_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    if not summary:
        return {}
    return _normalize_metrics(summary)


def _normalize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    confusion = (
        metrics.get("llm_confusion")
        or metrics.get("sample_confusion")
        or metrics.get("dkt_confusion")
        or metrics.get("prob_confusion")
        or metrics.get("confusion")
    )
    predicted_rate = _first_present(
        metrics,
        "llm_predicted_correct_rate",
        "sampled_correct_rate",
        "dkt_predicted_correct_rate",
        "prob_predicted_correct_rate",
        "predicted_correct_rate",
    )
    return {
        **metrics,
        "count": metrics.get("count") or metrics.get("n"),
        "acc": _first_present(
            metrics,
            "llm_response_acc",
            "sample_match_acc",
            "dkt_acc_at_threshold",
            "prob_acc_at_threshold",
            "acc",
            "accuracy",
        ),
        "f1": _first_present(
            metrics,
            "llm_response_f1",
            "sample_f1",
            "dkt_f1_at_threshold",
            "prob_f1_at_threshold",
            "f1",
        ),
        "balanced_acc": _first_present(
            metrics,
            "response_balanced_accuracy",
            "dkt_balanced_accuracy",
            "prob_balanced_accuracy",
            "balanced_acc",
            "balanced_accuracy",
        ),
        "specificity": _first_present(
            metrics,
            "response_specificity",
            "dkt_specificity",
            "prob_specificity",
            "specificity",
        ),
        "mcc": _first_present(
            metrics,
            "response_mcc",
            "dkt_mcc",
            "prob_mcc",
            "mcc",
        ),
        "auc": _first_present(
            metrics,
            "answer_confidence_auc",
            "llm_auc",
            "dkt_auc",
            "prob_auc",
            "auc",
        ),
        "real_correct_rate": metrics.get("real_correct_rate"),
        "predicted_correct_rate": predicted_rate,
        "confusion": confusion,
        "learner_distribution_error": metrics.get("learner_distribution_error"),
        "concept_distribution_error": metrics.get("concept_distribution_error"),
        "mastery_response_monotonicity": _score_or_value(
            metrics.get("mastery_response_monotonicity")
        ),
        "mastery_confidence_monotonicity": _score_or_value(
            metrics.get("mastery_confidence_monotonicity")
        ),
        "four_tier_answer_confidence_ece": metrics.get(
            "four_tier_answer_confidence_ece"
        ),
        "four_tier_mean_answer_confidence": metrics.get(
            "four_tier_mean_answer_confidence"
        ),
        "four_tier_mean_reasoning_confidence": metrics.get(
            "four_tier_mean_reasoning_confidence"
        ),
        "four_tier_high_confidence_accuracy": metrics.get(
            "four_tier_high_confidence_accuracy"
        ),
        "four_tier_low_confidence_accuracy": metrics.get(
            "four_tier_low_confidence_accuracy"
        ),
        "task1_state_inference_count": metrics.get("task1_state_inference_count"),
        "task2_concept_accuracy": metrics.get("task2_concept_accuracy"),
        "task3_response_acc": metrics.get("task3_response_acc"),
        "task3_response_balanced_accuracy": metrics.get(
            "task3_response_balanced_accuracy"
        ),
        "task4_mean_abs_mastery_delta": metrics.get(
            "task4_mean_abs_mastery_delta"
        ),
    }


def _validity(report: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    validity = report.get("validity") if isinstance(report.get("validity"), dict) else {}
    if validity:
        return {
            "valid": validity.get("valid"),
            "reason": validity.get("reason"),
            "llm_error_count": validity.get("llm_error_count"),
            "parsed_llm_steps": validity.get("parsed_llm_steps"),
            "expected_llm_steps": validity.get("expected_llm_steps"),
        }
    return {
        "valid": summary.get("valid"),
        "reason": summary.get("validity_reason"),
        "llm_error_count": None,
        "parsed_llm_steps": None,
        "expected_llm_steps": None,
    }


def _runtime(report: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    runtime = report.get("runtime") if isinstance(report.get("runtime"), dict) else {}
    if runtime:
        return runtime
    return {"total_human": summary.get("runtime")}


def _single_report_name(payload: dict[str, Any], source: str) -> str:
    if payload.get("experiment"):
        return str(payload["experiment"])
    if payload.get("checkpoint") or "dkt" in source.lower():
        return "dkt"
    return Path(source).stem if source else "result"


def _first_present(metrics: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metrics and metrics[key] is not None:
            return metrics[key]
    return None


def _score_or_value(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("score")
    return value

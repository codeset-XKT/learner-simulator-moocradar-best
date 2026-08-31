from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Any

from learner_simulator.evaluation import balanced_accuracy, f1_score


FORMAL_METRIC_VERSION = "formal_response_metrics_v6"


def _step_key(step: dict[str, Any]) -> tuple[str, str, str]:
    return (str(step.get("uid")), str(step.get("step_index")), str(step.get("qid")))


def _binary_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        step for step in steps
        if step.get("real_response") in {0, 1}
        and step.get("simulated_response") in {0, 1}
    ]


def build_ability_difficulty_partition(full_steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Create the one fixed 2x2 IRT partition used by every compared method."""
    eligible: list[tuple[dict[str, Any], float, float]] = []
    for step in _binary_steps(full_steps):
        irt = step.get("irt_ability_difficulty_evidence") or {}
        theta = irt.get("learner_theta")
        beta = irt.get("item_beta")
        if theta is None or beta is None:
            continue
        try:
            eligible.append((step, float(theta), float(beta)))
        except (TypeError, ValueError):
            continue
    if not eligible:
        return {
            "version": "irt_2x2_global_full_reference_v1",
            "available": False,
            "reason": "full reference steps have no usable IRT ability/difficulty evidence",
            "groups": {},
        }
    theta_median = median([theta for _, theta, _ in eligible])
    beta_median = median([beta for _, _, beta in eligible])
    groups: dict[tuple[str, str, str], str] = {}
    group_counts: dict[str, int] = defaultdict(int)
    for step, theta, beta in eligible:
        ability = "high_ability" if theta >= theta_median else "low_ability"
        difficulty = "high_difficulty" if beta >= beta_median else "low_difficulty"
        name = f"{ability}__{difficulty}"
        groups[_step_key(step)] = name
        group_counts[name] += 1
    return {
        "version": "irt_2x2_global_full_reference_v1",
        "available": True,
        "reference_method": "full",
        "theta_median": round(float(theta_median), 8),
        "beta_median": round(float(beta_median), 8),
        "reference_step_count": len(eligible),
        "group_counts": dict(sorted(group_counts.items())),
        # Internal key map is intentionally retained in the JSON report so a
        # baseline evaluator can reuse precisely the same partition offline.
        "groups": {"|".join(key): value for key, value in groups.items()},
    }


def _partition_group(partition: dict[str, Any], step: dict[str, Any]) -> str | None:
    return (partition.get("groups") or {}).get("|".join(_step_key(step)))


def evaluate_formal_response_metrics(
    steps: list[dict[str, Any]],
    partition: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute the paper-facing BAA, balanced accuracy, F1, and ADCDE."""
    usable = _binary_steps(steps)
    if not usable:
        return {
            "version": FORMAL_METRIC_VERSION,
            "count": 0,
            "baa": None,
            "balanced_accuracy": None,
            "f1": None,
            "adcde": None,
        }
    y_true = [int(step["real_response"]) for step in usable]
    y_pred = [int(step["simulated_response"]) for step in usable]
    positive = [pred for truth, pred in zip(y_true, y_pred) if truth == 1]
    negative = [pred for truth, pred in zip(y_true, y_pred) if truth == 0]
    acc_plus = sum(positive) / len(positive) if positive else None
    acc_minus = sum(1 - pred for pred in negative) / len(negative) if negative else None
    bacc = balanced_accuracy(y_true, y_pred)
    baa = (
        ((acc_plus + acc_minus) / 2) * (1 - abs(acc_plus - acc_minus))
        if acc_plus is not None and acc_minus is not None
        else None
    )
    result: dict[str, Any] = {
        "version": FORMAL_METRIC_VERSION,
        "count": len(usable),
        "acc_plus": _round(acc_plus),
        "acc_minus": _round(acc_minus),
        "baa": _round(baa),
        "balanced_accuracy": _round(bacc),
        "f1": _round(f1_score(y_true, y_pred)),
        "adcde": None,
        "adcde_definition": "unavailable_without_fixed_full_irt_2x2_partition",
    }
    if partition and partition.get("available"):
        grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for step in usable:
            group = _partition_group(partition, step)
            if group is not None:
                grouped[group].append((int(step["real_response"]), int(step["simulated_response"])))
        group_details = {}
        gaps = []
        for group in sorted(grouped):
            pairs = grouped[group]
            real_rate = sum(real for real, _ in pairs) / len(pairs)
            simulated_rate = sum(pred for _, pred in pairs) / len(pairs)
            gap = abs(real_rate - simulated_rate)
            gaps.append(gap)
            group_details[group] = {
                "count": len(pairs),
                "real_correct_rate": _round(real_rate),
                "simulated_correct_rate": _round(simulated_rate),
                "absolute_gap": _round(gap),
            }
        result.update({
            "adcde": _round(sum(gaps) / len(gaps)) if gaps else None,
            "adcde_definition": "macro_mean_absolute_correct_rate_gap_over_nonempty_fixed_full_irt_2x2_groups",
            "adcde_partition_version": partition.get("version"),
            "adcde_partition_coverage": len([step for step in usable if _partition_group(partition, step) is not None]),
            "adcde_groups": group_details,
        })
    return result


def _round(value: float | None) -> float | None:
    return round(float(value), 6) if value is not None else None

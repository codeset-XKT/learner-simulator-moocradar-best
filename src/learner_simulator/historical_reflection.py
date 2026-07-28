from __future__ import annotations

from collections import defaultdict
from typing import Any

from learner_simulator.data import clean_sequence


def build_historical_reflective_calibration(
    history_row: dict[str, str] | None,
    replay_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize replay errors inside observed history only.

    The target simulation labels must not be used here. The module compares an
    internal pre-target prediction signal with observed labels from the tail of
    the already-observed history and produces a compact calibration profile.
    """

    history_length = len(clean_sequence(history_row)) if history_row is not None else 0
    if not replay_records:
        adaptive_policy = _adaptive_policy(
            bias_direction="unavailable",
            error_pattern="unavailable",
            evidence_confidence="low",
            replay_acc=None,
            rate_gap=None,
            false_positive_rate=None,
            false_negative_rate=None,
        )
        return {
            "module": "historical_reflective_calibration",
            "source": "observed_history_only",
            "available": False,
            "history_length": history_length,
            "calibration_window": 0,
            "bias_direction": "unavailable",
            "error_pattern": "unavailable",
            "evidence_confidence": "low",
            "guidance": (
                "No historical replay calibration is available; use the stable "
                "profile, memory, and current proficiency evidence."
            ),
            "adaptive_policy": adaptive_policy,
        }

    labels = [int(item["real_response"]) for item in replay_records]
    probs = [float(item["predicted_probability"]) for item in replay_records]
    preds = [int(prob >= 0.5) for prob in probs]
    count = len(labels)
    actual_rate = _mean(labels)
    predicted_rate = _mean(probs)
    acc = _mean(int(y == p) for y, p in zip(labels, preds))
    false_positive = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    false_negative = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    negatives = sum(1 for y in labels if y == 0)
    positives = sum(1 for y in labels if y == 1)
    fp_rate = _safe_div(false_positive, negatives)
    fn_rate = _safe_div(false_negative, positives)
    rate_gap = actual_rate - predicted_rate

    if rate_gap >= 0.12:
        bias_direction = "underestimates_success"
    elif rate_gap <= -0.12:
        bias_direction = "overestimates_success"
    else:
        bias_direction = "approximately_calibrated"

    if fn_rate is not None and fp_rate is not None and fn_rate >= fp_rate + 0.15:
        error_pattern = "missed_successes"
    elif fn_rate is not None and fp_rate is not None and fp_rate >= fn_rate + 0.15:
        error_pattern = "overpredicted_successes"
    elif false_positive + false_negative == 0:
        error_pattern = "historically_aligned"
    else:
        error_pattern = "mixed_errors"

    concept_adjustments = _concept_adjustments(replay_records)
    evidence_confidence = (
        "high" if count >= 15 else "medium" if count >= 8 else "low"
    )
    adaptive_policy = _adaptive_policy(
        bias_direction=bias_direction,
        error_pattern=error_pattern,
        evidence_confidence=evidence_confidence,
        replay_acc=acc,
        rate_gap=rate_gap,
        false_positive_rate=fp_rate,
        false_negative_rate=fn_rate,
    )
    return {
        "module": "historical_reflective_calibration",
        "source": "observed_history_only",
        "available": True,
        "history_length": history_length,
        "calibration_window": count,
        "split": {
            "prefix_interactions": max(0, history_length - count),
            "replay_tail_interactions": count,
        },
        "actual_correct_rate": round(actual_rate, 6),
        "predicted_correct_rate": round(predicted_rate, 6),
        "rate_gap_actual_minus_predicted": round(rate_gap, 6),
        "replay_acc_at_threshold": round(acc, 6),
        "false_positive_rate": _round_optional(fp_rate),
        "false_negative_rate": _round_optional(fn_rate),
        "false_positive_count": false_positive,
        "false_negative_count": false_negative,
        "bias_direction": bias_direction,
        "error_pattern": error_pattern,
        "evidence_confidence": evidence_confidence,
        "concept_adjustments": concept_adjustments,
        "guidance": _guidance(bias_direction, error_pattern, evidence_confidence),
        "adaptive_policy": adaptive_policy,
    }


def _adaptive_policy(
    bias_direction: str,
    error_pattern: str,
    evidence_confidence: str,
    replay_acc: float | None,
    rate_gap: float | None,
    false_positive_rate: float | None,
    false_negative_rate: float | None,
) -> dict[str, Any]:
    """Convert replay diagnostics into a target-stage simulation policy.

    This policy is learned only from observed historical replay. It is not a
    target-step correction rule and must not override the generated response.
    """

    gap = float(rate_gap or 0.0)
    strength = _strength(abs(gap), evidence_confidence)
    trust = _ncdm_trust(replay_acc, evidence_confidence)
    if bias_direction == "underestimates_success":
        response_bias = "preserve_plausible_success"
        success_adjustment = strength
        failure_adjustment = "conservative"
        confidence_adjustment = "raise_when_ncdm_and_memory_support"
        instruction = (
            "During target simulation, do not make the learner unnecessarily "
            "pessimistic. When NCDM proficiency, recent memory, and item demand "
            "support success, preserve a learner-level correct attempt."
        )
    elif bias_direction == "overestimates_success":
        response_bias = "guard_against_overconfidence"
        success_adjustment = "conservative"
        failure_adjustment = strength
        confidence_adjustment = "lower_when_memory_or_item_demand_is_weak"
        instruction = (
            "During target simulation, guard against overconfident correct "
            "attempts. Require stronger item-specific support before producing "
            "a confident successful answer."
        )
    else:
        response_bias = "balanced"
        success_adjustment = "neutral"
        failure_adjustment = "neutral"
        confidence_adjustment = "follow_ncdm_memory_and_item_evidence"
        instruction = (
            "Historical replay is approximately calibrated. Use NCDM, memory, "
            "item demand, and learner profile without adding systematic optimism "
            "or pessimism."
        )

    if error_pattern == "missed_successes":
        slip_guess_policy = (
            "reduce_false_negative_simulation; avoid turning uncertainty into "
            "automatic errors on familiar items"
        )
    elif error_pattern == "overpredicted_successes":
        slip_guess_policy = (
            "increase_slip_sensitivity; weak memory or high demand can produce "
            "incorrect learner-level attempts"
        )
    elif error_pattern == "historically_aligned":
        slip_guess_policy = "preserve_observed_alignment"
    else:
        slip_guess_policy = "use_balanced_slip_and_guess_behavior"

    return {
        "module": "historical_replay_policy_adaptation",
        "source": "observed_history_replay_only",
        "target_label_access": False,
        "ncdm_trust": trust,
        "response_bias": response_bias,
        "success_plausibility_adjustment": success_adjustment,
        "failure_sensitivity_adjustment": failure_adjustment,
        "confidence_adjustment": confidence_adjustment,
        "slip_guess_policy": slip_guess_policy,
        "policy_strength": strength,
        "false_positive_rate": _round_optional(false_positive_rate),
        "false_negative_rate": _round_optional(false_negative_rate),
        "instruction": instruction,
        "application_scope": (
            "Apply consistently across target steps as a learner-specific "
            "simulation policy; do not revise it using target labels."
        ),
    }


def _strength(gap_abs: float, evidence_confidence: str) -> str:
    if evidence_confidence == "low":
        return "weak"
    if gap_abs >= 0.25:
        return "strong"
    if gap_abs >= 0.12:
        return "moderate"
    return "weak"


def _ncdm_trust(replay_acc: float | None, evidence_confidence: str) -> str:
    if replay_acc is None or evidence_confidence == "low":
        return "use_as_primary_but_uncertain"
    if replay_acc >= 0.75:
        return "high"
    if replay_acc >= 0.55:
        return "moderate"
    return "low_requires_profile_and_memory_correction"


def _concept_adjustments(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        concept = str(item.get("concept") or item.get("cid"))
        grouped[concept].append(item)

    adjustments: list[dict[str, Any]] = []
    for concept, items in grouped.items():
        if len(items) < 2:
            continue
        labels = [int(item["real_response"]) for item in items]
        probs = [float(item["predicted_probability"]) for item in items]
        gap = _mean(labels) - _mean(probs)
        if gap >= 0.20:
            direction = "underestimated"
        elif gap <= -0.20:
            direction = "overestimated"
        else:
            direction = "aligned"
        adjustments.append(
            {
                "concept": concept,
                "count": len(items),
                "actual_correct_rate": round(_mean(labels), 6),
                "predicted_correct_rate": round(_mean(probs), 6),
                "gap": round(gap, 6),
                "direction": direction,
            }
        )

    adjustments.sort(key=lambda item: (abs(float(item["gap"])), item["count"]), reverse=True)
    return adjustments[:5]


def _guidance(
    bias_direction: str,
    error_pattern: str,
    evidence_confidence: str,
) -> str:
    prefix = (
        f"Historical replay calibration confidence is {evidence_confidence}. "
        "Use this as an observed-history calibration signal, not as a target label. "
    )
    if bias_direction == "underestimates_success":
        return (
            prefix
            + "The internal state estimator underestimated this learner's historical successes; "
            "avoid unnecessary pessimism when current proficiency and memory are supportive."
        )
    if bias_direction == "overestimates_success":
        return (
            prefix
            + "The internal state estimator overestimated this learner's historical successes; "
            "require stronger current evidence before producing confident correct attempts."
        )
    if error_pattern == "missed_successes":
        return (
            prefix
            + "Historical replay missed several successful attempts; preserve plausible "
            "student-level success on familiar items."
        )
    if error_pattern == "overpredicted_successes":
        return (
            prefix
            + "Historical replay overpredicted some successes; lower confidence when "
            "memory or item demand is weak."
        )
    return (
        prefix
        + "Historical replay is roughly calibrated; use current item evidence, memory, "
        "and proficiency without adding a systematic optimism or pessimism correction."
    )


def _mean(values: Any) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(float(value) for value in values) / len(values)


def _safe_div(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _round_optional(value: float | None) -> float | None:
    return round(float(value), 6) if value is not None else None

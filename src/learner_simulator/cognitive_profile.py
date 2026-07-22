from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from learner_simulator.data import clean_sequence


def build_cognitive_profile(
    sequence: list[dict[str, int]],
    questions: dict[str, dict[str, Any]] | None = None,
    recent_window: int = 10,
    volatility_window: int = 5,
) -> dict[str, Any]:
    """Build a statistical cognitive profile from observed history only.

    The affective fields are behavioral proxies inspired by DEKT's affective
    dimensions. They are not treated as ground-truth emotions.
    """

    questions = questions or {}
    history = list(sequence)
    total = len(history)
    responses = [int(step["response"]) for step in history]
    overall_success = _mean(responses, default=0.5)
    recent = responses[-recent_window:] if responses else []
    early = responses[: len(recent)] if recent else []
    recent_success = _mean(recent, default=overall_success)
    early_success = _mean(early, default=overall_success)
    success_trend = recent_success - early_success

    window_rates = _sliding_rates(responses, volatility_window)
    response_volatility = _std(window_rates)
    mastery_stability = 1.0 - _clip(response_volatility / 0.5)
    error_streak_ratio, max_error_streak = _error_streak_stats(responses)

    concept_stats = _concept_stats(history)
    high_mastery = {
        cid
        for cid, stat in concept_stats.items()
        if stat["total"] >= 3 and stat["correct"] / stat["total"] >= 0.75
    }
    low_mastery = {
        cid
        for cid, stat in concept_stats.items()
        if stat["total"] >= 3 and stat["correct"] / stat["total"] <= 0.35
    }
    carelessness = _wrong_rate_on_concepts(history, high_mastery)
    guessing = _correct_rate_on_concepts(history, low_mastery)
    repeated_error_rate = _repeated_error_rate(history, questions)
    error_recovery = _error_recovery_rate(history, questions)

    parent_groups = _parent_concept_rates(history, questions)
    parent_variances = [
        _std(rates)
        for rates in parent_groups.values()
        if len(rates) >= 2
    ]
    confusion_risk = _clip(_mean(parent_variances, default=0.0) / 0.5)

    concentration = _clip(
        0.4 * recent_success
        + 0.4 * mastery_stability
        + 0.2 * (1.0 - error_streak_ratio)
    )
    frustration = _clip(0.5 * error_streak_ratio + 0.5 * repeated_error_rate)
    high_familiarity_rate = _concept_attempt_share(history, high_mastery)
    recent_slip_rate = _wrong_rate_on_concepts(
        history[-recent_window:] if history else [],
        high_mastery,
    )
    boredom = _clip(high_familiarity_rate * recent_slip_rate)

    same_parent_transfer, transfer_fragility = _transfer_rates(history, questions)

    return {
        "module": "statistical_cognitive_profile",
        "source": "observed_history",
        "control_traits": {
            "overall_success_rate": _round(overall_success),
            "recent_success_rate": _round(recent_success),
            "success_trend": _round(success_trend),
            "response_volatility": _round(response_volatility),
            "mastery_stability": _round(mastery_stability),
            "overall_success_level": _level(overall_success),
            "recent_success_level": _level(recent_success),
            "success_trend_level": _trend_level(success_trend),
            "mastery_stability_level": _level(mastery_stability),
        },
        "cognitive_affective_proxies": {
            "concentration_proxy": _round(concentration),
            "frustration_risk": _round(frustration),
            "confusion_risk": _round(confusion_risk),
            "boredom_risk": _round(boredom),
            "concentration_level": _level(concentration),
            "frustration_level": _level(frustration),
            "confusion_level": _level(confusion_risk),
            "boredom_level": _level(boredom),
            "proxy_note": (
                "Behavioral proxies inspired by affective KT; not observed emotion labels."
            ),
        },
        "error_generation_traits": {
            "carelessness_tendency": _round(carelessness),
            "guessing_tendency": _round(guessing),
            "misconception_persistence": _round(repeated_error_rate),
            "error_recovery_rate": _round(error_recovery),
            "error_streak_ratio": _round(error_streak_ratio),
            "max_error_streak": max_error_streak,
            "carelessness_level": _level(carelessness),
            "guessing_level": _level(guessing),
            "misconception_persistence_level": _level(repeated_error_rate),
            "error_recovery_level": _level(error_recovery),
        },
        "transfer_traits": {
            "same_parent_transfer_success": _round(same_parent_transfer),
            "transfer_fragility": _round(transfer_fragility),
            "same_parent_transfer_level": _level(same_parent_transfer),
            "transfer_fragility_level": _level(transfer_fragility),
        },
    }


def build_cognitive_profiles(
    rows: list[dict[str, str]],
    questions: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        row["uid"]: build_cognitive_profile(clean_sequence(row), questions)
        for row in rows
    }


def _concept_stats(sequence: list[dict[str, int]]) -> dict[int, dict[str, int]]:
    stats: dict[int, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for step in sequence:
        cid = int(step["cid"])
        stats[cid]["correct"] += int(step["response"] == 1)
        stats[cid]["total"] += 1
    return dict(stats)


def _sliding_rates(values: list[int], window: int) -> list[float]:
    if not values:
        return []
    if len(values) <= window:
        return [_mean(values, default=0.5)]
    return [
        _mean(values[index : index + window], default=0.5)
        for index in range(0, len(values) - window + 1)
    ]


def _error_streak_stats(responses: list[int]) -> tuple[float, int]:
    if not responses:
        return 0.0, 0
    streak_error_count = 0
    current = 0
    max_streak = 0
    for response in responses:
        if response == 0:
            current += 1
            max_streak = max(max_streak, current)
            if current >= 2:
                streak_error_count += 1
        else:
            current = 0
    return _clip(streak_error_count / len(responses)), max_streak


def _wrong_rate_on_concepts(
    sequence: list[dict[str, int]],
    concepts: set[int],
) -> float:
    selected = [step for step in sequence if int(step["cid"]) in concepts]
    if not selected:
        return 0.0
    return _mean([1 - int(step["response"]) for step in selected], default=0.0)


def _correct_rate_on_concepts(
    sequence: list[dict[str, int]],
    concepts: set[int],
) -> float:
    selected = [step for step in sequence if int(step["cid"]) in concepts]
    if not selected:
        return 0.0
    return _mean([int(step["response"]) for step in selected], default=0.0)


def _concept_attempt_share(
    sequence: list[dict[str, int]],
    concepts: set[int],
) -> float:
    if not sequence or not concepts:
        return 0.0
    count = sum(1 for step in sequence if int(step["cid"]) in concepts)
    return count / len(sequence)


def _repeated_error_rate(
    sequence: list[dict[str, int]],
    questions: dict[str, dict[str, Any]],
) -> float:
    previous_by_key: dict[str, int] = {}
    repeated = 0
    denominator = 0
    for step in sequence:
        key = _related_key(step, questions)
        response = int(step["response"])
        if key in previous_by_key:
            denominator += 1
            if previous_by_key[key] == 0 and response == 0:
                repeated += 1
        previous_by_key[key] = response
    return repeated / denominator if denominator else 0.0


def _error_recovery_rate(
    sequence: list[dict[str, int]],
    questions: dict[str, dict[str, Any]],
) -> float:
    previous_by_key: dict[str, int] = {}
    recovered = 0
    denominator = 0
    for step in sequence:
        key = _related_key(step, questions)
        response = int(step["response"])
        if previous_by_key.get(key) == 0:
            denominator += 1
            if response == 1:
                recovered += 1
        previous_by_key[key] = response
    return recovered / denominator if denominator else 0.5


def _parent_concept_rates(
    sequence: list[dict[str, int]],
    questions: dict[str, dict[str, Any]],
) -> dict[str, list[float]]:
    grouped: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for step in sequence:
        parent = _parent_key(step, questions)
        cid = int(step["cid"])
        grouped[parent][cid].append(int(step["response"]))
    result: dict[str, list[float]] = {}
    for parent, concept_items in grouped.items():
        result[parent] = [
            _mean(values, default=0.5)
            for values in concept_items.values()
            if values
        ]
    return result


def _transfer_rates(
    sequence: list[dict[str, int]],
    questions: dict[str, dict[str, Any]],
) -> tuple[float, float]:
    successes = 0
    failures = 0
    opportunities = 0
    for previous, current in zip(sequence, sequence[1:]):
        if int(previous["cid"]) == int(current["cid"]):
            continue
        if _parent_key(previous, questions) != _parent_key(current, questions):
            continue
        if int(previous["response"]) != 1:
            continue
        opportunities += 1
        if int(current["response"]) == 1:
            successes += 1
        else:
            failures += 1
    if opportunities == 0:
        return 0.5, 0.5
    return successes / opportunities, failures / opportunities


def _related_key(step: dict[str, int], questions: dict[str, dict[str, Any]]) -> str:
    return _parent_key(step, questions) or str(step["cid"])


def _parent_key(step: dict[str, int], questions: dict[str, dict[str, Any]]) -> str:
    qmeta = questions.get(str(step["qid"]), {})
    routes = qmeta.get("kc_routes") or []
    if routes:
        route = str(routes[0])
        parts = [
            part.strip()
            for part in route.replace("----", "/").split("/")
            if part.strip()
        ]
        if len(parts) >= 2:
            return "/".join(parts[:-1])
        if parts:
            return parts[0]
    return str(step["cid"])


def _mean(values: list[float] | list[int], default: float) -> float:
    if not values:
        return default
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _clip(value: float) -> float:
    return min(1.0, max(0.0, value))


def _round(value: float) -> float:
    return round(_clip(value), 4)


def _level(value: float) -> str:
    if value >= 0.66:
        return "high"
    if value >= 0.33:
        return "medium"
    return "low"


def _trend_level(value: float) -> str:
    if value >= 0.10:
        return "improving"
    if value <= -0.10:
        return "declining"
    return "stable"

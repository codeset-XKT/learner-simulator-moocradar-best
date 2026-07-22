from __future__ import annotations

from collections import defaultdict
from typing import Any

from learner_simulator.data import clean_sequence


def build_ability_profile(
    sequence: list[dict[str, int]],
    questions: dict[str, dict[str, Any]] | None = None,
    normalization_rows: list[dict[str, str]] | None = None,
    recent_window: int = 10,
) -> dict[str, Any]:
    """Build an ability profile from observed behavior and global item stats.

    Unlike the cognitive profile, this module focuses on capability signals:
    overall response ability, knowledge coverage, difficulty adaptation, and
    near-transfer generalization. It uses only historical responses and global
    item statistics from the fitting corpus.
    """

    questions = questions or {}
    history = list(sequence)
    global_stats = _global_stats(normalization_rows or [], questions)
    concept_ids = [int(step["cid"]) for step in history]
    concept_stats = _concept_stats(history)
    distinct_concepts = len(concept_stats)
    coverage_ratio = (
        distinct_concepts / global_stats["distinct_concept_count"]
        if global_stats["distinct_concept_count"]
        else 0.0
    )
    practice_depth = len(history) / distinct_concepts if distinct_concepts else 0.0

    hard_steps, easy_steps = _difficulty_groups(history, global_stats["item_rates"])
    hard_success = _mean([int(step["response"]) for step in hard_steps], default=0.5)
    challenge_adaptation = _clip(hard_success)

    _, cross_parent_transfer = _transfer_ability(history, questions)
    reliability = _ability_reliability(
        interaction_count=len(history),
        distinct_concepts=distinct_concepts,
        hard_observations=len(hard_steps),
    )

    return {
        "module": "compact_ability_summary",
        "source": "observed_history",
        "knowledge_breadth": _level(coverage_ratio),
        "knowledge_breadth_value": _round(coverage_ratio),
        "practice_depth": _depth_level(practice_depth),
        "practice_depth_value": round(practice_depth, 4),
        "challenge_adaptation": _level(challenge_adaptation),
        "challenge_adaptation_value": _round(challenge_adaptation),
        "cross_domain_generalization": _level(cross_parent_transfer),
        "cross_domain_generalization_value": _round(cross_parent_transfer),
        "profile_confidence": reliability,
        "evidence_counts": {
            "interactions": len(history),
            "distinct_concepts": distinct_concepts,
            "hard_item_observations": len(hard_steps),
            "easy_item_observations": len(easy_steps),
        },
    }


def build_ability_profiles(
    rows: list[dict[str, str]],
    questions: dict[str, dict[str, Any]] | None = None,
    normalization_rows: list[dict[str, str]] | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        row["uid"]: build_ability_profile(
            clean_sequence(row),
            questions=questions,
            normalization_rows=normalization_rows or rows,
        )
        for row in rows
    }


def _global_stats(
    rows: list[dict[str, str]],
    questions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    item_values: dict[int, list[int]] = defaultdict(list)
    concepts: set[int] = set()
    for row in rows:
        for step in clean_sequence(row):
            qid = int(step["qid"])
            cid = int(step["cid"])
            item_values[qid].append(int(step["response"]))
            concepts.add(cid)
    return {
        "item_rates": {
            qid: _mean(values, default=0.5)
            for qid, values in item_values.items()
        },
        "distinct_concept_count": len(concepts),
    }


def _concept_stats(sequence: list[dict[str, int]]) -> dict[int, dict[str, int]]:
    stats: dict[int, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for step in sequence:
        cid = int(step["cid"])
        stats[cid]["correct"] += int(step["response"] == 1)
        stats[cid]["total"] += 1
    return dict(stats)


def _difficulty_groups(
    sequence: list[dict[str, int]],
    item_rates: dict[int, float],
) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    hard: list[dict[str, int]] = []
    easy: list[dict[str, int]] = []
    for step in sequence:
        rate = item_rates.get(int(step["qid"]))
        if rate is None:
            continue
        if rate <= 0.45:
            hard.append(step)
        elif rate >= 0.75:
            easy.append(step)
    return hard, easy


def _transfer_ability(
    sequence: list[dict[str, int]],
    questions: dict[str, dict[str, Any]],
) -> tuple[float, float]:
    same_parent: list[int] = []
    cross_parent: list[int] = []
    for previous, current in zip(sequence, sequence[1:]):
        if int(previous["response"]) != 1:
            continue
        if int(previous["cid"]) == int(current["cid"]):
            continue
        previous_parent = _parent_key(previous, questions)
        current_parent = _parent_key(current, questions)
        if previous_parent == current_parent:
            same_parent.append(int(current["response"]))
        else:
            cross_parent.append(int(current["response"]))
    return (
        _mean(same_parent, default=0.5),
        _mean(cross_parent, default=0.5),
    )


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


def _ability_reliability(
    interaction_count: int,
    distinct_concepts: int,
    hard_observations: int,
) -> str:
    if interaction_count >= 80 and distinct_concepts >= 20 and hard_observations >= 5:
        return "reliable"
    if interaction_count >= 30 and distinct_concepts >= 8:
        return "moderate"
    return "uncertain"


def _mean(values: list[int] | list[float], default: float) -> float:
    if not values:
        return default
    return sum(values) / len(values)


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


def _depth_level(value: float) -> str:
    if value >= 5:
        return "deep"
    if value >= 2:
        return "moderate"
    return "shallow"

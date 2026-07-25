from __future__ import annotations

import re
from typing import Any


def build_item_conditioned_ability(
    question: dict[str, Any],
    profile_context: dict[str, Any],
    memory_context: dict[str, Any],
    proficiency: dict[str, Any] | None,
    behavior_factors: dict[str, float] | None,
    tendency_calibration: dict[str, Any] | None,
) -> dict[str, Any]:
    """Describe how historical ability is activated by the current item.

    This is an activation profile, not an error model. It avoids correctness
    labels and probabilities, treats missing memory as unobserved rather than
    negative evidence, and uses KT mastery as current-concept state evidence.
    """

    cognitive = profile_context.get("cognitive_profile") or {}
    ability = profile_context.get("ability_profile") or {}
    transfer = cognitive.get("transfer_traits") or {}
    control = cognitive.get("control_traits") or {}
    affective = cognitive.get("cognitive_affective_proxies") or {}
    errors = cognitive.get("error_generation_traits") or {}

    mastery = _optional_float((proficiency or {}).get("value"))
    kt_anchor = _kt_decision_anchor(mastery)
    item_demand_score = _item_demand_score(question)
    item_demand = _demand_level(item_demand_score)
    memory = _memory_support(question, memory_context)

    breadth = _level_value(ability.get("knowledge_breadth"))
    practice = _level_value(ability.get("practice_depth"))
    challenge = _level_value(ability.get("challenge_adaptation"))
    generalization = _level_value(ability.get("cross_domain_generalization"))
    stability = _level_value(
        control.get("mastery_stability_level") or control.get("mastery_stability")
    )
    transfer_fragility = _level_value(
        transfer.get("transfer_fragility_level") or transfer.get("transfer_fragility")
    )
    concentration = _level_value(
        affective.get("concentration_level") or affective.get("concentration")
    )
    carelessness = _level_value(
        errors.get("carelessness_level") or errors.get("carelessness")
    )

    mastery_signal = mastery if mastery is not None else 0.5
    knowledge_alignment_score = (
        0.72 * mastery_signal
        + 0.16 * _scale_level(stability)
        + 0.12 * _scale_level(breadth)
    )
    if _mastery_protected(mastery, item_demand):
        knowledge_alignment_score = max(knowledge_alignment_score, 0.72)
    knowledge_alignment = _alignment_level(knowledge_alignment_score)

    practice_alignment = _practice_alignment(memory, practice)

    demand_fit_score = (
        0.28 * _scale_level(practice)
        + 0.24 * _scale_level(challenge)
        + 0.18 * _scale_level(breadth)
        + 0.18 * _scale_level(generalization)
        + 0.12 * _scale_level(stability)
        - 0.07 * item_demand_score
        - 0.04 * max(0, transfer_fragility - 2)
    )
    if _mastery_protected(mastery, item_demand):
        demand_fit_score = max(demand_fit_score, 0.50)
    demand_alignment = _alignment_level(demand_fit_score)

    memory_support = _memory_level(memory)
    transfer_burden = _transfer_burden(
        item_demand=item_demand,
        generalization=generalization,
        transfer_fragility=transfer_fragility,
        memory=memory,
    )

    activation_score = (
        0.38 * _alignment_score(knowledge_alignment)
        + 0.28 * _alignment_score(demand_alignment)
        + 0.20 * _practice_score(practice_alignment)
        + 0.08 * _scale_level(stability)
        + 0.06 * _behavior_support(behavior_factors, concentration, carelessness)
    )
    if transfer_burden == "high" and memory["related_outcome"] == "mostly_incorrect":
        activation_score -= 0.06
    if _mastery_protected(mastery, item_demand):
        activation_score = max(activation_score, 0.70)

    ability_activation = (
        "available" if activation_score >= 0.68
        else "partial" if activation_score >= 0.42
        else "limited"
    )
    activation_level = ability_activation

    if ability_activation == "available":
        ability_expression = "fluent"
    elif ability_activation == "limited":
        ability_expression = "tentative"
    else:
        ability_expression = "bounded"

    return {
        "module": "item_conditioned_ability",
        "method": "activation_from_profile_memory_proficiency_and_item",
        "knowledge_alignment": knowledge_alignment,
        "practice_alignment": practice_alignment,
        "demand_alignment": demand_alignment,
        "memory_support": memory_support,
        "transfer_burden": transfer_burden,
        "ability_activation": ability_activation,
        "activation_level": activation_level,
        "ability_expression": ability_expression,
        "kt_decision_anchor": kt_anchor,
        "evidence_role": (
            "qualitative_activation_profile_with_kt_as_primary_predictive_anchor_not_posthoc_override"
        ),
        "evidence": {
            "concept": (proficiency or {}).get("concept"),
            "kt_probability": kt_anchor["probability"],
            "kt_predicted_response": kt_anchor["predicted_response"],
            "kt_confidence_band": kt_anchor["confidence_band"],
            "mastery_level": (proficiency or {}).get("level"),
            "mastery_protected": _mastery_protected(mastery, item_demand),
            "item_demand": item_demand,
            "item_demand_score": item_demand_score,
            "practice_depth": ability.get("practice_depth"),
            "related_memory_count": memory["related_total"],
            "related_memory_outcome": memory["related_outcome"],
            "identical_correct_count": memory["identical_correct_count"],
            "knowledge_breadth": ability.get("knowledge_breadth"),
            "practice_depth": ability.get("practice_depth"),
            "challenge_adaptation": ability.get("challenge_adaptation"),
            "cross_domain_generalization": ability.get("cross_domain_generalization"),
            "transfer_fragility": transfer.get("transfer_fragility_level")
            or transfer.get("transfer_fragility"),
            "mastery_stability": control.get("mastery_stability_level")
            or control.get("mastery_stability"),
        },
        "response_guidance": _response_guidance(
            ability_activation=ability_activation,
            ability_expression=ability_expression,
            knowledge_alignment=knowledge_alignment,
            practice_alignment=practice_alignment,
            demand_alignment=demand_alignment,
            memory_support=memory_support,
            transfer_burden=transfer_burden,
        ),
    }


def _response_guidance(
    ability_activation: str,
    ability_expression: str,
    knowledge_alignment: str,
    practice_alignment: str,
    demand_alignment: str,
    memory_support: str,
    transfer_burden: str,
) -> str:
    if ability_expression == "fluent":
        return (
            "The current item activates usable ability. Generate a concise "
            "learner-level attempt; do not add teacher-style correction."
        )
    if ability_expression == "tentative":
        return (
            "The current item activates limited ability. Use simpler reasoning "
            "or lower confidence, but do not assume the answer must be wrong."
        )
    return (
        "The current item partially activates the learner's ability. Match the "
        f"reasoning depth to {knowledge_alignment} knowledge alignment, "
        f"{practice_alignment} practice alignment, {demand_alignment} demand alignment, "
        f"{memory_support} memory evidence, "
        f"and {transfer_burden} transfer demand."
    )


def _kt_decision_anchor(mastery: float | None) -> dict[str, Any]:
    if mastery is None:
        return {
            "probability": None,
            "predicted_response": None,
            "confidence_band": "unavailable",
            "decision_weight": "fallback",
            "instruction": (
                "No external KT probability is available; use learner profile, "
                "memory, and item evidence."
            ),
        }
    probability = min(1.0, max(0.0, float(mastery)))
    margin = abs(probability - 0.5)
    confidence_band = (
        "strong" if margin >= 0.25
        else "moderate" if margin >= 0.12
        else "uncertain"
    )
    predicted = int(probability >= 0.5)
    direction = "success" if predicted == 1 else "failure"
    return {
        "probability": round(probability, 3),
        "predicted_response": predicted,
        "confidence_band": confidence_band,
        "decision_weight": (
            "primary" if confidence_band in {"strong", "moderate"} else "secondary"
        ),
        "instruction": (
            f"Use the KT state as the main predictive anchor for likely {direction}. "
            "It may be revised only when recent same-concept memory, item demand, "
            "or non-cognitive evidence gives a clear educational reason."
        ),
    }


def _mastery_protected(mastery: float | None, item_demand: str) -> bool:
    return mastery is not None and mastery >= 0.75 and item_demand in {"low", "medium"}


def _item_demand_score(question: dict[str, Any]) -> int:
    text = _normalize_text(question.get("content"))
    options = question.get("options") or question.get("option") or []
    if isinstance(options, dict):
        option_count = len(options)
        option_text = " ".join(str(value) for value in options.values())
    elif isinstance(options, list):
        option_count = len(options)
        option_text = " ".join(str(value) for value in options)
    else:
        option_count = 0
        option_text = str(options)
    combined = f"{text} {_normalize_text(option_text)}"
    score = 0
    if len(text) > 90:
        score += 1
    if len(text) > 180:
        score += 1
    if option_count >= 4:
        score += 1
    if len(question.get("kc_routes") or []) >= 2:
        score += 1
    if re.search(
        r"calculate|compare|infer|derive|which of the following|relationship|why|how|least|most",
        combined,
    ):
        score += 1
    return min(score, 5)


def _demand_level(score: int) -> str:
    if score >= 4:
        return "high"
    if score >= 2:
        return "medium"
    return "low"


def _memory_support(
    question: dict[str, Any],
    memory_context: dict[str, Any],
) -> dict[str, Any]:
    current_routes = {str(item) for item in question.get("kc_routes") or []}
    short_memory = memory_context.get("short_memory") or []
    related_total = 0
    related_correct = 0
    identical_correct = 0
    for item in short_memory:
        routes = {str(route) for route in item.get("kc_routes") or []}
        related = bool(current_routes and routes & current_routes)
        if related:
            related_total += 1
            if int(item.get("real_response", item.get("simulated_response", 0)) or 0) == 1:
                related_correct += 1
        if (
            str(item.get("qid")) == str(question.get("qid"))
            and int(item.get("real_response", 0) or 0) == 1
        ):
            identical_correct += 1
    if related_total:
        related_rate = related_correct / related_total
        related_outcome = (
            "mostly_successful" if related_rate >= 0.67
            else "mixed" if related_rate >= 0.34
            else "mostly_incorrect"
        )
    else:
        related_rate = None
        related_outcome = "not_observed"
    return {
        "related_total": related_total,
        "related_correct": related_correct,
        "related_correct_rate": round(related_rate, 3) if related_rate is not None else None,
        "related_outcome": related_outcome,
        "identical_correct_count": identical_correct,
    }


def _practice_alignment(memory: dict[str, Any], practice_depth: int) -> str:
    if memory["identical_correct_count"] > 0:
        return "strong"
    rate = memory["related_correct_rate"]
    if rate is not None:
        if rate >= 0.67 and memory["related_total"] >= 2:
            return "strong"
        if rate >= 0.34:
            return "mixed"
        return "weak"
    if practice_depth >= 3:
        return "mixed"
    if practice_depth <= 1:
        return "not_observed"
    return "mixed"


def _memory_level(memory: dict[str, Any]) -> str:
    if memory["identical_correct_count"] > 0:
        return "strong"
    rate = memory["related_correct_rate"]
    if rate is None:
        return "not_observed"
    if rate >= 0.67 and memory["related_total"] >= 2:
        return "strong"
    if rate >= 0.34:
        return "mixed"
    return "weak"


def _transfer_burden(
    item_demand: str,
    generalization: int,
    transfer_fragility: int,
    memory: dict[str, Any],
) -> str:
    score = {"low": 0, "medium": 1, "high": 2}.get(item_demand, 1)
    if generalization <= 1:
        score += 1
    if transfer_fragility >= 3:
        score += 1
    if memory["related_outcome"] == "mostly_incorrect":
        score += 1
    if score >= 4:
        return "high"
    if score >= 2:
        return "medium"
    return "low"


def _behavior_support(
    behavior_factors: dict[str, float] | None,
    concentration_level: int,
    carelessness_level: int,
) -> float:
    if not behavior_factors:
        return 0.5
    carelessness = _optional_float(behavior_factors.get("carelessness")) or 0.0
    fatigue = _optional_float(behavior_factors.get("fatigue")) or 0.0
    guessing = _optional_float(behavior_factors.get("guessing")) or 0.0
    support = 1.0 - (0.35 * carelessness + 0.20 * fatigue + 0.15 * guessing)
    support += 0.04 * (_scale_level(concentration_level) - 0.5)
    support -= 0.03 * (_scale_level(carelessness_level) - 0.5)
    return min(1.0, max(0.0, support))


def _alignment_level(score: float) -> str:
    if score >= 0.68:
        return "high"
    if score >= 0.42:
        return "medium"
    return "low"


def _alignment_score(level: str) -> float:
    return {"high": 0.85, "medium": 0.55, "low": 0.25}.get(level, 0.5)


def _practice_score(level: str) -> float:
    return {
        "strong": 0.85,
        "mixed": 0.58,
        "weak": 0.35,
        "not_observed": 0.50,
    }.get(level, 0.50)


def _level_value(value: Any) -> int:
    if value is None:
        return 2
    text = str(value).strip().lower()
    if text in {"high", "deep", "strong", "stable", "broad"}:
        return 3
    if text in {"medium", "moderate", "mixed", "partial"}:
        return 2
    if text in {"low", "shallow", "weak", "fragile", "narrow"}:
        return 1
    try:
        numeric = float(text)
    except ValueError:
        return 2
    if numeric >= 0.67:
        return 3
    if numeric >= 0.34:
        return 2
    return 1


def _scale_level(value: int) -> float:
    return {1: 0.25, 2: 0.55, 3: 0.85}.get(value, 0.55)


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()

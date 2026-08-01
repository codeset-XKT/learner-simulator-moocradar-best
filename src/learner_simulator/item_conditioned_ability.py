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
    irt_evidence: dict[str, Any] | None = None,
    learning_tool_state: dict[str, Any] | None = None,
    historical_reflection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe how historical ability is activated by the current item.

    This is an activation profile, not an error model. It avoids correctness
    labels in prompts, treats missing memory as unobserved rather than negative
    evidence, and uses proficiency/mastery as current-concept state evidence.
    """

    cognitive = profile_context.get("cognitive_profile") or {}
    ability = profile_context.get("ability_profile") or {}
    control = cognitive.get("control_traits") or {}
    response_probability = _optional_float((proficiency or {}).get("response_probability"))
    concept_mastery = _optional_float((proficiency or {}).get("concept_mastery_value"))
    mastery = _optional_float((proficiency or {}).get("value"))
    readiness_probability = (
        response_probability
        if response_probability is not None
        else mastery
    )
    kt_anchor = _kt_decision_anchor(
        readiness_probability,
        source=(proficiency or {}).get("response_probability_source")
        or (proficiency or {}).get("source"),
    )
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
    knowledge_alignment = _knowledge_alignment(
        mastery=readiness_probability,
        memory=memory,
        item_demand=item_demand,
        stability=stability,
    )
    knowledge_alignment = _apply_historical_replay_calibration(
        knowledge_alignment=knowledge_alignment,
        memory=memory,
        item_demand=item_demand,
        historical_reflection=historical_reflection,
    )

    practice_alignment = _practice_alignment(memory, practice)

    demand_alignment = _demand_alignment(
        item_demand=item_demand,
        practice=practice,
        challenge=challenge,
        breadth=breadth,
        generalization=generalization,
        stability=stability,
    )
    demand_alignment = _apply_irt_relative_challenge(
        demand_alignment,
        irt_evidence,
        mastery=readiness_probability,
    )

    memory_support = _memory_level(memory)
    ability_activation = _ability_activation(
        knowledge_alignment=knowledge_alignment,
        demand_alignment=demand_alignment,
        practice_alignment=practice_alignment,
        memory_support=memory_support,
        mastery=readiness_probability,
        item_demand=item_demand,
    )
    ability_activation = _apply_learning_tool_state(
        ability_activation,
        learning_tool_state,
    )
    activation_level = ability_activation

    if ability_activation == "available":
        ability_expression = "fluent"
    elif ability_activation == "limited":
        ability_expression = "tentative"
    else:
        ability_expression = "bounded"

    kt_proficiency_state = {
        "concept": (proficiency or {}).get("concept"),
        "source": (proficiency or {}).get("source"),
        "value": kt_anchor["probability"],
        "response_probability": kt_anchor["probability"],
        "response_probability_source": (proficiency or {}).get(
            "response_probability_source"
        ),
        "concept_mastery_value": (
            round(concept_mastery, 3)
            if concept_mastery is not None
            else round(mastery, 3)
            if mastery is not None
            else None
        ),
        "concept_mastery_source": (proficiency or {}).get("concept_mastery_source"),
        "level": (proficiency or {}).get("level"),
        "confidence_band": kt_anchor["confidence_band"],
        "state_role": (
            "current_item_response_readiness_not_sampled_response_label"
        ),
    }

    return {
        "module": "item_conditioned_ability",
        "method": "proficiency_first_rule_based_activation_from_profile_memory_and_item",
        "knowledge_alignment": knowledge_alignment,
        "practice_alignment": practice_alignment,
        "demand_alignment": demand_alignment,
        "memory_support": memory_support,
        "ability_activation": ability_activation,
        "ability_expression": ability_expression,
        "activation_level": activation_level,
        "irt_relative_challenge": (irt_evidence or {}).get("relative_challenge"),
        "irt_boundary_band": (irt_evidence or {}).get("boundary_band"),
        "learning_tool_state": _compact_learning_tool_state(learning_tool_state),
        "kt_proficiency_state": kt_proficiency_state,
        "kt_decision_anchor": kt_anchor,
        "evidence_role": (
            "qualitative_activation_profile_with_proficiency_as_current_state_not_posthoc_override"
        ),
        "evidence": {
            "concept": (proficiency or {}).get("concept"),
            "kt_source": (proficiency or {}).get("source"),
            "kt_probability": kt_anchor["probability"],
            "current_item_response_probability": kt_anchor["probability"],
            "current_item_response_source": (proficiency or {}).get(
                "response_probability_source"
            ),
            "concept_mastery_value": (
                round(concept_mastery, 3)
                if concept_mastery is not None
                else round(mastery, 3)
                if mastery is not None
                else None
            ),
            "kt_predicted_response": kt_anchor["predicted_response"],
            "kt_confidence_band": kt_anchor["confidence_band"],
            "kt_proficiency_state": kt_proficiency_state,
            "mastery_level": (proficiency or {}).get("level"),
            "mastery_protected": _mastery_protected(
                readiness_probability,
                item_demand,
            ),
            "item_demand": item_demand,
            "item_demand_score": item_demand_score,
            "irt_ability_difficulty": _compact_irt_evidence(irt_evidence),
            "learning_tool_state": _compact_learning_tool_state(learning_tool_state),
            "historical_replay_calibration": _compact_historical_reflection(
                historical_reflection
            ),
            "practice_depth": ability.get("practice_depth"),
            "related_memory_count": memory["related_total"],
            "related_memory_outcome": memory["related_outcome"],
            "identical_correct_count": memory["identical_correct_count"],
            "knowledge_breadth": ability.get("knowledge_breadth"),
            "practice_depth": ability.get("practice_depth"),
            "challenge_adaptation": ability.get("challenge_adaptation"),
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
        ),
    }


def _response_guidance(
    ability_activation: str,
    ability_expression: str,
    knowledge_alignment: str,
    practice_alignment: str,
    demand_alignment: str,
    memory_support: str,
) -> str:
    if ability_expression == "fluent":
        return (
            "The current item activates usable ability. Generate a concise "
            "learner-level attempt."
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
        f"and {memory_support} memory evidence."
    )


def _compact_irt_evidence(irt_evidence: dict[str, Any] | None) -> dict[str, Any] | None:
    if not irt_evidence:
        return None
    return {
        "learner_ability_level": irt_evidence.get("learner_ability_level"),
        "item_difficulty_level": irt_evidence.get("item_difficulty_level"),
        "relative_challenge": irt_evidence.get("relative_challenge"),
        "boundary_band": irt_evidence.get("boundary_band"),
        "evidence_role": irt_evidence.get("evidence_role"),
    }


def _compact_learning_tool_state(
    learning_tool_state: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not learning_tool_state:
        return None
    return {
        "joint_readiness_state": learning_tool_state.get("joint_readiness_state"),
        "decision_anchor": learning_tool_state.get("decision_anchor"),
        "state_commitment": learning_tool_state.get("state_commitment"),
        "response_planning": learning_tool_state.get("response_planning"),
        "four_tier_generation_policy": learning_tool_state.get("four_tier_generation_policy"),
        "knowledge_tool": learning_tool_state.get("knowledge_tool"),
    }


def _compact_historical_reflection(
    historical_reflection: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not historical_reflection:
        return None
    policy = historical_reflection.get("adaptive_policy") or {}
    return {
        "available": historical_reflection.get("available"),
        "bias_direction": historical_reflection.get("bias_direction"),
        "error_pattern": historical_reflection.get("error_pattern"),
        "evidence_confidence": historical_reflection.get("evidence_confidence"),
        "actual_correct_rate": historical_reflection.get("actual_correct_rate"),
        "predicted_correct_rate": historical_reflection.get("predicted_correct_rate"),
        "response_bias": policy.get("response_bias"),
        "ncdm_trust": policy.get("ncdm_trust"),
        "slip_guess_policy": policy.get("slip_guess_policy"),
    }


def _apply_historical_replay_calibration(
    knowledge_alignment: str,
    memory: dict[str, Any],
    item_demand: str,
    historical_reflection: dict[str, Any] | None,
) -> str:
    """Use observed-history replay to calibrate readiness interpretation.

    This is not target feedback and not a post-hoc label correction. It only
    changes how strict the simulator should be when interpreting a low/medium
    NCDM mastery value for this learner.
    """

    if not historical_reflection or not historical_reflection.get("available"):
        return knowledge_alignment
    confidence = str(historical_reflection.get("evidence_confidence") or "low")
    if confidence == "low":
        return knowledge_alignment
    policy = historical_reflection.get("adaptive_policy") or {}
    response_bias = str(policy.get("response_bias") or "")
    error_pattern = str(historical_reflection.get("error_pattern") or "")
    memory_outcome = str(memory.get("related_outcome") or "not_observed")

    if response_bias == "preserve_plausible_success" or error_pattern == "missed_successes":
        if item_demand != "high" and memory_outcome != "mostly_incorrect":
            return _raise_level(knowledge_alignment)
        if knowledge_alignment == "low" and memory_outcome in {"mixed", "mostly_successful"}:
            return "medium"

    if response_bias == "guard_against_overconfidence":
        if item_demand == "high" or memory_outcome == "mostly_incorrect":
            return _lower_level(knowledge_alignment)

    return knowledge_alignment


def _apply_learning_tool_state(
    ability_activation: str,
    learning_tool_state: dict[str, Any] | None,
) -> str:
    # The learning tool state is already represented as the KT decision anchor.
    # Do not raise activation again here; that double-counts NCDM and makes the
    # simulator more optimistic than the KT/CDM model itself.
    return ability_activation


def _raise_activation(value: str) -> str:
    return {
        "limited": "partial",
        "partial": "available",
        "available": "available",
    }.get(value, value)


def _lower_activation(value: str) -> str:
    return {
        "available": "partial",
        "partial": "limited",
        "limited": "limited",
    }.get(value, value)


def _apply_irt_relative_challenge(
    demand_alignment: str,
    irt_evidence: dict[str, Any] | None,
    mastery: float | None,
) -> str:
    if not irt_evidence or irt_evidence.get("ablated"):
        return demand_alignment
    challenge = irt_evidence.get("relative_challenge")
    boundary = irt_evidence.get("boundary_band")
    if challenge == "above_learner_ability" and boundary in {"clear", "moderate"}:
        if mastery is not None and mastery >= 0.65:
            return demand_alignment
        return _lower_level(demand_alignment)
    if challenge == "below_learner_ability" and boundary == "clear":
        return _raise_level(demand_alignment)
    return demand_alignment


def _kt_decision_anchor(mastery: float | None, source: Any = None) -> dict[str, Any]:
    if mastery is None:
        return {
            "source": source or "unavailable",
            "probability": None,
            "predicted_response": None,
            "confidence_band": "unavailable",
            "evidence_priority": "fallback",
            "instruction": (
                "No external KT probability is available; use learner profile, "
                "memory, and item evidence."
            ),
        }
    probability = min(1.0, max(0.0, float(mastery)))
    anchor = _anchor_name(probability)
    confidence_band = _anchor_confidence(anchor)
    threshold_reference = int(probability >= 0.5)
    return {
        "source": source or "dynamic",
        "probability": round(probability, 3),
        "predicted_response": threshold_reference,
        "threshold_reference_response": threshold_reference,
        "anchor": anchor,
        "direction": (
            "correct" if probability >= 0.60
            else "incorrect" if probability < 0.45
            else "uncertain"
        ),
        "confidence_band": confidence_band,
        "evidence_priority": (
            "primary" if confidence_band in {"strong", "moderate"} else "secondary"
        ),
        "instruction": (
            f"Use the KT state as the main symmetric decision anchor: {anchor}. "
            "When the anchor is strong or moderate, LearnerCorrect should normally "
            "follow its direction. Deviate only when concrete same-concept memory, "
            "item demand, or learner profile evidence clearly contradicts it. "
            "When the anchor is boundary, decide from memory, item demand, and profile."
        ),
    }


def _mastery_protected(mastery: float | None, item_demand: str) -> bool:
    return mastery is not None and mastery >= 0.75 and item_demand in {"low", "medium"}


def _anchor_name(probability: float) -> str:
    if probability >= 0.75:
        return "kt_strong_correct_anchor"
    if probability >= 0.60:
        return "kt_lean_correct_anchor"
    if probability >= 0.45:
        return "kt_boundary_anchor"
    if probability >= 0.30:
        return "kt_lean_incorrect_anchor"
    return "kt_strong_incorrect_anchor"


def _anchor_confidence(anchor: str) -> str:
    if anchor in {"kt_strong_correct_anchor", "kt_strong_incorrect_anchor"}:
        return "strong"
    if anchor in {"kt_lean_correct_anchor", "kt_lean_incorrect_anchor"}:
        return "moderate"
    return "uncertain"


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


def _knowledge_alignment(
    mastery: float | None,
    memory: dict[str, Any],
    item_demand: str,
    stability: int,
) -> str:
    """Infer current-concept readiness without hand-tuned weighted fusion."""

    if mastery is None:
        level = "medium"
    elif mastery >= 0.70:
        level = "high"
    elif mastery >= 0.40:
        level = "medium"
    else:
        level = "low"

    if memory["identical_correct_count"] > 0:
        level = _raise_level(level)
    elif memory["related_outcome"] == "mostly_successful" and stability >= 2:
        level = _raise_level(level)
    elif memory["related_outcome"] == "mostly_incorrect" and not _mastery_protected(mastery, item_demand):
        level = _lower_level(level)

    if item_demand == "high" and not _mastery_protected(mastery, item_demand):
        level = _lower_level(level)
    return level


def _demand_alignment(
    item_demand: str,
    practice: int,
    challenge: int,
    breadth: int,
    generalization: int,
    stability: int,
) -> str:
    if item_demand == "low":
        level = "high" if practice >= 2 or stability >= 2 else "medium"
    elif item_demand == "medium":
        level = "high" if practice >= 3 and challenge >= 2 else "medium"
    else:
        level = "medium" if challenge >= 3 and breadth >= 2 and generalization >= 2 else "low"

    if generalization >= 3 and challenge >= 3 and item_demand != "low":
        level = _raise_level(level)
    return level


def _ability_activation(
    knowledge_alignment: str,
    demand_alignment: str,
    practice_alignment: str,
    memory_support: str,
    mastery: float | None,
    item_demand: str,
) -> str:
    if _mastery_protected(mastery, item_demand) and demand_alignment != "low":
        return "available"
    if knowledge_alignment == "high":
        if demand_alignment == "low":
            return "partial"
        return "available"
    if knowledge_alignment == "medium":
        if (
            demand_alignment == "high"
            or practice_alignment == "weak"
        ):
            return "limited"
        return "partial"
    if memory_support == "strong" and demand_alignment != "high":
        return "partial"
    return "limited"


def _raise_level(level: str) -> str:
    return {"low": "medium", "medium": "high", "high": "high"}.get(level, "medium")


def _lower_level(level: str) -> str:
    return {"high": "medium", "medium": "low", "low": "low"}.get(level, "medium")


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

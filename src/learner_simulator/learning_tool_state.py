from __future__ import annotations

from typing import Any


def build_learning_tool_state(
    question: dict[str, Any],
    proficiency: dict[str, Any] | None,
    irt_evidence: dict[str, Any] | None,
    memory_context: dict[str, Any],
) -> dict[str, Any]:
    """Encode KT/CDM and IRT tools as a pre-response learner state.

    This module is intentionally qualitative. It translates NCDM proficiency
    and IRT ability-difficulty evidence into a latent readiness state before
    the LLM generates an answer. It does not sample a label and does not
    override the LLM response after generation.
    """

    response_probability = _optional_float((proficiency or {}).get("response_probability"))
    concept_mastery = _optional_float((proficiency or {}).get("concept_mastery_value"))
    mastery = _optional_float((proficiency or {}).get("value"))
    readiness_probability = (
        response_probability
        if response_probability is not None
        else mastery
    )
    kt_level = _knowledge_readiness(readiness_probability)
    irt_challenge = (irt_evidence or {}).get("relative_challenge")
    irt_boundary = (irt_evidence or {}).get("boundary_band")
    memory = _related_memory(question, memory_context)
    readiness = _joint_readiness(
        kt_level=kt_level,
        mastery=readiness_probability,
        irt_challenge=irt_challenge,
        irt_boundary=irt_boundary,
        memory_pattern=memory["pattern"],
    )
    decision_anchor = _decision_anchor(readiness_probability)
    return {
        "module": "learning_tool_state_encoder",
        "method": "agent4edu_style_ncdm_irt_tool_conditioning",
        "evidence_role": "pre_response_state_conditioning_not_posthoc_override",
        "knowledge_tool": {
            "tool": "NCDM" if (proficiency or {}).get("source") else "unavailable",
            "source": (proficiency or {}).get("source"),
            "concept": (proficiency or {}).get("concept"),
            "mastery_level": (proficiency or {}).get("level") or kt_level,
            "readiness": kt_level,
            "readiness_probability": (
                round(readiness_probability, 3)
                if readiness_probability is not None
                else None
            ),
            "item_response_probability": (
                round(response_probability, 3)
                if response_probability is not None
                else None
            ),
            "item_response_probability_source": (proficiency or {}).get(
                "response_probability_source"
            ),
            "concept_mastery_value": (
                round(concept_mastery, 3)
                if concept_mastery is not None
                else round(mastery, 3)
                if mastery is not None
                else None
            ),
            "concept_mastery_source": (proficiency or {}).get(
                "concept_mastery_source"
            ),
            "mastery_value": (
                round(readiness_probability, 3)
                if readiness_probability is not None
                else None
            ),
            "role": "current_item_response_readiness_with_concept_mastery_context",
        },
        "ability_difficulty_tool": {
            "tool": "IRT" if irt_evidence and not irt_evidence.get("ablated") else "unavailable",
            "learner_ability_level": (irt_evidence or {}).get("learner_ability_level"),
            "item_difficulty_level": (irt_evidence or {}).get("item_difficulty_level"),
            "relative_challenge": irt_challenge,
            "boundary_band": irt_boundary,
            "role": "learner_ability_vs_item_difficulty_state",
        },
        "related_memory_tool": memory,
        "joint_readiness_state": readiness,
        "decision_anchor": decision_anchor,
        "state_commitment": _state_commitment(readiness, decision_anchor),
        "response_planning": decision_anchor["anchor"],
        "four_tier_generation_policy": _four_tier_generation_policy(decision_anchor),
        "constraints": [
            "Read Profile, Memory, and Tool Evidence before the action step.",
            "Use the NCDM current-item response probability as the primary readiness tool when available.",
            "Use NCDM concept mastery as supporting knowledge-state context.",
            "Use IRT as a secondary ability-difficulty tool.",
            "Use memory and profile mainly to shape reasoning style and confidence.",
            "Use the KT decision anchor as a symmetric prior for LearnerCorrect.",
            "Do not turn readiness into a success guarantee or weakness into an automatic error.",
        ],
    }


def prompt_learning_tool_state(state: dict[str, Any] | None) -> dict[str, Any]:
    if not state:
        return {}
    return {
        "module": state.get("module"),
        "method": state.get("method"),
        "evidence_role": state.get("evidence_role"),
        "knowledge_tool": state.get("knowledge_tool"),
        "ability_difficulty_tool": state.get("ability_difficulty_tool"),
        "related_memory_tool": state.get("related_memory_tool"),
        "joint_readiness_state": state.get("joint_readiness_state"),
        "decision_anchor": state.get("decision_anchor"),
        "state_commitment": state.get("state_commitment"),
        "response_planning": state.get("response_planning"),
        "four_tier_generation_policy": state.get("four_tier_generation_policy"),
        "constraints": state.get("constraints"),
    }


def _joint_readiness(
    kt_level: str,
    mastery: float | None,
    irt_challenge: Any,
    irt_boundary: Any,
    memory_pattern: str,
) -> str:
    if kt_level == "strong":
        if irt_challenge == "above_learner_ability" and irt_boundary == "clear":
            return "ncdm_supported_challenged"
        if memory_pattern == "recent_related_failures" and mastery is not None and mastery < 0.8:
            return "ncdm_supported_unstable"
        return "ncdm_supported"
    if kt_level == "usable":
        if irt_challenge == "below_learner_ability" and memory_pattern in {
            "recent_related_successes",
            "mixed_or_sparse",
        }:
            return "ncdm_developing_supported"
        if irt_challenge == "above_learner_ability":
            return "ncdm_developing_challenged"
        return "ncdm_developing"
    if memory_pattern == "recent_related_successes" and irt_challenge != "above_learner_ability":
        return "ncdm_weak_with_memory"
    if kt_level == "weak":
        if irt_challenge == "below_learner_ability":
            return "ncdm_fragile_but_accessible"
        if mastery is not None and mastery >= 0.25 and memory_pattern != "recent_related_failures":
            return "ncdm_fragile_developing"
        return "ncdm_unsupported"
    return "ncdm_uncertain"


def _state_commitment(readiness: str, decision_anchor: dict[str, Any]) -> str:
    anchor = decision_anchor.get("anchor", "kt_boundary_anchor")
    direction = decision_anchor.get("direction", "uncertain")
    mapping = {
        "ncdm_supported": (
            "NCDM strongly supports current-concept readiness. Use this as a strong "
            "correct-leaning anchor, while still checking concrete item and memory evidence."
        ),
        "ncdm_supported_challenged": (
            "NCDM supports readiness while IRT marks challenge. Keep the KT anchor "
            "primary, but treat the item as a possible boundary case."
        ),
        "ncdm_supported_unstable": (
            "NCDM supports readiness, but recent related memory is mixed. Use the "
            "anchor direction with lower confidence."
        ),
        "ncdm_developing_supported": (
            "NCDM indicates developing readiness with favorable challenge or memory "
            "evidence. Treat this as a lean-correct rather than strong-correct case."
        ),
        "ncdm_developing": (
            "NCDM indicates developing readiness. Treat this as a boundary or "
            "lean-correct case depending on the KT probability."
        ),
        "ncdm_developing_challenged": (
            "NCDM indicates developing readiness and IRT marks extra challenge. "
            "Use a boundary decision unless memory strongly supports one side."
        ),
        "ncdm_weak_with_memory": (
            "NCDM is weak but related memory exists. Let memory moderate the weak "
            "anchor, without flipping it automatically."
        ),
        "ncdm_fragile_but_accessible": (
            "NCDM is weak while ability-difficulty evidence suggests accessibility. "
            "Treat this as a weak or boundary anchor, not as guaranteed success."
        ),
        "ncdm_fragile_developing": (
            "NCDM is fragile rather than absent. Use a lean-incorrect or boundary "
            "anchor with calibrated confidence."
        ),
        "ncdm_unsupported": (
            "NCDM gives little support for this concept. Use an incorrect-leaning "
            "anchor unless concrete related memory contradicts it."
        ),
        "ncdm_uncertain": (
            "Tool evidence is uncertain. Use the remaining memory, profile, and visible item cues."
        ),
    }
    return (
        f"KT anchor={anchor}, direction={direction}. "
        + mapping.get(readiness, mapping["ncdm_uncertain"])
    )


def _decision_anchor(probability: float | None) -> dict[str, Any]:
    if probability is None:
        return {
            "anchor": "kt_unavailable_anchor",
            "direction": "unknown",
            "probability": None,
            "predicted_response": None,
            "confidence_band": "unavailable",
        }
    p = min(1.0, max(0.0, float(probability)))
    if p >= 0.75:
        anchor = "kt_strong_correct_anchor"
        direction = "correct"
        confidence = "strong"
    elif p >= 0.60:
        anchor = "kt_lean_correct_anchor"
        direction = "correct"
        confidence = "moderate"
    elif p >= 0.45:
        anchor = "kt_boundary_anchor"
        direction = "uncertain"
        confidence = "uncertain"
    elif p >= 0.30:
        anchor = "kt_lean_incorrect_anchor"
        direction = "incorrect"
        confidence = "moderate"
    else:
        anchor = "kt_strong_incorrect_anchor"
        direction = "incorrect"
        confidence = "strong"
    return {
        "anchor": anchor,
        "direction": direction,
        "probability": round(p, 3),
        "predicted_response": int(p >= 0.5),
        "confidence_band": confidence,
    }


def _four_tier_generation_policy(decision_anchor: dict[str, Any]) -> dict[str, str]:
    anchor = str(decision_anchor.get("anchor") or "kt_boundary_anchor")
    if anchor == "kt_strong_correct_anchor":
        answer_policy = "follow_strong_correct_anchor_unless_specific_conflict"
        confidence_policy = "medium_or_high_answer_confidence"
        reasoning_policy = "short_fluent_learner_reasoning"
    elif anchor == "kt_lean_correct_anchor":
        answer_policy = "lean_correct_with_item_and_memory_check"
        confidence_policy = "medium_answer_confidence"
        reasoning_policy = "brief_checked_learner_reasoning"
    elif anchor == "kt_boundary_anchor":
        answer_policy = "decide_from_memory_item_demand_and_profile"
        confidence_policy = "medium_or_low_answer_confidence"
        reasoning_policy = "partial_learner_reasoning"
    elif anchor == "kt_lean_incorrect_anchor":
        answer_policy = "lean_incorrect_unless_specific_memory_supports_success"
        confidence_policy = "low_or_medium_answer_confidence"
        reasoning_policy = "limited_or_uncertain_learner_reasoning"
    elif anchor == "kt_strong_incorrect_anchor":
        answer_policy = "follow_strong_incorrect_anchor_unless_specific_conflict"
        confidence_policy = "low_answer_confidence"
        reasoning_policy = "limited_learner_reasoning"
    else:
        answer_policy = "use_remaining_evidence"
        confidence_policy = "medium_or_low_answer_confidence"
        reasoning_policy = "brief_learner_reasoning"
    return {
        "tool_conditioning": "ncdm_primary_irt_secondary",
        "output_target": "four_tier_answer_reasoning_confidence",
        "answer_policy": answer_policy,
        "confidence_policy": confidence_policy,
        "reasoning_policy": reasoning_policy,
    }


def _knowledge_readiness(mastery: float | None) -> str:
    if mastery is None:
        return "unknown"
    if mastery >= 0.70:
        return "strong"
    if mastery >= 0.40:
        return "usable"
    return "weak"


def _related_memory(
    question: dict[str, Any],
    memory_context: dict[str, Any],
) -> dict[str, Any]:
    routes = {str(route) for route in question.get("kc_routes") or []}
    related_total = 0
    related_correct = 0
    for item in memory_context.get("short_memory") or []:
        item_routes = {str(route) for route in item.get("kc_routes") or []}
        if not routes or not (routes & item_routes):
            continue
        related_total += 1
        response = item.get("real_response", item.get("simulated_response", 0))
        related_correct += int(int(response or 0) == 1)
    if related_total == 0:
        pattern = "not_observed"
        rate = None
    else:
        rate = related_correct / related_total
        if rate >= 0.67:
            pattern = "recent_related_successes"
        elif rate <= 0.33:
            pattern = "recent_related_failures"
        else:
            pattern = "mixed_or_sparse"
    return {
        "scope": "short_memory_same_concept_or_route",
        "related_total": related_total,
        "related_correct": related_correct,
        "related_correct_rate": round(rate, 3) if rate is not None else None,
        "pattern": pattern,
        "role": "observed_history_support_not_target_feedback",
    }


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

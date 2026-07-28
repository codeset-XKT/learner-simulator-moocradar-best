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

    mastery = _optional_float((proficiency or {}).get("value"))
    kt_level = _knowledge_readiness(mastery)
    irt_challenge = (irt_evidence or {}).get("relative_challenge")
    irt_boundary = (irt_evidence or {}).get("boundary_band")
    memory = _related_memory(question, memory_context)
    readiness = _joint_readiness(
        kt_level=kt_level,
        mastery=mastery,
        irt_challenge=irt_challenge,
        irt_boundary=irt_boundary,
        memory_pattern=memory["pattern"],
    )
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
            "mastery_value": round(mastery, 3) if mastery is not None else None,
            "role": "current_concept_knowledge_state",
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
        "state_commitment": _state_commitment(readiness),
        "response_planning": _response_planning(readiness),
        "four_tier_generation_policy": _four_tier_generation_policy(readiness),
        "constraints": [
            "Read Profile, Memory, and Tool Evidence before the action step.",
            "Use NCDM as the primary current knowledge-state tool.",
            "Use IRT as a secondary ability-difficulty tool.",
            "Use memory and profile mainly to shape reasoning style and confidence.",
            "Generate a four-tier learner response rather than a Task4 Yes/No label.",
            "Do not copy any tool value as a correctness label.",
            "Do not perform a final expert correction pass after drafting the answer.",
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
        return "ncdm_unsupported"
    return "ncdm_uncertain"


def _state_commitment(readiness: str) -> str:
    mapping = {
        "ncdm_supported": (
            "NCDM strongly supports current-concept readiness. Preserve a concise, "
            "ordinary student-level successful first attempt unless the item clearly requires unsupported knowledge."
        ),
        "ncdm_supported_challenged": (
            "NCDM supports readiness, while IRT marks the item as challenging. Keep success plausible and mainly reduce confidence or reasoning fluency."
        ),
        "ncdm_supported_unstable": (
            "NCDM supports readiness, but recent related memory is mixed. Preserve success on routine items; express uncertainty through confidence or brief reasoning."
        ),
        "ncdm_developing_supported": (
            "NCDM indicates developing readiness with favorable challenge or memory evidence. A correct first attempt is plausible with moderate confidence."
        ),
        "ncdm_developing": (
            "NCDM indicates developing readiness. Use partial learner reasoning and moderate confidence without defaulting to failure."
        ),
        "ncdm_developing_challenged": (
            "NCDM indicates developing readiness and IRT marks extra challenge. Use tentative reasoning; both success and mistake remain plausible."
        ),
        "ncdm_weak_with_memory": (
            "NCDM is weak but related memory exists. A cue-based answer can still be correct, usually with modest confidence."
        ),
        "ncdm_unsupported": (
            "NCDM gives little support for this concept. Use limited learner reasoning and low confidence; do not create a polished expert solution."
        ),
        "ncdm_uncertain": (
            "Tool evidence is uncertain. Use the remaining memory, profile, and visible item cues while avoiding teacher-style derivation."
        ),
    }
    return mapping.get(readiness, mapping["ncdm_uncertain"])


def _response_planning(readiness: str) -> str:
    if readiness in {"ncdm_supported", "ncdm_supported_challenged", "ncdm_supported_unstable"}:
        return "preserve_ncdm_supported_success"
    if readiness in {"ncdm_developing_supported", "ncdm_developing"}:
        return "success_plausible_with_moderate_confidence"
    if readiness == "ncdm_developing_challenged":
        return "mixed_attempt_with_reduced_confidence"
    if readiness == "ncdm_weak_with_memory":
        return "cue_based_success_possible"
    return "unsupported_attempt_with_low_confidence"


def _four_tier_generation_policy(readiness: str) -> dict[str, str]:
    policies = {
        "ncdm_supported": {
            "answer_policy": "generate_successful_first_attempt_when_item_is_routine",
            "confidence_policy": "medium_or_high_answer_confidence",
            "reasoning_policy": "short_fluent_learner_reasoning",
        },
        "ncdm_supported_challenged": {
            "answer_policy": "success_remains_plausible_under_challenge",
            "confidence_policy": "medium_answer_confidence",
            "reasoning_policy": "short_reasoning_with_possible_hesitation",
        },
        "ncdm_supported_unstable": {
            "answer_policy": "preserve_success_on_routine_items_allow_minor_uncertainty",
            "confidence_policy": "medium_answer_confidence",
            "reasoning_policy": "brief_reasoning_reflecting_unstable_memory",
        },
        "ncdm_developing_supported": {
            "answer_policy": "correct_attempt_plausible_from_memory_or_visible_cues",
            "confidence_policy": "medium_answer_confidence",
            "reasoning_policy": "partial_but_sufficient_learner_reasoning",
        },
        "ncdm_developing": {
            "answer_policy": "mixed_outcome_depends_on_item_cues",
            "confidence_policy": "medium_or_low_answer_confidence",
            "reasoning_policy": "partial_learner_reasoning",
        },
        "ncdm_developing_challenged": {
            "answer_policy": "mistake_plausible_but_not_required",
            "confidence_policy": "low_or_medium_answer_confidence",
            "reasoning_policy": "tentative_partial_reasoning",
        },
        "ncdm_weak_with_memory": {
            "answer_policy": "cue_based_correct_attempt_possible",
            "confidence_policy": "low_or_medium_answer_confidence",
            "reasoning_policy": "shallow_memory_based_reasoning",
        },
        "ncdm_unsupported": {
            "answer_policy": "unsupported_attempt_often_incomplete",
            "confidence_policy": "low_answer_confidence",
            "reasoning_policy": "limited_learner_reasoning",
        },
        "ncdm_uncertain": {
            "answer_policy": "use_remaining_evidence_without_expert_solution",
            "confidence_policy": "medium_or_low_answer_confidence",
            "reasoning_policy": "brief_learner_reasoning",
        },
    }
    return {
        "tool_conditioning": "ncdm_primary_irt_secondary",
        "output_target": "four_tier_answer_reasoning_confidence",
        **policies.get(readiness, policies["ncdm_uncertain"]),
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

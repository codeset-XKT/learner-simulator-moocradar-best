from __future__ import annotations

from typing import Any

from learner_simulator.agent4edu_prompt import build_action_prompt


def build_cognitive_profile_view(
    profile_context: dict[str, Any],
) -> dict[str, Any]:
    cognitive = profile_context.get("cognitive_profile") or {}
    control = cognitive.get("control_traits") or {}
    affective = cognitive.get("cognitive_affective_proxies") or {}
    errors = cognitive.get("error_generation_traits") or {}
    transfer = cognitive.get("transfer_traits") or {}
    ability = profile_context.get("ability_profile") or {}
    return {
        "module": "learner_profile_evidence",
        "scope": "learner_level_stable_traits",
        "general_performance_level": control.get("overall_success_level"),
        "learning_stability": {
            "recent_success": control.get("recent_success_level"),
            "success_trend": control.get("success_trend_level"),
            "mastery_stability": control.get("mastery_stability_level"),
        },
        "error_tendency": {
            "carelessness": errors.get("carelessness_level"),
            "guessing": errors.get("guessing_level"),
            "misconception_persistence": errors.get("misconception_persistence_level"),
            "error_recovery": errors.get("error_recovery_level"),
        },
        "affective_state_proxy": {
            "concentration": affective.get("concentration_level"),
            "confusion": affective.get("confusion_level"),
            "frustration": affective.get("frustration_level"),
            "boredom": affective.get("boredom_level"),
        },
        "transfer_trait": {
            "same_parent_transfer": transfer.get("same_parent_transfer_level"),
            "transfer_fragility": transfer.get("transfer_fragility_level"),
        },
        "ability_trait": {
            "knowledge_breadth": ability.get("knowledge_breadth"),
            "challenge_adaptation": ability.get("challenge_adaptation"),
            "cross_domain_generalization": ability.get("cross_domain_generalization"),
            "profile_confidence": ability.get("profile_confidence"),
        },
    }


def build_response_prompt(
    question: dict[str, Any],
    short_memory: list[dict[str, Any]],
    long_memory: dict[str, Any],
    concept_options: list[str],
    behavior_factors: dict[str, float] | None,
    cognitive_profile: dict[str, Any],
    item_conditioned_ability: dict[str, Any] | None = None,
    response_format: str = "four_tier",
) -> str:
    prompt_item_evidence = _prompt_item_conditioned_evidence(
        item_conditioned_ability or {}
    )
    base = build_action_prompt(
        question=question,
        short_memory=short_memory,
        long_memory=long_memory,
        concept_options=concept_options,
        proficiency=None,
        behavior_factors=behavior_factors,
        response_format=response_format,
        cognitive_strategy=None,
        tendency_calibration=None,
    )
    prefix = "\n".join(
        [
            "# Response Agent #",
            "Generate this learner's single first attempt using the educational constraints below.",
            "The item-conditioned ability profile describes how the learner's historical ability is activated by this specific item.",
            "It is qualitative evidence. Use it to shape response style, reasoning depth, confidence, and likely correctness.",
            "Do not add an expert correction pass, and do not output correctness.",
            "The KT mastery/proficiency signal is part of the current learner state. "
            "Use it together with profile, memory, item demand, and non-cognitive evidence; do not turn it into a deterministic answer label.",
            "",
            "# Learner Profile Evidence #",
            _compact(cognitive_profile),
            "",
            "# Item-conditioned Evidence #",
            _compact(prompt_item_evidence),
            "",
            "# KT State Guidance #",
            _compact(_kt_state_guidance(prompt_item_evidence)),
            "",
            "# Process-to-Response Constraints #",
            "- Decide the learner state before solving. Use KT state, memory, item demand, learner profile, and non-cognitive state together.",
            "- Do not force the submitted answer to equal the KT prediction after generation; let the evidence shape the latent learner state before the answer is produced.",
            "- If ability_activation is available, the attempt may be direct, concise, and learner-level.",
            "- If ability_activation is partial, include bounded reasoning depth and moderate confidence.",
            "- If ability_activation is limited, use simpler reasoning or lower confidence, but do not assume the final answer must be wrong.",
            "- Strong knowledge alignment supports concise recall or routine application.",
            "- Strong practice alignment supports familiar learner-level solution patterns.",
            "- Missing related memory is unobserved evidence, not evidence of inability.",
            "- Negative related memory may lower confidence or simplify reasoning, but it should not force an incorrect answer.",
            "- Careless or high-load evidence may affect verification depth, but avoid inventing mistakes unsupported by the item and learner evidence.",
            "- Weak knowledge evidence supports shallow cues or elimination rather than a full derivation.",
            "- If strong current knowledge alignment and low/medium item demand both support success, the attempt can be direct and correct at learner level.",
            "- Broad profile weaknesses should not override strong current-concept activation unless current item demand clearly exceeds available evidence.",
            "- Do not invent a wrong answer solely to appear learner-like when current state, memory, and item demand all support success.",
            "",
            "# Four-tier Output Requirement #",
            "The response must include answer, confidence, reasoning, and reasoning confidence so that a separate diagnostic module can assess it.",
            "",
        ]
    )
    return prefix + base


def _kt_state_guidance(item_conditioned_ability: dict[str, Any] | None) -> dict[str, Any]:
    ability = item_conditioned_ability or {}
    knowledge_alignment = ability.get("knowledge_alignment")
    practice_alignment = ability.get("practice_alignment")
    activation_level = ability.get("activation_level")
    demand_alignment = ability.get("demand_alignment")
    if knowledge_alignment is None:
        band = "unknown"
        instruction = "Use profile, memory, and item-conditioned ability evidence because knowledge alignment is unavailable."
    elif knowledge_alignment == "high":
        band = "strong"
        if activation_level == "available":
            instruction = (
                "The learner has strong current-concept alignment and available item-level ability. "
                "Generate a fluent learner-level first attempt unless memory or careless evidence "
                "clearly supports a mistake."
            )
        else:
            instruction = (
                "The learner has strong current-concept alignment, but item-level activation is not fully available. "
                "Reflect bounded reasoning or reduced confidence without forcing an error."
            )
    elif knowledge_alignment == "medium":
        band = "favorable"
        instruction = (
            "The learner has usable knowledge. Generate a plausible learner attempt "
            "that may be correct or partially flawed according to memory and item-level ability activation."
        )
    elif knowledge_alignment == "low":
        band = "weak"
        instruction = (
            "The learner has weak access to this knowledge. Avoid teacher-like full "
            "derivations; any correct answer should have a plausible memory cue, "
            "shortcut, or lucky shallow route."
        )
    else:
        band = "mixed"
        instruction = (
            "The learner is mixed. Let memory, item demand, and ability activation shape the "
            "attempt without forcing agreement with KT."
        )
    return {
        "module": "kt_state_guidance",
        "knowledge_band": band,
        "knowledge_alignment": knowledge_alignment,
        "practice_alignment": practice_alignment,
        "ability_activation": activation_level,
        "demand_alignment": demand_alignment,
        "kt_state": ability.get("kt_state"),
        "instruction": instruction,
    }


def _kt_decision_anchor(item_conditioned_ability: dict[str, Any] | None) -> dict[str, Any]:
    ability = item_conditioned_ability or {}
    anchor = ability.get("kt_decision_anchor")
    if isinstance(anchor, dict):
        return {
            "module": "kt_decision_anchor",
            "source": "external_kt_state",
            "probability": anchor.get("probability"),
            "predicted_response": anchor.get("predicted_response"),
            "confidence_band": anchor.get("confidence_band"),
            "decision_weight": anchor.get("decision_weight"),
            "instruction": anchor.get("instruction"),
        }
    return {
        "module": "kt_decision_anchor",
        "source": "unavailable",
        "probability": None,
        "predicted_response": None,
        "confidence_band": "unavailable",
        "decision_weight": "fallback",
        "instruction": (
            "No external KT state is available; rely on profile, memory, "
            "item evidence, and non-cognitive state."
        ),
    }


def _prompt_item_conditioned_evidence(
    item_conditioned_ability: dict[str, Any],
) -> dict[str, Any]:
    evidence = item_conditioned_ability.get("evidence") or {}
    anchor = item_conditioned_ability.get("kt_decision_anchor") or {}
    return {
        "module": item_conditioned_ability.get("module"),
        "method": item_conditioned_ability.get("method"),
        "knowledge_alignment": item_conditioned_ability.get("knowledge_alignment"),
        "practice_alignment": item_conditioned_ability.get("practice_alignment"),
        "demand_alignment": item_conditioned_ability.get("demand_alignment"),
        "memory_support": item_conditioned_ability.get("memory_support"),
        "transfer_burden": item_conditioned_ability.get("transfer_burden"),
        "ability_activation": item_conditioned_ability.get("ability_activation"),
        "activation_level": item_conditioned_ability.get("activation_level"),
        "ability_expression": item_conditioned_ability.get("ability_expression"),
        "kt_state": {
            "concept": evidence.get("concept"),
            "mastery_level": evidence.get("mastery_level"),
            "confidence_band": anchor.get("confidence_band"),
        },
        "evidence_role": (
            "balanced_state_evidence_not_correctness_label_or_error_trigger"
        ),
        "evidence": {
            "item_demand": evidence.get("item_demand"),
            "practice_depth": evidence.get("practice_depth"),
            "related_memory_count": evidence.get("related_memory_count"),
            "related_memory_outcome": evidence.get("related_memory_outcome"),
            "identical_correct_count": evidence.get("identical_correct_count"),
            "knowledge_breadth": evidence.get("knowledge_breadth"),
            "challenge_adaptation": evidence.get("challenge_adaptation"),
            "cross_domain_generalization": evidence.get(
                "cross_domain_generalization"
            ),
            "transfer_fragility": evidence.get("transfer_fragility"),
            "mastery_stability": evidence.get("mastery_stability"),
        },
        "response_guidance": item_conditioned_ability.get("response_guidance"),
    }


def build_four_tier_response_record(
    action: dict[str, Any],
    four_tier: dict[str, Any],
    item_conditioned_ability: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "module": "four_tier_response",
        "generation": {
            "attempt": action.get("attempt"),
            "identified_concept": action.get("identified_concept"),
            "student_answer": action.get("student_answer"),
            "answer_confidence": action.get("answer_confidence"),
            "student_reasoning": action.get("student_reasoning"),
            "reasoning_confidence": action.get("reasoning_confidence"),
        },
        "diagnostic_scoring": {
            "answer_correct": four_tier.get("answer_correct"),
            "reasoning_correct": four_tier.get("reasoning_correct"),
            "answer_confidence": four_tier.get("answer_confidence"),
            "reasoning_confidence": four_tier.get("reasoning_confidence"),
            "diagnosis": four_tier.get("diagnosis"),
            "fully_scored": four_tier.get("fully_scored"),
            "scoring_note": four_tier.get("scoring_note"),
        },
        "conditioning": {
            "item_conditioned_ability": {
                "activation_level": (item_conditioned_ability or {}).get("activation_level"),
                "ability_expression": (item_conditioned_ability or {}).get("ability_expression"),
                "knowledge_alignment": (item_conditioned_ability or {}).get("knowledge_alignment"),
                "practice_alignment": (item_conditioned_ability or {}).get("practice_alignment"),
                "demand_alignment": (item_conditioned_ability or {}).get("demand_alignment"),
                "memory_support": (item_conditioned_ability or {}).get("memory_support"),
                "transfer_burden": (item_conditioned_ability or {}).get("transfer_burden"),
                "kt_decision_anchor": (item_conditioned_ability or {}).get("kt_decision_anchor"),
            },
        },
    }


def _compact(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 3200 else text[:3200] + "...[truncated]"


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_optional(value: Any) -> float | None:
    number = _optional_float(value)
    return round(number, 3) if number is not None else None



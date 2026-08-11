from __future__ import annotations

from typing import Any

from learner_simulator.agent4edu_prompt import build_action_prompt
from learner_simulator.irt_evidence import prompt_irt_evidence


def build_cognitive_profile_view(
    profile_context: dict[str, Any],
) -> dict[str, Any]:
    cognitive = profile_context.get("cognitive_profile") or {}
    control = cognitive.get("control_traits") or {}
    ability = profile_context.get("ability_profile") or {}
    return {
        "module": "learner_state_profile",
        "scope": "stable_traits_estimated_from_observed_history",
        "general_performance_level": control.get("overall_success_level"),
        "learning_stability": {
            "recent_success": control.get("recent_success_level"),
            "success_trend": control.get("success_trend_level"),
            "mastery_stability": control.get("mastery_stability_level"),
        },
        "ability_trait": {
            "knowledge_breadth": ability.get("knowledge_breadth"),
            "challenge_adaptation": ability.get("challenge_adaptation"),
            "profile_confidence": ability.get("profile_confidence"),
        },
        "profile_use": (
            "Use stable profile evidence for reasoning style and confidence. "
            "Do not let broad traits override strong current-item knowledge evidence."
        ),
    }


def build_response_prompt(
    question: dict[str, Any],
    short_memory: list[dict[str, Any]],
    long_memory: dict[str, Any],
    concept_options: list[str],
    proficiency: dict[str, Any] | None = None,
    historical_reflection: dict[str, Any] | None = None,
    item_conditioned_ability: dict[str, Any] | None = None,
    irt_evidence: dict[str, Any] | None = None,
    response_format: str = "four_tier",
    include_profile_evidence: bool = True,
    include_item_conditioned_evidence: bool = True,
    include_historical_reflection: bool = True,
    include_irt_evidence: bool = True,
    include_ncdm_evidence: bool = True,
) -> str:
    if response_format not in {"four_tier", "reduced_response", "answer_only"}:
        raise ValueError(f"Unsupported response format: {response_format}")

    reflection = historical_reflection or {}
    item_evidence = _prompt_item_conditioned_evidence(
        item_conditioned_ability or {},
        include_profile_evidence=include_profile_evidence,
    )
    chunks = [
        "# Response Agent #",
        "Simulate this learner's single first attempt from the evidence below.",
    ]
    if include_ncdm_evidence and proficiency:
        chunks.extend(
            [
                "",
                "# NCDM State Evidence #",
                _compact(_prompt_ncdm_evidence(proficiency)),
                (
                    "Concept mastery is knowledge-state evidence, not a sampled "
                    "answer label or a current-item correctness probability."
                ),
            ]
        )
    elif proficiency:
        chunks.extend(
            [
                "",
                "# History-based Knowledge Evidence #",
                _compact(_prompt_history_state(proficiency)),
            ]
        )
    if include_historical_reflection:
        chunks.extend(
            [
                "",
                "# Observed-History Replay Calibration #",
                _compact(_prompt_reflection(reflection)),
            ]
        )
    if include_irt_evidence:
        chunks.extend(
            [
                "",
                "# IRT Ability-Difficulty Evidence #",
                _compact(prompt_irt_evidence(irt_evidence)),
            ]
        )
    if include_item_conditioned_evidence:
        chunks.extend(
            [
                "",
                "# Item-conditioned Integration #",
                _compact(item_evidence),
            ]
        )

    constraints = [
        "- Use only evidence sections that are present; never reconstruct an ablated module.",
        "- Missing related memory is unobserved evidence, not evidence of inability.",
        "- Generate one plausible first attempt; do not deliberately insert an error.",
    ]
    if response_format in {"four_tier", "reduced_response"}:
        constraints.insert(
            2,
            "- Decide LearnerCorrect before using the reference answer to render StudentAnswer.",
        )
    if include_ncdm_evidence:
        constraints.append(
            "- Treat NCDM concept mastery as the primary knowledge-state evidence, while allowing concrete memory and item demand to resolve boundary cases."
        )
    if include_irt_evidence:
        constraints.append(
            "- IRT calibrates relative challenge and confidence; it does not directly determine correctness."
        )
    if include_historical_reflection:
        constraints.append(
            "- Replay calibration uses observed history only and must never be updated from target labels inside response generation."
        )
    if include_item_conditioned_evidence:
        constraints.append(
            (
                "- Item-conditioned integration adjusts reasoning depth and confidence without duplicating or replacing NCDM evidence."
                if include_ncdm_evidence
                else "- Item-conditioned integration adjusts reasoning depth and confidence without inventing unavailable model evidence."
            )
        )
    chunks.extend(["", "# Decision Constraints #", *constraints, ""])

    # Memory, item content, reference answer, concept options, and the output
    # contract are rendered once by the shared action prompt.
    base = build_action_prompt(
        question=question,
        short_memory=short_memory,
        long_memory=long_memory,
        concept_options=concept_options,
        proficiency=None,
        behavior_factors=None,
        response_format=response_format,
        cognitive_strategy=None,
        tendency_calibration=None,
        include_profile_evidence=include_profile_evidence,
    )
    return "\n".join(chunks) + base


def _prompt_ncdm_evidence(proficiency: dict[str, Any]) -> dict[str, Any]:
    return {
        "module": "ncdm_state_evidence",
        "concept": proficiency.get("concept"),
        "concept_mastery": proficiency.get("concept_mastery_value"),
        "concept_mastery_level": proficiency.get("level"),
    }


def _prompt_history_state(proficiency: dict[str, Any]) -> dict[str, Any]:
    return {
        "module": "history_based_knowledge_evidence",
        "concept": proficiency.get("concept"),
        "mastery": proficiency.get("value"),
        "level": proficiency.get("level"),
        "source": proficiency.get("source"),
    }


def _prompt_reflection(reflection: dict[str, Any]) -> dict[str, Any]:
    policy = reflection.get("adaptive_policy") or {}
    return {
        "available": reflection.get("available"),
        "calibration_window": reflection.get("calibration_window"),
        "actual_correct_rate": reflection.get("actual_correct_rate"),
        "predicted_correct_rate": reflection.get("predicted_correct_rate"),
        "bias_direction": reflection.get("bias_direction"),
        "error_pattern": reflection.get("error_pattern"),
        "evidence_confidence": reflection.get("evidence_confidence"),
        "state_model_trust": policy.get("state_model_trust"),
        "response_bias": policy.get("response_bias"),
        "instruction": policy.get("instruction"),
    }


def _prompt_item_conditioned_evidence(
    item_conditioned_ability: dict[str, Any],
    include_profile_evidence: bool,
) -> dict[str, Any]:
    evidence = item_conditioned_ability.get("evidence") or {}
    result = {
        "module": item_conditioned_ability.get("module"),
        "knowledge_alignment": item_conditioned_ability.get("knowledge_alignment"),
        "practice_alignment": item_conditioned_ability.get("practice_alignment"),
        "demand_alignment": item_conditioned_ability.get("demand_alignment"),
        "memory_support": item_conditioned_ability.get("memory_support"),
        "ability_activation": item_conditioned_ability.get("ability_activation"),
        "item_demand": evidence.get("item_demand"),
        "related_memory_count": evidence.get("related_memory_count"),
        "related_memory_outcome": evidence.get("related_memory_outcome"),
        "response_guidance": item_conditioned_ability.get("response_guidance"),
    }
    if include_profile_evidence:
        result["profile_support"] = {
            "knowledge_breadth": evidence.get("knowledge_breadth"),
            "challenge_adaptation": evidence.get("challenge_adaptation"),
            "mastery_stability": evidence.get("mastery_stability"),
        }
    return result


def build_four_tier_response_record(
    action: dict[str, Any],
    four_tier: dict[str, Any],
    item_conditioned_ability: dict[str, Any] | None = None,
    irt_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "module": "four_tier_response",
        "generation": {
            "attempt": action.get("attempt"),
            "identified_concept": action.get("identified_concept"),
            "learner_correct": action.get("learner_correct"),
            "response_decision_source": action.get("response_decision_source"),
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
            "item_conditioned_integration": {
                "ability_activation": (item_conditioned_ability or {}).get(
                    "ability_activation"
                ),
                "knowledge_alignment": (item_conditioned_ability or {}).get(
                    "knowledge_alignment"
                ),
                "practice_alignment": (item_conditioned_ability or {}).get(
                    "practice_alignment"
                ),
                "demand_alignment": (item_conditioned_ability or {}).get(
                    "demand_alignment"
                ),
                "memory_support": (item_conditioned_ability or {}).get(
                    "memory_support"
                ),
            },
            "irt_ability_difficulty_evidence": prompt_irt_evidence(irt_evidence),
        },
    }


def _compact(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 2600 else text[:2600] + "...[truncated]"

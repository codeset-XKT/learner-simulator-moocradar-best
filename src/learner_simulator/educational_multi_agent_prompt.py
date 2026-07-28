from __future__ import annotations

from typing import Any

from learner_simulator.agent4edu_prompt import build_action_prompt
from learner_simulator.irt_evidence import prompt_irt_evidence
from learner_simulator.learning_tool_state import prompt_learning_tool_state


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
    historical_reflection: dict[str, Any] | None = None,
    item_conditioned_ability: dict[str, Any] | None = None,
    irt_evidence: dict[str, Any] | None = None,
    learning_tool_state: dict[str, Any] | None = None,
    response_format: str = "four_tier",
) -> str:
    prompt_item_evidence = _prompt_item_conditioned_evidence(
        item_conditioned_ability or {}
    )
    prompt_irt = prompt_irt_evidence(irt_evidence)
    prompt_tool_state = prompt_learning_tool_state(learning_tool_state)
    kt_proficiency = _proficiency_from_item_evidence(prompt_item_evidence)
    adaptive_policy = (historical_reflection or {}).get("adaptive_policy") or {}
    base = build_action_prompt(
        question=question,
        short_memory=short_memory,
        long_memory=long_memory,
        concept_options=concept_options,
        proficiency=kt_proficiency,
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
            "The KT proficiency state is the current knowledge-state estimate inferred from historical interactions, "
            "playing the same role as an external CDM/KT proficiency module in learner simulation. "
            "Treat it as the primary evidence for whether the learner is ready to answer this knowledge component, "
            "then use profile, memory, item demand, and non-cognitive evidence to shape how that readiness is expressed. "
            "It is not a sampled response label and must not be copied as a correctness label.",
            "",
            "# Learner Profile Evidence #",
            _compact(cognitive_profile),
            "",
            "# Historical Reflective Calibration #",
            _compact(historical_reflection or {}),
            "",
            "# Historical Replay Policy Adaptation #",
            _compact(adaptive_policy),
            "",
            "# KT Proficiency State #",
            _compact(kt_proficiency),
            "",
            "# IRT Ability-Difficulty Evidence #",
            _compact(prompt_irt or {}),
            "",
            "# Learning Tool State #",
            _compact(prompt_tool_state),
            "",
            "# Agent-style Internal Task Protocol #",
            _compact(_agent_style_task_protocol(prompt_tool_state)),
            "",
            "# Item-conditioned Evidence #",
            _compact(prompt_item_evidence),
            "",
            "# KT State Guidance #",
            _compact(_kt_state_guidance(prompt_item_evidence)),
            "",
            "# Process-to-Response Constraints #",
            "- Follow the internal task protocol in order: concept perception, tool-state reading, four-tier planning, then final learner response.",
            "- Treat the Learning Tool State as Agent-style tool evidence: NCDM is the primary knowledge-state tool; IRT is the secondary ability-difficulty tool; memory and profile shape expression.",
            "- Use the KT proficiency state before answer generation. Do not perform a post-generation correction that forces agreement with KT; the final answer must still be generated by the simulated learner.",
            "- Use IRT Ability-Difficulty Evidence as learner-level ability versus item difficulty evidence only. It can calibrate effort, confidence, and relative challenge; it is not concept mastery and is not a correctness label.",
            "- When IRT indicates the item is below learner ability, use it to support fluency and confidence when KT proficiency is at least developing.",
            "- When IRT indicates the item is above learner ability, lower confidence or simplify reasoning, but do not overturn strong current-concept proficiency.",
            "- Use historical reflective calibration only as observed-history replay evidence. It is computed before target simulation and must not be treated as feedback from the current target item.",
            "- Apply Historical Replay Policy Adaptation as this learner's stable target-stage simulation policy. It adapts response tendency and confidence from observed replay errors, but it must not override the generated answer after the fact.",
            "- If the policy says preserve_plausible_success, avoid unnecessary pessimism when NCDM proficiency and memory support success.",
            "- If the policy says guard_against_overconfidence, require stronger current evidence before producing confident correct attempts.",
            "- Never update this policy with target labels; target labels are unavailable during simulation.",
            "- If ability_activation is available, the attempt may be direct, concise, and learner-level.",
            "- If ability_activation is partial, include bounded reasoning depth and moderate confidence.",
            "- If ability_activation is limited, use simpler reasoning or lower confidence, but do not assume the final answer must be wrong.",
            "- Use four_tier_generation_policy to decide answer tendency, confidence tendency, and reasoning depth before producing StudentAnswer.",
            "- If response_planning says preserve_ncdm_supported_success, preserve a likely successful learner-level first attempt unless the visible item clearly requires unsupported knowledge.",
            "- If response_planning says success_plausible_with_moderate_confidence or cue_based_success_possible, allow a correct first attempt with modest confidence and short reasoning.",
            "- If response_planning says unsupported_attempt_with_low_confidence, limit reasoning depth and confidence without intentionally inserting an error.",
            "- Strong KT proficiency on a low- or medium-demand item supports a concise successful first attempt at learner level unless there is clear contradictory evidence.",
            "- Weak KT proficiency should limit reasoning depth and confidence; related memory or easy visible cues can still produce a correct answer.",
            "- Strong practice alignment supports familiar learner-level solution patterns.",
            "- Missing related memory is unobserved evidence, not evidence of inability.",
            "- Negative related memory may lower confidence or simplify reasoning, but it should not force an incorrect answer.",
            "- Careless or high-load evidence may affect verification depth, but avoid inventing mistakes unsupported by the item and learner evidence.",
            "- Weak knowledge evidence supports short cue-based reasoning rather than a full derivation.",
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
                "The learner has strong current-concept proficiency and available item-level ability. "
                "For routine items, preserve a concise successful first attempt unless the visible item "
                "or learner evidence gives a clear educational reason for a slip."
            )
        else:
            instruction = (
                "The learner has strong current-concept proficiency, but item-level activation is not fully available. "
                "Reflect bounded reasoning or reduced confidence without turning uncertainty into an automatic error."
            )
    elif knowledge_alignment == "medium":
        band = "favorable"
        instruction = (
            "The learner has usable current knowledge. On familiar or low-demand items, a correct first attempt "
            "remains plausible; on demanding or weak-memory items, partial flaws remain plausible."
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
        "irt_ability_difficulty": (
            (ability.get("evidence") or {}).get("irt_ability_difficulty") or {}
        ),
        "learning_tool_state": ability.get("learning_tool_state"),
        "kt_state": ability.get("kt_state"),
        "kt_proficiency_state": ability.get("kt_proficiency_state"),
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
            "evidence_priority": anchor.get("evidence_priority"),
            "instruction": anchor.get("instruction"),
        }
    return {
        "module": "kt_decision_anchor",
        "source": "unavailable",
        "probability": None,
        "predicted_response": None,
        "confidence_band": "unavailable",
        "evidence_priority": "fallback",
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
    kt_state = item_conditioned_ability.get("kt_proficiency_state")
    if not isinstance(kt_state, dict):
        kt_state = (evidence.get("kt_proficiency_state") or {})
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
        "irt_relative_challenge": item_conditioned_ability.get("irt_relative_challenge"),
        "irt_boundary_band": item_conditioned_ability.get("irt_boundary_band"),
        "learning_tool_state": item_conditioned_ability.get("learning_tool_state"),
        "kt_state": {
            "concept": kt_state.get("concept") or evidence.get("concept"),
            "source": kt_state.get("source") or evidence.get("kt_source"),
            "mastery_value": kt_state.get("value") or anchor.get("probability"),
            "mastery_level": kt_state.get("level") or evidence.get("mastery_level"),
            "confidence_band": anchor.get("confidence_band"),
            "state_role": kt_state.get("state_role"),
        },
        "kt_proficiency_state": {
            "concept": kt_state.get("concept") or evidence.get("concept"),
            "source": kt_state.get("source") or evidence.get("kt_source"),
            "mastery_value": kt_state.get("value") or anchor.get("probability"),
            "mastery_level": kt_state.get("level") or evidence.get("mastery_level"),
            "confidence_band": anchor.get("confidence_band"),
            "state_role": kt_state.get("state_role"),
        },
        "evidence_role": (
            "balanced_state_evidence_not_correctness_label_or_error_trigger"
        ),
        "evidence": {
            "item_demand": evidence.get("item_demand"),
            "irt_ability_difficulty": evidence.get("irt_ability_difficulty"),
            "learning_tool_state": evidence.get("learning_tool_state"),
            "kt_source": evidence.get("kt_source"),
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


def _agent_style_task_protocol(
    learning_tool_state: dict[str, Any] | None,
) -> dict[str, Any]:
    state = learning_tool_state or {}
    return {
        "module": "agent_style_four_tier_action_protocol",
        "difference_from_agent4edu": (
            "Agent4Edu Task4 predicts Yes/No correctness; this protocol generates "
            "answer, answer confidence, reasoning, and reasoning confidence."
        ),
        "task1_concept_perception": (
            "Identify the knowledge concept tested by the exercise using the concept options."
        ),
        "task2_tool_state_reading": {
            "objective": "Read NCDM and IRT tools before action generation.",
            "knowledge_tool": state.get("knowledge_tool"),
            "ability_difficulty_tool": state.get("ability_difficulty_tool"),
            "related_memory_tool": state.get("related_memory_tool"),
        },
        "task3_four_tier_planning": {
            "objective": (
                "Plan answer tendency, confidence tendency, and reasoning depth "
                "from tool evidence without producing a correctness label."
            ),
            "joint_readiness_state": state.get("joint_readiness_state"),
            "response_planning": state.get("response_planning"),
            "four_tier_generation_policy": state.get("four_tier_generation_policy"),
        },
        "task4_four_tier_response_generation": (
            "Generate StudentAnswer, AnswerConfidence, StudentReasoning, and "
            "ReasoningConfidence as the learner's first attempt."
        ),
    }


def _proficiency_from_item_evidence(
    prompt_item_evidence: dict[str, Any],
) -> dict[str, Any] | None:
    kt_state = prompt_item_evidence.get("kt_state") or {}
    value = _optional_float(kt_state.get("mastery_value"))
    concept = kt_state.get("concept")
    if value is None or concept in {None, ""}:
        return None
    return {
        "concept": concept,
        "value": value,
        "level": kt_state.get("mastery_level") or (
            "high" if value >= 0.7 else "medium" if value >= 0.4 else "low"
        ),
        "source": kt_state.get("source"),
        "state_role": kt_state.get("state_role"),
    }


def build_four_tier_response_record(
    action: dict[str, Any],
    four_tier: dict[str, Any],
    item_conditioned_ability: dict[str, Any] | None = None,
    irt_evidence: dict[str, Any] | None = None,
    learning_tool_state: dict[str, Any] | None = None,
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
                "irt_relative_challenge": (item_conditioned_ability or {}).get("irt_relative_challenge"),
                "learning_tool_state": (item_conditioned_ability or {}).get("learning_tool_state"),
            },
            "irt_ability_difficulty_evidence": prompt_irt_evidence(irt_evidence),
            "learning_tool_state": prompt_learning_tool_state(learning_tool_state),
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



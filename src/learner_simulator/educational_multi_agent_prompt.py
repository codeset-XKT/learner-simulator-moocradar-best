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
        "ability_trait": {
            "knowledge_breadth": ability.get("knowledge_breadth"),
            "challenge_adaptation": ability.get("challenge_adaptation"),
            "profile_confidence": ability.get("profile_confidence"),
        },
        "profile_use": (
            "Use stable profile evidence to calibrate confidence and reasoning style. "
            "Do not use broad or weakly observed traits to overturn current KT/CDM readiness."
        ),
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
    kt_anchor = _kt_decision_anchor(prompt_item_evidence)
    adaptive_policy = (historical_reflection or {}).get("adaptive_policy") or {}
    response_tendency_state = _build_pre_response_tendency_state(
        prompt_item_evidence=prompt_item_evidence,
        learning_tool_state=prompt_tool_state,
        historical_reflection=historical_reflection or {},
    )
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
            "It is qualitative evidence. Use it to shape response style, reasoning depth, confidence, and likely correctness without inventing unsupported mistakes.",
            "Output LearnerCorrect only as the Task4 learner-simulation decision; do not add extra correctness commentary.",
            "The KT/CDM response state is the current-item readiness estimate inferred from historical interactions, "
            "playing the same role as an external CDM/KT proficiency module in learner simulation. "
            "When current-item response probability is available, treat it as the primary evidence for whether the learner is ready to answer this item, "
            "then use concept mastery, profile, memory, item demand, and non-cognitive evidence to shape how that readiness is expressed. "
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
            "# KT/CDM Response State #",
            _compact(kt_proficiency),
            "",
            "# KT Decision Anchor #",
            _compact(kt_anchor),
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
            "# Pre-response Tendency State #",
            _compact(response_tendency_state),
            "",
            "# Process-to-Response Constraints #",
            "- Follow the internal task protocol in order: concept perception, tool-state reading, four-tier planning, then final learner response.",
            "- First read the KT Decision Anchor as the primary symmetric prior for LearnerCorrect.",
            "- If the KT anchor is strong or moderate, LearnerCorrect should normally follow its direction.",
            "- Deviate from a strong/moderate KT anchor only when concrete same-concept memory, item demand, or profile evidence clearly contradicts it.",
            "- If the KT anchor is boundary or unavailable, decide from related memory, item demand, learner profile, and visible item evidence.",
            "- Before solving, privately commit to the Pre-response Tendency State. It guides answer tendency and confidence, not post-hoc correction.",
            "- If the tendency is correct_leaning, produce a learner response consistent with a likely correct first attempt.",
            "- If the tendency is balanced, let current item demand, memory, and visible cues decide the attempt without systematic pessimism or optimism.",
            "- If the tendency is incorrect_leaning, produce a limited learner response consistent with a likely incorrect first attempt.",
            "- Treat the Learning Tool State as Agent-style tool evidence: current-item KT/CDM response probability is the primary readiness tool; concept mastery supports it; IRT is secondary ability-difficulty context; memory and profile shape expression.",
            "- Use IRT Ability-Difficulty Evidence only to calibrate effort and confidence. It should not overturn a strong/moderate KT anchor by itself.",
            "- Use historical reflective calibration only as observed-history replay evidence. It is computed before target simulation and must not be treated as feedback from the current target item.",
            "- Apply Historical Replay Policy Adaptation as a calibration signal for response tendency and confidence.",
            "- If the policy says preserve_plausible_success, apply it only to boundary anchors or weakly contradictory evidence.",
            "- If the policy says guard_against_overconfidence, require stronger item-specific support before deviating from an incorrect or boundary anchor.",
            "- Never update this policy with target labels; target labels are unavailable during simulation.",
            "- If ability_activation is available, the attempt may be direct and concise, but LearnerCorrect is still anchored by KT.",
            "- If ability_activation is partial, include bounded reasoning depth and moderate confidence.",
            "- If ability_activation is limited, simplify reasoning or lower confidence, while keeping LearnerCorrect aligned with the KT anchor unless evidence conflicts.",
            "- Use four_tier_generation_policy to decide answer tendency, confidence tendency, and reasoning depth before producing StudentAnswer.",
            "- If response_planning is kt_strong_correct_anchor or kt_lean_correct_anchor, LearnerCorrect should lean Yes unless concrete contradictory evidence exists.",
            "- If response_planning is kt_strong_incorrect_anchor or kt_lean_incorrect_anchor, LearnerCorrect should lean No unless concrete supportive evidence exists.",
            "- If response_planning is kt_boundary_anchor, decide from memory, item demand, and profile without forcing either side.",
            "- Strong current-item KT/CDM readiness supports a correct first attempt; weak KT/CDM readiness supports an incorrect first attempt. Both are anchors, not guarantees.",
            "- Missing related memory is unobserved evidence, not evidence of inability.",
            "- Negative related memory may lower confidence or shift boundary decisions, but it cannot override a strong KT anchor by itself.",
            "- Broad profile traits should not override current-item KT/CDM evidence unless item-specific evidence is clearly contradictory.",
            "",
            "# Four-tier Output Requirement #",
            "The response must include LearnerCorrect, answer, confidence, reasoning, and reasoning confidence so that Task4 correctness and four-tier diagnostics can be assessed separately.",
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
                "Use the KT anchor as the primary decision signal; profile and item evidence may "
                "adjust confidence and reasoning style."
            )
        else:
            instruction = (
                "The learner has strong current-concept proficiency, but item-level activation is not fully available. "
                "Reflect bounded reasoning or reduced confidence without changing the KT anchor direction by default."
            )
    elif knowledge_alignment == "medium":
        band = "favorable"
        instruction = (
            "The learner has usable current knowledge. Treat the item as a lean or boundary decision "
            "depending on the KT probability, memory, and item demand."
        )
    elif knowledge_alignment == "low":
        band = "weak"
        instruction = (
            "The learner has limited current evidence for this knowledge. Treat this as an "
            "incorrect-leaning or boundary decision unless concrete related memory supports success."
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


def _build_pre_response_tendency_state(
    prompt_item_evidence: dict[str, Any],
    learning_tool_state: dict[str, Any],
    historical_reflection: dict[str, Any],
) -> dict[str, Any]:
    kt_state = prompt_item_evidence.get("kt_proficiency_state") or {}
    evidence = prompt_item_evidence.get("evidence") or {}
    mastery = _optional_float(kt_state.get("mastery_value"))
    mastery_level = str(kt_state.get("mastery_level") or "unknown")
    memory_outcome = str(evidence.get("related_memory_outcome") or "not_observed")
    item_demand = str(evidence.get("item_demand") or "unknown")
    activation = str(prompt_item_evidence.get("activation_level") or "unknown")
    response_planning = str(learning_tool_state.get("response_planning") or "unknown")
    readiness = str(learning_tool_state.get("joint_readiness_state") or "unknown")
    policy = historical_reflection.get("adaptive_policy") or {}
    replay_bias = str(policy.get("response_bias") or "unknown")
    replay_confidence = str(historical_reflection.get("evidence_confidence") or "low")

    supportive = []
    cautious = []
    if mastery is not None:
        if mastery >= 0.70:
            supportive.append("strong_current_knowledge_state")
        elif mastery >= 0.50:
            supportive.append("usable_current_knowledge_state")
        elif mastery < 0.20:
            cautious.append("very_low_current_knowledge_state")
    if memory_outcome == "mostly_successful":
        supportive.append("related_memory_success")
    elif memory_outcome == "mostly_incorrect":
        cautious.append("related_memory_errors")
    if item_demand in {"low", "medium"}:
        supportive.append("accessible_item_demand")
    elif item_demand == "high":
        cautious.append("high_item_demand")
    if activation == "available":
        supportive.append("available_item_level_activation")
    elif activation == "limited":
        cautious.append("limited_item_level_activation")
    if response_planning in {
        "kt_strong_correct_anchor",
        "kt_lean_correct_anchor",
    }:
        supportive.append(f"tool_plan_{response_planning}")
    elif response_planning in {
        "kt_strong_incorrect_anchor",
        "kt_lean_incorrect_anchor",
    }:
        cautious.append("tool_plan_low_support")
    if replay_confidence != "low" and replay_bias == "preserve_plausible_success":
        supportive.append("historical_replay_underestimated_success")
    elif replay_confidence != "low" and replay_bias == "guard_against_overconfidence":
        cautious.append("historical_replay_overestimated_success")

    if (
        "strong_current_knowledge_state" in supportive
        and "high_item_demand" not in cautious
    ) or (
        "usable_current_knowledge_state" in supportive
        and "accessible_item_demand" in supportive
        and "related_memory_errors" not in cautious
    ) or (
        "historical_replay_underestimated_success" in supportive
        and item_demand != "high"
        and mastery is not None
        and mastery >= 0.35
    ):
        tendency = "correct_leaning"
        instruction = (
            "Start from the KT correct-leaning anchor. Keep reasoning concise and "
            "student-level; reduce confidence only if memory or item demand is unstable."
        )
    elif (
        "very_low_current_knowledge_state" in cautious
        and "high_item_demand" in cautious
        and "related_memory_errors" in cautious
    ) or (
        "historical_replay_overestimated_success" in cautious
        and "available_item_level_activation" not in supportive
    ):
        tendency = "incorrect_leaning"
        instruction = (
            "Start from the KT incorrect-leaning anchor. Use limited reasoning and "
            "lower confidence unless concrete related evidence supports success."
        )
    else:
        tendency = "balanced"
        instruction = (
            "Use the item content, memory, and tool state without systematic optimism or "
            "pessimism. A brief correct or partially flawed attempt may both be plausible."
        )

    return {
        "module": "pre_response_tendency_state",
        "source": "history_tool_item_evidence_before_answer_generation",
        "target_label_access": False,
        "tendency": tendency,
        "mastery_value": _round_optional(mastery),
        "mastery_level": mastery_level,
        "item_demand": item_demand,
        "memory_outcome": memory_outcome,
        "activation_level": activation,
        "learning_tool_readiness": readiness,
        "learning_tool_response_planning": response_planning,
        "historical_replay_bias": replay_bias,
        "supportive_evidence": supportive,
        "cautious_evidence": cautious,
        "instruction": instruction,
        "boundary": (
            "This state guides the learner's first-attempt tendency only. It must not be "
            "copied as a correctness label and must not override the generated answer after scoring."
        ),
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
        "ability_activation": item_conditioned_ability.get("ability_activation"),
        "activation_level": item_conditioned_ability.get("activation_level"),
        "irt_relative_challenge": item_conditioned_ability.get("irt_relative_challenge"),
        "irt_boundary_band": item_conditioned_ability.get("irt_boundary_band"),
        "learning_tool_state": item_conditioned_ability.get("learning_tool_state"),
        "kt_state": {
            "concept": kt_state.get("concept") or evidence.get("concept"),
            "source": kt_state.get("source") or evidence.get("kt_source"),
            "mastery_value": kt_state.get("value") or anchor.get("probability"),
            "item_response_probability": kt_state.get("response_probability")
            or evidence.get("current_item_response_probability")
            or anchor.get("probability"),
            "item_response_probability_source": kt_state.get(
                "response_probability_source"
            )
            or evidence.get("current_item_response_source"),
            "concept_mastery_value": kt_state.get("concept_mastery_value")
            or evidence.get("concept_mastery_value"),
            "concept_mastery_source": kt_state.get("concept_mastery_source"),
            "mastery_level": kt_state.get("level") or evidence.get("mastery_level"),
            "confidence_band": anchor.get("confidence_band"),
            "state_role": kt_state.get("state_role"),
        },
        "kt_proficiency_state": {
            "concept": kt_state.get("concept") or evidence.get("concept"),
            "source": kt_state.get("source") or evidence.get("kt_source"),
            "mastery_value": kt_state.get("value") or anchor.get("probability"),
            "item_response_probability": kt_state.get("response_probability")
            or evidence.get("current_item_response_probability")
            or anchor.get("probability"),
            "item_response_probability_source": kt_state.get(
                "response_probability_source"
            )
            or evidence.get("current_item_response_source"),
            "concept_mastery_value": kt_state.get("concept_mastery_value")
            or evidence.get("concept_mastery_value"),
            "concept_mastery_source": kt_state.get("concept_mastery_source"),
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
            "current_item_response_probability": evidence.get(
                "current_item_response_probability"
            ),
            "current_item_response_source": evidence.get(
                "current_item_response_source"
            ),
            "concept_mastery_value": evidence.get("concept_mastery_value"),
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
            "Agent4Edu Task4 predicts Yes/No correctness; this protocol predicts "
            "LearnerCorrect and also renders four-tier answer, confidence, reasoning, "
            "and reasoning confidence as behavioral diagnostics."
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
                "from tool evidence before rendering the submitted answer."
            ),
            "joint_readiness_state": state.get("joint_readiness_state"),
            "response_planning": state.get("response_planning"),
            "four_tier_generation_policy": state.get("four_tier_generation_policy"),
        },
        "task4_four_tier_response_generation": (
            "Generate LearnerCorrect, StudentAnswer, AnswerConfidence, "
            "StudentReasoning, and ReasoningConfidence as the learner's first attempt."
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
        "response_probability": kt_state.get("item_response_probability") or value,
        "response_probability_source": kt_state.get("item_response_probability_source"),
        "concept_mastery_value": kt_state.get("concept_mastery_value"),
        "concept_mastery_source": kt_state.get("concept_mastery_source"),
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
            "learner_correct": action.get("learner_correct"),
            "task4_decision_source": action.get("task4_decision_source"),
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
                "knowledge_alignment": (item_conditioned_ability or {}).get("knowledge_alignment"),
                "practice_alignment": (item_conditioned_ability or {}).get("practice_alignment"),
                "demand_alignment": (item_conditioned_ability or {}).get("demand_alignment"),
                "memory_support": (item_conditioned_ability or {}).get("memory_support"),
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



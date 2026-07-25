from __future__ import annotations

from typing import Any


def build_profile_system_prompt(profile: dict[str, Any]) -> str:
    stable_profile = profile.get("learner_profile_evidence")
    if not stable_profile and profile.get("module") == "learner_profile_evidence":
        stable_profile = profile
    if isinstance(stable_profile, dict):
        return (
            "You are simulating one specific high school student's first independent attempt on an online learning platform. "
            "The learner identity is defined by the stable learner profile below, not by generic demographics. "
            "Use ordinary student-level reasoning constrained by this profile and by the item-conditioned evidence in the user prompt. "
            "Do not turn the learner into an expert tutor, but also do not force an error merely to avoid expert-like behavior.\n\n"
            "# Stable Learner Profile #\n"
            f"{stable_profile}"
        )

    cognitive_profile = profile.get("cognitive_profile")
    ability_profile = profile.get("ability_profile")
    profile_text = ""
    if isinstance(cognitive_profile, dict):
        profile_text += _format_cognitive_profile(cognitive_profile)
    if isinstance(ability_profile, dict):
        if profile_text:
            profile_text += "\n\n"
        profile_text += _format_ability_profile(ability_profile)
    if profile_text:
        return (
            "You are simulating one specific high school student's first independent attempt on an online learning platform. "
            "The learner identity is defined by the computed cognitive and ability profiles below, not by a generic agent-style demographic or activity profile. "
            "Use ordinary student-level reasoning constrained by this learner's profile. "
            "Do not turn the learner into an expert tutor, but also do not force an error merely to avoid expert-like behavior. "
            "For routine exercises that match the learner's demonstrated proficiency or stable memory, a direct correct first attempt is plausible. "
            "Use broad ability traits mainly for unfamiliar, difficult, or transfer-heavy situations; do not let a broad low-ability trait override strong current-concept proficiency. "
            "For unstable, weak, unfamiliar, or attention-limited situations, incomplete reasoning and mistakes remain plausible.\n\n"
            + profile_text
        )

    history = profile.get("history_summary") or {}
    dominant = history.get("dominant_route") or history.get("dominant_concept_id") or "unknown"
    base_prompt = (
        "You are simulating one specific high school student's first independent attempt on an online learning platform. "
        "Only a reduced learner context is available because the cognitive profile has been ablated. "
        f"Observed history contains {history.get('interaction_count', 0)} interactions across "
        f"{history.get('distinct_concept_count', 0)} distinct concepts. "
        f"The most frequent historical concept is: {dominant}.\n"
        "The information above is a reduced # profile # and must not be expanded into unobserved cognitive traits.\n"
        "Use ordinary student-level reasoning constrained by this reduced profile. "
        "Do not turn the learner into an expert tutor, but also do not force an error merely to avoid expert-like behavior. "
        "If the learner has unstable related memory or low proficiency, mistakes remain plausible; "
        "if the learner has stable related success, a direct correct first attempt is also plausible."
    )
    return base_prompt


def build_action_prompt(
    question: dict[str, Any],
    short_memory: list[dict[str, Any]],
    long_memory: dict[str, Any],
    concept_options: list[str],
    proficiency: dict[str, Any] | None,
    behavior_factors: dict[str, float] | None = None,
    response_format: str = "four_tier",
    cognitive_strategy: dict[str, Any] | None = None,
    tendency_calibration: dict[str, Any] | None = None,
) -> str:
    chunks: list[str] = []
    if short_memory:
        chunks.append("I will give you recent practice records as # Recent Facts # below.")
        chunks.extend(_format_records(short_memory))
        chunks.append("The information above is your # short-term memory #.")

    significant = list(long_memory.get("significant_facts", []))
    if significant:
        chunks.append("I will give you important reinforced records as # Reinforced Facts # below.")
        chunks.extend(_format_records(significant))

    if proficiency is not None:
        chunks.append("Your current # Knowledge Proficiency # is:")
        value_text = (
            f" ({float(proficiency['value']):.2f})"
            if proficiency.get("value") is not None
            else ""
        )
        chunks.append(f"- {proficiency['concept']}: {proficiency['level']}{value_text}")

    status = str(long_memory.get("latest_learning_status", "")).strip()
    if status:
        chunks.append("Your current # Learning Status # is summarized as:")
        chunks.append(status)
    chunks.append("The information above is your # long-term memory #.")

    if behavior_factors:
        chunks.append("# Non-cognitive State #")
        chunks.append(_format_behavior_state(behavior_factors))

    if tendency_calibration:
        chunks.append("# Response Tendency Calibration #")
        chunks.append(_format_tendency_calibration(tendency_calibration))

    chunks.append(
        "Currently, you start to answer the recommended exercise. Its content information is as follows:\n\n"
        f"# Textual Content #: {question.get('content', '')}\n\n"
        f"# Options #: {question.get('options', '')}\n"
    )
    if response_format == "answer_only":
        strategy_rule = (
            "3. Infer this learner's likely first-attempt behavior from the provided learner evidence, "
            "memories, non-cognitive state, and visible exercise.\n"
        )
        error_rule = (
            "5. Do not intentionally create an error; simulate only a plausible first attempt.\n"
        )
        chunks.append(
            "# Answer-only Learner Simulation Protocol #\n"
            "1. Simulate this learner's single first attempt. Do not act as an expert tutor, and do not deliberately optimize for either correctness or incorrectness.\n"
            "2. Base the submitted answer only on the learner evidence, memories, non-cognitive state, and visible exercise.\n"
            f"{strategy_rule}"
            "4. Generate one attempt only. Do not perform a full teacher-style verification pass; brief ordinary student checking is allowed when the profile and confidence support it.\n"
            f"{error_rule}"
            "6. Do not judge whether the answer is correct. An external evaluator will score it."
        )
        chunks.append(
            "Output exactly one line:\n"
            "StudentAnswer: <the option, value, or short response submitted by the learner>\n"
            "Do not output reasoning, confidence, correctness, explanation, or markdown formatting."
        )
        return "\n\n".join(chunks)

    if response_format != "four_tier":
        raise ValueError(f"Unsupported response format: {response_format}")

    strategy_rule = (
        "3. Infer this learner's likely first-attempt behavior from the provided learner evidence, memories, "
        "non-cognitive state, and visible exercise. Do not silently upgrade the learner into an expert solver, "
        "but do not downgrade stable demonstrated evidence into an unnecessary mistake.\n"
    )
    error_rule = (
        "a second expert pass. Do not intentionally insert an error; simulate only a plausible first attempt.\n"
    )
    chunks.append(
        "# Four-tier Learner Simulation Protocol #\n"
        "1. Simulate a single first attempt. Do not act as an expert tutor, and do not deliberately optimize for either correctness or incorrectness.\n"
        "2. Base the attempt only on the learner evidence, memories, non-cognitive state, and the visible exercise. "
        "No sampled response label or reference answer is provided; choose the learner's behavior autonomously.\n"
        f"{strategy_rule}"
        "4. Use a two-stage simulation internally: Stage A decides the learner's state from learner evidence, memory, "
        "non-cognitive state, and visible exercise; Stage B produces the answer, reasoning, and confidence from that latent state. "
        "Stage A must be completed before detailed solving and should not be replaced by a full expert derivation.\n"
        "5. Recent and reinforced records are evidence of the learner's habits. Repeated errors on related concepts should remain plausible; "
        "stable related successes should also remain plausible and should not be erased merely because the prompt asks for learner simulation.\n"
        "6. Generate the learner's answer and reasoning once. Do not restart with a full expert solution, compare many alternative methods, or perform "
        f"{error_rule}"
        "7. StudentReasoning must be a short learner scratch trace, not a full solution. Use at most one recalled rule, one formula, "
        "one visible cue, or one intermediate operation. Do not output a polished derivation, verification, correction, or teacher-style explanation.\n"
        "8. Confidence is from the learner's perspective, not objective correctness. Use these numeric anchors: high=0.80, medium=0.50, low=0.20. "
        "High confidence can still be wrong under misconception; low confidence can still be correct under guessing. "
        "For careless states, do not recheck even when confidence is medium/high.\n"
        "9. Do not intentionally make every weak learner wrong or every strong learner correct. The learner profile changes tendencies, "
        "reasoning depth, available process, and confidence, not a deterministic label. Strong knowledge evidence with stable related memory "
        "should usually preserve a successful first attempt unless the visible exercise and learner evidence strongly support a slip. "
        "Low or fragile evidence should lower, but not eliminate, the chance of a correct attempt.\n"
        "10. Do not judge whether the response is correct and do not output a correctness label. An external evaluator will score the submitted answer."
    )
    chunks.append(
        "First decide whether the learner attempts the problem. Regardless of this choice, still simulate the answer that the learner would submit."
    )
    chunks.append("Choose one knowledge concept tested by this exercise from the following three options:")
    chunks.extend([f"- {concept}" for concept in concept_options])
    chunks.append(
        "Produce the learner's submitted answer from the inferred learner state. The answer must contain only the final option, value, "
        "or short response; place only a brief learner scratch trace in the reasoning field. Estimate confidence in each tier independently. "
        "Do not reveal the private latent state in the final output."
    )
    chunks.append(
        "Output exactly in this format:\n"
        "Attempt: <Yes or No>\n"
        "IdentifiedConcept: <one concept from the provided options>\n"
        "StudentAnswer: <the learner's submitted answer only>\n"
        "AnswerConfidence: <0.80 for high, 0.50 for medium, or 0.20 for low>\n"
        "StudentReasoning: <one short learner scratch step, which may be incomplete or mistaken>\n"
        "ReasoningConfidence: <0.80 for high, 0.50 for medium, or 0.20 for low>\n"
        "Return only these six fields. Do not output the private commitment, correctness, an expert solution, a correction, or markdown formatting."
    )
    return "\n\n".join(chunks)


def proficiency_context(concept: str, mastery: float) -> dict[str, Any]:
    level = "high" if mastery >= 0.7 else "medium" if mastery >= 0.4 else "low"
    return {"concept": concept, "value": float(mastery), "level": level}


def _format_cognitive_profile(cognitive_profile: dict[str, Any]) -> str:
    control = cognitive_profile.get("control_traits") or {}
    affective = cognitive_profile.get("cognitive_affective_proxies") or {}
    error = cognitive_profile.get("error_generation_traits") or {}
    transfer = cognitive_profile.get("transfer_traits") or {}
    return (
        "# Computed Cognitive Profile #\n"
        "The following profile is computed from observed historical responses only. "
        "Affective fields are behavioral proxies, not ground-truth emotion labels.\n"
        f"- control: overall_success={control.get('overall_success_level', 'unknown')} "
        f"({control.get('overall_success_rate')}), recent_success={control.get('recent_success_level', 'unknown')} "
        f"({control.get('recent_success_rate')}), trend={control.get('success_trend_level', 'unknown')} "
        f"({control.get('success_trend')}), stability={control.get('mastery_stability_level', 'unknown')} "
        f"({control.get('mastery_stability')})\n"
        f"- cognitive-affective proxies: concentration={affective.get('concentration_level', 'unknown')} "
        f"({affective.get('concentration_proxy')}), frustration={affective.get('frustration_level', 'unknown')} "
        f"({affective.get('frustration_risk')}), confusion={affective.get('confusion_level', 'unknown')} "
        f"({affective.get('confusion_risk')}), boredom={affective.get('boredom_level', 'unknown')} "
        f"({affective.get('boredom_risk')})\n"
        f"- error generation: carelessness={error.get('carelessness_level', 'unknown')} "
        f"({error.get('carelessness_tendency')}), guessing={error.get('guessing_level', 'unknown')} "
        f"({error.get('guessing_tendency')}), misconception_persistence={error.get('misconception_persistence_level', 'unknown')} "
        f"({error.get('misconception_persistence')}), error_recovery={error.get('error_recovery_level', 'unknown')} "
        f"({error.get('error_recovery_rate')})\n"
        f"- transfer adaptation: same_parent_transfer={transfer.get('same_parent_transfer_level', 'unknown')} "
        f"({transfer.get('same_parent_transfer_success')}), transfer_fragility="
        f"{transfer.get('transfer_fragility_level', 'unknown')} ({transfer.get('transfer_fragility')})\n"
        "Use this computed cognitive profile to constrain attention stability, error tendencies, confidence, "
        "and transfer behavior during first-attempt simulation. Do not treat a generally successful learner as correct "
        "by default when recent performance, repeated errors, confusion, or fragile transfer indicate instability. "
        "Preserve plausible mistakes instead of repairing the answer with expert reasoning."
    )


def _format_ability_profile(ability_profile: dict[str, Any]) -> str:
    return (
        "# Ability Summary #\n"
        "The following compact profile is computed from observed history and global item statistics.\n"
        f"- knowledge breadth: {ability_profile.get('knowledge_breadth', 'unknown')} "
        f"({ability_profile.get('knowledge_breadth_value')})\n"
        f"- practice depth: {ability_profile.get('practice_depth', 'unknown')} "
        f"({ability_profile.get('practice_depth_value')})\n"
        f"- challenge adaptation: {ability_profile.get('challenge_adaptation', 'unknown')} "
        f"({ability_profile.get('challenge_adaptation_value')})\n"
        f"- cross-domain generalization: {ability_profile.get('cross_domain_generalization', 'unknown')} "
        f"({ability_profile.get('cross_domain_generalization_value')})\n"
        f"- profile confidence: {ability_profile.get('profile_confidence', 'unknown')}\n"
        "Use this summary to calibrate breadth, depth, difficulty tolerance, and cross-domain transfer. "
        "It is a broad historical capability signal, not a current-item correctness label. "
        "For routine same-concept items, current knowledge proficiency and related memory are more specific evidence than broad ability traits. "
        "Do not treat ability as guaranteed correctness or guaranteed failure."
    )


def _format_tendency_calibration(calibration: dict[str, Any]) -> str:
    return (
        "This calibration is a tendency signal, not an answer label and not a probability formula. "
        "Use it to avoid systematic over-pessimism or over-optimism across similar first attempts.\n"
        f"- calibrated tendency: {calibration.get('band', 'unknown')} "
        f"({calibration.get('score', 'unavailable')})\n"
        f"- historical tendency: {calibration.get('history_level', 'unknown')} "
        f"({calibration.get('history_rate', 'unavailable')})\n"
        f"- current knowledge tendency: {calibration.get('mastery_level', 'unknown')} "
        f"({calibration.get('mastery', 'unavailable')})\n"
        f"- related concept tendency: {calibration.get('concept_level', 'unknown')} "
        f"({calibration.get('concept_rate', 'unavailable')})\n"
        "If this tendency is strong or favorable, do not default to an incorrect answer just because the role is a learner; "
        "a concise correct student attempt is plausible. If it is fragile or weak, mistakes and low confidence are plausible. "
        "The visible exercise and memory can still override the tendency when they provide clear evidence."
    )


def _format_behavior_state(factors: dict[str, float]) -> str:
    return (
        f"- attention: {_three_level(float(factors.get('attention', 0.7)), 0.55, 0.8)}\n"
        f"- fatigue: {_three_level(float(factors.get('fatigue', 0.2)), 0.15, 0.35)}\n"
        f"- carelessness: {_three_level(float(factors.get('carelessness', 0.1)), 0.08, 0.2)}\n"
        f"- guessing tendency: {_three_level(float(factors.get('guessing', 0.1)), 0.08, 0.2)}"
    )


def _format_records(records: list[dict[str, Any]]) -> list[str]:
    formatted: list[str] = []
    for index, item in enumerate(records, start=1):
        result = "rightly" if int(item.get("simulated_response", 0)) == 1 else "wrongly"
        content = item.get("content") or item.get("content_preview", "")
        concept = _record_concept(item)
        formatted.append(
            f"Record{index}: You {result} answered an exercise.\n"
            f"- # Textual Content #: {content}\n"
            f"- # Knowledge Concept #: {concept}"
        )
    return formatted


def _record_concept(record: dict[str, Any]) -> str:
    routes = record.get("kc_routes") or []
    return str(routes[0]) if routes else str(record.get("cid", "unknown"))


def _binary_level(value: float) -> str:
    return "high" if value >= 0.5 else "low"


def _format_optional_float(value: Any) -> str:
    if value is None:
        return "unavailable"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _three_level(value: float, low: float, high: float) -> str:
    if value > high:
        return "high"
    if value > low:
        return "medium"
    return "low"

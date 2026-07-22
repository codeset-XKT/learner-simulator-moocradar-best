from __future__ import annotations

import time
from typing import Any

from learner_simulator.action import parse_agent_response
from learner_simulator.agent4edu_prompt import (
    build_action_prompt,
    build_profile_system_prompt,
    proficiency_context,
)
from learner_simulator.behavior import non_cognitive_factors
from learner_simulator.data import clean_sequence
from learner_simulator.four_tier import (
    assess_four_tier_response,
    compare_answers,
    parse_answer_only_response,
)
from learner_simulator.llm import call_openai_compatible_chat
from learner_simulator.simulators.random_simulator import (
    RandomLearnerSimulator,
    build_sequence_summary,
)


class LLMLearnerSimulator(RandomLearnerSimulator):
    """Autonomous LLM learner simulator.

    The statistical layer is retained as a baseline and state estimator. The
    LLM produces a four-tier learner response: answer, answer confidence,
    reasoning, and reasoning confidence. Correctness is assessed externally
    when the submitted answer can be matched to question metadata.
    """

    def simulate_sequence(
        self,
        row: dict[str, str],
        questions: dict[str, dict[str, Any]],
        llm_config: dict[str, Any] | None = None,
        call_llm: bool = False,
        include_prompt: bool = False,
        progress_callback: Any | None = None,
        history_row: dict[str, str] | None = None,
        response_format: str = "four_tier",
        include_profile: bool = True,
        include_memory: bool = True,
        include_proficiency: bool = True,
        include_behavior: bool = True,
        include_cognitive_strategy: bool = True,
        include_cognitive_profile: bool = True,
        include_ability_profile: bool = True,
    ) -> dict[str, Any]:
        uid = row["uid"]
        state, memory = self.initialize_from_history(
            uid,
            history_row,
            questions,
        )
        profile = self.get_profile(uid)
        full_profile_context = profile.to_context()
        profile_context = (
            full_profile_context
            if include_cognitive_profile
            else _without_cognitive_profile(full_profile_context)
        )
        if not include_ability_profile:
            profile_context = _without_ability_profile(profile_context)
        simulated_steps = []

        for step in clean_sequence(row):
            qid = step["qid"]
            cid = step["cid"]
            real_response = step["response"]
            components = self.probability_components(uid, qid, cid, state)
            behavior_factors = non_cognitive_factors(profile_context, int(step.get("position") or 0), self.random)
            mastery_before = components["mastery"]
            fallback_response = int(mastery_before >= 0.5)
            tendency_calibration = _build_tendency_calibration(
                profile_context,
                components,
            )

            qmeta = questions.get(str(qid), {})
            kc_routes = qmeta.get("kc_routes", [])
            memory_context = (
                memory.snapshot(cid, kc_routes, mastery_before)
                if include_memory
                else {"short_memory": [], "long_memory": {}}
            )
            cognitive_strategy = None
            true_concept = str(kc_routes[0]) if kc_routes else str(cid)
            concept_options = self.concept_options(
                true_concept=true_concept,
                seed=self.irt_model.seed + int(qid) + int(step.get("position") or 0),
            )
            profile_system_prompt = (
                build_profile_system_prompt(profile_context)
                if include_profile
                else (
                    "You are simulating one high school student's first independent attempt. "
                    "Stay within an ordinary learner's cognitive boundary and do not act as an expert tutor."
                )
            )
            llm_prompt = build_action_prompt(
                question=qmeta,
                short_memory=memory_context.get("short_memory", []),
                long_memory=memory_context.get("long_memory", {}),
                concept_options=concept_options,
                proficiency=(
                    proficiency_context(true_concept, mastery_before)
                    if include_proficiency
                    else None
                ),
                behavior_factors=behavior_factors if include_behavior else None,
                response_format=response_format,
                cognitive_strategy=cognitive_strategy,
                tendency_calibration=tendency_calibration,
            )

            step_result = {
                "uid": uid,
                "source": "simulated",
                "step_index": step.get("position"),
                "timestamp": step.get("timestamp"),
                "qid": qid,
                "cid": cid,
                "probability_components": components,
                "non_cognitive_factors": behavior_factors,
                "tendency_calibration": tendency_calibration,
                "simulated_response": fallback_response,
                "real_response": real_response,
                "question_type": qmeta.get("type"),
                "content": qmeta.get("content"),
                "options": qmeta.get("options"),
                "answer": qmeta.get("answer"),
                "analysis": qmeta.get("analysis"),
                "content_preview": str(qmeta.get("content", ""))[:80],
                "kc_routes": kc_routes,
                "learner_profile": profile_context,
                "memory_context": memory_context,
                "agent_action": {
                    "attempt": "yes",
                    "identified_concept": kc_routes[0] if kc_routes else str(cid),
                    "student_answer": "",
                    "simulated_correct": fallback_response,
                    "error_type": "unscored_llm_fallback" if fallback_response == 0 else "none",
                    "confidence": 0.5,
                },
                "concept_options": concept_options,
                "enabled_modules": {
                    "profile": include_profile,
                    "memory": include_memory,
                    "proficiency": include_proficiency,
                    "behavior": include_behavior,
                    "cognitive_strategy": False,
                    "cognitive_profile": include_cognitive_profile,
                    "ability_profile": include_ability_profile and bool(profile_context.get("ability_profile")),
                    "four_tier": response_format == "four_tier",
                },
            }
            if include_prompt:
                step_result["llm_system_prompt"] = profile_system_prompt
                step_result["llm_prompt"] = llm_prompt
            if call_llm:
                if llm_config is None:
                    raise ValueError("llm_config is required when call_llm=True")
                call_start = time.perf_counter()
                try:
                    step_result["llm_result"] = call_openai_compatible_chat(
                        llm_config,
                        llm_prompt,
                        system_prompt=profile_system_prompt,
                    )
                    parsed_action = (
                        parse_answer_only_response(step_result["llm_result"])
                        if response_format == "answer_only"
                        else parse_agent_response(step_result["llm_result"])
                    )
                    if parsed_action is None:
                        raise ValueError(
                            f"LLM response did not match the {response_format} output contract"
                        )
                    step_result["llm_parsed_action"] = parsed_action
                    step_result["agent_action"] = parsed_action
                    if response_format == "answer_only":
                        answer_correct = compare_answers(
                            parsed_action.get("student_answer"),
                            qmeta.get("answer"),
                        )
                        step_result["ablation"] = "no_four_tier"
                    else:
                        four_tier = assess_four_tier_response(
                            parsed_action,
                            reference_answers=qmeta.get("answer"),
                            reference_reasoning=str(qmeta.get("analysis", "")),
                        )
                        parsed_action["four_tier_assessment"] = four_tier
                        step_result["four_tier_assessment"] = four_tier
                        answer_correct = four_tier.get("answer_correct")
                    if answer_correct is not None:
                        parsed_action["simulated_correct"] = int(answer_correct)
                        if response_format == "four_tier":
                            parsed_action["confidence"] = four_tier["answer_confidence"]
                            parsed_action["error_type"] = (
                                "none" if answer_correct else four_tier["diagnosis"]
                            )
                        llm_correct = int(parsed_action["simulated_correct"])
                        step_result["llm_behavior_correct"] = llm_correct
                        step_result["simulated_response"] = llm_correct
                except Exception as exc:
                    step_result["llm_error"] = f"{exc.__class__.__name__}: {exc}"
                step_result["llm_elapsed_seconds"] = round(time.perf_counter() - call_start, 3)
                if progress_callback is not None:
                    progress_callback(step_result)

            state.update(cid, int(step_result["simulated_response"]), learning_rate=self.learning_rate)
            memory.observe(step_result)
            simulated_steps.append(step_result)

        summary = build_sequence_summary(uid, simulated_steps)
        summary["history_interactions"] = (
            len(clean_sequence(history_row)) if history_row is not None else 0
        )
        summary["target_interactions"] = len(simulated_steps)
        return summary


def _without_cognitive_profile(profile_context: dict[str, Any]) -> dict[str, Any]:
    stripped = dict(profile_context)
    stripped.pop("cognitive_profile", None)
    return stripped


def _without_ability_profile(profile_context: dict[str, Any]) -> dict[str, Any]:
    stripped = dict(profile_context)
    stripped.pop("ability_profile", None)
    return stripped


def _build_tendency_calibration(
    profile_context: dict[str, Any],
    components: dict[str, Any],
) -> dict[str, Any]:
    cognitive_profile = profile_context.get("cognitive_profile") or {}
    control = cognitive_profile.get("control_traits") or {}
    history_rate = _optional_float(control.get("recent_success_rate"))
    if history_rate is None:
        history_rate = _optional_float(control.get("overall_success_rate"))
    if history_rate is None:
        history_rate = _optional_float(components.get("user_rate"))
    mastery = _optional_float(components.get("mastery"))
    concept_rate = _optional_float(components.get("concept_rate"))
    item_rate = _optional_float(components.get("item_rate"))

    evidence = [
        (history_rate, 0.30),
        (mastery, 0.35),
        (concept_rate, 0.20),
        (item_rate, 0.15),
    ]
    weighted_sum = sum(value * weight for value, weight in evidence if value is not None)
    total_weight = sum(weight for value, weight in evidence if value is not None)
    score = weighted_sum / total_weight if total_weight else 0.5
    score = min(0.9, max(0.1, score))
    return {
        "score": round(score, 3),
        "band": _tendency_band(score),
        "history_rate": _round_optional(history_rate),
        "history_level": _rate_level(history_rate),
        "mastery": _round_optional(mastery),
        "mastery_level": _rate_level(mastery),
        "concept_rate": _round_optional(concept_rate),
        "concept_level": _rate_level(concept_rate),
        "item_rate": _round_optional(item_rate),
    }


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_optional(value: float | None) -> float | str:
    return round(value, 3) if value is not None else "unavailable"


def _rate_level(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value >= 0.72:
        return "high"
    if value >= 0.58:
        return "favorable"
    if value >= 0.42:
        return "mixed"
    if value >= 0.28:
        return "fragile"
    return "low"


def _tendency_band(value: float) -> str:
    if value >= 0.72:
        return "strong-correct-leaning"
    if value >= 0.58:
        return "correct-leaning"
    if value >= 0.42:
        return "mixed"
    if value >= 0.28:
        return "error-leaning"
    return "strong-error-leaning"

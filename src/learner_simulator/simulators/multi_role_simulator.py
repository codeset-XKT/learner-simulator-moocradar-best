from __future__ import annotations

import math
import time
from typing import Any

from learner_simulator.action import parse_agent_response
from learner_simulator.agent4edu_prompt import (
    build_profile_system_prompt,
    proficiency_context,
)
from learner_simulator.behavior import non_cognitive_factors
from learner_simulator.data import clean_sequence
from learner_simulator.educational_multi_agent_prompt import (
    build_ability_boundary_evidence,
    build_cognitive_profile_view,
    build_cognitive_route_prompt,
    build_four_tier_response_record,
    build_response_prompt,
    fallback_ability_boundary,
    fallback_cognitive_route,
    infer_cognitive_route_evidence,
    parse_cognitive_route,
)
from learner_simulator.four_tier import (
    assess_four_tier_response,
    compare_answers,
    parse_answer_only_response,
)
from learner_simulator.llm import call_openai_compatible_chat
from learner_simulator.simulators.llm_simulator import (
    _build_tendency_calibration,
    _without_ability_profile,
    _without_cognitive_profile,
)
from learner_simulator.simulators.random_simulator import (
    RandomLearnerSimulator,
    build_sequence_summary,
)


class MultiRoleLearnerSimulator(RandomLearnerSimulator):
    """Educational multi-agent learner simulator.

    The historical command name is still ``multi-role`` for compatibility, but
    the implementation is now a functional educational-agent pipeline:

    1. Cognitive Profile Evidence summarizes historical cognitive evidence.
    2. Ability Boundary Evidence constrains what the learner can plausibly use.
    3. Cognitive Route Agent selects the first-attempt cognitive path.
    4. Four-tier Response Module generates and externally scores the answer.
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
        feedback_mode: str = "rollout",
    ) -> dict[str, Any]:
        if response_format not in {"four_tier", "answer_only"}:
            raise ValueError(f"Unsupported response_format: {response_format}")
        if feedback_mode not in {"rollout", "teacher-forcing"}:
            raise ValueError(f"Unsupported feedback_mode: {feedback_mode}")

        uid = row["uid"]
        state, memory = self.initialize_from_history(uid, history_row, questions)
        profile = self.get_profile(uid)
        full_profile_context = profile.to_context()
        profile_context = (
            full_profile_context
            if include_cognitive_profile
            else _without_cognitive_profile(full_profile_context)
        )
        if not include_ability_profile:
            profile_context = _without_ability_profile(profile_context)

        simulated_steps: list[dict[str, Any]] = []
        for step in clean_sequence(row):
            step_start = time.perf_counter()
            qid = step["qid"]
            cid = step["cid"]
            real_response = step["response"]
            components = self.probability_components(uid, qid, cid, state)
            mastery_before = components["mastery"]
            kt_state_probability = float(mastery_before)
            external_kt_raw_probability = self._dkt_raw_probability(uid, cid)
            dkt_predicted_response = (
                int(external_kt_raw_probability >= 0.5)
                if external_kt_raw_probability is not None
                else None
            )
            fallback_response = int(mastery_before >= 0.5)
            behavior_factors = non_cognitive_factors(
                profile_context,
                int(step.get("position") or 0),
                self.random,
            )
            active_behavior = behavior_factors if include_behavior else None
            visible_components = _visible_probability_components(
                components,
                include_proficiency=include_proficiency,
            )
            tendency_calibration = _build_tendency_calibration(
                profile_context,
                visible_components,
            ) if include_proficiency else None

            qmeta = questions.get(str(qid), {})
            kc_routes = qmeta.get("kc_routes", [])
            memory_context = (
                memory.snapshot(cid, kc_routes, mastery_before)
                if include_memory
                else {"short_memory": [], "long_memory": {}}
            )
            true_concept = str(kc_routes[0]) if kc_routes else str(cid)
            concept_options = self.concept_options(
                true_concept=true_concept,
                seed=self.irt_model.seed + int(qid) + int(step.get("position") or 0),
            )
            proficiency = (
                proficiency_context(true_concept, mastery_before)
                if include_proficiency
                else None
            )
            profile_system_prompt = (
                build_profile_system_prompt(profile_context)
                if include_profile
                else (
                    "You are simulating one high school student's first independent attempt. "
                    "Use only the reduced evidence provided by the current module prompts."
                )
            )
            cognitive_profile_view = build_cognitive_profile_view(
                profile_context=profile_context,
                proficiency=proficiency,
                behavior_factors=active_behavior,
                tendency_calibration=tendency_calibration,
            )

            step_result: dict[str, Any] = {
                "uid": uid,
                "source": "simulated",
                "step_index": step.get("position"),
                "timestamp": step.get("timestamp"),
                "qid": qid,
                "cid": cid,
                "probability_components": components,
                "visible_probability_components": visible_components,
                "kt_state_probability": round(kt_state_probability, 6),
                "external_proficiency_source": self._external_proficiency_source(),
                "external_kt_raw_probability": (
                    round(external_kt_raw_probability, 6)
                    if external_kt_raw_probability is not None
                    else None
                ),
                "external_kt_correct_probability": (
                    round(external_kt_raw_probability, 6)
                    if external_kt_raw_probability is not None
                    else None
                ),
                "external_kt_predicted_response": dkt_predicted_response,
                "dkt_raw_probability": (
                    round(external_kt_raw_probability, 6)
                    if external_kt_raw_probability is not None
                    else None
                ),
                "dkt_correct_probability": (
                    round(external_kt_raw_probability, 6)
                    if external_kt_raw_probability is not None
                    else None
                ),
                "dkt_predicted_response": dkt_predicted_response,
                "non_cognitive_factors": active_behavior,
                "tendency_calibration": tendency_calibration,
                "feedback_mode": feedback_mode,
                "simulated_response": fallback_response,
                "real_response": real_response,
                "feedback_response": (
                    real_response if feedback_mode == "teacher-forcing" else fallback_response
                ),
                "question_type": qmeta.get("type"),
                "content": qmeta.get("content"),
                "options": qmeta.get("options"),
                "answer": qmeta.get("answer"),
                "analysis": qmeta.get("analysis"),
                "content_preview": str(qmeta.get("content", ""))[:80],
                "kc_routes": kc_routes,
                "learner_profile": profile_context,
                "memory_context": memory_context,
                "cognitive_profile_evidence": cognitive_profile_view,
                "agent_action": {
                    "attempt": "yes",
                    "identified_concept": true_concept,
                    "student_answer": "",
                    "simulated_correct": fallback_response,
                    "error_type": "educational_multi_agent_fallback",
                    "confidence": 0.5,
                },
                "concept_options": concept_options,
                "enabled_modules": {
                    "profile": include_profile,
                    "memory": include_memory,
                    "proficiency": include_proficiency,
                    "behavior": include_behavior,
                    "cognitive_strategy": include_cognitive_strategy,
                    "cognitive_profile": include_cognitive_profile,
                    "ability_profile": include_ability_profile
                    and bool(profile_context.get("ability_profile")),
                    "four_tier": True,
                    "cognitive_profile_evidence": True,
                    "ability_boundary_evidence": include_ability_profile,
                    "cognitive_route_agent": include_cognitive_strategy,
                    "four_tier_response_module": True,
                },
            }

            if call_llm:
                if llm_config is None:
                    raise ValueError("llm_config is required when call_llm=True")
                ability_boundary = self._build_ability_boundary_evidence(
                    question=qmeta,
                    cognitive_profile=cognitive_profile_view,
                    memory_context=memory_context,
                    proficiency=proficiency,
                    enabled=include_ability_profile,
                )
                ability_boundary_view = self._public_agent_view(ability_boundary)
                route_evidence = infer_cognitive_route_evidence(
                    question=qmeta,
                    cognitive_profile=cognitive_profile_view,
                    ability_boundary=ability_boundary_view,
                    proficiency=proficiency,
                    behavior_factors=active_behavior,
                    tendency_calibration=tendency_calibration,
                )
                cognitive_route = self._call_cognitive_route_agent(
                    llm_config=llm_config,
                    question=qmeta,
                    cognitive_profile=cognitive_profile_view,
                    ability_boundary=ability_boundary_view,
                    proficiency=proficiency,
                    behavior_factors=active_behavior,
                    tendency_calibration=tendency_calibration,
                    route_evidence=route_evidence,
                    system_prompt=profile_system_prompt,
                    enabled=include_cognitive_strategy,
                )
                cognitive_route_view = (
                    self._public_agent_view(cognitive_route)
                    if cognitive_route is not None
                    else None
                )
                latent_state = self._build_latent_learner_state(
                    components=components,
                    tendency_calibration=tendency_calibration,
                    behavior_factors=active_behavior,
                    ability_boundary=ability_boundary_view,
                    cognitive_route=cognitive_route_view,
                    route_evidence=route_evidence,
                )
                response_prompt = build_response_prompt(
                    question=qmeta,
                    short_memory=memory_context.get("short_memory", []),
                    long_memory=memory_context.get("long_memory", {}),
                    concept_options=concept_options,
                    proficiency=proficiency,
                    behavior_factors=active_behavior,
                    tendency_calibration=tendency_calibration,
                    cognitive_profile=cognitive_profile_view,
                    ability_boundary=ability_boundary_view,
                    cognitive_route=cognitive_route_view,
                    latent_state=latent_state,
                    response_format=response_format,
                )
                output_contract = (
                    "Four-tier learner response"
                    if response_format == "four_tier"
                    else "answer-only learner response"
                )
                response_system_prompt = (
                    f"{profile_system_prompt}\n\n"
                    "You are the Response Agent. Follow the Cognitive Profile Agent, "
                    "Ability Boundary Evidence, and Cognitive Route Agent outputs. "
                    f"Return only the required {output_contract}."
                )
                if include_prompt:
                    step_result["cognitive_route_prompt"] = (
                        cognitive_route or {}
                    ).get("prompt")
                    step_result["response_agent_prompt"] = response_prompt
                    step_result["response_agent_system_prompt"] = response_system_prompt
                step_result["ability_boundary_evidence"] = ability_boundary
                step_result["cognitive_route_evidence"] = route_evidence
                step_result["cognitive_route_agent"] = cognitive_route
                step_result["latent_learner_state"] = latent_state
                try:
                    response_start = time.perf_counter()
                    raw = call_openai_compatible_chat(
                        llm_config,
                        response_prompt,
                        system_prompt=response_system_prompt,
                    )
                    action = (
                        parse_agent_response(raw)
                        if response_format == "four_tier"
                        else parse_answer_only_response(raw)
                    )
                    if action is None:
                        raise ValueError(
                            f"Response Agent output did not match {response_format} contract"
                        )
                    action["raw"] = raw
                    action["elapsed_seconds"] = round(
                        time.perf_counter() - response_start,
                        3,
                    )
                    four_tier = None
                    if response_format == "four_tier":
                        four_tier = assess_four_tier_response(
                            action,
                            reference_answers=qmeta.get("answer"),
                            reference_reasoning=str(qmeta.get("analysis", "")),
                        )
                        answer_correct = four_tier.get("answer_correct")
                    else:
                        answer_correct = compare_answers(
                            action.get("student_answer"),
                            qmeta.get("answer"),
                        )
                        step_result["ablation"] = "no_four_tier"
                    if answer_correct is not None:
                        final_correct = int(answer_correct)
                        answer_confidence = (
                            four_tier["answer_confidence"]
                            if four_tier is not None
                            else 0.5
                        )
                        dkt_conditioning = self._dkt_alignment_diagnostics(
                            llm_correct=final_correct,
                            answer_confidence=answer_confidence,
                            components=components,
                            cognitive_route=cognitive_route_view,
                            ability_boundary=ability_boundary_view,
                            tendency_calibration=tendency_calibration,
                            latent_state=latent_state,
                        )
                        action["simulated_correct"] = final_correct
                        action["confidence"] = answer_confidence
                        action["error_type"] = "none" if final_correct else (
                            four_tier["diagnosis"]
                            if four_tier is not None
                            else "answer_only_incorrect"
                        )
                        step_result["simulated_response"] = final_correct
                        step_result["llm_behavior_correct"] = final_correct
                        step_result["dkt_conditioning"] = dkt_conditioning
                    step_result["agent_action"] = action
                    step_result["llm_parsed_action"] = action
                    if four_tier is not None:
                        step_result["four_tier_assessment"] = four_tier
                        step_result["four_tier_response_module"] = build_four_tier_response_record(
                            action=action,
                            four_tier=four_tier,
                            cognitive_route=cognitive_route_view,
                            ability_boundary=ability_boundary_view,
                        )
                    step_result["llm_result"] = raw
                except Exception as exc:
                    step_result["llm_error"] = f"{exc.__class__.__name__}: {exc}"
                step_result["llm_elapsed_seconds"] = round(
                    time.perf_counter() - step_start,
                    3,
                )
                if progress_callback is not None:
                    progress_callback(step_result)

            feedback_response = (
                int(step_result["real_response"])
                if feedback_mode == "teacher-forcing"
                else int(step_result["simulated_response"])
            )
            step_result["feedback_response"] = feedback_response
            state.update(cid, feedback_response, learning_rate=self.learning_rate)
            memory_record = dict(step_result)
            if feedback_mode == "teacher-forcing":
                memory_record["source"] = "teacher_forced_feedback"
                memory_record["model_simulated_response"] = step_result["simulated_response"]
                memory_record["simulated_response"] = feedback_response
            memory.observe(memory_record)
            simulated_steps.append(step_result)

        summary = build_sequence_summary(uid, simulated_steps)
        summary["history_interactions"] = (
            len(clean_sequence(history_row)) if history_row is not None else 0
        )
        summary["target_interactions"] = len(simulated_steps)
        summary["simulator_type"] = "educational_multi_agent"
        summary["feedback_mode"] = feedback_mode
        return summary

    def _build_ability_boundary_evidence(
        self,
        question: dict[str, Any],
        cognitive_profile: dict[str, Any],
        memory_context: dict[str, Any],
        proficiency: dict[str, Any] | None,
        enabled: bool,
    ) -> dict[str, Any]:
        if not enabled:
            boundary = fallback_ability_boundary(cognitive_profile, proficiency)
            boundary["ablated"] = True
            return boundary
        boundary = build_ability_boundary_evidence(
            question=question,
            cognitive_profile=cognitive_profile,
            memory_context=memory_context,
            proficiency=proficiency,
        )
        boundary["elapsed_seconds"] = 0.0
        return boundary

    def _call_cognitive_route_agent(
        self,
        llm_config: dict[str, Any],
        question: dict[str, Any],
        cognitive_profile: dict[str, Any],
        ability_boundary: dict[str, Any],
        proficiency: dict[str, Any] | None,
        behavior_factors: dict[str, float] | None,
        tendency_calibration: dict[str, Any] | None,
        route_evidence: dict[str, Any],
        system_prompt: str,
        enabled: bool,
    ) -> dict[str, Any] | None:
        if not enabled:
            return None
        prompt = build_cognitive_route_prompt(
            question=question,
            cognitive_profile=cognitive_profile,
            ability_boundary=ability_boundary,
            proficiency=proficiency,
            behavior_factors=behavior_factors,
            tendency_calibration=tendency_calibration,
            route_evidence=route_evidence,
        )
        started = time.perf_counter()
        try:
            raw = call_openai_compatible_chat(
                llm_config,
                prompt,
                system_prompt=(
                    f"{system_prompt}\n\n"
                    "You are the Cognitive Route Agent. Select the route only; do not solve."
                ),
            )
            parsed = parse_cognitive_route(raw)
            if parsed is None:
                raise ValueError("Cognitive Route Agent output did not match contract")
            parsed["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            parsed["prompt"] = prompt
            parsed["route_evidence"] = route_evidence
            return self._apply_route_guard(parsed, route_evidence)
        except Exception as exc:
            route = fallback_cognitive_route(
                cognitive_profile=cognitive_profile,
                ability_boundary=ability_boundary,
                proficiency=proficiency,
                behavior_factors=behavior_factors,
                tendency_calibration=tendency_calibration,
                route_evidence=route_evidence,
            )
            route["error"] = f"{exc.__class__.__name__}: {exc}"
            route["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            route["prompt"] = prompt
            return self._apply_route_guard(route, route_evidence)

    @staticmethod
    def _public_agent_view(agent_output: dict[str, Any] | None) -> dict[str, Any]:
        if not agent_output:
            return {}
        hidden = {"prompt", "raw"}
        return {
            key: value
            for key, value in agent_output.items()
            if key not in hidden and not key.endswith("_prompt")
        }

    @staticmethod
    def _apply_route_guard(
        route: dict[str, Any],
        route_evidence: dict[str, Any],
    ) -> dict[str, Any]:
        if route.get("cognitive_route") != "mastery_retrieval":
            return route
        if route_evidence.get("mastery_retrieval_allowed") is not False:
            return route
        suggested = route_evidence.get("suggested_route")
        if suggested not in {
            "partial_reasoning",
            "misconception_transfer",
            "careless_execution",
            "uncertain_guessing",
        }:
            suggested = "partial_reasoning"
        adjusted = dict(route)
        adjusted["original_cognitive_route"] = "mastery_retrieval"
        adjusted["cognitive_route"] = suggested
        adjusted["route_guard_override"] = True
        adjusted["route_guard_reason"] = (
            "Mastery retrieval was disallowed by cognitive route evidence."
        )
        adjusted["expected_fluency"] = (
            "hesitant"
            if suggested == "uncertain_guessing"
            else "uneven"
            if suggested in {"misconception_transfer", "careless_execution"}
            else "constrained"
        )
        adjusted["expected_verification_depth"] = "none" if suggested != "mastery_retrieval" else "light"
        return adjusted

    @staticmethod
    def _build_latent_learner_state(
        components: dict[str, Any],
        tendency_calibration: dict[str, Any] | None,
        behavior_factors: dict[str, float] | None,
        ability_boundary: dict[str, Any],
        cognitive_route: dict[str, Any] | None,
        route_evidence: dict[str, Any],
    ) -> dict[str, Any]:
        mastery = _safe_float(components.get("mastery"), default=0.5)
        history_rate = _safe_float((tendency_calibration or {}).get("history_rate"), default=0.5)
        item_rate = _safe_float(components.get("item_rate"), default=0.5)
        memory_support = ability_boundary.get("memory_support") or {}
        related_rate = _safe_float(memory_support.get("related_correct_rate"), default=0.5)
        identical_correct = int(memory_support.get("identical_correct_count", 0) or 0)
        carelessness = _safe_float((behavior_factors or {}).get("carelessness"), default=0.1)
        fatigue = _safe_float((behavior_factors or {}).get("fatigue"), default=0.1)
        guessing = _safe_float((behavior_factors or {}).get("guessing"), default=0.1)
        route = str((cognitive_route or {}).get("cognitive_route") or route_evidence.get("suggested_route") or "partial_reasoning")
        boundary_risk = str(ability_boundary.get("boundary_risk", "medium"))
        item_demand = str(ability_boundary.get("item_demand", "medium"))

        base_probability = min(
            0.95,
            max(
                0.05,
                0.55 * mastery + 0.20 * history_rate + 0.15 * item_rate + 0.10 * related_rate,
            ),
        )
        score = _logit(base_probability)
        score += 0.45 * (history_rate - 0.5)
        score += 0.35 * (related_rate - 0.5)
        score += 0.12 * min(2, identical_correct)
        score -= 0.70 * carelessness
        score -= 0.55 * fatigue
        score -= 0.45 * guessing
        score += {
            "mastery_retrieval": 0.30,
            "partial_reasoning": 0.0,
            "misconception_transfer": -0.55,
            "careless_execution": -0.30,
            "uncertain_guessing": -0.75,
        }.get(route, 0.0)
        score += {"low": 0.12, "medium": 0.0, "high": -0.28}.get(item_demand, 0.0)
        score += {"low": 0.10, "medium": -0.08, "high": -0.35}.get(boundary_risk, -0.08)
        anchor_probability = _sigmoid(score)

        access_score = 0.60 * mastery + 0.20 * history_rate + 0.20 * related_rate
        if route == "uncertain_guessing":
            access_score -= 0.15
        elif route == "misconception_transfer":
            access_score -= 0.08
        knowledge_access = _three_band(access_score, low=0.45, high=0.72)

        execution_score = 1.0 - (0.50 * carelessness + 0.25 * fatigue + 0.25 * guessing)
        execution_score += {"low": 0.08, "medium": 0.0, "high": -0.12}.get(boundary_risk, 0.0)
        execution_quality = (
            "stable" if execution_score >= 0.72
            else "variable" if execution_score >= 0.48
            else "fragile"
        )
        confidence_band = _three_band(anchor_probability, low=0.42, high=0.72)

        return {
            "module": "latent_learner_state",
            "p_correct_anchor": round(anchor_probability, 3),
            "knowledge_access": knowledge_access,
            "execution_quality": execution_quality,
            "confidence_band": confidence_band,
            "route": route,
            "expected_fluency": (cognitive_route or {}).get("expected_fluency"),
            "expected_verification_depth": (cognitive_route or {}).get("expected_verification_depth"),
            "boundary_risk": boundary_risk,
            "item_demand": item_demand,
            "anchor_rationale": (
                f"anchor combines mastery={mastery:.2f}, history={history_rate:.2f}, "
                f"memory={related_rate:.2f}, route={route}, and execution risks "
                f"(carelessness={carelessness:.2f}, fatigue={fatigue:.2f}, guessing={guessing:.2f})."
            ),
        }

    @staticmethod
    def _dkt_alignment_diagnostics(
        llm_correct: int,
        answer_confidence: float,
        components: dict[str, Any],
        cognitive_route: dict[str, Any] | None,
        ability_boundary: dict[str, Any],
        tendency_calibration: dict[str, Any] | None,
        latent_state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Record LLM-vs-DKT alignment without changing the LLM outcome."""

        mastery = _safe_float(components.get("mastery"), default=0.5)
        dkt_predicted = int(mastery >= 0.5)
        route = str((cognitive_route or {}).get("cognitive_route", "unknown"))
        boundary_risk = str(ability_boundary.get("boundary_risk", "medium"))
        item_demand = str(ability_boundary.get("item_demand", "medium"))
        tendency_band = str((tendency_calibration or {}).get("band", "mixed"))
        history_level = str((tendency_calibration or {}).get("history_level", "unknown"))
        confidence = min(1.0, max(0.0, _safe_float(answer_confidence, default=0.5)))

        conflict_direction = "aligned"
        if int(llm_correct) != dkt_predicted:
            conflict_direction = (
                "llm_more_pessimistic_than_dkt"
                if int(llm_correct) == 0
                else "llm_more_optimistic_than_dkt"
            )
        strong_conflict = (
            conflict_direction == "llm_more_pessimistic_than_dkt"
            and mastery >= 0.70
            and boundary_risk != "high"
            and confidence < 0.75
        ) or (
            conflict_direction == "llm_more_optimistic_than_dkt"
            and mastery <= 0.30
            and history_level != "high"
            and confidence < 0.75
        )

        return {
            "module": "dkt_prompt_conditioning_diagnostics",
            "method": "prompt_only_no_posthoc_override",
            "kt_state_probability": round(mastery, 6),
            "kt_state_predicted_response": dkt_predicted,
            "llm_response": int(llm_correct),
            "final_response": int(llm_correct),
            "overrode_llm": False,
            "agreement": int(llm_correct) == dkt_predicted,
            "conflict_direction": conflict_direction,
            "strong_conflict": strong_conflict,
            "answer_confidence": round(confidence, 6),
            "cognitive_route": route,
            "boundary_risk": boundary_risk,
            "item_demand": item_demand,
            "tendency_band": tendency_band,
            "history_level": history_level,
            "p_correct_anchor": _safe_float((latent_state or {}).get("p_correct_anchor"), default=0.5),
            "knowledge_access": str((latent_state or {}).get("knowledge_access", "unknown")),
            "execution_quality": str((latent_state or {}).get("execution_quality", "unknown")),
        }

    def _dkt_raw_probability(self, uid: str, cid: int) -> float | None:
        if getattr(self, "mikt_proficiency", None) is not None and self.mikt_proficiency.available():
            value = self.mikt_proficiency.value(uid, cid)
            if value is not None:
                return min(1.0, max(0.0, float(value)))
        if not self.dkt_proficiency.available():
            return None
        value = self.dkt_proficiency.value(uid, cid)
        if value is None:
            return None
        return min(1.0, max(0.0, float(value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _logit(probability: float) -> float:
    clipped = min(0.999, max(0.001, probability))
    return math.log(clipped / (1.0 - clipped))


def _three_band(value: float, low: float, high: float) -> str:
    if value >= high:
        return "high"
    if value >= low:
        return "medium"
    return "low"


def _visible_probability_components(
    components: dict[str, Any],
    include_proficiency: bool,
) -> dict[str, Any]:
    visible = dict(components)
    if not include_proficiency:
        for key in [
            "mastery",
            "mastery_source",
            "irt_probability",
            "irt_theta",
            "irt_beta",
        ]:
            visible.pop(key, None)
    return visible

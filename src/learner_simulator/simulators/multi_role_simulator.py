from __future__ import annotations

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
    build_cognitive_profile_view,
    build_four_tier_response_record,
    build_response_prompt,
)
from learner_simulator.four_tier import (
    assess_four_tier_response,
    compare_answers,
    parse_answer_only_response,
)
from learner_simulator.item_conditioned_ability import build_item_conditioned_ability
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
    """Multi-role learner simulator with three explicit research modules.

    The command name remains ``multi-role`` for compatibility. Internally the
    implementation is organized as a compact educational simulation pipeline:

    1. Learner Profile Encoder: stable learner-level traits from observed
       history.
    2. Item-conditioned Evidence Encoder: current-item knowledge alignment,
       practice alignment, demand fit, and transfer burden.
    3. Four-tier Response Simulator: learner-level answer generation; external
       diagnostic scoring is used only for evaluation.
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
        include_item_conditioned_ability: bool = True,
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
            qmeta_for_modules = dict(qmeta)
            qmeta_for_modules.setdefault("qid", qid)
            qmeta_for_modules.setdefault("cid", cid)
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
            cognitive_profile_view = build_cognitive_profile_view(
                profile_context=profile_context,
            )
            profile_system_prompt = (
                build_profile_system_prompt(
                    {"learner_profile_evidence": cognitive_profile_view}
                )
                if include_profile
                else (
                    "You are simulating one high school student's first independent attempt. "
                    "Use only the reduced evidence provided by the current module prompts."
                )
            )
            learner_profile_encoder = self._build_learner_profile_encoder(
                profile_context=profile_context,
                memory_context=memory_context,
                behavior_factors=active_behavior,
                cognitive_profile_view=cognitive_profile_view,
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
                "learner_profile_evidence": cognitive_profile_view,
                "learner_profile_encoder": learner_profile_encoder,
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
                    "learner_profile_encoder": include_profile,
                    "item_conditioned_evidence_encoder": include_item_conditioned_ability,
                    "four_tier_response_simulation": True,
                    "details": {
                        "profile": include_profile,
                        "memory": include_memory,
                        "proficiency": include_proficiency,
                        "behavior": include_behavior,
                        "cognitive_profile": include_cognitive_profile,
                        "ability_profile": include_ability_profile
                        and bool(profile_context.get("ability_profile")),
                        "item_conditioned_evidence": include_item_conditioned_ability,
                        "four_tier": True,
                    },
                },
            }

            if call_llm:
                if llm_config is None:
                    raise ValueError("llm_config is required when call_llm=True")
                item_conditioned_ability = (
                    build_item_conditioned_ability(
                        question=qmeta_for_modules,
                        profile_context=profile_context,
                        memory_context=memory_context,
                        proficiency=proficiency,
                        behavior_factors=active_behavior,
                        tendency_calibration=tendency_calibration,
                    )
                    if include_item_conditioned_ability
                    else {
                        "module": "item_conditioned_ability",
                        "ablated": True,
                        "response_guidance": (
                            "Item-conditioned ability evidence is removed for this ablation."
                        ),
                    }
                )
                response_prompt = build_response_prompt(
                    question=qmeta,
                    short_memory=memory_context.get("short_memory", []),
                    long_memory=memory_context.get("long_memory", {}),
                    concept_options=concept_options,
                    behavior_factors=active_behavior,
                    cognitive_profile=cognitive_profile_view,
                    item_conditioned_ability=item_conditioned_ability,
                    response_format=response_format,
                )
                output_contract = (
                    "Four-tier learner response"
                    if response_format == "four_tier"
                    else "answer-only learner response"
                )
                response_system_prompt = (
                    f"{profile_system_prompt}\n\n"
                    "You are the Response Agent. Follow the learner evidence profile, "
                    "item-conditioned ability profile, and four-tier output contract. "
                    f"Return only the required {output_contract}."
                )
                if include_prompt:
                    step_result["response_agent_prompt"] = response_prompt
                    step_result["response_agent_system_prompt"] = response_system_prompt
                step_result["item_conditioned_ability"] = item_conditioned_ability
                step_result["simulation_tasks"] = self._build_simulation_tasks(
                    true_concept=true_concept,
                    concept_options=concept_options,
                    learner_profile_encoder=learner_profile_encoder,
                    item_conditioned_ability=item_conditioned_ability,
                    action=None,
                    four_tier=None,
                )
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
                            tendency_calibration=tendency_calibration,
                            item_conditioned_ability=item_conditioned_ability,
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
                    step_result["simulation_tasks"] = self._build_simulation_tasks(
                        true_concept=true_concept,
                        concept_options=concept_options,
                        learner_profile_encoder=learner_profile_encoder,
                        item_conditioned_ability=item_conditioned_ability,
                        action=action,
                        four_tier=four_tier,
                    )
                    if four_tier is not None:
                        step_result["four_tier_assessment"] = four_tier
                        four_tier_record = build_four_tier_response_record(
                            action=action,
                            four_tier=four_tier,
                            item_conditioned_ability=item_conditioned_ability,
                        )
                        step_result["four_tier_response_module"] = four_tier_record
                        step_result["four_tier_response_simulation"] = four_tier_record
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
            mastery_for_update = state.get_mastery(cid, 0.5)
            state.update(cid, feedback_response, learning_rate=self.learning_rate)
            updated_mastery = state.get_mastery(cid, 0.5)
            step_result["state_evolution"] = self._build_state_evolution_task(
                cid=cid,
                feedback_response=feedback_response,
                feedback_mode=feedback_mode,
                mastery_before=mastery_for_update,
                mastery_after=updated_mastery,
            )
            if isinstance(step_result.get("simulation_tasks"), dict):
                step_result["simulation_tasks"]["task4_state_evolution"] = (
                    step_result["state_evolution"]
                )
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

    @staticmethod
    def _build_simulation_tasks(
        true_concept: str,
        concept_options: list[str],
        learner_profile_encoder: dict[str, Any],
        item_conditioned_ability: dict[str, Any],
        action: dict[str, Any] | None,
        four_tier: dict[str, Any] | None,
    ) -> dict[str, Any]:
        selected_concept = action.get("identified_concept") if action else None
        return {
            "task1_learner_state_inference": {
                "objective": "infer stable learner state from observed history",
                "output": learner_profile_encoder,
                "evaluation": "indirect_profile_and_history_consistency",
            },
            "task2_item_conditioned_activation": {
                "objective": "infer how the current item activates learner knowledge and ability",
                "true_concept": true_concept,
                "concept_options": concept_options,
                "selected_concept": selected_concept,
                "concept_match": (
                    _concept_match(selected_concept, true_concept)
                    if selected_concept is not None
                    else None
                ),
                "item_conditioned_ability": item_conditioned_ability,
                "kt_decision_anchor": item_conditioned_ability.get("kt_decision_anchor"),
                "evaluation": "concept_label_and_kt_anchor_consistency",
            },
            "task3_four_tier_response_generation": {
                "objective": "generate learner answer, reasoning, and metacognitive confidence",
                "attempt": action.get("attempt") if action else None,
                "student_answer": action.get("student_answer") if action else None,
                "student_reasoning": action.get("student_reasoning") if action else None,
                "answer_confidence": action.get("answer_confidence") if action else None,
                "reasoning_confidence": (
                    action.get("reasoning_confidence") if action else None
                ),
                "answer_correct": four_tier.get("answer_correct") if four_tier else None,
                "diagnosis": four_tier.get("diagnosis") if four_tier else None,
                "evaluation": "external_answer_scoring_and_confidence_calibration",
            },
        }

    @staticmethod
    def _build_state_evolution_task(
        cid: int,
        feedback_response: int,
        feedback_mode: str,
        mastery_before: float,
        mastery_after: float,
    ) -> dict[str, Any]:
        return {
            "objective": "update learner state after the current interaction",
            "cid": cid,
            "feedback_mode": feedback_mode,
            "feedback_response": feedback_response,
            "mastery_before": round(float(mastery_before), 6),
            "mastery_after": round(float(mastery_after), 6),
            "mastery_delta": round(float(mastery_after) - float(mastery_before), 6),
            "evaluation": "trajectory_distribution_and_future_behavior_consistency",
        }

    @staticmethod
    def _build_learner_profile_encoder(
        profile_context: dict[str, Any],
        memory_context: dict[str, Any],
        behavior_factors: dict[str, float] | None,
        cognitive_profile_view: dict[str, Any],
    ) -> dict[str, Any]:
        long_memory = memory_context.get("long_memory") or {}
        return {
            "module": "learner_profile_encoder",
            "stable_profile": cognitive_profile_view,
            "history_exposure": profile_context.get("history_summary"),
            "memory_summary": {
                "short_memory_count": len(memory_context.get("short_memory", [])),
                "reinforced_memory_count": len(
                    long_memory.get("significant_facts", [])
                ),
                "current_concept_memory": long_memory.get("current_concept"),
            },
            "non_cognitive_state": behavior_factors,
        }

    @staticmethod
    def _dkt_alignment_diagnostics(
        llm_correct: int,
        answer_confidence: float,
        components: dict[str, Any],
        tendency_calibration: dict[str, Any] | None,
        item_conditioned_ability: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Record LLM-vs-DKT alignment without changing the LLM outcome."""

        mastery = _safe_float(components.get("mastery"), default=0.5)
        dkt_predicted = int(mastery >= 0.5)
        tendency_band = str((tendency_calibration or {}).get("band", "mixed"))
        history_level = str((tendency_calibration or {}).get("history_level", "unknown"))
        activation = item_conditioned_ability or {}
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
            and activation.get("activation_level") != "limited"
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
            "item_conditioned_ability": {
                "activation_level": activation.get("activation_level"),
                "ability_expression": activation.get("ability_expression"),
                "knowledge_alignment": activation.get("knowledge_alignment"),
                "practice_alignment": activation.get("practice_alignment"),
                "demand_alignment": activation.get("demand_alignment"),
                "memory_support": activation.get("memory_support"),
                "transfer_burden": activation.get("transfer_burden"),
            },
            "tendency_band": tendency_band,
            "history_level": history_level,
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


def _concept_match(selected: Any, true_concept: str) -> bool:
    selected_text = str(selected or "").strip()
    true_text = str(true_concept or "").strip()
    if not selected_text or not true_text:
        return False
    return selected_text == true_text


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

from __future__ import annotations

import time
from typing import Any

from learner_simulator.action import parse_agent_response
from learner_simulator.agent4edu_prompt import build_profile_system_prompt, proficiency_context
from learner_simulator.data import clean_sequence, sequence_row_from_steps
from learner_simulator.dneuralcdm import DNeuralCDMResponsePredictor
from learner_simulator.educational_multi_agent_prompt import (
    build_cognitive_profile_view,
    build_four_tier_response_record,
    build_response_prompt,
)
from learner_simulator.four_tier import (
    assess_four_tier_response,
    compare_answers,
    parse_answer_only_response,
    parse_reduced_response,
)
from learner_simulator.historical_reflection import build_historical_reflective_calibration
from learner_simulator.irt_evidence import build_irt_ability_item_evidence
from learner_simulator.item_conditioned_ability import build_item_conditioned_ability
from learner_simulator.llm import call_openai_compatible_chat
from learner_simulator.simulators.llm_simulator import (
    _without_ability_profile,
    _without_cognitive_profile,
)
from learner_simulator.simulators.random_simulator import (
    RandomLearnerSimulator,
    build_sequence_summary,
)


class MultiRoleLearnerSimulator(RandomLearnerSimulator):
    """NCDM-grounded learner simulator with explicit, ablatable evidence modules."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.dneuralcdm_response_predictor = DNeuralCDMResponsePredictor(
            self.dneuralcdm_checkpoint_path
        )

    def summary(self) -> dict[str, Any]:
        result = super().summary()
        result["ncdm_checkpoint_grounding"] = {
            "available": self.dneuralcdm_response_predictor.available(),
            "checkpoint": str(self.dneuralcdm_checkpoint_path),
            "checkpoint_sha256": self.dneuralcdm_response_predictor.checkpoint_sha256,
            "prompt_evidence": "concept_mastery_only",
            "item_response_probability_use": "independent_baseline_evaluation_only",
        }
        return result

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
        include_cognitive_profile: bool = True,
        include_ability_profile: bool = True,
        include_item_conditioned_ability: bool = True,
        include_irt_evidence: bool = True,
        include_ncdm_evidence: bool = True,
        include_historical_reflection: bool = True,
        include_dynamic_state_evolution: bool = True,
        feedback_mode: str = "rollout",
    ) -> dict[str, Any]:
        if response_format not in {"four_tier", "reduced_response", "answer_only"}:
            raise ValueError(f"Unsupported response_format: {response_format}")
        if feedback_mode not in {"rollout", "teacher-forcing"}:
            raise ValueError(f"Unsupported feedback_mode: {feedback_mode}")
        if call_llm and llm_config is None:
            raise ValueError("llm_config is required when call_llm=True")
        uid = str(row["uid"])
        if include_ncdm_evidence:
            self._require_ncdm(uid)
        state, memory = self.initialize_from_history(uid, history_row, questions)
        response_prefix = list(clean_sequence(history_row)) if history_row else []
        ncdm_state = (
            self._initialize_ncdm_runtime_state(response_prefix)
            if include_ncdm_evidence
            else None
        )
        profile_context = self._profile_context(
            uid,
            include_profile=include_profile,
            include_cognitive_profile=include_cognitive_profile,
            include_ability_profile=include_ability_profile,
        )
        historical_reflection = self._build_historical_reflection(
            history_row=history_row,
            questions=questions,
            enabled=include_historical_reflection,
            include_ncdm_evidence=include_ncdm_evidence,
            include_profile_evidence=include_profile,
        )

        target_sequence = clean_sequence(row)
        simulated_steps: list[dict[str, Any]] = []
        for step in target_sequence:
            step_start = time.perf_counter()
            qid = int(step["qid"])
            cid = int(step["cid"])
            real_response = int(step["response"])
            qmeta = dict(questions.get(str(qid), {}))
            qmeta.setdefault("qid", qid)
            qmeta.setdefault("cid", cid)

            components = self.probability_components(uid, qid, cid, state)
            dynamic_mastery = float(components.get("mastery", 0.5))
            concept_mastery = (
                float(ncdm_state.get(cid, dynamic_mastery))
                if ncdm_state is not None
                else dynamic_mastery
            )
            response_probability = (
                self._ncdm_response_probability(response_prefix, qid, cid)
                if include_ncdm_evidence
                else None
            )
            current_state_probability = (
                float(response_probability)
                if response_probability is not None
                else concept_mastery
            )
            fallback_response = int(current_state_probability >= 0.5)

            routes = qmeta.get("kc_routes") or []
            true_concept = str(routes[0]) if routes else str(cid)
            concept_options = self.concept_options(
                true_concept=true_concept,
                seed=self.irt_model.seed + qid + int(step.get("position") or 0),
            )
            memory_context = (
                memory.snapshot(cid, routes, concept_mastery)
                if include_memory
                else {"short_memory": [], "long_memory": {}}
            )
            # NCDM's item-response probability is retained in the trace for baseline
            # evaluation. Prompt-facing modules consume only latent concept mastery.
            proficiency = proficiency_context(true_concept, concept_mastery)
            proficiency.update(
                {
                    "source": (
                        "dneuralcdm"
                        if include_ncdm_evidence
                        else "observed_history_dynamic_state"
                    ),
                    "concept_mastery_value": round(concept_mastery, 6),
                    "concept_mastery_source": (
                        "dneuralcdm_latent_state"
                        if include_ncdm_evidence
                        else "observed_history_dynamic_state"
                    ),
                }
            )
            irt_evidence = (
                build_irt_ability_item_evidence(self.irt_model, uid, qid)
                if include_irt_evidence
                else None
            )
            profile_view = (
                build_cognitive_profile_view(profile_context)
                if include_profile
                else {"module": "learner_state_profile", "ablated": True}
            )
            profile_encoder = (
                self._build_learner_profile_encoder(profile_context, profile_view)
                if include_profile
                else {"module": "learner_state_profile", "ablated": True}
            )
            item_integration = (
                build_item_conditioned_ability(
                    question=qmeta,
                    profile_context=profile_context,
                    memory_context=memory_context,
                    proficiency=proficiency,
                    irt_evidence=irt_evidence,
                    historical_reflection=historical_reflection,
                    include_profile_evidence=include_profile,
                )
                if include_item_conditioned_ability
                else {"module": "item_conditioned_integration", "ablated": True}
            )

            step_result: dict[str, Any] = {
                "uid": uid,
                "source": "simulated",
                "step_index": step.get("position"),
                "timestamp": step.get("timestamp"),
                "qid": qid,
                "cid": cid,
                "real_response": real_response,
                "simulated_response": fallback_response,
                "prediction_valid": not call_llm,
                "prediction_source": "ncdm_threshold_fallback",
                "feedback_mode": feedback_mode,
                "history_state_components": (
                    components if not include_ncdm_evidence else None
                ),
                "ncdm_evidence_in_prompt": include_ncdm_evidence,
                "ncdm_correct_probability": (
                    round(float(response_probability), 6)
                    if response_probability is not None
                    else None
                ),
                "ncdm_predicted_response": (
                    int(response_probability >= 0.5)
                    if response_probability is not None
                    else None
                ),
                "ncdm_concept_mastery": (
                    round(concept_mastery, 6) if include_ncdm_evidence else None
                ),
                "history_state_probability": (
                    round(current_state_probability, 6)
                    if not include_ncdm_evidence
                    else None
                ),
                "irt_evidence_in_prompt": include_irt_evidence,
                "irt_ability_difficulty_evidence": irt_evidence,
                "historical_reflective_calibration": historical_reflection,
                "question_type": qmeta.get("type"),
                "content": qmeta.get("content"),
                "options": qmeta.get("options"),
                "answer": qmeta.get("answer"),
                "analysis": qmeta.get("analysis"),
                "content_preview": str(qmeta.get("content", ""))[:80],
                "kc_routes": routes,
                "learner_profile": profile_context,
                "memory_context": memory_context,
                "learner_profile_evidence": profile_view,
                "learner_profile_encoder": profile_encoder,
                "item_conditioned_integration": item_integration,
                "concept_options": concept_options,
                "enabled_modules": {
                    "learner_state_profile": include_profile,
                    "ncdm_state_evidence": include_ncdm_evidence,
                    "irt_ability_difficulty": include_irt_evidence,
                    "observed_history_replay": include_historical_reflection,
                    "item_conditioned_integration": include_item_conditioned_ability,
                    "four_tier_response": response_format == "four_tier",
                    "dynamic_state_evolution": include_dynamic_state_evolution,
                },
            }

            action: dict[str, Any] | None = None
            four_tier: dict[str, Any] | None = None
            rendered_answer_correct: bool | None = None
            if call_llm:
                response_prompt = build_response_prompt(
                    question=qmeta,
                    short_memory=memory_context.get("short_memory", []),
                    long_memory=memory_context.get("long_memory", {}),
                    concept_options=concept_options,
                    proficiency=proficiency,
                    historical_reflection=historical_reflection,
                    item_conditioned_ability=item_integration,
                    irt_evidence=irt_evidence,
                    response_format=response_format,
                    include_profile_evidence=include_profile,
                    include_item_conditioned_evidence=include_item_conditioned_ability,
                    include_historical_reflection=include_historical_reflection,
                    include_irt_evidence=include_irt_evidence,
                    include_ncdm_evidence=include_ncdm_evidence,
                )
                system_prompt = self._profile_system_prompt(profile_view, include_profile)
                if include_prompt:
                    step_result["response_agent_prompt"] = response_prompt
                    step_result["response_agent_system_prompt"] = system_prompt
                try:
                    raw = call_openai_compatible_chat(
                        llm_config or {},
                        response_prompt,
                        system_prompt=system_prompt,
                    )
                    # Preserve malformed model output for post-run diagnosis;
                    # parsing failures must not erase the evidence needed to repair them.
                    step_result["llm_result"] = raw
                    if response_format == "four_tier":
                        action = parse_agent_response(raw)
                    elif response_format == "reduced_response":
                        action = parse_reduced_response(raw)
                    else:
                        action = parse_answer_only_response(raw)
                    if action is None or (
                        response_format != "answer_only"
                        and action.get("learner_correct") is None
                    ):
                        raise ValueError(
                            f"Response Agent output did not match {response_format} contract"
                        )
                    action["raw"] = raw
                    if response_format == "four_tier":
                        four_tier = assess_four_tier_response(
                            action,
                            reference_answers=qmeta.get("answer"),
                            reference_reasoning=str(qmeta.get("analysis", "")),
                        )
                    rendered_answer_correct = compare_answers(
                        action.get("student_answer"),
                        qmeta.get("answer"),
                    )
                    if response_format == "answer_only":
                        if rendered_answer_correct is None:
                            raise ValueError(
                                "Answer-only response could not be scored against the reference answer"
                            )
                        final_correct = int(rendered_answer_correct)
                        decision_source = "externally_scored_student_answer"
                        action["learner_correct"] = final_correct
                    else:
                        final_correct = int(action["learner_correct"])
                        decision_source = "learner_correct"
                    action["simulated_correct"] = final_correct
                    action["response_decision_source"] = decision_source
                    step_result.update(
                        {
                            "simulated_response": final_correct,
                            "prediction_valid": True,
                            "prediction_source": (
                                "llm_rendered_answer"
                                if response_format == "answer_only"
                                else "llm_learner_correct"
                            ),
                            "llm_parsed_action": action,
                            "agent_action": action,
                            "response_learner_correct": final_correct,
                            "rendered_answer_correct": rendered_answer_correct,
                            "response_decision_source": decision_source,
                        }
                    )
                    if four_tier is not None:
                        step_result["four_tier_assessment"] = four_tier
                        step_result["four_tier_response_module"] = (
                            build_four_tier_response_record(
                                action,
                                four_tier,
                                item_conditioned_ability=item_integration,
                                irt_evidence=irt_evidence,
                            )
                        )
                except Exception as exc:
                    step_result["llm_error"] = f"{exc.__class__.__name__}: {exc}"
                    step_result["prediction_valid"] = False
                    step_result["prediction_source"] = "invalid_llm_fallback"

            step_result["simulation_tasks"] = self._build_simulation_tasks(
                true_concept=true_concept,
                profile_encoder=profile_encoder,
                item_integration=item_integration,
                action=action,
                four_tier=four_tier,
                rendered_answer_correct=rendered_answer_correct,
            )
            step_result["llm_elapsed_seconds"] = round(
                time.perf_counter() - step_start,
                3,
            )
            if progress_callback is not None:
                progress_callback(step_result)

            mastery_before = concept_mastery
            feedback_response = (
                real_response
                if feedback_mode == "teacher-forcing"
                else int(step_result["simulated_response"])
            )
            step_result["feedback_response"] = feedback_response
            # Keep the independent NCDM response baseline and target memory on the
            # same observed prefix in every ablation. The dynamic-state switch only
            # controls whether the prompt-facing latent knowledge state is refreshed.
            response_prefix.append({**step, "response": feedback_response})
            if include_dynamic_state_evolution:
                state.update(cid, feedback_response, learning_rate=self.learning_rate)
                if ncdm_state is not None:
                    refreshed = self.dneuralcdm_response_predictor.knowledge_state(
                        response_prefix
                    )
                    if refreshed:
                        ncdm_state.clear()
                        ncdm_state.update(refreshed)
                    mastery_after = float(ncdm_state.get(cid, mastery_before))
                    state_source = "dneuralcdm_latent_state_update"
                else:
                    mastery_after = float(state.get_mastery(cid, 0.5))
                    state_source = "observed_history_dynamic_update"
                step_result["state_evolution"] = self._build_state_evolution_task(
                    cid=cid,
                    feedback_response=feedback_response,
                    feedback_mode=feedback_mode,
                    mastery_before=mastery_before,
                    mastery_after=mastery_after,
                    state_source=state_source,
                )
            else:
                step_result["state_evolution"] = {
                    "module": "dynamic_state_evolution",
                    "ablated": True,
                    "feedback_mode": feedback_mode,
                    "feedback_response": feedback_response,
                    "concept_id": cid,
                    "mastery_before": round(mastery_before, 6),
                    "mastery_after": round(mastery_before, 6),
                    "mastery_delta": 0.0,
                    "state_source": "frozen_history_state",
                    "target_feedback_consumed": True,
                    "knowledge_state_feedback_consumed": False,
                    "memory_feedback_consumed": True,
                }
            memory_record = dict(step_result)
            if feedback_mode == "teacher-forcing":
                memory_record["source"] = "teacher_forced_feedback"
                memory_record["model_simulated_response"] = step_result[
                    "simulated_response"
                ]
                memory_record["simulated_response"] = feedback_response
            memory.observe(memory_record)
            step_result["simulation_tasks"]["task4_dynamic_state_evolution"] = (
                step_result["state_evolution"]
            )
            simulated_steps.append(step_result)

        summary = build_sequence_summary(uid, simulated_steps)
        summary["history_interactions"] = len(clean_sequence(history_row)) if history_row else 0
        summary["target_interactions"] = len(simulated_steps)
        summary["feedback_mode"] = feedback_mode
        summary["knowledge_state_artifact"] = (
            {
                "model": "dneuralcdm",
                "checkpoint": str(self.dneuralcdm_checkpoint_path),
                "checkpoint_sha256": self.dneuralcdm_response_predictor.checkpoint_sha256,
            }
            if include_ncdm_evidence
            else {"model": None, "ablated": True}
        )
        summary["steps"] = simulated_steps
        return summary

    def _profile_context(
        self,
        uid: str,
        include_profile: bool,
        include_cognitive_profile: bool,
        include_ability_profile: bool,
    ) -> dict[str, Any]:
        if not include_profile:
            return {"uid": uid}
        context = self.get_profile(uid).to_context()
        if not include_cognitive_profile:
            context = _without_cognitive_profile(context)
        if not include_ability_profile:
            context = _without_ability_profile(context)
        return context

    @staticmethod
    def _profile_system_prompt(
        profile_view: dict[str, Any],
        include_profile: bool,
    ) -> str:
        if not include_profile:
            return (
                "You are simulating one learner's first independent attempt. "
                "Use only evidence explicitly present in the user prompt."
            )
        return build_profile_system_prompt({"learner_profile_evidence": profile_view})

    @staticmethod
    def _build_learner_profile_encoder(
        profile_context: dict[str, Any],
        profile_view: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "module": "learner_state_profile",
            "stable_profile": profile_view,
            "history_exposure": profile_context.get("history_summary"),
        }

    @staticmethod
    def _build_simulation_tasks(
        true_concept: str,
        profile_encoder: dict[str, Any],
        item_integration: dict[str, Any],
        action: dict[str, Any] | None,
        four_tier: dict[str, Any] | None,
        rendered_answer_correct: bool | None,
    ) -> dict[str, Any]:
        selected = action.get("identified_concept") if action else None
        return {
            "task1_learner_state_profile": {
                "objective": "estimate stable learner state from observed history",
                "output": profile_encoder,
                "ablated": bool(profile_encoder.get("ablated")),
            },
            "task2_item_conditioned_integration": {
                "objective": "integrate available learner, memory, knowledge, and item evidence",
                "true_concept": true_concept,
                "selected_concept": selected,
                "concept_match": (
                    _concept_match(selected, true_concept) if selected is not None else None
                ),
                "output": item_integration,
                "ablated": bool(item_integration.get("ablated")),
            },
            "task3_learner_response_generation": {
                "objective": "generate the learner's first-attempt response",
                "learner_correct": action.get("learner_correct") if action else None,
                "student_answer": action.get("student_answer") if action else None,
                "student_reasoning": action.get("student_reasoning") if action else None,
                "answer_confidence": action.get("answer_confidence") if action else None,
                "reasoning_confidence": action.get("reasoning_confidence") if action else None,
                "answer_correct": (
                    four_tier.get("answer_correct")
                    if four_tier
                    else rendered_answer_correct
                ),
            },
        }

    @staticmethod
    def _build_state_evolution_task(
        cid: int,
        feedback_response: int,
        feedback_mode: str,
        mastery_before: float,
        mastery_after: float,
        state_source: str,
    ) -> dict[str, Any]:
        return {
            "module": "dynamic_state_evolution",
            "objective": "update learner state after the current interaction",
            "concept_id": cid,
            "state_source": state_source,
            "feedback_mode": feedback_mode,
            "feedback_response": feedback_response,
            "mastery_before": round(float(mastery_before), 6),
            "mastery_after": round(float(mastery_after), 6),
            "mastery_delta": round(float(mastery_after) - float(mastery_before), 6),
            "target_feedback_consumed": feedback_mode == "teacher-forcing",
        }

    def _require_ncdm(self, uid: str) -> None:
        if not self.dneuralcdm_response_predictor.available():
            raise RuntimeError(
                "Multi-role Full requires a trained --dneuralcdm-checkpoint."
            )
        if not self.dneuralcdm_response_predictor.is_user_held_out(uid):
            raise RuntimeError(
                "Multi-role Full requires a leakage-safe NCDM checkpoint whose "
                f"training metadata proves uid={uid} was excluded from training. "
                "Retrain with --cohort-file using the current training script."
            )

    def _initialize_ncdm_runtime_state(
        self,
        history_steps: list[dict[str, Any]],
    ) -> dict[int, float]:
        values = self.dneuralcdm_response_predictor.knowledge_state(history_steps)
        if not values:
            raise RuntimeError(
                "NCDM could not infer a learner state from the observed history."
            )
        return {
            int(cid): min(1.0, max(0.0, float(value)))
            for cid, value in values.items()
        }

    def _ncdm_response_probability(
        self,
        prefix_steps: list[dict[str, Any]],
        qid: int,
        cid: int,
    ) -> float | None:
        value = self.dneuralcdm_response_predictor.probability(
            prefix_steps=prefix_steps,
            target_qid=qid,
            target_cid=cid,
        )
        return None if value is None else min(1.0, max(0.0, float(value)))

    def _seed_state_from_external_proficiency(self, uid: str, state: Any) -> None:
        # Full owns one canonical NCDM state computed from the checkpoint. The
        # base history state remains independent for the no-NCDM ablation.
        return None

    def _mastery_source(self, uid: str, cid: int) -> str | None:
        return None

    def _build_historical_reflection(
        self,
        history_row: dict[str, str] | None,
        questions: dict[str, dict[str, Any]],
        enabled: bool,
        include_ncdm_evidence: bool,
        include_profile_evidence: bool,
    ) -> dict[str, Any]:
        if not enabled or history_row is None:
            return build_historical_reflective_calibration(
                history_row,
                [],
                include_profile_evidence=include_profile_evidence,
            )
        history = clean_sequence(history_row)
        if len(history) < 12:
            return build_historical_reflective_calibration(
                history_row,
                [],
                include_profile_evidence=include_profile_evidence,
            )

        replay_window = min(10, max(5, len(history) // 9))
        prefix = list(history[:-replay_window])
        replay = history[-replay_window:]
        replay_state, _ = self.initialize_from_history(
            str(history_row["uid"]),
            sequence_row_from_steps(str(history_row["uid"]), prefix),
            questions,
        )
        ncdm_state = (
            self.dneuralcdm_response_predictor.knowledge_state(prefix)
            if include_ncdm_evidence
            else None
        )
        records: list[dict[str, Any]] = []
        for step in replay:
            qid = int(step["qid"])
            cid = int(step["cid"])
            components = self.probability_components(
                str(history_row["uid"]), qid, cid, replay_state
            )
            probability = (
                self._ncdm_response_probability(prefix, qid, cid)
                if include_ncdm_evidence
                else None
            )
            if probability is None:
                probability = float(
                    (ncdm_state or {}).get(cid, components.get("mastery", 0.5))
                )
            qmeta = questions.get(str(qid), {})
            routes = qmeta.get("kc_routes") or []
            records.append(
                {
                    "uid": str(history_row["uid"]),
                    "qid": qid,
                    "cid": cid,
                    "concept": str(routes[0]) if routes else str(cid),
                    "predicted_probability": probability,
                    "prediction_source": (
                        "dneuralcdm_history_replay"
                        if include_ncdm_evidence
                        else "observed_history_dynamic_replay"
                    ),
                    "real_response": int(step["response"]),
                }
            )
            replay_state.update(cid, int(step["response"]), self.learning_rate)
            prefix.append(step)
            if include_ncdm_evidence:
                ncdm_state = self.dneuralcdm_response_predictor.knowledge_state(prefix)
        return build_historical_reflective_calibration(
            history_row,
            records,
            include_profile_evidence=include_profile_evidence,
        )


def _concept_match(selected: Any, true_concept: str) -> bool:
    return str(selected or "").strip() == str(true_concept or "").strip()

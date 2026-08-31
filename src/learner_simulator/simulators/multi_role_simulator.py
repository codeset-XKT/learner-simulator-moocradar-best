from __future__ import annotations

import time
from typing import Any

from learner_simulator.action import parse_agent_response
from learner_simulator.agent4edu_prompt import proficiency_context
from learner_simulator.data import clean_sequence
from learner_simulator.dneuralcdm import DNeuralCDMResponsePredictor
from learner_simulator.educational_multi_agent_prompt import (
    build_four_tier_response_record,
    build_response_prompt,
)
from learner_simulator.four_tier import (
    assess_four_tier_response,
    compare_answers,
    parse_answer_only_response,
    parse_reduced_response,
)
from learner_simulator.irt_evidence import build_irt_ability_item_evidence
from learner_simulator.llm import call_openai_compatible_chat
from learner_simulator.process_evidence import (
    build_process_consistency,
    build_state_item_alignment,
    build_traceable_evidence_repository,
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
        include_irt_evidence: bool = True,
        include_ncdm_evidence: bool = True,
        include_dynamic_state_evolution: bool = True,
        include_evidence_representation: bool = True,
        include_state_item_alignment: bool = True,
        feedback_mode: str = "rollout",
    ) -> dict[str, Any]:
        if response_format not in {"four_tier", "reduced_response"}:
            raise ValueError(
                "MultiRoleLearnerSimulator v6 requires LearnerCorrect; "
                f"unsupported response_format={response_format!r}"
            )
        if feedback_mode not in {"rollout", "teacher-forcing"}:
            raise ValueError(f"Unsupported feedback_mode: {feedback_mode}")
        if call_llm and llm_config is None:
            raise ValueError("llm_config is required when call_llm=True")
        uid = str(row["uid"])
        if include_ncdm_evidence:
            self._require_ncdm(uid)
        state, _legacy_memory = self.initialize_from_history(uid, history_row, questions)
        response_prefix = list(clean_sequence(history_row)) if history_row else []
        ncdm_state = (
            self._initialize_ncdm_runtime_state(response_prefix)
            if include_ncdm_evidence
            else None
        )
        evidence_repository = build_traceable_evidence_repository(
            history_row,
            questions,
        )
        evidence_repository_trace = {
            key: value
            for key, value in evidence_repository.items()
            if key != "events"
        }
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

            routes = [str(value) for value in (qmeta.get("kc_routes") or [])]
            # The formal simulator uses the first metadata route as the sole
            # current-item concept. The complete selected route remains a
            # hierarchy for evidence selection; additional metadata routes are
            # archived but never fused into state or prompting.
            true_concepts = [routes[0]] if routes else [str(cid)]
            true_concept = true_concepts[0]
            concept_options = (
                self.concept_options(
                    true_concept=true_concepts,
                    seed=self.irt_model.seed + qid + int(step.get("position") or 0),
                )
                if include_state_item_alignment
                else None
            )
            # v3 keeps complete observed history in a read-only evidence
            # repository. It never injects Agent4Edu-style short/long memory.
            memory_context = {
                "module": "legacy_agent_memory",
                "used": False,
                "short_memory": [],
                "long_memory": {},
            }
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
            learner_evidence_representation = (
                {
                    **evidence_repository_trace,
                    "ncdm_concept_state": {
                        str(key): round(float(value), 6)
                        for key, value in sorted((ncdm_state or {}).items())
                    },
                    "irt_learner_ability": (irt_evidence or {}).get("learner_theta"),
                    "learner_specific": True,
                }
                if include_evidence_representation
                else {
                    "module": "traceable_learner_evidence_representation",
                    "ablated": True,
                    "learner_specific": False,
                }
            )
            alignment_mastery = concept_mastery if include_evidence_representation else 0.5
            alignment_irt = dict(irt_evidence or {})
            if not include_evidence_representation:
                # Learner theta belongs to Module 1. Item beta remains because
                # it is a current-item property owned by Module 2.
                alignment_irt.pop("learner_theta", None)
            state_item_alignment = build_state_item_alignment(
                evidence_repository,
                qmeta,
                alignment_mastery,
                alignment_irt,
                include_evidence_representation=include_evidence_representation,
                include_alignment=include_state_item_alignment,
            )
            state_item_alignment["learner_evidence_available"] = bool(
                include_evidence_representation
            )
            if not include_evidence_representation:
                state_item_alignment["concept_mastery"] = None
                state_item_alignment["irt_ability"] = None
                state_item_alignment["selected_evidence"] = []
                state_item_alignment["selected_evidence_ids"] = []
                state_item_alignment["selected_evidence_summary"] = {
                    "support_count": 0,
                    "observed_correct_count": 0,
                    "observed_error_count": 0,
                    "observed_accuracy": None,
                    "observed_error_event_ids": [],
                }
                state_item_alignment["relevant_stable_patterns"] = []
            elif not include_state_item_alignment:
                values = [float(value) for value in (ncdm_state or {}).values()]
                state_item_alignment["concept_mastery"] = None
                state_item_alignment["current_concepts"] = []
                state_item_alignment["relevant_stable_patterns"] = []
                state_item_alignment["unaligned_learner_state_summary"] = {
                    "concept_count": len(values),
                    "mean_mastery": round(sum(values) / len(values), 6) if values else None,
                    "minimum_mastery": round(min(values), 6) if values else None,
                    "maximum_mastery": round(max(values), 6) if values else None,
                    "irt_learner_ability": (irt_evidence or {}).get("learner_theta"),
                    "policy": "learner_state_without_current_item_mapping",
                }
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
                "ncdm_evidence_in_prompt": bool(
                    include_ncdm_evidence and include_evidence_representation
                ),
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
                "question_type": qmeta.get("type"),
                "content": qmeta.get("content"),
                "options": qmeta.get("options"),
                "answer": qmeta.get("answer"),
                "analysis": qmeta.get("analysis"),
                "content_preview": str(qmeta.get("content", ""))[:80],
                "kc_routes": routes,
                "memory_context": memory_context,
                "traceable_learner_evidence_representation": learner_evidence_representation,
                "cognitive_state_item_alignment": state_item_alignment,
                "concept_options": concept_options,
                "enabled_modules": {
                    "traceable_learner_evidence_representation": include_evidence_representation,
                    "cognitive_state_item_alignment": include_state_item_alignment,
                    "ncdm_learner_state_in_module1": bool(
                        include_ncdm_evidence and include_evidence_representation
                    ),
                    "irt_learner_ability_in_module1": bool(
                        include_irt_evidence and include_evidence_representation
                    ),
                    "irt_item_difficulty_in_module2": include_irt_evidence,
                    "structured_response_generation": response_format == "four_tier",
                    "response_contract": response_format,
                    "dynamic_state_evolution": include_dynamic_state_evolution,
                    "agent_memory": False,
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
                    irt_evidence=irt_evidence,
                    response_format=response_format,
                    include_profile_evidence=include_profile,
                    include_irt_evidence=include_irt_evidence,
                    include_ncdm_evidence=(
                        include_ncdm_evidence and include_evidence_representation
                    ),
                    evidence_repository=None,
                    state_item_alignment=state_item_alignment,
                    include_evidence_representation=include_evidence_representation,
                    include_state_item_alignment=include_state_item_alignment,
                )
                system_prompt = self._profile_system_prompt()
                if include_prompt:
                    step_result["response_agent_prompt"] = response_prompt
                    step_result["response_agent_system_prompt"] = system_prompt
                try:
                    request_config = dict(llm_config or {})
                    raw = call_openai_compatible_chat(
                        request_config,
                        response_prompt,
                        system_prompt=system_prompt,
                    )
                    # Exactly one logical API call per item. Malformed output is
                    # archived and excluded; contract repair is local/offline.
                    step_result["llm_result"] = raw
                    if response_format == "four_tier":
                        action = parse_agent_response(raw)
                    elif response_format == "reduced_response":
                        action = parse_reduced_response(raw)
                    else:
                        action = parse_answer_only_response(raw)
                    contract_valid = action is not None and (
                        response_format == "answer_only"
                        or (
                            bool(str(action.get("student_answer", "")).strip())
                            and (
                                response_format != "four_tier"
                                or action.get("learner_correct") is not None
                            )
                        )
                    )
                    if not contract_valid:
                        raise ValueError(
                            "Response Agent output did not match "
                            f"{response_format} contract"
                        )
                    action["raw"] = raw
                    # LearnerCorrect is the LLM's behavioral prediction. The
                    # reference answer is used only to realize a submitted answer
                    # that is mechanically consistent with that prediction.
                    rendered_answer_correct, final_correct, decision_source = (
                        enforce_learner_correct_answer_consistency(action, qmeta)
                    )
                    # Score the final archived learner response, never the raw
                    # pre-materialization answer. Targeted repair uses this same
                    # order, so fresh and repaired steps have identical semantics.
                    if response_format == "four_tier":
                        four_tier = assess_four_tier_response(
                            action,
                            reference_answers=qmeta.get("answer"),
                            reference_reasoning=str(qmeta.get("analysis", "")),
                        )
                    step_result.update(
                        {
                            "simulated_response": final_correct,
                            "prediction_valid": True,
                            "prediction_source": "llm_learner_correct",
                            "llm_parsed_action": action,
                            "agent_action": action,
                            "response_learner_correct": final_correct,
                            "response_declared_learner_correct": action.get(
                                "declared_learner_correct"
                            ),
                            "learner_correct_answer_consistent": action[
                                "learner_correct_consistent_with_answer"
                            ],
                            "rendered_answer_correct": rendered_answer_correct,
                            "response_decision_source": decision_source,
                        }
                    )
                    if four_tier is not None:
                        # Module 2 is deterministic evidence. The response model
                        # may cite it but must not invent a self-justifying
                        # categorical StateAlignment before LearnerCorrect.
                        action["state_alignment"] = state_item_alignment
                        process_consistency = build_process_consistency(
                            action,
                            state_item_alignment,
                            true_concepts,
                        )
                        step_result["process_consistency"] = process_consistency
                        step_result["four_tier_assessment"] = four_tier
                        step_result["four_tier_response_module"] = (
                            build_four_tier_response_record(
                                action,
                                four_tier,
                                irt_evidence=irt_evidence,
                                state_item_alignment=state_item_alignment,
                                process_consistency=process_consistency,
                            )
                        )
                except Exception as exc:
                    step_result["llm_error"] = f"{exc.__class__.__name__}: {exc}"
                    # Retain the model output already captured above for failed
                    # diagnostic contracts as well; no retry can be debugged
                    # fairly when the rejected response is discarded.
                    if "llm_result" in step_result:
                        step_result["llm_raw_on_error"] = step_result["llm_result"]
                    step_result["prediction_valid"] = False
                    step_result["prediction_source"] = "invalid_llm_fallback"

            step_result["simulation_tasks"] = self._build_simulation_tasks(
                true_concepts=true_concepts,
                evidence_representation=step_result[
                    "traceable_learner_evidence_representation"
                ],
                state_item_alignment=state_item_alignment,
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
            # Keep the independent NCDM response baseline on the same observed
            # prefix in every ablation. No target response is written into an
            # Agent-style textual memory.
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
                    "module": "auditable_state_evolution",
                    "ablated": True,
                    "feedback_mode": feedback_mode,
                    "feedback_response": feedback_response,
                    "concept_id": cid,
                    "mastery_before": round(mastery_before, 6),
                    "mastery_after": round(mastery_before, 6),
                    "mastery_delta": 0.0,
                    "state_source": "frozen_history_state",
                    "target_feedback_consumed": feedback_mode == "teacher-forcing",
                    "knowledge_state_feedback_consumed": False,
                    "memory_feedback_consumed": False,
                    "llm_called": False,
                }
            step_result["simulation_tasks"]["module4_auditable_state_evolution"] = (
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
        summary["traceable_evidence_repository"] = evidence_repository
        summary["steps"] = simulated_steps
        return summary

    @staticmethod
    def _profile_system_prompt() -> str:
        return (
            "Simulate one learner's first independent attempt using only the "
            "traceable evidence and cognitive state explicitly present in the "
            "user prompt. Do not invent demographics, biography, reflection, "
            "memory, or unobserved traits. Produce observable structured process "
            "fields, not hidden chain-of-thought."
        )

    @staticmethod
    def _build_simulation_tasks(
        true_concepts: list[str],
        evidence_representation: dict[str, Any],
        state_item_alignment: dict[str, Any],
        action: dict[str, Any] | None,
        four_tier: dict[str, Any] | None,
        rendered_answer_correct: bool | None,
    ) -> dict[str, Any]:
        selected = action.get("identified_concept") if action else None
        return {
            "module1_traceable_learner_evidence_representation": {
                "objective": "derive provenance-preserving evidence from observed history",
                "output": evidence_representation,
                "ablated": bool(evidence_representation.get("ablated")),
            },
            "module2_cognitive_state_item_alignment": {
                "objective": "align cognitive state and bounded historical evidence with the current item",
                "output": state_item_alignment,
                # The authoritative reference set has exactly one entry: the
                # complete primary kc route used by the formal simulator.
                "true_concept": true_concepts[0] if true_concepts else None,
                "reference_concepts": list(true_concepts),
                "selected_concept": selected,
                "concept_match": (
                    _concept_match(selected, true_concepts) if selected is not None else None
                ),
            },
            "module3_structured_response_generation": {
                "objective": "generate an attributable structured learner response",
                "response_contract": "single_pass_process_verifiable_v6_label_conditioned_answer_realization" if four_tier is not None else "single_pass_direct_response_v6",
                "state_alignment": state_item_alignment,
                "evidence_refs": action.get("evidence_refs") if action else None,
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
            "module": "auditable_state_evolution",
            "objective": "update learner state after the current interaction",
            "concept_id": cid,
            "state_source": state_source,
            "feedback_mode": feedback_mode,
            "feedback_response": feedback_response,
            "mastery_before": round(float(mastery_before), 6),
            "mastery_after": round(float(mastery_after), 6),
            "mastery_delta": round(float(mastery_after) - float(mastery_before), 6),
            "target_feedback_consumed": feedback_mode == "teacher-forcing",
            "llm_called": False,
            "recomputable_from_saved_prefix_and_checkpoint": True,
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


def enforce_learner_correct_answer_consistency(
    action: dict[str, Any],
    question: dict[str, Any],
) -> tuple[bool, int, str]:
    """Make the rendered answer agree with the LLM's predicted outcome.

    LearnerCorrect is the response-level prediction. The answer key is a
    realization aid: a correct prediction is rendered as the reference answer,
    while an incorrect prediction retains the LLM's wrong answer whenever
    possible and otherwise uses a deterministic non-reference option.
    """
    declared = action.get("learner_correct")
    if declared not in (0, 1):
        raise ValueError("LearnerCorrect is required for label-conditioned rendering")
    reference = question.get("answer")
    original_answer = action.get("student_answer")
    original_scored = compare_answers(original_answer, reference)
    if original_scored is None:
        raise ValueError("StudentAnswer could not be scored against the reference answer")

    materialization = "llm_answer_retained"
    if declared == 1:
        # Preserve the native reference representation (including multi-answer
        # questions) so deterministic evaluation recognizes the correct answer.
        action["student_answer"] = reference
        materialization = "reference_answer_for_predicted_correct"
    elif original_scored is True:
        action["student_answer"] = _first_incorrect_option(question, reference)
        materialization = "fallback_incorrect_answer_for_predicted_incorrect"

    rendered = compare_answers(action.get("student_answer"), reference)
    if rendered is None or int(rendered) != int(declared):
        raise ValueError("Could not render StudentAnswer consistent with LearnerCorrect")

    action["declared_learner_correct"] = int(declared)
    action["student_answer_before_consistency_rendering"] = original_answer
    action["student_answer_materialization"] = materialization
    action["learner_correct_consistent_with_answer"] = True
    action["learner_correct"] = int(declared)
    action["simulated_correct"] = int(declared)
    source = "llm_learner_correct_conditioned_answer_realization"
    action["response_decision_source"] = source
    return bool(rendered), int(declared), source


def _first_incorrect_option(question: dict[str, Any], reference: Any) -> Any:
    options = question.get("options")
    candidates: list[Any] = []
    if isinstance(options, dict):
        candidates.extend(options.keys())
    elif isinstance(options, (list, tuple)):
        for option in options:
            text = str(option).strip()
            # Datasets commonly serialize choices as ``A. text``. The label
            # is the submitted answer expected by the evaluator.
            if len(text) >= 1 and text[0].upper() in "ABCDEFGH":
                candidates.append(text[0].upper())
            candidates.append(option)
    for candidate in candidates:
        if compare_answers(candidate, reference) is not False:
            continue
        # A full option string such as ``A. correct`` does not compare equal
        # to the label ``A`` under every dataset adapter. Do not accidentally
        # treat that same labelled option as a distractor.
        text = str(candidate).strip()
        if text and text[0].upper() in "ABCDEFGH":
            if compare_answers(text[0].upper(), reference) is True:
                continue
        return candidate
    # Free-response fallback: used only when the model declared an incorrect
    # outcome yet emitted the reference and no distractor is stored.
    return "__SIMULATED_INCORRECT_ANSWER__"

def _concept_match(selected: Any, true_concepts: list[str] | tuple[str, ...]) -> bool:
    value = str(selected or "").strip()
    return bool(value) and value in {str(concept).strip() for concept in true_concepts}

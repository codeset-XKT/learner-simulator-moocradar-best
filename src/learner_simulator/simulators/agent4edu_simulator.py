from __future__ import annotations

import time
from typing import Any

from learner_simulator.agent4edu_baseline import (
    build_agent4edu_action_prompt,
    build_agent4edu_profile_prompt,
    build_agent4edu_reflection_prompt,
    parse_agent4edu_action,
)
from learner_simulator.agent4edu_prompt import proficiency_context
from learner_simulator.data import clean_sequence
from learner_simulator.dneuralcdm import DNeuralCDMResponsePredictor
from learner_simulator.llm import call_openai_compatible_chat
from learner_simulator.simulators.random_simulator import (
    RandomLearnerSimulator,
    build_sequence_summary,
)


class Agent4EduBaselineSimulator(RandomLearnerSimulator):
    """Isolated Agent4Edu reproduction baseline.

    It follows the official four-task action prompt, exposes the reference
    answer and analysis, uses Task4 as the predicted response, and performs a
    post-response reflection call. As in the official implementation, the
    action is followed by the *observed* exercise outcome, which updates memory
    and later-step state; Task4 itself is generated before that feedback.
    Its current-concept proficiency is derived online from the configured
    DNeuralCDM checkpoint; dynamic mastery is retained only as an explicit
    fallback when a checkpoint cannot encode an item or concept.
    """

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
            "prompt_evidence": "current_concept_latent_mastery",
            "item_response_probability_use": "trace_only_not_prompt_label",
        }
        return result

    def simulate_sequence(
        self,
        row: dict[str, str],
        questions: dict[str, dict[str, Any]],
        llm_config: dict[str, Any],
        history_row: dict[str, str] | None = None,
        include_prompt: bool = False,
        progress_callback: Any | None = None,
        enable_reflection: bool = True,
    ) -> dict[str, Any]:
        uid = row["uid"]
        state, memory = self.initialize_from_history(uid, history_row, questions)
        profile = self.get_profile(uid)
        profile_context = self._agent4edu_profile_context(
            profile.to_context(),
            questions,
            irt_ability=self.irt_model.ability_score(uid),
            irt_theta=self.irt_model.user_theta(uid),
        )
        system_prompt = build_agent4edu_profile_prompt(profile_context)
        simulated_steps: list[dict[str, Any]] = []
        response_prefix = list(clean_sequence(history_row)) if history_row else []
        ncdm_state = self._ncdm_knowledge_state(response_prefix)

        for step in clean_sequence(row):
            qid = step["qid"]
            cid = step["cid"]
            real_response = int(step["response"])
            qmeta = questions.get(str(qid), {})
            routes = qmeta.get("kc_routes", [])
            true_concept = str(routes[0]) if routes else str(cid)
            components = self.probability_components(uid, qid, cid, state)
            dynamic_mastery = float(components["mastery"])
            ncdm_mastery = (
                float(ncdm_state[cid])
                if ncdm_state is not None and cid in ncdm_state
                else None
            )
            current_mastery = ncdm_mastery if ncdm_mastery is not None else dynamic_mastery
            mastery_source = "dneuralcdm_checkpoint" if ncdm_mastery is not None else "dynamic_fallback"
            ncdm_response_probability = self._ncdm_response_probability(
                response_prefix, qid, cid
            )
            memory_context = memory.snapshot(cid, routes, current_mastery)
            prompt = build_agent4edu_action_prompt(
                question=qmeta,
                short_memory=memory_context["short_memory"],
                long_memory=memory_context["long_memory"],
                concept_options=self.concept_options(
                    true_concept,
                    self.irt_model.seed + int(qid) + int(step.get("position") or 0),
                ),
                proficiency={
                    **proficiency_context(true_concept, current_mastery),
                    "source": mastery_source,
                },
            )
            result: dict[str, Any] = {
                "uid": uid,
                "source": "simulated",
                "step_index": step.get("position"),
                "timestamp": step.get("timestamp"),
                "qid": qid,
                "cid": cid,
                "real_response": real_response,
                "simulated_response": 0,
                "probability_components": components,
                "irt_ability_score": round(self.irt_model.ability_score(uid), 6),
                "irt_theta": round(self.irt_model.user_theta(uid), 6),
                "irt_item_beta": round(self.irt_model.item_beta(qid), 6),
                "ncdm_concept_mastery": round(current_mastery, 6),
                "ncdm_mastery_source": mastery_source,
                "ncdm_correct_probability": (
                    round(float(ncdm_response_probability), 6)
                    if ncdm_response_probability is not None
                    else None
                ),
                "learner_profile": profile_context,
                "memory_context": memory_context,
                "kc_routes": routes,
                "question_type": qmeta.get("type"),
                "content": qmeta.get("content"),
                "options": qmeta.get("options"),
                "answer": qmeta.get("answer"),
                "analysis": qmeta.get("analysis"),
                "reference_answer_exposed": True,
                "reference_analysis_exposed": True,
                "agent4edu_reflection_enabled": enable_reflection,
                "agent4edu_feedback_mode": "official_observed_outcome",
            }
            if include_prompt:
                result["llm_system_prompt"] = system_prompt
                result["llm_prompt"] = prompt

            started = time.perf_counter()
            try:
                raw = call_openai_compatible_chat(
                    llm_config,
                    prompt,
                    system_prompt=system_prompt,
                )
                parsed = parse_agent4edu_action(raw)
                if parsed is None:
                    raise ValueError("LLM response did not match Agent4Edu Task1-Task4 format")
                result["llm_result"] = raw
                result["llm_parsed_action"] = parsed
                result["agent_action"] = parsed
                result["simulated_response"] = int(parsed["simulated_correct"])

                simulated_response = int(parsed["simulated_correct"])
                if enable_reflection:
                    reflection_prompt = build_agent4edu_reflection_prompt(
                        qmeta,
                        parsed,
                        real_response,
                        true_concept,
                        short_memory=memory_context["short_memory"],
                        long_memory=memory_context["long_memory"],
                    )
                    if include_prompt:
                        result["reflection_prompt"] = reflection_prompt
                    reflection_started = time.perf_counter()
                    result["reflection"] = call_openai_compatible_chat(
                        llm_config,
                        reflection_prompt,
                        system_prompt=system_prompt,
                    )
                    result["reflection_elapsed_seconds"] = round(
                        time.perf_counter() - reflection_started,
                        3,
                    )
                    memory.learning_status.append(str(result["reflection"]))
            except Exception as exc:
                result["llm_error"] = f"{exc.__class__.__name__}: {exc}"
            result["llm_elapsed_seconds"] = round(time.perf_counter() - started, 3)

            # Official Agent4Edu observes the practice score after the action,
            # then uses this outcome in reflection, factual memory, and the
            # subsequent interaction state.  The current Task4 prediction was
            # already made before this update.
            state.update(cid, real_response, learning_rate=self.learning_rate)
            memory_record = dict(result)
            memory_record["simulated_response"] = real_response
            memory_record["source"] = "observed_feedback"
            memory.observe(memory_record)
            memory.forget(
                time_step=(
                    len(clean_sequence(history_row))
                    if history_row is not None
                    else 0
                )
                + len(simulated_steps)
                + 1,
                forget_lambda=0.99,
            )
            # Official feedback is now part of the learner's observed prefix,
            # so the next exercise receives a fresh NCDM latent state without
            # exposing the current response before its Task4 prediction.
            response_prefix.append({"qid": qid, "cid": cid, "response": real_response})
            ncdm_state = self._ncdm_knowledge_state(response_prefix)
            simulated_steps.append(result)
            if progress_callback is not None:
                progress_callback(result)

        summary = build_sequence_summary(uid, simulated_steps)
        summary["history_interactions"] = (
            len(clean_sequence(history_row)) if history_row is not None else 0
        )
        summary["target_interactions"] = len(simulated_steps)
        return summary

    @staticmethod
    def _agent4edu_profile_context(
        profile: dict[str, Any],
        questions: dict[str, dict[str, Any]],
        irt_ability: float,
        irt_theta: float,
    ) -> dict[str, Any]:
        """Convert project profile statistics to Agent4Edu's original ratios."""

        context = dict(profile)
        exercise_count = max(1, len(questions))
        knowledge_space = {
            str(route)
            for question in questions.values()
            for route in (question.get("kc_routes") or [])
        }
        history = context.get("history_summary") or {}
        context["interaction_count"] = history.get("interaction_count", context.get("interaction_count", 0))
        context["distinct_concept_count"] = history.get(
            "distinct_concept_count",
            context.get("distinct_concept_count", 0),
        )
        context["activity_ratio"] = (
            float(context.get("interaction_count", 0)) / exercise_count
        )
        context["diversity_ratio"] = (
            float(context.get("distinct_concept_count", 0))
            / max(1, len(knowledge_space))
        )
        cognitive_profile = context.get("cognitive_profile") or {}
        control = cognitive_profile.get("control_traits") or {}
        context["success_rate"] = float(control.get("overall_success_rate", 0.5) or 0.5)
        context["effective_ability"] = float(irt_ability)
        context["irt_theta"] = float(irt_theta)
        context["irt_ability_source"] = "history_fitted_rasch1pl"
        context["preference_cid"] = history.get("dominant_concept_id")
        context["preference_route"] = history.get("dominant_route") or context["preference_cid"]
        context["profile_statistics"] = "agent4edu_original_ratio_definitions"
        return context

    def _ncdm_knowledge_state(
        self,
        prefix_steps: list[dict[str, Any]],
    ) -> dict[int, float] | None:
        state = self.dneuralcdm_response_predictor.knowledge_state(prefix_steps)
        if state is None:
            return None
        return {
            int(cid): min(1.0, max(0.0, float(value)))
            for cid, value in state.items()
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

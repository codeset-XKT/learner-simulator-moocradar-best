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
from learner_simulator.llm import call_openai_compatible_chat
from learner_simulator.simulators.random_simulator import (
    RandomLearnerSimulator,
    build_sequence_summary,
)


class Agent4EduBaselineSimulator(RandomLearnerSimulator):
    """Isolated Agent4Edu reproduction baseline.

    It follows the official four-task action prompt, exposes the reference
    answer and analysis, uses Task4 as the predicted response, and performs a
    post-response reflection call. The project's mastery estimator replaces
    Agent4Edu's original DNeuralCDM proficiency module.
    """

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
        )
        system_prompt = build_agent4edu_profile_prompt(profile_context)
        simulated_steps: list[dict[str, Any]] = []

        for step in clean_sequence(row):
            qid = step["qid"]
            cid = step["cid"]
            real_response = int(step["response"])
            qmeta = questions.get(str(qid), {})
            routes = qmeta.get("kc_routes", [])
            true_concept = str(routes[0]) if routes else str(cid)
            components = self.probability_components(uid, qid, cid, state)
            memory_context = memory.snapshot(cid, routes, components["mastery"])
            prompt = build_agent4edu_action_prompt(
                question=qmeta,
                short_memory=memory_context["short_memory"],
                long_memory=memory_context["long_memory"],
                concept_options=self.concept_options(
                    true_concept,
                    self.irt_model.seed + int(qid) + int(step.get("position") or 0),
                ),
                proficiency=proficiency_context(true_concept, components["mastery"]),
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
                "learner_profile": profile_context,
                "memory_context": memory_context,
                "kc_routes": routes,
                "content": qmeta.get("content"),
                "options": qmeta.get("options"),
                "answer": qmeta.get("answer"),
                "analysis": qmeta.get("analysis"),
                "reference_answer_exposed": True,
                "reference_analysis_exposed": True,
                "agent4edu_reflection_enabled": enable_reflection,
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
                        simulated_response,
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

            # Use self-rollout feedback for fair comparison with the full
            # simulator: target-step state and memory must not observe labels.
            state.update(cid, int(result["simulated_response"]), learning_rate=self.learning_rate)
            memory_record = dict(result)
            memory_record["simulated_response"] = int(result["simulated_response"])
            memory_record["source"] = "simulated_feedback"
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
        ability = context.get("ability_estimate") or {}
        context["success_rate"] = float(control.get("overall_success_rate", 0.5) or 0.5)
        context["effective_ability"] = float(
            ability.get("irt_ability", context["success_rate"]) or context["success_rate"]
        )
        context["preference_cid"] = history.get("dominant_concept_id")
        context["preference_route"] = history.get("dominant_route") or context["preference_cid"]
        context["profile_statistics"] = "agent4edu_original_ratio_definitions"
        return context

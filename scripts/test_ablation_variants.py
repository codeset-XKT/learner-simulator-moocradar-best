from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from experiments.ablation.run_ablation import ABLATIONS  # noqa: E402
from learner_simulator.data import sequence_row_from_steps  # noqa: E402
from learner_simulator.simulators import MultiRoleLearnerSimulator  # noqa: E402
import learner_simulator.simulators.multi_role_simulator as module  # noqa: E402


class FakeNCDMPredictor:
    checkpoint_sha256 = "fakehash"

    def available(self):
        return True

    def is_user_held_out(self, uid):
        return uid == "u1"

    def probability(self, prefix_steps, target_qid, target_cid):
        return 0.72

    def knowledge_state(self, prefix_steps):
        return {1: min(0.9, 0.60 + 0.001 * len(prefix_steps))}


def main() -> None:
    history = [
        {"qid": i, "cid": 1, "response": int(i % 3 != 0), "timestamp": i}
        for i in range(90)
    ]
    target = [
        {"qid": 90, "cid": 1, "response": 1, "timestamp": 90},
        {"qid": 91, "cid": 1, "response": 0, "timestamp": 91},
    ]
    questions = {
        str(i): {
            "content": f"Question {i}",
            "options": ["A. correct", "B. wrong"],
            "answer": ["A"],
            "analysis": "A is correct.",
            "kc_routes": ["concept"],
        }
        for i in range(92)
    }
    history_row = sequence_row_from_steps("u1", history)
    target_row = sequence_row_from_steps("u1", target)

    with tempfile.TemporaryDirectory() as temp_dir:
        simulator = MultiRoleLearnerSimulator(
            seed=42,
            dneuralcdm_checkpoint_path=str(Path(temp_dir) / "best_model.pt"),
        )
        simulator.dneuralcdm_response_predictor = FakeNCDMPredictor()
        simulator.fit([history_row], questions=questions)
        original = module.call_openai_compatible_chat

        def fake_chat(config, prompt, system_prompt=""):
            if "# Direct Response Generation Contract #" in prompt:
                return "IdentifiedConcept: concept\nLearnerCorrect: Yes\nStudentAnswer: A"
            return (
                "EvidenceRefs: hist_090\n"
                "IdentifiedConcept: concept\nLearnerCorrect: Yes\n"
                "StudentAnswer: A\nAnswerConfidence: 0.80\n"
                "StudentReasoning: familiar rule\nReasoningConfidence: 0.80"
            )

        module.call_openai_compatible_chat = fake_chat
        try:
            full = _run(simulator, target_row, history_row, questions)
            no_evidence = _run(
                simulator,
                target_row,
                history_row,
                questions,
                include_evidence_representation=False,
            )
            no_alignment = _run(
                simulator,
                target_row,
                history_row,
                questions,
                include_state_item_alignment=False,
            )
            no_process = _run(
                simulator,
                target_row,
                history_row,
                questions,
                response_format="reduced_response",
            )
            no_evolution = _run(
                simulator,
                target_row,
                history_row,
                questions,
                include_dynamic_state_evolution=False,
            )
        finally:
            module.call_openai_compatible_chat = original

    first = full["steps"][0]
    assert len(first["cognitive_state_item_alignment"]["selected_evidence"]) <= 4
    alignment = first["cognitive_state_item_alignment"]
    assert alignment["selection_policy"] == "hierarchical_role_based_evidence_selection_v1"
    assert all("selection_tier" in event for event in alignment["selected_evidence"])
    assert all("selection_score" not in event for event in alignment["selected_evidence"])
    assert {event["selection_tier"] for event in alignment["selected_evidence"]} >= {
        "same_concept_recent_success", "same_concept_recent_error"
    }
    assert "events" not in first["traceable_learner_evidence_representation"]
    assert len(full["traceable_evidence_repository"]["events"]) == 90
    assert first["traceable_learner_evidence_representation"]["ncdm_concept_state"]
    assert "irt_learner_ability" in first["traceable_learner_evidence_representation"]
    assert first["response_agent_prompt"].count("# Cognitive State-Item Alignment #") == 1
    assert first["memory_context"]["used"] is False
    assert "short-term memory" not in first["response_agent_prompt"].lower()
    assert "long-term memory" not in first["response_agent_prompt"].lower()
    assert first["agent_action"]["evidence_refs"] == ["hist_090"]
    assert first["process_consistency"]["valid"] is True
    assert set(first["simulation_tasks"]) == {
        "module1_traceable_learner_evidence_representation",
        "module2_cognitive_state_item_alignment",
        "module3_structured_response_generation",
        "module4_auditable_state_evolution",
    }
    assert no_evidence["steps"][0]["traceable_learner_evidence_representation"]["ablated"]
    assert no_evidence["steps"][0]["cognitive_state_item_alignment"]["selected_evidence"] == []
    assert no_evidence["steps"][0]["cognitive_state_item_alignment"]["concept_mastery"] is None
    assert no_evidence["steps"][0]["cognitive_state_item_alignment"]["irt_ability"] is None
    assert "NCDM concept mastery" not in no_evidence["steps"][0]["response_agent_prompt"]
    assert "# NCDM State Evidence #" not in first["response_agent_prompt"]
    assert "# IRT Ability-Difficulty Evidence #" not in first["response_agent_prompt"]
    assert "# Traceable Learner Evidence Representation #" not in first["response_agent_prompt"]
    assert no_alignment["steps"][0]["cognitive_state_item_alignment"]["ablated"]
    assert no_alignment["steps"][0]["cognitive_state_item_alignment"]["concept_mastery"] is None
    assert no_alignment["steps"][0]["cognitive_state_item_alignment"]["current_concepts"] == []
    assert "unaligned_learner_state_summary" in no_alignment["steps"][0]["cognitive_state_item_alignment"]
    assert "# Direct Response Generation Contract #" in no_process["steps"][0]["response_agent_prompt"]
    assert "Strong aligned readiness" not in no_process["steps"][0]["response_agent_prompt"]
    assert "four_tier_assessment" not in no_process["steps"][0]
    assert no_evolution["steps"][0]["state_evolution"]["ablated"] is True
    # LearnerCorrect is the simulated behavioral label. StudentAnswer is
    # deterministically materialized so it cannot contradict that label.
    def mismatch_chat(config, prompt, system_prompt=""):
        return (
            "EvidenceRefs: hist_090\n"
            "IdentifiedConcept: concept\nLearnerCorrect: No\n"
            "StudentAnswer: A\nAnswerConfidence: 0.80\n"
            "StudentReasoning: familiar rule\nReasoningConfidence: 0.80"
        )
    module.call_openai_compatible_chat = mismatch_chat
    try:
        mismatch = _run(simulator, target_row, history_row, questions)
    finally:
        module.call_openai_compatible_chat = original
    mismatch_step = mismatch["steps"][0]
    assert mismatch_step["simulated_response"] == 0
    assert mismatch_step["response_learner_correct"] == 0
    assert mismatch_step["response_declared_learner_correct"] == 0
    assert mismatch_step["learner_correct_answer_consistent"] is True
    assert mismatch_step["prediction_source"] == "llm_learner_correct"
    assert mismatch_step["agent_action"]["student_answer"] == "B"
    assert "# Reference Answer for Response Rendering #" in mismatch_step["response_agent_prompt"]

    assert set(ABLATIONS) == {
        "full",
        "no-evidence-representation",
        "no-state-item-alignment",
        "no-structured-response-process",
        "no-dynamic-state-evolution",
        "direct-response-generation",
    }
    print("ablation_variants_v3_test_ok")


def _run(simulator, target, history, questions, **kwargs):
    return simulator.simulate_sequence(
        target,
        questions=questions,
        llm_config={"model": "fake", "response_contract_retries": 99},
        call_llm=True,
        include_prompt=True,
        history_row=history,
        feedback_mode="teacher-forcing",
        **kwargs,
    )


if __name__ == "__main__":
    main()

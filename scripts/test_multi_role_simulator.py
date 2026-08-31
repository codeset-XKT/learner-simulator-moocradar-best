from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from learner_simulator.data import sequence_row_from_steps  # noqa: E402
from learner_simulator.simulators import MultiRoleLearnerSimulator  # noqa: E402
import learner_simulator.simulators.multi_role_simulator as module  # noqa: E402


class FakeNCDMPredictor:
    checkpoint_sha256 = "fakehash"

    def available(self): return True
    def is_user_held_out(self, uid): return uid == "u1"
    def probability(self, prefix_steps, target_qid, target_cid): return 0.72
    def knowledge_state(self, prefix_steps): return {1: 0.65}


def main() -> None:
    history = [{"qid": i, "cid": 1, "response": i % 2, "timestamp": i} for i in range(90)]
    target = [{"qid": 90, "cid": 1, "response": 1, "timestamp": 90}]
    questions = {
        str(i): {
            "content": f"Question {i}", "options": ["A", "B"],
            "answer": ["A"], "analysis": "A", "kc_routes": ["addition"],
        }
        for i in range(91)
    }
    history_row = sequence_row_from_steps("u1", history)
    target_row = sequence_row_from_steps("u1", target)
    calls = []

    def fake_chat(config, prompt, system_prompt=""):
        calls.append(prompt)
        return (
            "EvidenceRefs: hist_090\n"
            "IdentifiedConcept: addition\nLearnerCorrect: Yes\nStudentAnswer: A\n"
            "AnswerConfidence: 0.50\nStudentReasoning: apply the known rule\n"
            "ReasoningConfidence: 0.50"
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        simulator = MultiRoleLearnerSimulator(
            seed=42,
            dneuralcdm_checkpoint_path=str(Path(temp_dir) / "best_model.pt"),
        )
        simulator.dneuralcdm_response_predictor = FakeNCDMPredictor()
        simulator.fit([history_row], questions=questions)
        original = module.call_openai_compatible_chat
        module.call_openai_compatible_chat = fake_chat
        try:
            result = simulator.simulate_sequence(
                target_row,
                questions=questions,
                llm_config={"model": "fake", "response_contract_retries": 5},
                call_llm=True,
                include_prompt=True,
                history_row=history_row,
                feedback_mode="teacher-forcing",
            )
        finally:
            module.call_openai_compatible_chat = original

    assert len(calls) == 1, "one target item must make exactly one logical API call"
    step = result["steps"][0]
    assert step["prediction_valid"] is True
    assert step["agent_action"]["state_alignment"]["module"] == "cognitive_state_item_alignment"
    assert "knowledge_gap" not in step["agent_action"]["state_alignment"]
    assert step["agent_action"]["evidence_refs"] == ["hist_090"]
    assert step["four_tier_response_module"]["response_contract"] == "single_pass_process_verifiable_v6_label_conditioned_answer_realization"
    assert step["state_evolution"]["llm_called"] is False
    assert "Attempt:" not in step["response_agent_prompt"]
    assert "# Structured Response Process Contract #" in step["response_agent_prompt"]
    assert "StateAlignment:" not in step["response_agent_prompt"]
    print("multi_role_simulator_v6_test_ok")


if __name__ == "__main__":
    main()

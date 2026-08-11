from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from learner_simulator.data import sequence_row_from_steps  # noqa: E402
from learner_simulator.item_conditioned_ability import (  # noqa: E402
    build_item_conditioned_ability,
)
from learner_simulator.simulators import MultiRoleLearnerSimulator  # noqa: E402
import learner_simulator.simulators.multi_role_simulator as module  # noqa: E402


class FakeNCDMPredictor:
    checkpoint_sha256 = "fakehash"

    def available(self) -> bool:
        return True

    def is_user_held_out(self, uid: str) -> bool:
        return uid == "u1"

    def probability(self, prefix_steps, target_qid, target_cid):
        return 0.72

    def knowledge_state(self, prefix_steps):
        return {1: min(0.9, 0.60 + 0.001 * len(prefix_steps))}


def main() -> None:
    item = build_item_conditioned_ability(
        question={"content": "2+2", "options": ["3", "4"]},
        profile_context={
            "cognitive_profile": {
                "control_traits": {"mastery_stability_level": "high"}
            },
            "ability_profile": {
                "knowledge_breadth": "broad",
                "practice_depth": "deep",
                "challenge_adaptation": "strong",
            },
        },
        memory_context={"short_memory": [], "long_memory": {}},
        proficiency={
            "concept": "addition",
            "value": 0.70,
            "response_probability": 0.72,
            "response_probability_source": "dneuralcdm_response_predictor",
            "concept_mastery_value": 0.70,
            "concept_mastery_source": "dneuralcdm_latent_state",
            "source": "dneuralcdm",
            "level": "high",
        },
        irt_evidence=None,
        historical_reflection=None,
    )
    assert "ncdm_evidence" not in item
    assert "kt_decision_anchor" not in item
    assert "learning_tool_state" not in item

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
            "kc_routes": ["addition"],
        }
        for i in range(92)
    }
    history_row = sequence_row_from_steps("u1", history)
    target_row = sequence_row_from_steps("u1", target)

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir) / "best_model.pt"
        simulator = MultiRoleLearnerSimulator(
            seed=42,
            dneuralcdm_checkpoint_path=str(checkpoint_path),
        )
        simulator.dneuralcdm_response_predictor = FakeNCDMPredictor()
        simulator.fit([history_row], questions=questions)

        original = module.call_openai_compatible_chat
        module.call_openai_compatible_chat = lambda *args, **kwargs: (
            "Attempt: Yes\n"
            "IdentifiedConcept: addition\n"
            "LearnerCorrect: Yes\n"
            "StudentAnswer: A\n"
            "AnswerConfidence: 0.80\n"
            "StudentReasoning: I use the familiar rule.\n"
            "ReasoningConfidence: 0.80"
        )
        try:
            result = simulator.simulate_sequence(
                target_row,
                questions=questions,
                llm_config={"model": "fake"},
                call_llm=True,
                include_prompt=True,
                history_row=history_row,
                feedback_mode="teacher-forcing",
            )
        finally:
            module.call_openai_compatible_chat = original

    assert result["knowledge_state_artifact"] == {
        "model": "dneuralcdm",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": "fakehash",
    }
    assert len(result["steps"]) == 2
    for step in result["steps"]:
        prompt = step["response_agent_prompt"]
        assert prompt.count("# NCDM State Evidence #") == 1
        assert "current_item_response_probability" not in prompt
        assert "response_probability_source" not in prompt
        assert "KT Decision Anchor" not in prompt
        assert "KT State Guidance" not in prompt
        assert "Learning Tool State" not in prompt
        assert "Non-cognitive State" not in prompt
        assert step["prediction_valid"] is True
        assert step["prediction_source"] == "llm_learner_correct"
        assert step["state_evolution"]["state_source"] == "dneuralcdm_latent_state_update"
        assert set(step["simulation_tasks"]) == {
            "task1_learner_state_profile",
            "task2_item_conditioned_integration",
            "task3_learner_response_generation",
            "task4_dynamic_state_evolution",
        }
    print("multi_role_simulator_test_ok")


if __name__ == "__main__":
    main()

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
        return min(0.99, 0.5 + (0.001 * len(prefix_steps)))

    def knowledge_state(self, prefix_steps):
        return {1: min(0.99, 0.5 + (0.001 * len(prefix_steps)))}


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
        checkpoint_path = Path(temp_dir) / "best_model.pt"
        simulator = MultiRoleLearnerSimulator(
            seed=42,
            dneuralcdm_checkpoint_path=str(checkpoint_path),
        )
        simulator.dneuralcdm_response_predictor = FakeNCDMPredictor()
        simulator.fit([history_row], questions=questions)
        original = module.call_openai_compatible_chat

        def fake_chat(config, prompt, system_prompt=""):
            if "# Reduced-response Learner Simulation Protocol #" in prompt:
                return (
                    "Attempt: Yes\nIdentifiedConcept: concept\n"
                    "LearnerCorrect: Yes\nStudentAnswer: A"
                )
            if "LearnerCorrect:" not in prompt:
                return "StudentAnswer: A"
            return (
                "Attempt: Yes\nIdentifiedConcept: concept\nLearnerCorrect: Yes\n"
                "StudentAnswer: A\nAnswerConfidence: 0.80\n"
                "StudentReasoning: familiar rule\nReasoningConfidence: 0.80"
            )

        module.call_openai_compatible_chat = fake_chat
        try:
            outputs = {
                "full": _run(simulator, target_row, history_row, questions),
                "no-profile": _run(
                    simulator,
                    target_row,
                    history_row,
                    questions,
                    include_profile=False,
                    include_cognitive_profile=False,
                    include_ability_profile=False,
                ),
                "no-item": _run(
                    simulator,
                    target_row,
                    history_row,
                    questions,
                    include_item_conditioned_ability=False,
                ),
                "no-four": _run(
                    simulator,
                    target_row,
                    history_row,
                    questions,
                    response_format="reduced_response",
                ),
                "no-evolution": _run(
                    simulator,
                    target_row,
                    history_row,
                    questions,
                    include_dynamic_state_evolution=False,
                ),
                "no-ncdm": _run(
                    simulator,
                    target_row,
                    history_row,
                    questions,
                    include_ncdm_evidence=False,
                ),
                "no-irt": _run(
                    simulator,
                    target_row,
                    history_row,
                    questions,
                    include_irt_evidence=False,
                ),
            }
        finally:
            module.call_openai_compatible_chat = original

    full_prompt = outputs["full"]["steps"][0]["response_agent_prompt"]
    assert full_prompt.count("# NCDM State Evidence #") == 1
    assert "current_item_response_probability" not in full_prompt
    assert "response_probability_source" not in full_prompt
    assert "KT Decision Anchor" not in full_prompt
    assert "KT State Guidance" not in full_prompt
    assert "Learning Tool State" not in full_prompt

    for step in outputs["no-profile"]["steps"]:
        assert step["learner_profile"] == {"uid": "u1"}
        assert step["learner_profile_encoder"]["ablated"] is True
        assert "profile" not in step["response_agent_prompt"].lower()
    for step in outputs["no-item"]["steps"]:
        assert step["item_conditioned_integration"]["ablated"] is True
        assert "# Item-conditioned Integration #" not in step["response_agent_prompt"]
        assert "# NCDM State Evidence #" in step["response_agent_prompt"]
    for step in outputs["no-four"]["steps"]:
        assert "# Reference Answer for Response Rendering #" in step["response_agent_prompt"]
        assert "LearnerCorrect:" in step["response_agent_prompt"]
        assert "# Four-tier Learner Simulation Protocol #" not in step["response_agent_prompt"]
        assert "# Reduced-response Learner Simulation Protocol #" in step["response_agent_prompt"]
        assert step["prediction_source"] == "llm_learner_correct"
        assert step["response_decision_source"] == "learner_correct"
        assert "four_tier_assessment" not in step
    for step in outputs["no-evolution"]["steps"]:
        assert step["state_evolution"]["ablated"] is True
        assert step["feedback_response"] == step["real_response"]
        assert step["state_evolution"]["knowledge_state_feedback_consumed"] is False
        assert step["state_evolution"]["memory_feedback_consumed"] is True
    assert [
        step["ncdm_correct_probability"] for step in outputs["no-evolution"]["steps"]
    ] == [step["ncdm_correct_probability"] for step in outputs["full"]["steps"]]
    assert (
        outputs["no-evolution"]["steps"][0]["ncdm_concept_mastery"]
        == outputs["no-evolution"]["steps"][1]["ncdm_concept_mastery"]
    )
    assert (
        outputs["full"]["steps"][0]["ncdm_concept_mastery"]
        < outputs["full"]["steps"][1]["ncdm_concept_mastery"]
    )
    for step in outputs["no-ncdm"]["steps"]:
        prompt = step["response_agent_prompt"]
        assert "NCDM" not in prompt, [
            line for line in prompt.splitlines() if "NCDM" in line
        ]
        assert step["ncdm_correct_probability"] is None
        assert step["history_state_probability"] is not None
    for step in outputs["no-irt"]["steps"]:
        assert "IRT" not in step["response_agent_prompt"]
        assert step["irt_ability_difficulty_evidence"] is None

    assert set(ABLATIONS) == {
        "full",
        "no-learner-state-profile",
        "no-item-conditioned-integration",
        "no-dynamic-state-evolution",
        "no-ncdm",
        "no-irt",
        "no-ncdm-irt",
        "no-four-tier",
    }
    print("ablation_variants_test_ok")


def _run(simulator, target, history, questions, **kwargs):
    return simulator.simulate_sequence(
        target,
        questions=questions,
        llm_config={"model": "fake"},
        call_llm=True,
        include_prompt=True,
        history_row=history,
        feedback_mode="teacher-forcing",
        **kwargs,
    )


if __name__ == "__main__":
    main()

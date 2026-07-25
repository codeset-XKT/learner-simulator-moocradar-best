from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from learner_simulator.data import sequence_row_from_steps  # noqa: E402
from learner_simulator.item_conditioned_ability import build_item_conditioned_ability  # noqa: E402
from learner_simulator.simulators import MultiRoleLearnerSimulator  # noqa: E402
import learner_simulator.simulators.multi_role_simulator as multi_role_module  # noqa: E402


def main() -> None:
    item_ability = build_item_conditioned_ability(
        question={
            "content": "Choose the correct method for this familiar concept.",
            "options": ["A. method 1", "B. method 2"],
            "kc_routes": ["addition"],
        },
        profile_context={
            "ability_profile": {
                "knowledge_breadth": "high",
                "practice_depth": "deep",
                "challenge_adaptation": "medium",
                "cross_domain_generalization": "medium",
            },
            "cognitive_profile": {
                "transfer_traits": {"transfer_fragility_level": "medium"},
                "control_traits": {"mastery_stability_level": "high"},
                "cognitive_affective_proxies": {"concentration_level": "high"},
                "error_generation_traits": {"carelessness_level": "low"},
            },
        },
        memory_context={
            "short_memory": [
                {
                    "qid": 1,
                    "kc_routes": ["addition"],
                    "real_response": 1,
                }
            ],
            "long_memory": {},
        },
        proficiency={"value": 0.86, "level": "high", "concept": "addition"},
        behavior_factors={"carelessness": 0.05, "fatigue": 0.05, "guessing": 0.05},
        tendency_calibration={"history_level": "high"},
    )
    assert item_ability["module"] == "item_conditioned_ability"
    assert item_ability["knowledge_alignment"] == "high"
    assert item_ability["practice_alignment"] in {"strong", "mixed", "weak", "not_observed"}
    assert item_ability["activation_level"] in {"available", "partial", "limited"}
    assert item_ability["ability_expression"] in {"fluent", "bounded", "tentative"}
    assert item_ability["kt_decision_anchor"]["predicted_response"] == 1
    assert item_ability["kt_decision_anchor"]["decision_weight"] == "primary"
    assert item_ability["memory_support"] != "unavailable"
    assert "p_correct" not in str(item_ability)

    history_steps = [
        {"qid": index, "cid": 1, "response": 1, "timestamp": index}
        for index in range(90)
    ]
    target_steps = [
        {"qid": 90, "cid": 1, "response": 1, "timestamp": 90}
    ]
    questions = {
        str(index): {
            "content": "1 + 1 = ?",
            "options": ["A. 2", "B. 3", "C. 1"],
            "answer": ["A"],
            "analysis": "Add one and one.",
            "kc_routes": ["addition"],
            "type": "unit_test",
        }
        for index in range(91)
    }
    history_row = sequence_row_from_steps("u1", history_steps)
    target_row = sequence_row_from_steps("u1", target_steps)
    simulator = MultiRoleLearnerSimulator(seed=42)
    simulator.fit([history_row], questions=questions)

    def fake_chat(config, prompt, system_prompt=""):
        return (
            "Attempt: Yes\n"
            "IdentifiedConcept: addition\n"
            "StudentAnswer: A\n"
            "AnswerConfidence: 0.80\n"
            "StudentReasoning: 1 plus 1 gives 2.\n"
            "ReasoningConfidence: 0.80"
        )

    original_chat = multi_role_module.call_openai_compatible_chat
    multi_role_module.call_openai_compatible_chat = fake_chat
    try:
        simulation = simulator.simulate_sequence(
            target_row,
            questions=questions,
            llm_config={"model": "fake", "base_url": "https://example.invalid"},
            call_llm=True,
            include_prompt=True,
            history_row=history_row,
            feedback_mode="teacher-forcing",
        )
    finally:
        multi_role_module.call_openai_compatible_chat = original_chat

    step = simulation["steps"][0]
    assert simulation["simulator_type"] == "educational_multi_agent"
    assert step["learner_profile_encoder"]["module"] == "learner_profile_encoder"
    assert step["learner_profile_evidence"]["module"] == "learner_profile_evidence"
    assert step["four_tier_response_simulation"]["module"] == "four_tier_response"
    assert step["enabled_modules"]["learner_profile_encoder"] is True
    assert step["enabled_modules"]["item_conditioned_evidence_encoder"] is True
    assert step["enabled_modules"]["four_tier_response_simulation"] is True
    assert step["item_conditioned_ability"]["module"] == "item_conditioned_ability"
    assert "concept_alignment" not in step["item_conditioned_ability"]
    assert step["item_conditioned_ability"]["knowledge_alignment"]
    assert step["item_conditioned_ability"]["practice_alignment"]
    assert step["item_conditioned_ability"]["kt_decision_anchor"]["predicted_response"] == 1
    assert "# KT State Guidance #" in step["response_agent_prompt"]
    assert "deterministic answer label" in step["response_agent_prompt"]
    assert "kt_predicted_response" not in step["response_agent_prompt"]
    assert "predicted_response" not in step["response_agent_prompt"]
    assert "kt_decision_anchor" not in step["response_agent_prompt"]
    assert "learner_evidence_profile" not in step
    assert "cognitive_profile_evidence" not in step
    assert "cognitive_process_constraint" not in step
    assert "ability_boundary_evidence" not in step
    assert "cognitive_route_evidence" not in step
    assert "latent_learner_state" not in step
    assert step["four_tier_response_module"]["diagnostic_scoring"]["answer_correct"] is True
    assert step["four_tier_response_module"]["conditioning"]["item_conditioned_ability"]["activation_level"]
    assert step["kt_state_probability"] >= 0.5
    assert step["dkt_correct_probability"] is None
    assert step["dkt_conditioning"]["method"] == "prompt_only_no_posthoc_override"
    assert step["dkt_conditioning"]["overrode_llm"] is False
    assert step["simulation_tasks"]["task1_learner_state_inference"]["output"]["module"] == "learner_profile_encoder"
    assert step["simulation_tasks"]["task2_item_conditioned_activation"]["concept_match"] is True
    assert step["simulation_tasks"]["task3_four_tier_response_generation"]["answer_correct"] is True
    assert step["simulation_tasks"]["task4_state_evolution"]["feedback_mode"] == "teacher-forcing"
    assert step["simulated_response"] == 1
    assert step["feedback_response"] == 1
    assert step["agent_action"]["student_answer"] == "A"
    print("multi_role_simulator_test_ok")


if __name__ == "__main__":
    main()

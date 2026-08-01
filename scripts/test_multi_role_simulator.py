from __future__ import annotations

import json
import sys
import tempfile
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
        irt_evidence={
            "module": "irt_ability_difficulty_evidence",
            "learner_ability_level": "high",
            "item_difficulty_level": "medium",
            "relative_challenge": "below_learner_ability",
            "boundary_band": "clear",
            "evidence_role": "test_irt_signal",
        },
        learning_tool_state={
            "module": "learning_tool_state_encoder",
            "joint_readiness_state": "ncdm_supported",
            "state_commitment": "The learner is ready for this item.",
            "response_planning": "preserve_ncdm_supported_success",
            "knowledge_tool": {"readiness": "strong"},
            "ability_difficulty_tool": {
                "relative_challenge": "below_learner_ability"
            },
        },
    )
    assert item_ability["module"] == "item_conditioned_ability"
    assert item_ability["knowledge_alignment"] == "high"
    assert item_ability["practice_alignment"] in {"strong", "mixed", "weak", "not_observed"}
    assert item_ability["activation_level"] in {"available", "partial", "limited"}
    assert item_ability["ability_expression"] in {"fluent", "bounded", "tentative"}
    assert item_ability["kt_decision_anchor"]["predicted_response"] == 1
    assert item_ability["kt_decision_anchor"]["evidence_priority"] == "primary"
    assert item_ability["memory_support"] != "unavailable"
    assert item_ability["irt_relative_challenge"] == "below_learner_ability"
    assert item_ability["learning_tool_state"]["joint_readiness_state"] == "ncdm_supported"
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
    temp = tempfile.TemporaryDirectory()
    ncdm_path = Path(temp.name) / "stu_know_proficiency.json"
    ncdm_path.write_text(
        json.dumps(
            {
                "_meta": {"concept_id_map": {"1": 0}, "exercise_id_map": {}},
                "students": {"u1": [[0.86] for _ in range(90)]},
            }
        ),
        encoding="utf-8",
    )
    simulator = MultiRoleLearnerSimulator(
        seed=42,
        dneuralcdm_proficiency_path=str(ncdm_path),
    )
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
    assert step["item_conditioned_ability"]["kt_proficiency_state"]["value"] >= 0.5
    assert step["irt_ability_difficulty_evidence"]["module"] == "irt_ability_difficulty_evidence"
    assert step["irt_ability_difficulty_evidence"]["source"] == "rasch_1pl_from_observed_history"
    assert step["irt_evidence_in_prompt"] is True
    assert step["learning_tool_state_in_prompt"] is True
    assert step["learning_tool_state"]["module"] == "learning_tool_state_encoder"
    assert step["learning_tool_state"]["method"] == "agent4edu_style_ncdm_irt_tool_conditioning"
    assert step["learning_tool_state"]["four_tier_generation_policy"]["output_target"] == "four_tier_answer_reasoning_confidence"
    assert step["learning_tool_state"]["joint_readiness_state"] in {
        "ncdm_supported",
        "ncdm_supported_challenged",
        "ncdm_supported_unstable",
        "ncdm_developing_supported",
        "ncdm_developing",
        "ncdm_developing_challenged",
        "ncdm_weak_with_memory",
        "ncdm_unsupported",
        "ncdm_uncertain",
    }
    assert step["learner_profile_encoder"]["irt_ability_difficulty_evidence"]["module"] == "irt_ability_difficulty_evidence"
    assert step["learner_profile_encoder"]["learning_tool_state"]["module"] == "learning_tool_state_encoder"
    assert "# KT/CDM Response State #" in step["response_agent_prompt"]
    assert "# KT Decision Anchor #" in step["response_agent_prompt"]
    assert "# IRT Ability-Difficulty Evidence #" in step["response_agent_prompt"]
    assert "# Learning Tool State #" in step["response_agent_prompt"]
    assert "# Agent-style Internal Task Protocol #" in step["response_agent_prompt"]
    assert "Agent4Edu Task4 predicts Yes/No correctness" in step["response_agent_prompt"]
    assert "# KT State Guidance #" in step["response_agent_prompt"]
    assert "primary symmetric prior for LearnerCorrect" in step["response_agent_prompt"]
    assert "sampled response label" in step["response_agent_prompt"]
    assert "rasch_expected_success" not in step["response_agent_prompt"]
    assert "p_correct" not in step["response_agent_prompt"]
    assert "learner_evidence_profile" not in step
    assert "cognitive_profile_evidence" not in step
    assert "cognitive_process_constraint" not in step
    assert "ability_boundary_evidence" not in step
    assert "cognitive_route_evidence" not in step
    assert "latent_learner_state" not in step
    assert step["four_tier_response_module"]["diagnostic_scoring"]["answer_correct"] is True
    assert step["four_tier_response_module"]["conditioning"]["item_conditioned_ability"]["activation_level"]
    assert step["four_tier_response_module"]["conditioning"]["learning_tool_state"]["module"] == "learning_tool_state_encoder"
    assert step["kt_state_probability"] >= 0.5
    assert step["knowledge_state_source"] == "dneuralcdm_online"
    assert step["ncdm_correct_probability"] >= 0.5
    assert step["dkt_correct_probability"] is None
    assert step["state_evolution"]["state_source"] == "dneuralcdm_online_update"
    assert step["dkt_conditioning"]["method"] == "prompt_only_no_posthoc_override"
    assert step["dkt_conditioning"]["overrode_llm"] is False
    assert step["simulation_tasks"]["task1_learner_state_inference"]["output"]["module"] == "learner_profile_encoder"
    assert step["simulation_tasks"]["task2_item_conditioned_activation"]["concept_match"] is True
    assert step["simulation_tasks"]["task2_item_conditioned_activation"]["learning_tool_state"]
    assert step["simulation_tasks"]["task3_four_tier_response_generation"]["answer_correct"] is True
    assert step["simulation_tasks"]["task4_state_evolution"]["feedback_mode"] == "teacher-forcing"
    assert step["simulated_response"] == 1
    assert step["feedback_response"] == 1
    assert step["agent_action"]["student_answer"] == "A"
    temp.cleanup()
    print("multi_role_simulator_test_ok")


if __name__ == "__main__":
    main()

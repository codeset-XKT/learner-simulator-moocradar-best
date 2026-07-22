from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from learner_simulator.data import sequence_row_from_steps  # noqa: E402
from learner_simulator.educational_multi_agent_prompt import (  # noqa: E402
    build_ability_boundary_evidence,
    fallback_ability_boundary,
    fallback_cognitive_route,
    parse_cognitive_route,
)
from learner_simulator.simulators import MultiRoleLearnerSimulator  # noqa: E402
import learner_simulator.simulators.multi_role_simulator as multi_role_module  # noqa: E402


def main() -> None:
    boundary = build_ability_boundary_evidence(
        question={"content": "1 + 1 = ?", "options": ["A. 2", "B. 3"]},
        cognitive_profile={
            "ability_summary": {
                "knowledge_breadth": "high",
                "practice_depth": "deep",
                "challenge_adaptation": "medium",
                "cross_domain_generalization": "medium",
            },
            "transfer_traits": {"transfer_fragility": "medium"},
            "control_traits": {"mastery_stability": "high"},
        },
        memory_context={"short_memory": [], "long_memory": {}},
        proficiency={"value": 0.8, "level": "high", "concept": "addition"},
    )
    assert boundary["module"] == "ability_boundary_evidence"
    assert boundary["item_demand"] == "low"
    assert "recall familiar concept" in boundary["available_abilities"]

    mastery_aligned_boundary = build_ability_boundary_evidence(
        question={
            "content": "Choose the correct method for this familiar concept.",
            "options": ["A. method 1", "B. method 2"],
            "kc_routes": ["addition"],
        },
        cognitive_profile={
            "ability_summary": {
                "knowledge_breadth": "low",
                "practice_depth": "shallow",
                "challenge_adaptation": "low",
                "cross_domain_generalization": "medium",
            },
            "transfer_traits": {"transfer_fragility": "medium"},
            "control_traits": {"mastery_stability": "medium"},
        },
        memory_context={
            "short_memory": [
                {
                    "content": "earlier related addition item",
                    "kc_routes": ["addition"],
                    "real_response": 1,
                }
            ],
            "long_memory": {},
        },
        proficiency={"value": 0.88, "level": "high", "concept": "addition"},
    )
    assert mastery_aligned_boundary["concept_mastery_overrides_global_profile_weakness"] is True
    assert mastery_aligned_boundary["boundary_risk"] == "low"
    assert "knowledge breadth for distinguishing alternatives" not in mastery_aligned_boundary["weak_abilities"]

    route = parse_cognitive_route(
        "CognitiveRoute: mastery_retrieval\n"
        "ExpectedCorrectnessTendency: favorable\n"
        "ExpectedConfidence: high\n"
        "RouteRationale: Stable memory supports direct retrieval."
    )
    assert route is not None
    assert route["cognitive_route"] == "mastery_retrieval"
    assert route["expected_confidence"] == "high"

    fallback_boundary = fallback_ability_boundary(
        {"ability_summary": {"practice_depth": "medium"}},
        {"value": 0.8, "level": "high", "concept": "addition"},
    )
    fallback_route = fallback_cognitive_route(
        {"error_traits": {}, "transfer_traits": {}},
        fallback_boundary,
        {"value": 0.8},
        {"carelessness": 0.05, "fatigue": 0.05},
        {"band": "strong-correct", "history_level": "high", "history_rate": 0.9},
    )
    assert fallback_route["cognitive_route"] == "mastery_retrieval"

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
        if "# Cognitive Route Agent #" in prompt:
            return (
                "CognitiveRoute: mastery_retrieval\n"
                "ExpectedCorrectnessTendency: favorable\n"
                "ExpectedConfidence: high\n"
                "RouteRationale: The learner has stable mastery on this concept."
            )
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
            history_row=history_row,
            feedback_mode="teacher-forcing",
        )
    finally:
        multi_role_module.call_openai_compatible_chat = original_chat

    step = simulation["steps"][0]
    assert simulation["simulator_type"] == "educational_multi_agent"
    assert step["ability_boundary_evidence"]["module"] == "ability_boundary_evidence"
    assert step["ability_boundary_evidence"]["method"] == "deterministic_from_profile_memory_and_item"
    assert step["ability_boundary_evidence"]["item_demand"] == "low"
    assert step["cognitive_route_agent"]["cognitive_route"] == "mastery_retrieval"
    assert step["four_tier_response_module"]["diagnostic_scoring"]["answer_correct"] is True
    assert step["kt_state_probability"] >= 0.5
    assert step["dkt_correct_probability"] is None
    assert step["dkt_conditioning"]["method"] == "prompt_only_no_posthoc_override"
    assert step["dkt_conditioning"]["overrode_llm"] is False
    assert step["simulated_response"] == 1
    assert step["feedback_response"] == 1
    assert step["agent_action"]["student_answer"] == "A"
    print("multi_role_simulator_test_ok")


if __name__ == "__main__":
    main()

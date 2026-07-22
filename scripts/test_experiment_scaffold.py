from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.ablation.run_ablation import ABLATIONS  # noqa: E402
from experiments.common import validate_cohort  # noqa: E402
from learner_simulator.data import sequence_row_from_steps  # noqa: E402
from learner_simulator.agent4edu_baseline import (  # noqa: E402
    build_agent4edu_action_prompt,
    parse_agent4edu_action,
)
from learner_simulator.simulators import Agent4EduBaselineSimulator  # noqa: E402
import learner_simulator.simulators.agent4edu_simulator as agent4edu_module  # noqa: E402


def main() -> None:
    prompt = build_agent4edu_action_prompt(
        question={
            "content": "1 + 1 = ?",
            "options": ["1", "2"],
            "answer": ["2"],
            "analysis": "Add the two numbers.",
        },
        short_memory=[],
        long_memory={},
        concept_options=["addition", "subtraction", "division"],
        proficiency={"concept": "addition", "level": "medium"},
    )
    assert "# Reference Answer #" in prompt
    assert "# Analysis #" in prompt
    assert "Task4:" in prompt

    parsed = parse_agent4edu_action(
        "Task1: Yes\n"
        "Task2: addition\n"
        "Task3: I add one and one.\n"
        "Task4: Yes"
    )
    assert parsed is not None
    assert parsed["simulated_correct"] == 1
    assert set(ABLATIONS) == {
        "full",
        "no-profile",
        "no-memory",
        "no-proficiency",
        "no-four-tier",
        "no-cognitive-selection",
        "no-cognitive-profile",
        "no-ability-profile",
    }

    history_steps = [
        {"qid": index, "cid": index % 3, "response": index % 2, "timestamp": index}
        for index in range(90)
    ]
    target_steps = [
        {
            "qid": 90 + index,
            "cid": index % 3,
            "response": index % 2,
            "timestamp": 90 + index,
        }
        for index in range(10)
    ]
    questions = {
        str(index): {
            "content": f"question {index}",
            "options": ["A", "B"],
            "answer": ["A"],
            "analysis": "reference analysis",
            "kc_routes": [f"concept {index % 3}"],
        }
        for index in range(100)
    }
    history_row = sequence_row_from_steps("u1", history_steps)
    target_row = sequence_row_from_steps("u1", target_steps)
    validate_cohort([history_row], [target_row])
    single_target_row = sequence_row_from_steps("u1", target_steps[:1])
    simulator = Agent4EduBaselineSimulator(seed=42, long_threshold=5)
    simulator.fit([history_row], questions=questions)
    calls = {"count": 0}

    def fake_chat(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] % 2 == 1:
            return (
                "Task1: Yes\n"
                "Task2: concept 0\n"
                "Task3: I choose A.\n"
                "Task4: Yes"
            )
        return "I answered correctly and should retain this method."

    original_chat = agent4edu_module.call_openai_compatible_chat
    agent4edu_module.call_openai_compatible_chat = fake_chat
    try:
        simulation = simulator.simulate_sequence(
            single_target_row,
            questions,
            llm_config={"model": "fake", "base_url": "https://example.invalid"},
            history_row=history_row,
        )
    finally:
        agent4edu_module.call_openai_compatible_chat = original_chat
    step = simulation["steps"][0]
    assert step["simulated_response"] == 1
    assert step["real_response"] == 0
    assert step["memory_context"]["long_memory"]["latest_learning_status"].startswith(
        "source=observed_history"
    )
    assert step["reference_answer_exposed"] is True
    assert step["learner_profile"]["profile_statistics"] == (
        "agent4edu_original_ratio_definitions"
    )
    assert "reflection" in step
    assert calls["count"] == 2
    second_target_row = sequence_row_from_steps("u1", target_steps[:2])
    calls["count"] = 0
    agent4edu_module.call_openai_compatible_chat = fake_chat
    try:
        simulation = simulator.simulate_sequence(
            second_target_row,
            questions,
            llm_config={"model": "fake", "base_url": "https://example.invalid"},
            history_row=history_row,
        )
    finally:
        agent4edu_module.call_openai_compatible_chat = original_chat
    second_step = simulation["steps"][1]
    assert second_step["memory_context"]["short_memory"][-1]["source"] == "simulated_feedback"
    assert second_step["memory_context"]["short_memory"][-1]["simulated_response"] == 1
    assert second_step["memory_context"]["short_memory"][-1]["real_response"] == 0
    print("experiment_scaffold_test_ok")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.ablation.run_ablation import ABLATIONS  # noqa: E402
from experiments.common import (  # noqa: E402
    add_shared_arguments,
    build_archive_metadata,
    exclude_cohort_users,
    initialize_experiment_run,
    save_report,
    validate_cohort,
    valid_metric_steps,
)
from learner_simulator.data import sequence_row_from_steps  # noqa: E402
from learner_simulator.evaluation import evaluate_steps  # noqa: E402
from learner_simulator.agent4edu_baseline import (  # noqa: E402
    build_agent4edu_action_prompt,
    parse_agent4edu_action,
)
from learner_simulator.simulators import Agent4EduBaselineSimulator  # noqa: E402
import learner_simulator.simulators.agent4edu_simulator as agent4edu_module  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    add_shared_arguments(parser)
    default_args = parser.parse_args([])
    assert default_args.include_prompt is True
    assert default_args.save_steps is True
    initialize_experiment_run(default_args)
    metadata = build_archive_metadata(
        default_args,
        {
            "model": "fake-model",
            "base_url": "https://example.invalid/v1",
            "api_key": "must-not-be-written",
            "api_key_env": "ALSO_SENSITIVE",
        },
    )
    assert metadata["storage_policy"]["all_steps_saved"] is True
    assert metadata["storage_policy"]["prompts_saved"] is True
    assert metadata["storage_policy"]["existing_files_overwritten"] is False
    assert metadata["llm_config"]["api_key"] == "<redacted>"
    assert metadata["llm_config"]["api_key_env"] == "<redacted>"

    with tempfile.TemporaryDirectory() as temp_dir:
        requested = Path(temp_dir) / "experiment.json"
        first = save_report({"run": 1}, requested)
        second = save_report({"run": 2}, requested)
        assert first == requested
        assert second != first
        assert first.exists() and second.exists()
        assert json.loads(first.read_text(encoding="utf-8"))["run"] == 1
        assert json.loads(second.read_text(encoding="utf-8"))["run"] == 2

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
    paper_variants = {
        "full",
        "no-learner-state-profile",
        "no-item-conditioned-integration",
        "no-dynamic-state-evolution",
        "no-ncdm",
        "no-irt",
        "no-ncdm-irt",
        "no-four-tier",
    }
    assert paper_variants == set(ABLATIONS)

    filtered, excluded = valid_metric_steps(
        "multi-role",
        [
            {"uid": "ok", "prediction_valid": True},
            {"uid": "bad", "prediction_valid": True},
            {"uid": "bad", "prediction_valid": False, "llm_error": "timeout"},
        ],
    )
    assert filtered == [{"uid": "ok", "prediction_valid": True}]
    assert excluded == ["bad"]
    assert exclude_cohort_users(
        [{"uid": "cohort"}, {"uid": "external"}],
        [{"uid": "cohort"}],
    ) == [{"uid": "external"}]
    partial_metrics = evaluate_steps(
        [
            {
                "uid": "u1",
                "cid": 1,
                "real_response": 1,
                "simulated_response": 1,
                "ncdm_correct_probability": 0.8,
            },
            {
                "uid": "u2",
                "cid": 1,
                "real_response": 0,
                "simulated_response": 0,
                "ncdm_correct_probability": None,
            },
        ]
    )
    assert partial_metrics["ncdm_probability_count"] == 1
    assert partial_metrics["ncdm_probability_coverage"] == 0.5

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

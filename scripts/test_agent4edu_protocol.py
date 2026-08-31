from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from learner_simulator.data import (  # noqa: E402
    clean_sequence,
    split_rows_agent4edu,
)
from learner_simulator.agent4edu_baseline import (  # noqa: E402
    AGENT4EDU_ACTIVITY_MEAN,
    AGENT4EDU_DIVERSITY_MEAN,
    build_agent4edu_profile_prompt,
    parse_agent4edu_action,
)
from learner_simulator.simulators import RandomLearnerSimulator  # noqa: E402


def make_row(uid: str, start: int, count: int) -> dict[str, str]:
    values = list(range(start, start + count))
    return {
        "uid": uid,
        "questions": ",".join(str(value) for value in values),
        "concepts": ",".join(str(value % 5) for value in values),
        "responses": ",".join(str(value % 2) for value in values),
        "timestamps": ",".join(str(1000 + value) for value in values),
    }


def main() -> None:
    rows = [
        make_row("u1", 0, 60),
        make_row("u2", 200, 100),
        make_row("u1", 60, 40),
        make_row("too_short", 500, 99),
    ]
    history_rows, target_rows = split_rows_agent4edu(
        rows,
        history_steps=90,
        target_steps=10,
    )

    assert [row["uid"] for row in history_rows] == ["u1", "u2"]
    assert [len(clean_sequence(row)) for row in history_rows] == [90, 90]
    assert [len(clean_sequence(row)) for row in target_rows] == [10, 10]
    assert clean_sequence(history_rows[0])[-1]["qid"] == 89
    assert clean_sequence(target_rows[0])[0]["qid"] == 90

    questions = {
        str(qid): {
            "content": f"question {qid}",
            "kc_routes": [f"concept {qid % 5}"],
            "answer": [str(qid % 2)],
        }
        for qid in range(300)
    }
    simulator = RandomLearnerSimulator(seed=42)
    simulator.fit(history_rows, questions=questions, profile_normalization_rows=rows)
    profile_context = simulator.get_profile("u1").to_context()
    assert profile_context["history_summary"]["interaction_count"] == 90
    assert "activity_ratio" not in profile_context
    simulation = simulator.simulate_sequence(
        target_rows[0],
        questions=questions,
        history_row=history_rows[0],
    )
    first = simulation["steps"][0]
    assert len(first["memory_context"]["short_memory"]) == 5
    assert {
        item["source"] for item in first["memory_context"]["short_memory"]
    } == {"observed_history"}
    assert (
        first["memory_context"]["long_memory"]["latest_learning_status"]
        .startswith("source=observed_history")
    )

    # The official parser normalizes both `Task1:` and the common `Task 1:`
    # spelling before extracting the four fields.
    parsed = parse_agent4edu_action(
        "Task 1: Yes\nTask 2: concept 1\nTask 3: solve\nTask 4: No"
    )
    assert parsed is not None
    assert parsed["attempt"] == "yes"
    assert parsed["simulated_correct"] == 0

    # Profile discretization must use the official global means rather than
    # cohort-dependent thresholds.
    profile_prompt = build_agent4edu_profile_prompt(
        {
            "activity_ratio": AGENT4EDU_ACTIVITY_MEAN + 1e-6,
            "diversity_ratio": AGENT4EDU_DIVERSITY_MEAN + 1e-6,
            "success_rate": 0.6,
            "effective_ability": 0.5,
            "preference_route": "concept 1",
        }
    )
    assert "high activity" in profile_prompt
    assert "high knowledge diversity" in profile_prompt
    print("agent4edu_protocol_test_ok")


if __name__ == "__main__":
    main()

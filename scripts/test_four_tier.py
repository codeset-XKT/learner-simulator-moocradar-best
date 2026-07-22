from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from learner_simulator.four_tier import (  # noqa: E402
    assess_four_tier_response,
    compare_answers,
    parse_answer_only_response,
    parse_four_tier_response,
)
from learner_simulator.evaluation import evaluate_steps  # noqa: E402


def main() -> None:
    raw = """Attempt: Yes
IdentifiedConcept: decimal notation
StudentAnswer: 500.050
AnswerConfidence: 0.82
StudentReasoning: The hundredths digit is 5 and the other decimal digits are zero.
ReasoningConfidence: 0.67"""
    parsed = parse_four_tier_response(raw)
    assert parsed is not None
    assert compare_answers(parsed["student_answer"], ["$$500.050$$"]) is True
    assert compare_answers("C", ["C"]) is True
    assert compare_answers("42", ["$$42$$"]) is True
    assert compare_answers("9, 11", ["$$9$$", "$$11$$"]) is True
    assert compare_answers("9", ["$$9$$", "$$11$$"]) is False

    assessment = assess_four_tier_response(
        parsed,
        reference_answers=["$$500.050$$"],
    )
    assert assessment["answer_correct"] is True
    assert assessment["reasoning_correct"] is None
    assert assessment["diagnosis"] == "confident_correct_answer_reason_unscored"

    metrics = evaluate_steps(
        [
            {
                "real_response": 1,
                "p_correct": 0.6,
                "simulated_response": 1,
                "four_tier_assessment": assessment,
                "llm_parsed_action": {
                    **parsed,
                    "simulated_correct": 1,
                    "confidence": parsed["answer_confidence"],
                    "four_tier_assessment": assessment,
                },
            }
        ]
    )
    assert metrics["four_tier_response_count"] == 1
    assert metrics["four_tier_answer_scored_count"] == 1
    assert metrics["four_tier_fully_scored_count"] == 0

    answer_only = parse_answer_only_response("StudentAnswer: B")
    assert answer_only == {
        "student_answer": "B",
        "response_format": "answer_only",
    }
    answer_only_metrics = evaluate_steps(
        [
            {
                "real_response": 1,
                "p_correct": 0.6,
                "simulated_response": 1,
                "llm_parsed_action": {
                    **answer_only,
                    "simulated_correct": 1,
                },
            }
        ]
    )
    assert answer_only_metrics["llm_response_acc"] == 1.0
    assert "llm_auc" not in answer_only_metrics
    assert "four_tier_response_count" not in answer_only_metrics
    print("four_tier_test_ok")


if __name__ == "__main__":
    main()

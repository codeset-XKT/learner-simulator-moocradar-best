from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from learner_simulator.agent4edu_prompt import build_action_prompt  # noqa: E402
from learner_simulator.cognitive_strategy import select_cognitive_strategy  # noqa: E402
from learner_simulator.evaluation import evaluate_steps  # noqa: E402


def main() -> None:
    memory = {
        "short_memory": [
            {
                "source": "observed_history",
                "step_index": -2,
                "qid": 1,
                "cid": 7,
                "kc_routes": ["math----fractions"],
                "simulated_response": 0,
            },
            {
                "source": "observed_history",
                "step_index": -1,
                "qid": 2,
                "cid": 7,
                "kc_routes": ["math----fractions"],
                "simulated_response": 0,
            },
            {
                "source": "observed_history",
                "step_index": -4,
                "qid": 4,
                "cid": 7,
                "kc_routes": ["math----fractions"],
                "simulated_response": 0,
            },
            {
                "source": "observed_history",
                "step_index": -5,
                "qid": 5,
                "cid": 7,
                "kc_routes": ["math----fractions"],
                "simulated_response": 0,
            },
            {
                "source": "observed_history",
                "step_index": -3,
                "qid": 3,
                "cid": 99,
                "kc_routes": ["geometry----angles"],
                "simulated_response": 1,
            },
        ],
        "long_memory": {"significant_facts": []},
    }
    strategy = select_cognitive_strategy(
        mastery=0.25,
        profile={},
        memory_context=memory,
        behavior_factors={
            "attention": 0.7,
            "fatigue": 0.2,
            "carelessness": 0.1,
            "guessing": 0.1,
        },
        has_options=True,
        current_cid=7,
        current_routes=["math----fractions"],
    )
    assert strategy["evidence_type"] == "cognitive_evidence"
    assert strategy["related_history"]["signal"] == "repeated_related_errors"
    assert strategy["related_history"]["history_count"] == 4
    assert strategy["related_history"]["observed_count"] == 4
    assert strategy["related_history"]["wrong_count"] == 4
    assert "mode" not in strategy
    assert "dominant_state" not in strategy
    assert "mode_weights" not in strategy
    assert "response_boundary" not in strategy
    assert "failure_risk" not in strategy
    assert "likely_failure_sources" not in strategy
    assert "correct" not in strategy

    prompt = build_action_prompt(
        question={"content": "1/2 + 1/3 = ?", "options": ["A", "B"]},
        short_memory=memory["short_memory"],
        long_memory=memory["long_memory"],
        concept_options=["fractions", "angles", "measurement"],
        proficiency={"concept": "fractions", "level": "low"},
        behavior_factors={
            "attention": 0.7,
            "fatigue": 0.2,
            "carelessness": 0.1,
            "guessing": 0.1,
        },
        response_format="four_tier",
        cognitive_strategy=strategy,
        tendency_calibration={
            "score": 0.64,
            "band": "correct-leaning",
            "history_level": "favorable",
            "history_rate": 0.62,
            "mastery_level": "favorable",
            "mastery": 0.66,
            "concept_level": "mixed",
            "concept_rate": 0.55,
        },
    )
    assert "# Cognitive Evidence #" not in prompt
    assert "soft state weights" not in prompt
    assert "predefined state selector" not in prompt
    assert "allowed cues" not in prompt
    assert "prohibited actions" not in prompt
    assert "cognitive evidence" not in prompt.lower()
    assert "failure_risk" not in prompt
    assert "failure sources" not in prompt
    assert "two-stage simulation" in prompt
    assert "# Response Tendency Calibration #" in prompt
    assert "do not default to an incorrect answer" in prompt
    assert "Do not restart, verify, recompute" not in prompt
    assert "Reference Answer" not in prompt
    clean_prompt = build_action_prompt(
        question={"content": "1/2 + 1/3 = ?", "options": ["A", "B"]},
        short_memory=memory["short_memory"],
        long_memory=memory["long_memory"],
        concept_options=["fractions", "angles", "measurement"],
        proficiency={"concept": "fractions", "level": "low"},
        behavior_factors={
            "attention": 0.7,
            "fatigue": 0.2,
            "carelessness": 0.1,
            "guessing": 0.1,
        },
        response_format="four_tier",
        cognitive_strategy=None,
    )
    assert "# Cognitive Evidence #" not in clean_prompt
    assert "boundary controller" not in clean_prompt
    assert "committed boundary" not in clean_prompt
    assert "learner response boundary changes tendencies" not in clean_prompt
    metrics = evaluate_steps(
        [
            {
                "real_response": 0,
                "p_correct": 0.4,
                "simulated_response": 0,
                "uid": "u1",
                "cid": 7,
                "probability_components": {"mastery": 0.2},
                "cognitive_strategy": strategy,
                "llm_parsed_action": {"simulated_correct": 0},
                "four_tier_assessment": {"answer_confidence": 0.2},
            },
            {
                "real_response": 1,
                "p_correct": 0.6,
                "simulated_response": 1,
                "uid": "u1",
                "cid": 8,
                "probability_components": {"mastery": 0.5},
                "four_tier_assessment": {"answer_confidence": 0.5},
            },
            {
                "real_response": 1,
                "p_correct": 0.8,
                "simulated_response": 1,
                "uid": "u2",
                "cid": 8,
                "probability_components": {"mastery": 0.8},
                "four_tier_assessment": {"answer_confidence": 0.8},
            }
        ]
    )
    assert metrics["cognitive_evidence_counts"] == {"repeated_related_errors": 1}
    assert metrics["cognitive_evidence_metrics"]["repeated_related_errors"]["acc"] == 1.0
    assert "cognitive_failure_risk_mean" not in metrics
    assert "cognitive_failure_source_counts" not in metrics
    assert metrics["response_specificity"] == 1.0
    assert metrics["response_balanced_accuracy"] == 1.0
    assert metrics["response_mcc"] == 1.0
    assert metrics["learner_distribution_error"] == 0.0
    assert metrics["concept_distribution_error"] == 0.0
    assert metrics["mastery_response_monotonicity"]["score"] == 1.0
    assert metrics["mastery_confidence_monotonicity"]["score"] == 1.0
    no_probability_metrics = evaluate_steps(
        [
            {
                "real_response": 1,
                "simulated_response": 1,
                "uid": "u1",
                "cid": 7,
                "probability_components": {"mastery": 0.8},
            }
        ]
    )
    assert "prob_acc_at_threshold" not in no_probability_metrics
    print("cognitive_strategy_test_ok")


if __name__ == "__main__":
    main()

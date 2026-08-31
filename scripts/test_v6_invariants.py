from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from learner_simulator.educational_multi_agent_prompt import build_response_prompt
from learner_simulator.formal_metrics import (
    build_ability_difficulty_partition,
    evaluate_formal_response_metrics,
)
from learner_simulator.four_tier import assess_four_tier_response
from learner_simulator.process_evidence import (
    build_process_consistency,
    build_state_item_alignment,
    primary_kc_route,
)
from learner_simulator.process_verification import verify_process_steps
from learner_simulator.simulators.multi_role_simulator import (
    enforce_learner_correct_answer_consistency,
)
from learner_simulator.simulators.random_simulator import RandomLearnerSimulator


def main() -> None:
    route = "math----counting----multiplication-principle"
    assert primary_kc_route(
        {"kc_routes": [route, "math----counting----addition-principle"]},
        fallback="fallback",
    ) == [route]
    alignment = build_state_item_alignment(
        {"events": [], "stable_patterns": []},
        {"kc_routes": [route, "math----counting----addition-principle"]},
        concept_mastery=0.5,
        irt_evidence=None,
    )
    assert alignment["current_concepts"] == [route]

    question = {"answer": ["A"], "options": ["A", "B"]}
    # Regression: raw A with declared No must be materialized to B before the
    # Four-tier diagnostic is scored.
    action = {
        "learner_correct": 0,
        "student_answer": "A",
        "answer_confidence": 0.5,
        "student_reasoning": "mistake",
        "reasoning_confidence": 0.5,
    }
    rendered, final, _ = enforce_learner_correct_answer_consistency(action, question)
    assessment = assess_four_tier_response(action, reference_answers=question["answer"])
    assert rendered is False and final == 0 and assessment["answer_correct"] is False

    simulator = RandomLearnerSimulator(seed=1)
    simulator.concept_labels = ["c1", "c2", "c3", "c4"]
    options = simulator.concept_options(["c1", "c2"], seed=3, count=3)
    assert {"c1", "c2"}.issubset(options)

    prompt = build_response_prompt(
        question={"content": "x", "options": ["A", "B"], "answer": ["A"]},
        short_memory=[], long_memory={}, concept_options=None,
        response_format="four_tier", state_item_alignment={"ablated": True},
    )
    assert "ConceptOptions:" not in prompt
    assert "inferred from item text" in prompt

    aligned = {"ablated": False, "current_concepts": ["c1", "c2"], "selected_evidence_ids": ["h1"]}
    check = build_process_consistency(
        {"identified_concept": "c2", "evidence_refs": ["h1"]}, aligned, ["c1", "c2"]
    )
    assert check["valid"]
    unaligned = {"ablated": True, "current_concepts": [], "selected_evidence_ids": []}
    check = build_process_consistency(
        {"identified_concept": "inferred-topic", "evidence_refs": []}, unaligned, ["c1", "c2"]
    )
    assert check["valid"] and not check["concept_reference_checked"]

    # BAA: TPR=1, TNR=.5 => BA=.75 and penalty=.5 => .375.
    steps = [
        {"uid": "u1", "step_index": 1, "qid": 1, "real_response": 1, "simulated_response": 1,
         "irt_ability_difficulty_evidence": {"learner_theta": 1.0, "item_beta": 1.0}},
        {"uid": "u1", "step_index": 2, "qid": 2, "real_response": 0, "simulated_response": 0,
         "irt_ability_difficulty_evidence": {"learner_theta": 1.0, "item_beta": 2.0}},
        {"uid": "u2", "step_index": 1, "qid": 3, "real_response": 0, "simulated_response": 1,
         "irt_ability_difficulty_evidence": {"learner_theta": 0.0, "item_beta": 1.0}},
    ]
    partition = build_ability_difficulty_partition(steps)
    metrics = evaluate_formal_response_metrics(steps, partition)
    assert metrics["baa"] == 0.375
    assert metrics["balanced_accuracy"] == 0.75
    assert metrics["adcde"] is not None

    audit = verify_process_steps([{
        "enabled_modules": {"structured_response_generation": True},
        "agent_action": {"evidence_refs": []},
        "cognitive_state_item_alignment": {"selected_evidence_ids": ["h1"]},
        "process_consistency": {"valid": False},
    }])
    assert audit["framework"] == "evidence_grounded_process_audit_v1"
    assert audit["evidence_reference_precision"] is None
    print("v6_invariants_test_ok")


if __name__ == "__main__":
    main()

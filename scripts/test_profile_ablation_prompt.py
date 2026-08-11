from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from learner_simulator.educational_multi_agent_prompt import build_response_prompt  # noqa: E402
from learner_simulator.item_conditioned_ability import build_item_conditioned_ability  # noqa: E402


def main() -> None:
    question = {
        "content": "2 + 2 = ?",
        "options": {"A": "3", "B": "4"},
        "answer": "B",
    }
    proficiency = {
        "concept": "addition",
        "source": "dneuralcdm",
        "value": 0.68,
        "level": "developing",
        "response_probability": 0.68,
        "response_probability_source": "dneuralcdm_response_predictor",
        "concept_mastery_value": 0.70,
    }
    leaked_context = {
        "uid": "student-1",
        "cognitive_profile": {"control_traits": {"carelessness": "high"}},
        "ability_profile": {"knowledge_breadth": "high"},
    }
    item_state = build_item_conditioned_ability(
        question=question,
        profile_context=leaked_context,
        memory_context={"short_memory": [], "long_memory": {}},
        proficiency=proficiency,
        irt_evidence=None,
        historical_reflection=None,
        include_profile_evidence=False,
    )
    changed_probability = dict(proficiency)
    changed_probability["response_probability"] = 0.01
    item_state_with_changed_baseline = build_item_conditioned_ability(
        question=question,
        profile_context=leaked_context,
        memory_context={"short_memory": [], "long_memory": {}},
        proficiency=changed_probability,
        irt_evidence=None,
        historical_reflection=None,
        include_profile_evidence=False,
    )
    assert item_state == item_state_with_changed_baseline
    prompt = build_response_prompt(
        question=question,
        short_memory=[],
        long_memory={},
        concept_options=["addition", "subtraction", "number"],
        proficiency=proficiency,
        item_conditioned_ability=item_state,
        include_profile_evidence=False,
        include_irt_evidence=False,
    )
    assert not re.search(r"\bprofile\b", prompt, flags=re.IGNORECASE)
    assert "carelessness" not in prompt
    assert "knowledge_breadth" not in prompt
    assert "# NCDM State Evidence #" in prompt
    print("profile_ablation_prompt_test_ok")


if __name__ == "__main__":
    main()

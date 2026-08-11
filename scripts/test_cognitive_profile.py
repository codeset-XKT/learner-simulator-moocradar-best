from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from learner_simulator.agent4edu_prompt import build_profile_system_prompt  # noqa: E402
from learner_simulator.behavior import non_cognitive_factors  # noqa: E402
from learner_simulator.cognitive_profile import build_cognitive_profile  # noqa: E402
from learner_simulator.profile import build_learner_profiles  # noqa: E402
from learner_simulator.simulators.llm_simulator import _without_cognitive_profile  # noqa: E402


def main() -> None:
    sequence = [
        {"qid": 1, "cid": 10, "response": 1, "timestamp": 1},
        {"qid": 2, "cid": 11, "response": 1, "timestamp": 2},
        {"qid": 3, "cid": 10, "response": 0, "timestamp": 3},
        {"qid": 4, "cid": 10, "response": 0, "timestamp": 4},
        {"qid": 5, "cid": 11, "response": 0, "timestamp": 5},
        {"qid": 6, "cid": 12, "response": 1, "timestamp": 6},
        {"qid": 7, "cid": 12, "response": 1, "timestamp": 7},
        {"qid": 8, "cid": 13, "response": 0, "timestamp": 8},
    ]
    questions = {
        "1": {"kc_routes": ["math----fraction----add"]},
        "2": {"kc_routes": ["math----fraction----sub"]},
        "3": {"kc_routes": ["math----fraction----add"]},
        "4": {"kc_routes": ["math----fraction----add"]},
        "5": {"kc_routes": ["math----fraction----sub"]},
        "6": {"kc_routes": ["math----geometry----angle"]},
        "7": {"kc_routes": ["math----geometry----angle"]},
        "8": {"kc_routes": ["math----geometry----area"]},
    }
    profile = build_cognitive_profile(sequence, questions)
    assert set(profile) == {
        "module",
        "source",
        "control_traits",
        "cognitive_affective_proxies",
        "error_generation_traits",
        "transfer_traits",
    }
    assert profile["module"] == "statistical_cognitive_profile"
    assert 0.0 <= profile["cognitive_affective_proxies"]["frustration_risk"] <= 1.0
    assert profile["error_generation_traits"]["max_error_streak"] == 3
    assert 0.0 <= profile["transfer_traits"]["transfer_fragility"] <= 1.0

    row = {
        "uid": "u1",
        "questions": ",".join(str(item["qid"]) for item in sequence),
        "concepts": ",".join(str(item["cid"]) for item in sequence),
        "responses": ",".join(str(item["response"]) for item in sequence),
        "timestamps": ",".join(str(item["timestamp"]) for item in sequence),
    }
    profiles = build_learner_profiles([row], questions)
    context = profiles["u1"].to_context()
    assert "history_summary" in context
    assert "cognitive_profile" in context
    assert "ability_profile" in context
    assert "activity_ratio" not in context
    assert "success_rate" not in context
    assert context["history_summary"]["interaction_count"] == len(sequence)
    assert context["history_summary"]["dominant_concept_id"] == 10
    prompt = build_profile_system_prompt(context)
    assert "# Computed Cognitive Profile #" in prompt
    assert "# Ability Summary #" in prompt
    assert "knowledge breadth" in prompt
    assert "challenge adaptation" in prompt
    assert "profile confidence" in prompt
    assert "irt_ability" not in prompt
    assert "generic agent-style demographic or activity profile" in prompt
    assert "During online study, you exhibit" not in prompt
    stripped_prompt = build_profile_system_prompt(_without_cognitive_profile(context))
    assert "# Computed Cognitive Profile #" not in stripped_prompt
    assert "# Ability Summary #" in stripped_prompt

    cognitive_factors = non_cognitive_factors(context, step_index=3, rng=random.Random(7))
    fallback_factors = non_cognitive_factors(
        _without_cognitive_profile(context),
        step_index=3,
        rng=random.Random(7),
    )
    assert cognitive_factors != fallback_factors

    print("cognitive_profile_test_ok")


if __name__ == "__main__":
    main()

from __future__ import annotations

import random
from typing import Any


ERROR_TYPES = [
    "concept_misunderstanding",
    "calculation_error",
    "careless_error",
    "strategy_error",
    "guessing",
    "reading_error",
    "memory_decay",
]


def non_cognitive_factors(
    profile: dict[str, Any],
    step_index: int,
    rng: random.Random,
) -> dict[str, float]:
    """Estimate lightweight non-cognitive factors for one simulated step."""

    cognitive_profile = profile.get("cognitive_profile")
    if isinstance(cognitive_profile, dict):
        return _cognitive_profile_factors(cognitive_profile, step_index, rng)

    history = profile.get("history_summary") or {}
    ability_estimate = profile.get("ability_estimate") or {}
    ability = float(ability_estimate.get("irt_ability", 0.5) or 0.5)
    interaction_count = int(history.get("interaction_count", 0) or 0)
    distinct_count = int(history.get("distinct_concept_count", 0) or 0)

    exposure_penalty = 0.06 if interaction_count < 20 else 0.03 if distinct_count < 5 else 0.02
    fatigue = min(0.45, 0.02 * step_index + exposure_penalty + rng.uniform(0.0, 0.04))
    attention = min(1.0, max(0.45, 0.88 - fatigue + (0.08 * ability) + rng.uniform(-0.04, 0.04)))
    carelessness = min(0.35, max(0.02, 0.18 - 0.12 * ability + fatigue * 0.35 + rng.uniform(0.0, 0.04)))
    guessing = min(0.35, max(0.03, 0.18 - 0.08 * ability + rng.uniform(-0.02, 0.04)))

    return {
        "attention": round(attention, 6),
        "fatigue": round(fatigue, 6),
        "carelessness": round(carelessness, 6),
        "guessing": round(guessing, 6),
    }


def _cognitive_profile_factors(
    cognitive_profile: dict[str, Any],
    step_index: int,
    rng: random.Random,
) -> dict[str, float]:
    control = cognitive_profile.get("control_traits") or {}
    affective = cognitive_profile.get("cognitive_affective_proxies") or {}
    error_traits = cognitive_profile.get("error_generation_traits") or {}
    transfer = cognitive_profile.get("transfer_traits") or {}

    overall_success = float(control.get("overall_success_rate", 0.5) or 0.5)
    recent_success = float(control.get("recent_success_rate", 0.5) or 0.5)
    stability = float(control.get("mastery_stability", 0.5) or 0.5)
    concentration = float(affective.get("concentration_proxy", 0.5) or 0.5)
    frustration = float(affective.get("frustration_risk", 0.0) or 0.0)
    confusion = float(affective.get("confusion_risk", 0.0) or 0.0)
    boredom = float(affective.get("boredom_risk", 0.0) or 0.0)
    carelessness_trait = float(error_traits.get("carelessness_tendency", 0.0) or 0.0)
    guessing_trait = float(error_traits.get("guessing_tendency", 0.0) or 0.0)
    misconception = float(error_traits.get("misconception_persistence", 0.0) or 0.0)
    error_streak = float(error_traits.get("error_streak_ratio", 0.0) or 0.0)
    transfer_fragility = float(transfer.get("transfer_fragility", 0.5) or 0.5)
    error_pressure = min(
        1.0,
        max(
            0.0,
            0.24 * (1.0 - overall_success)
            + 0.24 * (1.0 - recent_success)
            + 0.16 * error_streak
            + 0.14 * misconception
            + 0.10 * confusion
            + 0.08 * transfer_fragility,
        ),
    )

    fatigue = (
        0.015 * step_index
        + 0.10 * frustration
        + 0.06 * boredom
        + 0.05 * error_pressure
        + rng.uniform(0.0, 0.035)
    )
    attention = (
        0.42
        + 0.32 * concentration
        + 0.16 * recent_success
        + 0.10 * stability
        - 0.12 * frustration
        - 0.10 * error_pressure
        - 0.18 * fatigue
        + rng.uniform(-0.035, 0.035)
    )
    carelessness = (
        0.04
        + 0.38 * carelessness_trait
        + 0.10 * boredom
        + 0.12 * (1.0 - stability)
        + 0.16 * error_pressure
        + 0.18 * fatigue
        + rng.uniform(0.0, 0.035)
    )
    guessing = (
        0.04
        + 0.38 * guessing_trait
        + 0.14 * (1.0 - recent_success)
        + 0.10 * error_pressure
        + rng.uniform(-0.015, 0.035)
    )

    return {
        "attention": round(min(1.0, max(0.35, attention)), 6),
        "fatigue": round(min(0.55, max(0.0, fatigue)), 6),
        "carelessness": round(min(0.45, max(0.02, carelessness)), 6),
        "guessing": round(min(0.45, max(0.03, guessing)), 6),
    }


def apply_behavior_adjustment(
    cognitive_probability: float,
    factors: dict[str, float],
) -> float:
    """Convert cognitive correctness probability into final behavior probability."""

    attention = factors["attention"]
    fatigue = factors["fatigue"]
    carelessness = factors["carelessness"]
    guessing = factors["guessing"]
    adjusted = cognitive_probability * attention
    adjusted -= carelessness * (0.18 + fatigue * 0.25)
    adjusted += (1.0 - cognitive_probability) * guessing * 0.25
    return min(0.98, max(0.02, adjusted))


def sample_error_type(
    components: dict[str, float],
    factors: dict[str, float],
    rng: random.Random,
) -> str:
    mastery = float(components.get("mastery", 0.5))
    item_rate = float(components.get("item_rate", 0.5))
    fatigue = float(factors.get("fatigue", 0.0))
    carelessness = float(factors.get("carelessness", 0.0))
    guessing = float(factors.get("guessing", 0.0))

    weighted = {
        "concept_misunderstanding": max(0.05, 1.0 - mastery),
        "calculation_error": 0.35 + fatigue * 0.4,
        "careless_error": 0.2 + carelessness * 1.5,
        "strategy_error": max(0.05, 1.0 - item_rate) * 0.6,
        "guessing": guessing,
        "reading_error": 0.08 + fatigue * 0.3,
        "memory_decay": 0.12 + max(0.0, 0.6 - mastery) * 0.3,
    }
    total = sum(weighted.values())
    draw = rng.random() * total
    cumulative = 0.0
    for error_type, weight in weighted.items():
        cumulative += weight
        if draw <= cumulative:
            return error_type
    return "concept_misunderstanding"

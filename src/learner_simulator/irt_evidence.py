from __future__ import annotations

from typing import Any


def build_irt_ability_item_evidence(
    irt_model: Any,
    uid: str,
    qid: int,
) -> dict[str, Any]:
    """Build Agent4Edu-style IRT evidence for the current learner-item pair.

    The evidence separates learner-level ability from concept proficiency.
    ``theta`` and ``beta`` are Rasch/1PL parameters estimated from observed
    history; they should inform relative challenge and confidence, not provide
    a sampled correctness label.
    """

    theta = float(irt_model.user_theta(uid))
    beta = float(irt_model.item_beta(qid))
    margin = theta - beta
    ability_score = float(irt_model.ability_score(uid))
    difficulty_score = float(irt_model.item_difficulty_score(qid))
    return {
        "module": "irt_ability_difficulty_evidence",
        "source": "rasch_1pl_from_observed_history",
        "learner_theta": round(theta, 6),
        "item_beta": round(beta, 6),
        "theta_minus_beta": round(margin, 6),
        "learner_ability_level": _ability_level(ability_score),
        "item_difficulty_level": _difficulty_level(difficulty_score),
        "relative_challenge": _relative_challenge(margin),
        "boundary_band": _boundary_band(abs(margin)),
        "rasch_expected_success": round(float(irt_model.predict(uid, qid)), 6),
        "evidence_role": (
            "learner_ability_vs_item_difficulty_signal_not_knowledge_proficiency_or_response_label"
        ),
        "prompt_instruction": (
            "Use IRT as learner-level ability-difficulty evidence. It may change "
            "confidence, effort, and whether success is plausible on this item, "
            "but it must not replace NCDM/KT concept proficiency and must not be "
            "copied as a correctness label."
        ),
    }


def prompt_irt_evidence(evidence: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the prompt-safe view, omitting direct probability fields."""

    if not evidence or evidence.get("ablated"):
        return evidence
    return {
        "module": evidence.get("module"),
        "source": evidence.get("source"),
        "learner_theta": evidence.get("learner_theta"),
        "item_beta": evidence.get("item_beta"),
        "theta_minus_beta": evidence.get("theta_minus_beta"),
        "learner_ability_level": evidence.get("learner_ability_level"),
        "item_difficulty_level": evidence.get("item_difficulty_level"),
        "relative_challenge": evidence.get("relative_challenge"),
        "boundary_band": evidence.get("boundary_band"),
        "evidence_role": evidence.get("evidence_role"),
        "prompt_instruction": evidence.get("prompt_instruction"),
    }


def _ability_level(score: float) -> str:
    if score >= 0.67:
        return "high"
    if score >= 0.34:
        return "medium"
    return "low"


def _difficulty_level(score: float) -> str:
    if score >= 0.67:
        return "high"
    if score >= 0.34:
        return "medium"
    return "low"


def _relative_challenge(margin: float) -> str:
    if margin >= 0.75:
        return "below_learner_ability"
    if margin <= -0.75:
        return "above_learner_ability"
    return "near_learner_boundary"


def _boundary_band(abs_margin: float) -> str:
    if abs_margin >= 1.25:
        return "clear"
    if abs_margin >= 0.50:
        return "moderate"
    return "uncertain"

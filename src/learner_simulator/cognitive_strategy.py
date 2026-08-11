from __future__ import annotations

from typing import Any


def select_cognitive_strategy(
    mastery: float,
    profile: dict[str, Any],
    memory_context: dict[str, Any],
    behavior_factors: dict[str, float],
    has_options: bool,
    current_cid: int,
    current_routes: list[str],
) -> dict[str, Any]:
    """Build learner cognitive evidence without choosing a latent state.

    This module summarizes observable evidence from history, mastery,
    cognitive profile, and non-cognitive factors. It never emits a selected
    cognitive mode, soft weights, p_correct, a sampled label, the reference
    answer, or the target learner response.
    """

    related = _related_records(memory_context, current_cid, current_routes)
    observed = [
        record
        for record in related
        if record.get("source") == "observed_history"
    ]
    recent_simulated = [
        record
        for record in related
        if record.get("source") == "simulated"
    ]
    correct_count = sum(int(record.get("simulated_response", 0)) == 1 for record in observed)
    wrong_count = len(observed) - correct_count
    error_rate = wrong_count / len(observed) if observed else None
    observed_success_rate = correct_count / len(observed) if observed else None

    attention = float(behavior_factors.get("attention", 0.7))
    fatigue = float(behavior_factors.get("fatigue", 0.2))
    carelessness = float(behavior_factors.get("carelessness", 0.1))
    guessing = float(behavior_factors.get("guessing", 0.1))
    ability_estimate = profile.get("ability_estimate") or {}
    raw_ability = float(ability_estimate.get("irt_ability", 0.5) or 0.5)
    cognitive_profile = profile.get("cognitive_profile") or {}
    control_traits = cognitive_profile.get("control_traits") or {}
    affective_traits = cognitive_profile.get("cognitive_affective_proxies") or {}
    error_traits = cognitive_profile.get("error_generation_traits") or {}
    transfer_traits = cognitive_profile.get("transfer_traits") or {}
    if isinstance(cognitive_profile, dict) and cognitive_profile:
        overall_success = float(control_traits.get("overall_success_rate", raw_ability) or raw_ability)
        recent_success = float(control_traits.get("recent_success_rate", overall_success) or overall_success)
        mastery_stability = float(control_traits.get("mastery_stability", 0.5) or 0.5)
    else:
        overall_success = raw_ability
        recent_success = raw_ability
        mastery_stability = 0.5
    concentration_proxy = float(affective_traits.get("concentration_proxy", 0.5) or 0.5)
    frustration_risk = float(affective_traits.get("frustration_risk", 0.0) or 0.0)
    confusion_risk = float(affective_traits.get("confusion_risk", 0.0) or 0.0)
    boredom_risk = float(affective_traits.get("boredom_risk", 0.0) or 0.0)
    carelessness_trait = float(error_traits.get("carelessness_tendency", 0.0) or 0.0)
    guessing_trait = float(error_traits.get("guessing_tendency", 0.0) or 0.0)
    misconception_trait = float(error_traits.get("misconception_persistence", 0.0) or 0.0)
    error_streak = float(error_traits.get("error_streak_ratio", 0.0) or 0.0)
    transfer_fragility = float(transfer_traits.get("transfer_fragility", 0.5) or 0.5)
    error_pressure = _clip(
        0.18 * (1.0 - overall_success)
        + 0.18 * (1.0 - recent_success)
        + 0.12 * error_streak
        + 0.10 * misconception_trait
        + 0.08 * confusion_risk
        + 0.06 * transfer_fragility
    )
    success_support = _clip(
        0.45 * overall_success
        + 0.35 * recent_success
        + 0.20 * mastery_stability
    )
    attention = min(1.0, max(0.35, (0.80 * attention) + (0.20 * concentration_proxy) - (0.06 * frustration_risk)))
    fatigue = min(0.60, max(0.0, fatigue + (0.06 * frustration_risk) + (0.04 * boredom_risk)))
    carelessness = min(0.50, max(carelessness, (0.65 * carelessness) + (0.35 * carelessness_trait)))
    guessing = min(0.50, max(guessing, (0.70 * guessing) + (0.30 * guessing_trait)))

    strong_observed_success = (
        len(observed) >= 3 and (observed_success_rate or 0.0) >= 0.65
        and error_pressure < 0.55
    )
    repeated_observed_error = (
        mastery < 0.55
        and len(observed) >= 4
        and wrong_count >= 3
        and correct_count <= 1
        and (error_rate or 0.0) >= 0.75
    )
    related_signal = _related_signal(
        observed_count=len(observed),
        correct_count=correct_count,
        wrong_count=wrong_count,
        strong_observed_success=strong_observed_success,
        repeated_observed_error=repeated_observed_error,
    )
    stability_signal = _stability_signal(
        overall_success=overall_success,
        recent_success=recent_success,
        mastery_stability=mastery_stability,
        error_pressure=error_pressure,
    )
    behavior_signal = _behavior_signal(
        attention=attention,
        fatigue=fatigue,
        carelessness=carelessness,
        guessing=guessing,
    )
    transfer_signal = _band(1.0 - transfer_fragility, 0.35, 0.72)
    return {
        "evidence_type": "cognitive_evidence",
        "mastery_band": _band(mastery, 0.35, 0.72),
        "mastery_value": round(float(mastery), 4),
        "related_history": {
            "signal": related_signal,
            "history_count": len(related),
            "observed_count": len(observed),
            "simulated_count": len(recent_simulated),
            "correct_count": correct_count,
            "wrong_count": wrong_count,
            "error_rate": round(error_rate, 4) if error_rate is not None else None,
            "success_rate": round(observed_success_rate, 4)
            if observed_success_rate is not None
            else None,
        },
        "profile_evidence": {
            "stability_signal": stability_signal,
            "overall_success": round(overall_success, 4),
            "recent_success": round(recent_success, 4),
            "mastery_stability": round(mastery_stability, 4),
            "success_support": round(success_support, 4),
            "error_pressure": round(error_pressure, 4),
            "concentration": _band(concentration_proxy, 0.45, 0.75),
            "confusion": _band(confusion_risk, 0.18, 0.38),
            "carelessness_trait": _band(carelessness_trait, 0.08, 0.22),
            "guessing_trait": _band(guessing_trait, 0.08, 0.22),
            "misconception_persistence": _band(misconception_trait, 0.10, 0.30),
            "transfer_adaptability": transfer_signal,
        },
        "behavior_cues": {
            "signal": behavior_signal,
            "attention": _band(attention, 0.55, 0.8),
            "fatigue": _band(fatigue, 0.15, 0.35),
            "carelessness": _band(carelessness, 0.08, 0.2),
            "guessing": _band(guessing, 0.08, 0.2),
        },
        "evidence_summary": _evidence_summary(
            mastery_band=_band(mastery, 0.35, 0.72),
            related_signal=related_signal,
            stability_signal=stability_signal,
            behavior_signal=behavior_signal,
            transfer_signal=transfer_signal,
        ),
        "controller_note": (
            "These fields summarize observable cognitive evidence only. They do not select a "
            "cognitive mode, assign state weights, or prescribe whether the final answer is correct."
        ),
    }


def format_cognitive_strategy(strategy: dict[str, Any]) -> str:
    related = strategy.get("related_history") or {}
    profile = strategy.get("profile_evidence") or {}
    behavior = strategy.get("behavior_cues") or {}
    summary = strategy.get("evidence_summary") or []
    summary_text = "\n".join(f"- {item}" for item in summary)
    return (
        f"- mastery evidence: {strategy.get('mastery_band', 'unknown')} "
        f"({strategy.get('mastery_value', 'unknown')})\n"
        f"- related-history evidence: {related.get('signal', 'unknown')}; "
        f"observed={related.get('observed_count', 0)}, correct={related.get('correct_count', 0)}, "
        f"wrong={related.get('wrong_count', 0)}, error_rate={_format_optional(related.get('error_rate'))}\n"
        f"- profile stability evidence: {profile.get('stability_signal', 'unknown')}; "
        f"overall_success={_format_optional(profile.get('overall_success'))}, "
        f"recent_success={_format_optional(profile.get('recent_success'))}, "
        f"error_pressure={_format_optional(profile.get('error_pressure'))}\n"
        f"- cognitive-affective evidence: concentration={profile.get('concentration', 'unknown')}, "
        f"confusion={profile.get('confusion', 'unknown')}, carelessness_trait={profile.get('carelessness_trait', 'unknown')}, "
        f"guessing_trait={profile.get('guessing_trait', 'unknown')}, transfer_adaptability={profile.get('transfer_adaptability', 'unknown')}\n"
        f"- current behavior cues: {behavior.get('signal', 'unknown')}; "
        f"attention={behavior.get('attention', 'unknown')}, fatigue={behavior.get('fatigue', 'unknown')}, "
        f"carelessness={behavior.get('carelessness', 'unknown')}, guessing={behavior.get('guessing', 'unknown')}\n"
        f"{summary_text}\n"
        "- Use these as evidence for Four-tier learner-state inference. Do not treat them as a selected state, "
        "a predefined state selector, or a correctness label."
    )


def _related_signal(
    observed_count: int,
    correct_count: int,
    wrong_count: int,
    strong_observed_success: bool,
    repeated_observed_error: bool,
) -> str:
    if observed_count == 0:
        return "no_related_observed_history"
    if repeated_observed_error:
        return "repeated_related_errors"
    if strong_observed_success:
        return "stable_related_success"
    if wrong_count and correct_count:
        return "mixed_related_history"
    if wrong_count:
        return "limited_related_errors"
    return "limited_related_success"


def _stability_signal(
    overall_success: float,
    recent_success: float,
    mastery_stability: float,
    error_pressure: float,
) -> str:
    if overall_success >= 0.75 and recent_success >= 0.75 and mastery_stability >= 0.65:
        return "stable_high_success"
    if error_pressure >= 0.45 or (overall_success < 0.45 and recent_success < 0.55):
        return "unstable_or_error_prone"
    if abs(recent_success - overall_success) >= 0.25:
        return "recent_shift"
    return "moderate_or_mixed"


def _behavior_signal(
    attention: float,
    fatigue: float,
    carelessness: float,
    guessing: float,
) -> str:
    if attention < 0.55 or fatigue >= 0.35:
        return "low_attention_or_fatigue"
    if carelessness >= 0.2:
        return "carelessness_cue"
    if guessing >= 0.2:
        return "guessing_cue"
    return "ordinary_attention"


def _evidence_summary(
    mastery_band: str,
    related_signal: str,
    stability_signal: str,
    behavior_signal: str,
    transfer_signal: str,
) -> list[str]:
    return [
        f"mastery is {mastery_band}",
        f"related practice signal is {related_signal}",
        f"profile stability signal is {stability_signal}",
        f"current behavior signal is {behavior_signal}",
        f"transfer adaptability is {transfer_signal}",
    ]


def _format_optional(value: Any) -> str:
    if value is None:
        return "unavailable"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _related_records(
    memory_context: dict[str, Any],
    current_cid: int,
    current_routes: list[str],
) -> list[dict[str, Any]]:
    short = list(memory_context.get("short_memory", []))
    long_memory = memory_context.get("long_memory", {})
    significant = list(long_memory.get("significant_facts", []))
    seen: set[tuple[Any, Any, Any]] = set()
    records: list[dict[str, Any]] = []
    for record in [*short, *significant]:
        if not _is_related(record, current_cid, current_routes):
            continue
        key = (
            record.get("source"),
            record.get("step_index"),
            record.get("qid"),
        )
        if key in seen:
            continue
        seen.add(key)
        records.append(record)
    return records


def _is_related(
    record: dict[str, Any],
    current_cid: int,
    current_routes: list[str],
) -> bool:
    try:
        if int(record.get("cid", -1)) == int(current_cid):
            return True
    except (TypeError, ValueError):
        pass
    left = _leaf_routes(record.get("kc_routes") or [])
    right = _leaf_routes(current_routes)
    return bool(left and right and left.intersection(right))


def _leaf_routes(routes: list[str]) -> set[str]:
    leaves: set[str] = set()
    for route in routes:
        parts = [
            part.strip()
            for part in str(route).replace("----", "/").split("/")
            if part.strip()
        ]
        if parts:
            leaves.add(parts[-1])
    return leaves


def _band(value: float, low: float, high: float) -> str:
    if value >= high:
        return "high"
    if value >= low:
        return "medium"
    return "low"


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, value))

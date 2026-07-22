from __future__ import annotations

import re
from typing import Any

from learner_simulator.agent4edu_prompt import build_action_prompt


ROLE_POLICIES = {
    "competent": (
        "You are the Competent Learner Agent. Your independent goal is to model "
        "the path a normally prepared version of this same learner would take. "
        "Use demonstrated mastery, stable related memory, and ability evidence. "
        "Do not become a teacher, but do not suppress a plausible correct answer."
    ),
    "misconception": (
        "You are the Misconception Learner Agent. Your independent goal is to "
        "model a plausible concept-confusion or wrong-transfer path for this same "
        "learner when the evidence supports it. If the history and proficiency do "
        "not support a misconception, you may still produce a correct attempt, but "
        "your reasoning should remain learner-level and uncertainty-aware."
    ),
    "careless": (
        "You are the Careless Learner Agent. Your independent goal is to model a "
        "plausible first-attempt slip for this same learner under attention limits, "
        "option traps, incomplete reading, or calculation shortcuts. If the item is "
        "routine and the learner evidence is strong, a correct but terse attempt is "
        "still allowed."
    ),
}


def build_role_system_prompt(base_system_prompt: str, role: str) -> str:
    policy = ROLE_POLICIES[role]
    return (
        f"{base_system_prompt}\n\n"
        "# Independent Role Objective #\n"
        f"{policy}\n"
        "You produce one candidate learner response. You do not know the reference "
        "answer, and you must not output a correctness label."
    )


def build_role_action_prompt(
    role: str,
    question: dict[str, Any],
    short_memory: list[dict[str, Any]],
    long_memory: dict[str, Any],
    concept_options: list[str],
    proficiency: dict[str, Any] | None,
    behavior_factors: dict[str, float] | None,
    tendency_calibration: dict[str, Any] | None,
) -> str:
    base_prompt = build_action_prompt(
        question=question,
        short_memory=short_memory,
        long_memory=long_memory,
        concept_options=concept_options,
        proficiency=proficiency,
        behavior_factors=behavior_factors,
        response_format="four_tier",
        cognitive_strategy=None,
        tendency_calibration=tendency_calibration,
    )
    return (
        "# Candidate Role #\n"
        f"{ROLE_POLICIES[role]}\n\n"
        "# Role Constraint #\n"
        "Generate the response according to this role's independent cognitive policy. "
        "Do not coordinate with other roles and do not mention the role name in the final output.\n\n"
        f"{base_prompt}"
    )


def build_arbiter_system_prompt(base_system_prompt: str) -> str:
    return (
        f"{base_system_prompt}\n\n"
        "You are the Learning Performance Regulator Agent. Your goal is not to solve "
        "the item from scratch. Your goal is to choose which candidate response best "
        "matches how the learner's knowledge readiness, performance stability, "
        "transfer readiness, and cognitive load are likely to appear in this first attempt."
    )


def build_arbiter_prompt(
    question: dict[str, Any],
    profile_context: dict[str, Any],
    memory_context: dict[str, Any],
    proficiency: dict[str, Any] | None,
    behavior_factors: dict[str, float] | None,
    tendency_calibration: dict[str, Any] | None,
    candidates: dict[str, dict[str, Any]],
    learning_performance_regulation: dict[str, Any] | None = None,
) -> str:
    regulation = learning_performance_regulation or build_learning_performance_regulation(
        profile_context=profile_context,
        proficiency=proficiency,
        behavior_factors=behavior_factors,
        tendency_calibration=tendency_calibration,
        candidates=candidates,
    )
    chunks = [
        "# Arbitration Task #",
        "Choose exactly one candidate response as this learner's final first attempt.",
        "Do not create a new answer, do not repair a candidate, and do not solve the item independently.",
        "Use the candidate's StudentAnswer and confidence exactly as submitted.",
        "",
        "# Learner Evidence #",
        _compact_json_like(
            {
                "history_summary": profile_context.get("history_summary"),
                "cognitive_profile": profile_context.get("cognitive_profile"),
                "ability_profile": profile_context.get("ability_profile"),
                "proficiency": proficiency,
                "non_cognitive_state": behavior_factors,
                "response_tendency": tendency_calibration,
                "short_memory_count": len(memory_context.get("short_memory", [])),
                "reinforced_memory_count": len(
                    (memory_context.get("long_memory") or {}).get("significant_facts", [])
                ),
            }
        ),
        "",
        "# Learning Performance Regulation #",
        _compact_json_like(regulation),
        (
            "Interpretation: this regulation summarizes how knowledge readiness, performance stability, "
            "transfer readiness, and cognitive load may transform latent knowledge into observable first-attempt behavior. "
            "Stable performance supports the competent path; transfer-fragile performance supports the misconception path; "
            "load-induced slips support the careless path."
        ),
        "",
        "# Exercise #",
        f"Textual Content: {question.get('content', '')}",
        f"Options: {question.get('options', '')}",
        "",
        "# Candidate Responses #",
    ]
    for role, candidate in candidates.items():
        action = candidate.get("action") or {}
        chunks.extend(
            [
                f"Candidate={role}",
                f"Attempt: {action.get('attempt', '')}",
                f"IdentifiedConcept: {action.get('identified_concept', '')}",
                f"StudentAnswer: {action.get('student_answer', '')}",
                f"AnswerConfidence: {action.get('answer_confidence', '')}",
                f"StudentReasoning: {action.get('student_reasoning', '')}",
                f"ReasoningConfidence: {action.get('reasoning_confidence', '')}",
                "",
            ]
        )
    chunks.extend(
        [
            "# Selection Rules #",
            "1. Prefer the competent candidate when proficiency and related memory are stable and the response is learner-like.",
            "2. Prefer the misconception candidate when history, transfer fragility, or repeated errors make a concept-confusion path plausible.",
            "3. Prefer the careless candidate when non-cognitive state, item format, or option traps make a slip plausible.",
            "4. Do not maximize correctness. Also do not maximize incorrectness. Choose the most profile-consistent first attempt.",
            "5. You may consider disagreement among candidates as uncertainty evidence, but you must still select one candidate.",
            "6. If regulation_decision is transfer_fragile or load_induced_slip and candidates disagree, competent is not the default. Select competent only when the learner evidence clearly rejects both non-competent paths.",
            "7. If competent and a regulated alternative path give different answers with similar confidence, use preferred_response_path unless knowledge readiness and performance stability strongly support competent.",
            "",
            "Output exactly:",
            "SelectedAgent: <competent, misconception, or careless>",
            "ArbiterConfidence: <0.80 for high, 0.50 for medium, or 0.20 for low>",
            "ArbiterRationale: <one concise sentence about learning-performance consistency and why alternatives were rejected, not correctness>",
        ]
    )
    return "\n".join(chunks)


def parse_arbiter_response(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    selected_match = re.search(
        r"SelectedAgent\s*:\s*(competent|misconception|careless)",
        raw,
        flags=re.IGNORECASE,
    )
    if not selected_match:
        return None
    confidence_match = re.search(
        r"ArbiterConfidence\s*:\s*([01](?:\.\d+)?)",
        raw,
        flags=re.IGNORECASE,
    )
    rationale_match = re.search(
        r"ArbiterRationale\s*:\s*(.+)",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return {
        "selected_agent": selected_match.group(1).lower(),
        "arbiter_confidence": _confidence(
            confidence_match.group(1) if confidence_match else None
        ),
        "arbiter_rationale": (
            rationale_match.group(1).strip() if rationale_match else ""
        ),
    }


def fallback_arbiter_selection(
    candidates: dict[str, dict[str, Any]],
    tendency_calibration: dict[str, Any] | None,
) -> dict[str, Any]:
    available = [role for role, item in candidates.items() if item.get("action")]
    if not available:
        return {
            "selected_agent": "competent",
            "arbiter_confidence": 0.2,
            "arbiter_rationale": "No parsed candidates were available; defaulted to competent.",
            "fallback": True,
        }
    band = str((tendency_calibration or {}).get("band", "mixed"))
    if "strong-correct" in band or band == "correct-leaning":
        selected = "competent" if "competent" in available else available[0]
    elif "error" in band:
        selected = "misconception" if "misconception" in available else available[-1]
    else:
        selected = "careless" if "careless" in available else available[0]
    return {
        "selected_agent": selected,
        "arbiter_confidence": 0.2,
        "arbiter_rationale": "Fallback selection based on calibrated response tendency.",
        "fallback": True,
    }


def build_learning_performance_regulation(
    profile_context: dict[str, Any],
    proficiency: dict[str, Any] | None,
    behavior_factors: dict[str, float] | None,
    tendency_calibration: dict[str, Any] | None,
    candidates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cognitive = profile_context.get("cognitive_profile") or {}
    control = cognitive.get("control_traits") or {}
    affective = cognitive.get("cognitive_affective_proxies") or {}
    errors = cognitive.get("error_generation_traits") or {}
    transfer = cognitive.get("transfer_traits") or {}

    mastery = _optional_float((proficiency or {}).get("value"))
    attention = _optional_float((behavior_factors or {}).get("attention"))
    fatigue = _optional_float((behavior_factors or {}).get("fatigue"))
    carelessness = _optional_float((behavior_factors or {}).get("carelessness"))
    guessing = _optional_float((behavior_factors or {}).get("guessing"))
    trend = _optional_float(control.get("success_trend"))
    stability = _optional_float(control.get("mastery_stability"))
    misconception = _optional_float(errors.get("misconception_persistence"))
    recovery = _optional_float(errors.get("error_recovery_rate"))
    transfer_fragility = _optional_float(transfer.get("transfer_fragility"))
    confusion = _optional_float(affective.get("confusion_risk"))
    frustration = _optional_float(affective.get("frustration_risk"))

    answers = {
        role: _normalized_candidate_answer(item)
        for role, item in candidates.items()
        if item.get("action")
    }
    unique_answers = {answer for answer in answers.values() if answer}
    competent_answer = answers.get("competent")
    non_competent_disagree = any(
        answer and competent_answer and answer != competent_answer
        for role, answer in answers.items()
        if role != "competent"
    )

    instability_score = 0.0
    evidence: list[str] = []
    band = str((tendency_calibration or {}).get("band", "mixed"))
    if "error" in band:
        instability_score += 0.25
        evidence.append(f"response tendency is {band}")
    elif band == "mixed":
        instability_score += 0.12
        evidence.append("response tendency is mixed")
    elif "strong-correct" in band:
        instability_score -= 0.15
    elif "correct" in band:
        instability_score -= 0.05

    knowledge_evidence: list[str] = []
    if mastery is not None:
        if mastery < 0.42:
            instability_score += 0.25
            evidence.append("current mastery is low")
        elif mastery < 0.58:
            instability_score += 0.12
            evidence.append("current mastery is medium-fragile")
        elif mastery >= 0.78:
            instability_score -= 0.12
        knowledge_evidence.append(f"DKT mastery={round(mastery, 3)}")

    load_evidence: list[str] = []
    if attention is not None and attention < 0.55:
        instability_score += 0.18
        evidence.append("attention is low")
        load_evidence.append(f"attention={round(attention, 3)}")
    if fatigue is not None and fatigue > 0.35:
        instability_score += 0.12
        evidence.append("fatigue is high")
        load_evidence.append(f"fatigue={round(fatigue, 3)}")
    if carelessness is not None and carelessness > 0.20:
        instability_score += 0.18
        evidence.append("carelessness is high")
        load_evidence.append(f"carelessness={round(carelessness, 3)}")
    if guessing is not None and guessing > 0.20:
        instability_score += 0.12
        evidence.append("guessing tendency is high")
        load_evidence.append(f"guessing={round(guessing, 3)}")
    if misconception is not None and misconception > 0.25:
        instability_score += 0.18
        evidence.append("misconception persistence is visible")
    if transfer_fragility is not None and transfer_fragility > 0.55:
        instability_score += 0.12
        evidence.append("transfer is fragile")
    if confusion is not None and confusion > 0.20:
        instability_score += 0.12
        evidence.append("confusion proxy is high")
    if frustration is not None and frustration > 0.25:
        instability_score += 0.08
        evidence.append("frustration proxy is high")
    if trend is not None and trend < -0.10:
        instability_score += 0.10
        evidence.append("recent trend is declining")
    if stability is not None and stability < 0.45:
        instability_score += 0.10
        evidence.append("mastery stability is low")
    if recovery is not None and recovery < 0.35:
        instability_score += 0.08
        evidence.append("error recovery is weak")

    if len(unique_answers) > 1:
        instability_score += 0.18
        evidence.append("candidate answers disagree")
    if non_competent_disagree:
        instability_score += 0.12
        evidence.append("regulated alternatives disagree with competent")

    instability_score = min(1.0, max(0.0, instability_score))
    preferred = _preferred_alternative_agent(
        errors=errors,
        transfer=transfer,
        behavior_factors=behavior_factors or {},
        answers=answers,
    )
    knowledge_readiness = _readiness_level(mastery, high=0.72, low=0.42)
    stability_value = _mean_available(
        [
            stability,
            recovery,
            _invert_optional(abs(trend) if trend is not None else None),
        ]
    )
    performance_stability = _readiness_level(stability_value, high=0.70, low=0.45)
    transfer_value = None
    if transfer_fragility is not None:
        transfer_value = 1.0 - transfer_fragility
    transfer_readiness = _readiness_level(transfer_value, high=0.65, low=0.40)
    load_value = _mean_available(
        [
            _invert_optional(attention),
            fatigue,
            carelessness,
            guessing,
            confusion,
            frustration,
        ]
    )
    cognitive_load = _load_level(load_value)
    decision = _regulation_decision(
        instability_score=instability_score,
        knowledge_readiness=knowledge_readiness,
        performance_stability=performance_stability,
        transfer_readiness=transfer_readiness,
        cognitive_load=cognitive_load,
        preferred=preferred,
        non_competent_disagree=non_competent_disagree,
    )
    preferred_path = _preferred_response_path(decision, preferred)
    return {
        "knowledge_readiness": {
            "level": knowledge_readiness,
            "value": _round_optional(mastery),
            "evidence": knowledge_evidence,
        },
        "performance_stability": {
            "level": performance_stability,
            "value": _round_optional(stability_value),
            "evidence": [
                f"mastery_stability={_round_optional(stability)}",
                f"error_recovery={_round_optional(recovery)}",
                f"success_trend={_round_optional(trend)}",
            ],
        },
        "transfer_readiness": {
            "level": transfer_readiness,
            "value": _round_optional(transfer_value),
            "evidence": [f"transfer_fragility={_round_optional(transfer_fragility)}"],
        },
        "cognitive_load": {
            "level": cognitive_load,
            "value": _round_optional(load_value),
            "evidence": load_evidence,
        },
        "regulation_decision": decision,
        "preferred_response_path": preferred_path,
        "performance_instability_score": round(instability_score, 3),
        "candidate_answer_count": len(unique_answers),
        "candidate_answers_disagree": len(unique_answers) > 1,
        "non_competent_disagrees_with_competent": non_competent_disagree,
        "preferred_alternative_agent": preferred,
        "competent_guard_active": bool(
            decision in {"transfer_fragile", "load_induced_slip", "uncertain_performance"}
            and non_competent_disagree
            and preferred in candidates
        ),
        "main_evidence": evidence[:6],
    }


def build_arbiter_diagnostics(
    profile_context: dict[str, Any],
    proficiency: dict[str, Any] | None,
    behavior_factors: dict[str, float] | None,
    tendency_calibration: dict[str, Any] | None,
    candidates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return build_learning_performance_regulation(
        profile_context,
        proficiency,
        behavior_factors,
        tendency_calibration,
        candidates,
    )


def apply_competent_dominance_guard(
    arbiter: dict[str, Any],
    diagnostics: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if arbiter.get("selected_agent") != "competent":
        return arbiter
    if not diagnostics.get("competent_guard_active"):
        return arbiter
    preferred = str(diagnostics.get("preferred_alternative_agent") or "")
    if preferred not in candidates or not candidates[preferred].get("action"):
        return arbiter
    adjusted = dict(arbiter)
    adjusted["original_selected_agent"] = "competent"
    adjusted["selected_agent"] = preferred
    adjusted["guard_override"] = True
    adjusted["guard_reason"] = (
        "Learning-performance guard: the regulation decision and candidate disagreement "
        "require selecting the preferred alternative performance path."
    )
    if "arbiter_rationale" in adjusted:
        adjusted["arbiter_rationale"] = (
            str(adjusted["arbiter_rationale"])
            + " Guard override selected "
            + preferred
            + " due to learning-performance regulation."
        )
    return adjusted


def _compact_json_like(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 2500 else text[:2500] + "...[truncated]"


def _normalized_candidate_answer(candidate: dict[str, Any]) -> str:
    action = candidate.get("action") or {}
    return re.sub(r"\s+", "", str(action.get("student_answer", "")).strip().lower())


def _preferred_alternative_agent(
    errors: dict[str, Any],
    transfer: dict[str, Any],
    behavior_factors: dict[str, Any],
    answers: dict[str, str],
) -> str:
    carelessness = _optional_float(behavior_factors.get("carelessness")) or 0.0
    attention = _optional_float(behavior_factors.get("attention"))
    guessing = _optional_float(behavior_factors.get("guessing")) or 0.0
    misconception = _optional_float(errors.get("misconception_persistence")) or 0.0
    transfer_fragility = _optional_float(transfer.get("transfer_fragility")) or 0.0
    careless_score = carelessness + guessing + (0.3 if attention is not None and attention < 0.55 else 0.0)
    misconception_score = misconception + 0.5 * transfer_fragility
    if answers.get("misconception") and answers.get("misconception") != answers.get("competent"):
        misconception_score += 0.2
    if answers.get("careless") and answers.get("careless") != answers.get("competent"):
        careless_score += 0.2
    return "careless" if careless_score > misconception_score else "misconception"


def _regulation_decision(
    instability_score: float,
    knowledge_readiness: str,
    performance_stability: str,
    transfer_readiness: str,
    cognitive_load: str,
    preferred: str,
    non_competent_disagree: bool,
) -> str:
    if cognitive_load == "high" and non_competent_disagree:
        return "load_induced_slip"
    if transfer_readiness == "low" and non_competent_disagree:
        return "transfer_fragile"
    if (
        knowledge_readiness == "high"
        and performance_stability == "high"
        and cognitive_load == "low"
        and instability_score < 0.35
    ):
        return "stable_performance"
    if instability_score >= 0.55 and non_competent_disagree:
        return "load_induced_slip" if preferred == "careless" else "transfer_fragile"
    if instability_score >= 0.35 or performance_stability == "medium":
        return "partial_performance"
    return "stable_performance"


def _preferred_response_path(decision: str, preferred: str) -> str:
    if decision == "stable_performance":
        return "competent"
    if decision == "transfer_fragile":
        return "misconception"
    if decision == "load_induced_slip":
        return "careless"
    if decision == "partial_performance":
        return f"competent_or_{preferred}"
    return preferred


def _readiness_level(value: float | None, high: float, low: float) -> str:
    if value is None:
        return "unknown"
    if value >= high:
        return "high"
    if value >= low:
        return "medium"
    return "low"


def _load_level(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value >= 0.45:
        return "high"
    if value >= 0.25:
        return "medium"
    return "low"


def _mean_available(values: list[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    if not available:
        return None
    return sum(available) / len(available)


def _invert_optional(value: float | None) -> float | None:
    if value is None:
        return None
    return 1.0 - min(1.0, max(0.0, value))


def _round_optional(value: float | None) -> float | str:
    return round(value, 3) if value is not None else "unavailable"


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.5
    return min(1.0, max(0.0, confidence))

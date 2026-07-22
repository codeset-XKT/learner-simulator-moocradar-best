from __future__ import annotations

import re
from typing import Any

from learner_simulator.agent4edu_prompt import build_action_prompt


ROUTES = {
    "mastery_retrieval",
    "partial_reasoning",
    "misconception_transfer",
    "careless_execution",
    "uncertain_guessing",
}


def build_cognitive_profile_view(
    profile_context: dict[str, Any],
    proficiency: dict[str, Any] | None,
    behavior_factors: dict[str, float] | None,
    tendency_calibration: dict[str, Any] | None,
) -> dict[str, Any]:
    cognitive = profile_context.get("cognitive_profile") or {}
    control = cognitive.get("control_traits") or {}
    affective = cognitive.get("cognitive_affective_proxies") or {}
    errors = cognitive.get("error_generation_traits") or {}
    transfer = cognitive.get("transfer_traits") or {}
    ability = profile_context.get("ability_profile") or {}
    return {
        "module": "cognitive_profile_evidence",
        "history_summary": profile_context.get("history_summary"),
        "knowledge_signal": {
            "concept": (proficiency or {}).get("concept"),
            "mastery": _round_optional((proficiency or {}).get("value")),
            "level": (proficiency or {}).get("level"),
        },
        "control_traits": {
            "overall_success": control.get("overall_success_level"),
            "recent_success": control.get("recent_success_level"),
            "success_trend": control.get("success_trend_level"),
            "mastery_stability": control.get("mastery_stability_level"),
            "error_recovery": errors.get("error_recovery_level"),
        },
        "cognitive_affective_proxies": {
            "concentration": affective.get("concentration_level"),
            "confusion": affective.get("confusion_level"),
            "frustration": affective.get("frustration_level"),
            "boredom": affective.get("boredom_level"),
        },
        "error_traits": {
            "carelessness": errors.get("carelessness_level"),
            "guessing": errors.get("guessing_level"),
            "misconception_persistence": errors.get("misconception_persistence_level"),
        },
        "transfer_traits": {
            "same_parent_transfer": transfer.get("same_parent_transfer_level"),
            "transfer_fragility": transfer.get("transfer_fragility_level"),
        },
        "ability_summary": {
            "knowledge_breadth": ability.get("knowledge_breadth"),
            "practice_depth": ability.get("practice_depth"),
            "challenge_adaptation": ability.get("challenge_adaptation"),
            "cross_domain_generalization": ability.get("cross_domain_generalization"),
            "profile_confidence": ability.get("profile_confidence"),
        },
        "non_cognitive_state": behavior_factors,
        "response_tendency": tendency_calibration,
    }


def fallback_ability_boundary(
    cognitive_profile: dict[str, Any],
    proficiency: dict[str, Any] | None,
) -> dict[str, Any]:
    mastery = _optional_float((proficiency or {}).get("value"))
    ability = cognitive_profile.get("ability_summary") or {}
    depth = ability.get("practice_depth") or "medium"
    if mastery is not None and mastery >= 0.7:
        expected_depth = "medium" if depth in {"medium", "high"} else "shallow"
        available = ["recall familiar concept", "apply a practiced single-step rule"]
        weak = ["multi-step verification under unfamiliar wording"]
    elif mastery is not None and mastery < 0.4:
        expected_depth = "shallow"
        available = ["recognize surface cues", "attempt a familiar option pattern"]
        weak = ["stable concept application", "multi-step transfer"]
    else:
        expected_depth = "medium"
        available = ["partial concept recall", "simple calculation or elimination"]
        weak = ["reliable transfer", "complete self-checking"]
    return {
        "module": "ability_boundary_evidence",
        "available_abilities": available,
        "weak_abilities": weak,
        "forbidden_expert_behaviors": [
            "full teacher-style derivation",
            "advanced method not evidenced by learner history",
            "post-hoc expert correction",
        ],
        "expected_solution_depth": expected_depth,
        "boundary_rationale": "Fallback boundary inferred from mastery and ability summary.",
        "fallback": True,
    }


def build_ability_boundary_evidence(
    question: dict[str, Any],
    cognitive_profile: dict[str, Any],
    memory_context: dict[str, Any],
    proficiency: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a deterministic ability boundary from observed learner evidence.

    This replaces the earlier LLM-based ability-boundary generator. The boundary is
    evidence, not a generated response: it summarizes which abilities are
    available, which are fragile, and whether the current item demands more
    than direct recall.
    """

    mastery = _optional_float((proficiency or {}).get("value"))
    ability = cognitive_profile.get("ability_summary") or {}
    transfer = cognitive_profile.get("transfer_traits") or {}
    control = cognitive_profile.get("control_traits") or {}
    item_score = _item_demand_score(question)
    item_demand = "high" if item_score >= 3 else "medium" if item_score >= 1 else "low"
    memory = _memory_support(question, memory_context)

    breadth = _level_value(ability.get("knowledge_breadth"))
    practice = _level_value(ability.get("practice_depth"))
    challenge = _level_value(ability.get("challenge_adaptation"))
    generalization = _level_value(ability.get("cross_domain_generalization"))
    transfer_fragile = _level_value(transfer.get("transfer_fragility"))
    stability = _level_value(control.get("mastery_stability"))
    related_rate = _optional_float(memory.get("related_correct_rate"))
    strong_concept_evidence = (
        mastery is not None
        and item_demand != "high"
        and (
            mastery >= 0.85
            or (
                mastery >= 0.78
                and (
                    memory["identical_correct_count"] > 0
                    or (related_rate is not None and related_rate >= 0.8)
                )
            )
        )
    )

    available: list[str] = []
    weak: list[str] = []
    if mastery is not None and mastery >= 0.75:
        available.extend(["recall familiar concept", "apply a practiced rule"])
    elif mastery is not None and mastery >= 0.45:
        available.extend(["partial concept recall", "simple elimination"])
        weak.append("stable independent application")
    else:
        available.extend(["surface cue recognition", "guessing from familiar options"])
        weak.extend(["concept recall", "stable application"])

    if memory["identical_correct_count"] > 0:
        available.append("retrieve a recently seen or reinforced similar item")
    elif memory["related_correct_rate"] is not None and memory["related_correct_rate"] >= 0.7:
        available.append("use related successful practice")
    elif memory["related_total"] > 0:
        weak.append("reliable transfer from related memory")

    if item_demand in {"medium", "high"}:
        if practice <= 1 and not strong_concept_evidence:
            weak.append("multi-step reasoning under current item demand")
        if breadth <= 1 and not strong_concept_evidence:
            weak.append("knowledge breadth for distinguishing alternatives")
        if challenge <= 1 and not strong_concept_evidence:
            weak.append("adaptation to non-routine or difficult items")
    if item_demand == "high" or generalization <= 1 or transfer_fragile >= 3:
        weak.append("cross-concept transfer or comparison")
    if stability <= 1 and not strong_concept_evidence:
        weak.append("stable performance under uncertainty")

    reasoning_capacity = _reasoning_capacity(
        mastery=mastery,
        practice=practice,
        breadth=breadth,
        challenge=challenge,
        generalization=generalization,
        memory_support=memory,
    )
    if strong_concept_evidence and reasoning_capacity == "shallow":
        reasoning_capacity = "medium"
    if item_demand == "high" and reasoning_capacity != "deep":
        boundary_risk = "high"
    elif item_demand == "medium" and reasoning_capacity == "shallow":
        boundary_risk = "high"
    elif weak and not strong_concept_evidence:
        boundary_risk = "medium"
    else:
        boundary_risk = "low"

    return {
        "module": "ability_boundary_evidence",
        "method": "deterministic_from_profile_memory_and_item",
        "available_abilities": _dedupe(available),
        "weak_abilities": _dedupe(weak),
        "forbidden_expert_behaviors": [
            "full teacher-style derivation",
            "advanced method not evidenced by learner history",
            "post-hoc expert correction",
        ],
        "expected_solution_depth": reasoning_capacity,
        "learner_reasoning_capacity": reasoning_capacity,
        "item_demand": item_demand,
        "item_demand_score": item_score,
        "boundary_risk": boundary_risk,
        "concept_mastery_overrides_global_profile_weakness": strong_concept_evidence,
        "memory_support": memory,
        "boundary_rationale": _ability_boundary_rationale(
            item_demand=item_demand,
            reasoning_capacity=reasoning_capacity,
            boundary_risk=boundary_risk,
            mastery=mastery,
            memory=memory,
        ),
    }


def build_cognitive_route_prompt(
    question: dict[str, Any],
    cognitive_profile: dict[str, Any],
    ability_boundary: dict[str, Any],
    proficiency: dict[str, Any] | None,
    behavior_factors: dict[str, float] | None,
    tendency_calibration: dict[str, Any] | None,
    route_evidence: dict[str, Any] | None = None,
) -> str:
    evidence = route_evidence or infer_cognitive_route_evidence(
        question=question,
        cognitive_profile=cognitive_profile,
        ability_boundary=ability_boundary,
        proficiency=proficiency,
        behavior_factors=behavior_factors,
        tendency_calibration=tendency_calibration,
    )
    return "\n".join(
        [
            "# Cognitive Route Agent #",
            "Select the learner's likely first-attempt cognitive route before answer generation.",
            "Do not solve the item and do not output a final answer.",
            "",
            "Allowed routes:",
            "- mastery_retrieval: stable recall or routine application",
            "- partial_reasoning: incomplete but relevant understanding",
            "- misconception_transfer: wrong concept or fragile transfer path",
            "- careless_execution: knows enough but slips in reading, calculation, or option choice",
            "- uncertain_guessing: weak knowledge with guessing behavior",
            "",
            "# Cognitive Profile #",
            _compact(cognitive_profile),
            "",
            "# Ability Boundary #",
            _compact(ability_boundary),
            "",
            "# Cognitive Route Evidence #",
            _compact(evidence),
            "Interpretation rules:",
            "- mastery_retrieval is allowed only when knowledge, history, ability boundary, and item demand are all stable.",
            "- If mastery is medium/mixed, history is fragile, or ability_boundary_risk is high, do not choose mastery_retrieval.",
            "- If weak abilities mention ordering, sequence, procedure, comparison, transfer, or multi-step reasoning, prefer partial_reasoning or misconception_transfer unless the memory evidence is nearly identical and stable.",
            "- A shallow expected solution depth means limited reasoning capacity; it is not by itself evidence for mastery_retrieval.",
            "",
            "# Knowledge and State #",
            _compact(
                {
                    "proficiency": proficiency,
                    "non_cognitive_state": behavior_factors,
                    "response_tendency": tendency_calibration,
                }
            ),
            "",
            "# Exercise #",
            f"Textual Content: {question.get('content', '')}",
            f"Options: {question.get('options', '')}",
            "",
            "Output exactly:",
            "CognitiveRoute: <one allowed route>",
            "ExpectedCorrectnessTendency: <favorable, uncertain, or error-prone>",
            "ExpectedConfidence: <high, medium, or low>",
            "RouteRationale: <one concise sentence>",
        ]
    )


def parse_cognitive_route(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    route = (_field(raw, "CognitiveRoute") or "").strip().lower()
    route = re.sub(r"[^a-z_]", "", route)
    if route not in ROUTES:
        return None
    tendency = (_field(raw, "ExpectedCorrectnessTendency") or "uncertain").strip().lower()
    if tendency not in {"favorable", "uncertain", "error-prone"}:
        tendency = "uncertain"
    confidence = (_field(raw, "ExpectedConfidence") or "medium").strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"
    return {
        "agent": "cognitive_route_agent",
        "cognitive_route": route,
        "expected_correctness_tendency": tendency,
        "expected_confidence": confidence,
        "route_rationale": _field(raw, "RouteRationale") or "",
        "raw": raw,
    }


def fallback_cognitive_route(
    cognitive_profile: dict[str, Any],
    ability_boundary: dict[str, Any],
    proficiency: dict[str, Any] | None,
    behavior_factors: dict[str, float] | None,
    tendency_calibration: dict[str, Any] | None,
    route_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evidence = route_evidence or infer_cognitive_route_evidence(
        question={},
        cognitive_profile=cognitive_profile,
        ability_boundary=ability_boundary,
        proficiency=proficiency,
        behavior_factors=behavior_factors,
        tendency_calibration=tendency_calibration,
    )
    mastery = _optional_float((proficiency or {}).get("value")) or 0.5
    state = behavior_factors or {}
    tendency = str((tendency_calibration or {}).get("band", "mixed"))
    history_rate = _optional_float((tendency_calibration or {}).get("history_rate"))
    history_level = str((tendency_calibration or {}).get("history_level", "unknown"))
    error_traits = cognitive_profile.get("error_traits") or {}
    transfer = cognitive_profile.get("transfer_traits") or {}
    route = "partial_reasoning"
    boundary_risk = str(evidence.get("ability_boundary_risk", "medium"))
    demand = str(evidence.get("item_demand", "medium"))
    stable_history = (history_rate is not None and history_rate >= 0.75) or history_level == "high"
    if (
        mastery >= 0.78
        and "strong" in tendency
        and stable_history
        and boundary_risk != "high"
        and demand != "high"
    ):
        route = "mastery_retrieval"
    elif boundary_risk == "high" and demand in {"medium", "high"}:
        route = "misconception_transfer" if _level_is_high(transfer.get("transfer_fragility")) else "partial_reasoning"
    elif _level_is_high(transfer.get("transfer_fragility")):
        route = "misconception_transfer"
    elif float(state.get("carelessness", 0.0)) >= 0.2 or float(state.get("fatigue", 0.0)) >= 0.35:
        route = "careless_execution"
    elif mastery < 0.35 or history_level == "fragile" or _level_is_high(error_traits.get("guessing")):
        route = "uncertain_guessing"
    correctness = "favorable" if route == "mastery_retrieval" else "error-prone" if route in {
        "misconception_transfer",
        "uncertain_guessing",
    } else "uncertain"
    confidence = "high" if route == "mastery_retrieval" else "low" if route == "uncertain_guessing" else "medium"
    return {
        "agent": "cognitive_route_agent",
        "cognitive_route": route,
        "expected_correctness_tendency": correctness,
        "expected_confidence": confidence,
        "route_rationale": "Fallback route inferred from mastery, error traits, transfer, and non-cognitive state.",
        "route_evidence": evidence,
        "fallback": True,
    }


def build_response_prompt(
    question: dict[str, Any],
    short_memory: list[dict[str, Any]],
    long_memory: dict[str, Any],
    concept_options: list[str],
    proficiency: dict[str, Any] | None,
    behavior_factors: dict[str, float] | None,
    tendency_calibration: dict[str, Any] | None,
    cognitive_profile: dict[str, Any],
    ability_boundary: dict[str, Any],
    cognitive_route: dict[str, Any] | None,
    latent_state: dict[str, Any] | None = None,
    response_format: str = "four_tier",
) -> str:
    route_block = cognitive_route or {
        "cognitive_route": "not_provided",
        "route_rationale": "Cognitive route ablated.",
    }
    base = build_action_prompt(
        question=question,
        short_memory=short_memory,
        long_memory=long_memory,
        concept_options=concept_options,
        proficiency=proficiency,
        behavior_factors=behavior_factors,
        response_format=response_format,
        cognitive_strategy=None,
        tendency_calibration=tendency_calibration,
    )
    prefix = "\n".join(
        [
            "# Response Agent #",
            "Generate this learner's single first attempt using the educational constraints below.",
            "The ability boundary defines what the learner may plausibly use. The cognitive route defines how knowledge is expressed in this attempt.",
            "Do not solve outside the boundary, do not add an expert correction pass, and do not output correctness.",
            "The KT mastery/proficiency signal is state evidence, not a correctness label. Do not copy it as a decision rule; use it to shape how capable, fluent, hesitant, or error-prone the learner's attempt should look.",
            "",
            "# Cognitive Profile Evidence #",
            _compact(cognitive_profile),
            "",
            "# Ability Boundary Evidence #",
            _compact(ability_boundary),
            "",
            "# Cognitive Route Selection #",
            _compact(route_block),
            "",
            "# Learner State Evidence #",
            _compact(latent_state or {}),
            "",
            "# KT State Constraint #",
            _compact(_kt_state_constraint(proficiency, ability_boundary, route_block)),
            "",
            "# Route-to-Response Constraints #",
            "- Treat the selected cognitive route as a behavioral boundary, not a suggestion.",
            "- mastery_retrieval: use concise recall only when the route selected this path and the ability boundary supports it.",
            "- partial_reasoning: use one relevant but incomplete step; do not silently repair missing weak abilities.",
            "- misconception_transfer: preserve the wrong-transfer path instead of repairing it with expert reasoning.",
            "- careless_execution: allow a reading, arithmetic, or option-selection slip without rechecking.",
            "- uncertain_guessing: use shallow cues, low confidence, or elimination rather than a full derivation.",
            "- If mastery_retrieval is selected with strong KT mastery and low/medium item demand, the attempt should usually be direct and correct at learner level.",
            "- If the ability boundary lists a weak ability, the final reasoning must not rely on that weak ability as if it were mastered.",
            "- Do not invent a wrong answer solely to appear learner-like when KT mastery, memory, and item demand all support success.",
            "",
            "# Four-tier Output Requirement #",
            "The response must include answer, confidence, reasoning, and reasoning confidence so that a separate diagnostic module can assess it.",
            "",
        ]
    )
    return prefix + base


def _kt_state_constraint(
    proficiency: dict[str, Any] | None,
    ability_boundary: dict[str, Any],
    cognitive_route: dict[str, Any],
) -> dict[str, Any]:
    mastery = _optional_float((proficiency or {}).get("value"))
    route = cognitive_route.get("cognitive_route")
    boundary_risk = ability_boundary.get("boundary_risk")
    if mastery is None:
        band = "unknown"
        instruction = "Use profile, memory, and route evidence because KT mastery is unavailable."
    elif mastery >= 0.75:
        band = "strong"
        if ability_boundary.get("item_demand") in {"low", "medium"}:
            instruction = (
                "The learner has strong evidence for this concept. For low or medium "
                "demand items, generate a fluent first attempt; errors should be rare "
                "and only come from explicit careless, transfer, or item-demand evidence."
            )
        else:
            instruction = (
                "The learner should usually show fluent access to this knowledge. "
                "Errors are still possible, but they should arise from a plausible "
                "careless, transfer, or item-demand reason instead of invented helplessness."
            )
    elif mastery >= 0.55:
        band = "favorable"
        instruction = (
            "The learner has usable knowledge. Generate a plausible learner attempt "
            "that may be correct or partially flawed according to route and memory."
        )
    elif mastery <= 0.30:
        band = "weak"
        instruction = (
            "The learner has weak access to this knowledge. Avoid teacher-like full "
            "derivations; any correct answer should have a plausible memory cue, "
            "shortcut, or lucky shallow route."
        )
    else:
        band = "mixed"
        instruction = (
            "The learner is mixed. Let route, memory, and item demand shape the "
            "attempt without forcing agreement with KT."
        )
    return {
        "module": "kt_state_constraint",
        "mastery": _round_optional(mastery),
        "mastery_band": band,
        "cognitive_route": route,
        "boundary_risk": boundary_risk,
        "instruction": instruction,
    }


def build_four_tier_response_record(
    action: dict[str, Any],
    four_tier: dict[str, Any],
    cognitive_route: dict[str, Any] | None,
    ability_boundary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "module": "four_tier_response",
        "generation": {
            "attempt": action.get("attempt"),
            "identified_concept": action.get("identified_concept"),
            "student_answer": action.get("student_answer"),
            "answer_confidence": action.get("answer_confidence"),
            "student_reasoning": action.get("student_reasoning"),
            "reasoning_confidence": action.get("reasoning_confidence"),
        },
        "diagnostic_scoring": {
            "answer_correct": four_tier.get("answer_correct"),
            "reasoning_correct": four_tier.get("reasoning_correct"),
            "answer_confidence": four_tier.get("answer_confidence"),
            "reasoning_confidence": four_tier.get("reasoning_confidence"),
            "diagnosis": four_tier.get("diagnosis"),
            "fully_scored": four_tier.get("fully_scored"),
            "scoring_note": four_tier.get("scoring_note"),
        },
        "conditioning": {
            "cognitive_route": (cognitive_route or {}).get("cognitive_route"),
            "expected_solution_depth": ability_boundary.get("expected_solution_depth"),
            "boundary_risk": ability_boundary.get("boundary_risk"),
            "item_demand": ability_boundary.get("item_demand"),
        },
    }


def infer_cognitive_route_evidence(
    question: dict[str, Any],
    cognitive_profile: dict[str, Any],
    ability_boundary: dict[str, Any],
    proficiency: dict[str, Any] | None,
    behavior_factors: dict[str, float] | None,
    tendency_calibration: dict[str, Any] | None,
) -> dict[str, Any]:
    mastery = _optional_float((proficiency or {}).get("value"))
    history_rate = _optional_float((tendency_calibration or {}).get("history_rate"))
    history_level = str((tendency_calibration or {}).get("history_level", "unknown"))
    tendency_band = str((tendency_calibration or {}).get("band", "unknown"))
    weak_text = " ".join(str(x).lower() for x in ability_boundary.get("weak_abilities", []))
    boundary_risk_score = 0
    if ability_boundary.get("expected_solution_depth") == "shallow":
        boundary_risk_score += 1
    if any(
        key in weak_text
        for key in [
            "sequence",
            "ordering",
            "order",
            "procedure",
            "procedural",
            "multi-step",
            "transfer",
            "comparison",
            "differentiate",
            "inference",
        ]
    ):
        boundary_risk_score += 2
    if any(key in weak_text for key in ["low breadth", "shallow practice", "fragile"]):
        boundary_risk_score += 1
    item_demand_score = _item_demand_score(question)
    fragile_history = (
        history_level == "fragile"
        or (history_rate is not None and history_rate < 0.55)
    )
    if mastery is None:
        knowledge_readiness = "unknown"
    elif mastery >= 0.78 and not fragile_history:
        knowledge_readiness = "high"
    elif mastery >= 0.45:
        knowledge_readiness = "mixed"
    else:
        knowledge_readiness = "low"
    boundary_risk = "high" if boundary_risk_score >= 3 else "medium" if boundary_risk_score >= 1 else "low"
    item_demand = "high" if item_demand_score >= 3 else "medium" if item_demand_score >= 1 else "low"
    strong_concept_evidence = bool(
        ability_boundary.get("concept_mastery_overrides_global_profile_weakness")
    )
    if strong_concept_evidence and item_demand != "high" and boundary_risk == "medium":
        boundary_risk = "low"
    mastery_allowed = (
        knowledge_readiness == "high"
        and (history_level == "high" or strong_concept_evidence)
        and boundary_risk != "high"
        and item_demand != "high"
        and ("strong" in tendency_band or strong_concept_evidence)
    )
    if mastery_allowed:
        suggested_route = "mastery_retrieval"
    elif boundary_risk == "high" and item_demand in {"medium", "high"}:
        suggested_route = "partial_reasoning"
    elif fragile_history or knowledge_readiness == "low":
        suggested_route = "uncertain_guessing"
    else:
        suggested_route = "partial_reasoning"
    state = behavior_factors or {}
    if float(state.get("carelessness", 0.0)) >= 0.2 or float(state.get("fatigue", 0.0)) >= 0.35:
        suggested_route = "careless_execution"
    return {
        "knowledge_readiness": knowledge_readiness,
        "history_level": history_level,
        "history_rate": _round_optional(history_rate),
        "tendency_band": tendency_band,
        "ability_boundary_risk": boundary_risk,
        "item_demand": item_demand,
        "mastery_retrieval_allowed": mastery_allowed,
        "suggested_route": suggested_route,
        "evidence_reasons": _route_evidence_reasons(
            weak_text=weak_text,
            fragile_history=fragile_history,
            boundary_risk=boundary_risk,
            item_demand=item_demand,
            knowledge_readiness=knowledge_readiness,
        ),
    }


def _field(raw: str, name: str) -> str | None:
    pattern = rf"{re.escape(name)}\s*:\s*(.*?)(?=\n[A-Za-z][A-Za-z]+(?:[A-Za-z]+)?\s*:|\Z)"
    match = re.search(pattern, raw, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return match.group(1).strip().strip('"')


def _list_field(raw: str, name: str) -> list[str]:
    value = _field(raw, name) or ""
    parts = re.split(r"[,;锛岋紱\n]+", value)
    return [part.strip(" -\t") for part in parts if part.strip(" -\t")]


def _compact(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 3200 else text[:3200] + "...[truncated]"


def _item_demand_score(question: dict[str, Any]) -> int:
    text = (
        str(question.get("content", ""))
        + " "
        + str(question.get("options", ""))
        + " "
        + str(question.get("analysis", ""))
    ).lower()
    score = 0
    for key in [
        "sequence",
        "order",
        "first",
        "step",
        "procedure",
        "process",
        "which of the following",
        "compare",
        "different",
        "relationship",
        "infer",
        "reason",
        "calculate",
        "\u987a\u5e8f",
        "\u6b65\u9aa4",
        "\u6d41\u7a0b",
        "\u7a0b\u5e8f",
        "\u9996\u5148",
        "\u7b2c\u4e00",
        "\u4ee5\u4e0b",
        "\u54ea\u9879",
        "\u6bd4\u8f83",
        "\u533a\u522b",
        "\u5173\u7cfb",
        "\u63a8\u65ad",
        "\u8ba1\u7b97",
        "\u539f\u56e0",
    ]:
        if key in text:
            score += 1
    if len(str(question.get("options", ""))) > 120:
        score += 1
    if len(str(question.get("content", ""))) > 80:
        score += 1
    return min(score, 5)


def _memory_support(
    question: dict[str, Any],
    memory_context: dict[str, Any],
) -> dict[str, Any]:
    content = _normalize_text(question.get("content", ""))
    current_routes = {str(route) for route in question.get("kc_routes", [])}
    records = list(memory_context.get("short_memory", []))
    records.extend((memory_context.get("long_memory") or {}).get("significant_facts", []))
    related_total = 0
    related_correct = 0
    identical_correct = 0
    for record in records:
        response = int(record.get("simulated_response", record.get("real_response", 0)) or 0)
        preview = _normalize_text(record.get("content") or record.get("content_preview") or "")
        if preview and content and preview == content:
            if response == 1:
                identical_correct += 1
            related_total += 1
            related_correct += int(response == 1)
        elif _routes_overlap(current_routes, record.get("kc_routes", [])):
            related_total += 1
            related_correct += int(response == 1)
    related_rate = related_correct / related_total if related_total else None
    return {
        "related_total": related_total,
        "related_correct_count": related_correct,
        "related_correct_rate": _round_optional(related_rate),
        "identical_correct_count": identical_correct,
    }


def _routes_overlap(current_routes: set[str], record_routes: Any) -> bool:
    if not current_routes:
        return False
    return bool(current_routes & {str(route) for route in (record_routes or [])})


def _reasoning_capacity(
    mastery: float | None,
    practice: int,
    breadth: int,
    challenge: int,
    generalization: int,
    memory_support: dict[str, Any],
) -> str:
    score = 0
    if mastery is not None and mastery >= 0.75:
        score += 2
    elif mastery is not None and mastery >= 0.45:
        score += 1
    score += max(0, practice - 1)
    score += max(0, breadth - 1)
    score += max(0, challenge - 1)
    score += max(0, generalization - 1)
    if memory_support.get("identical_correct_count", 0) > 0:
        score += 1
    if score >= 6:
        return "deep"
    if score >= 3:
        return "medium"
    return "shallow"


def _ability_boundary_rationale(
    item_demand: str,
    reasoning_capacity: str,
    boundary_risk: str,
    mastery: float | None,
    memory: dict[str, Any],
) -> str:
    mastery_text = "unknown" if mastery is None else f"{mastery:.2f}"
    return (
        f"Item demand is {item_demand}; learner reasoning capacity is {reasoning_capacity}; "
        f"boundary risk is {boundary_risk}; mastery={mastery_text}; "
        f"identical_correct_memory={memory.get('identical_correct_count', 0)}."
    )


def _level_value(value: Any) -> int:
    text = str(value or "").strip().lower()
    if text in {"deep", "high", "strong", "higher"}:
        return 3
    if text in {"medium", "mixed", "moderate"}:
        return 2
    if text in {"shallow", "low", "weak", "fragile"}:
        return 1
    return 2


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip().lower()


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = str(value).strip()
        if clean and clean not in seen:
            result.append(clean)
            seen.add(clean)
    return result


def _route_evidence_reasons(
    weak_text: str,
    fragile_history: bool,
    boundary_risk: str,
    item_demand: str,
    knowledge_readiness: str,
) -> list[str]:
    reasons: list[str] = []
    if fragile_history:
        reasons.append("fragile historical performance")
    if boundary_risk == "high":
        reasons.append("weak abilities conflict with current item demand")
    if item_demand == "high":
        reasons.append("item requires more than direct recall")
    if knowledge_readiness in {"mixed", "low"}:
        reasons.append(f"{knowledge_readiness} knowledge readiness")
    if any(key in weak_text for key in ["sequence", "ordering", "procedure", "procedural"]):
        reasons.append("procedural or ordering weakness")
    if any(key in weak_text for key in ["transfer", "comparison", "differentiate", "inference"]):
        reasons.append("transfer/comparison weakness")
    return reasons[:5]


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_optional(value: Any) -> float | None:
    number = _optional_float(value)
    return round(number, 3) if number is not None else None


def _level_is_high(value: Any) -> bool:
    return str(value).strip().lower() in {"high", "higher", "strong"}


from __future__ import annotations

from typing import Any

from learner_simulator.irt_evidence import prompt_irt_evidence


def build_cognitive_profile_view(
    profile_context: dict[str, Any],
) -> dict[str, Any]:
    cognitive = profile_context.get("cognitive_profile") or {}
    control = cognitive.get("control_traits") or {}
    ability = profile_context.get("ability_profile") or {}
    return {
        "module": "learner_state_profile",
        "scope": "stable_traits_estimated_from_observed_history",
        "general_performance_level": control.get("overall_success_level"),
        "learning_stability": {
            "recent_success": control.get("recent_success_level"),
            "success_trend": control.get("success_trend_level"),
            "mastery_stability": control.get("mastery_stability_level"),
        },
        "ability_trait": {
            "knowledge_breadth": ability.get("knowledge_breadth"),
            "challenge_adaptation": ability.get("challenge_adaptation"),
            "profile_confidence": ability.get("profile_confidence"),
        },
        "profile_use": (
            "Use stable profile evidence for reasoning style and confidence. "
            "Do not let broad traits override strong current-item knowledge evidence."
        ),
    }


def build_response_prompt(
    question: dict[str, Any],
    short_memory: list[dict[str, Any]],
    long_memory: dict[str, Any],
    concept_options: list[str] | None,
    proficiency: dict[str, Any] | None = None,
    irt_evidence: dict[str, Any] | None = None,
    response_format: str = "four_tier",
    include_profile_evidence: bool = True,
    include_irt_evidence: bool = True,
    include_ncdm_evidence: bool = True,
    evidence_repository: dict[str, Any] | None = None,
    state_item_alignment: dict[str, Any] | None = None,
    include_evidence_representation: bool = True,
    include_state_item_alignment: bool = True,
) -> str:
    if response_format not in {
        "four_tier",
        "reduced_response",
        "answer_only",
    }:
        raise ValueError(f"Unsupported response format: {response_format}")

    chunks = [
        "# Response Agent #",
        "Simulate this learner's single first attempt from the evidence below.",
    ]
    # Strict V5 information bottleneck: Module 3 receives only the canonical
    # output of Module 2, never parallel raw NCDM, IRT, or history sections.
    if state_item_alignment:
        chunks.extend(
            [
                "",
                "# Cognitive State-Item Alignment #",
                _compact(state_item_alignment),
            ]
        )

    constraints = [
        "- Use only the canonical Module 2 output; never reconstruct an ablated upstream module.",
        "- Missing learner evidence is unobserved evidence, not evidence of inability.",
        "- Simulate one plausible first attempt without optimizing for either correctness or incorrectness.",
    ]
    if response_format in {"four_tier", "reduced_response"}:
        constraints.insert(
            2,
            "- Decide LearnerCorrect from the learner evidence first. Then use the reference answer only to render a StudentAnswer consistent with that decision: Yes must submit the reference answer; No must submit a different option or answer.",
        )
    if response_format == "four_tier":
        constraints.extend(
            [
                "- Cite only evidence IDs listed in selected_evidence_ids; never invent an ID.",
                "- IdentifiedConcept must be decided before LearnerCorrect.",
                "- Strong aligned readiness should normally lean correct; weak readiness should normally lean incorrect.",
                "- Repeated relevant errors keep a mistake plausible; repeated successes support a correct attempt without fixing the label.",
                "- Resolve boundary evidence using aligned mastery, selected events, recent performance, learner ability, and item difficulty together.",
                "- Commit to LearnerCorrect once, then render StudentAnswer, reasoning, and confidence consistently with that decision.",
            ]
        )
    if response_format == "four_tier" and include_ncdm_evidence:
        constraints.append(
            "- Treat NCDM concept mastery as the primary knowledge-state evidence, while allowing concrete memory and item demand to resolve boundary cases."
        )
    if response_format == "four_tier" and include_irt_evidence:
        constraints.append(
            "- IRT calibrates relative challenge and confidence; it does not directly determine correctness."
        )
    chunks.extend(["", "# Decision Constraints #", *constraints, ""])

    chunks.extend(
        [
            "",
            "# Current Exercise #",
            f"TextualContent: {question.get('content', '')}",
            f"Options: {question.get('options', '')}",
            *([f"ConceptOptions: {concept_options}"] if concept_options else []),
            "",
            "# Reference Answer for Response Rendering #",
            f"ReferenceAnswer: {question.get('answer', '')}",
        ]
    )
    chunks.extend(["", _response_contract(response_format, concept_options=concept_options)])
    return "\n".join(chunks)


def _response_contract(
    response_format: str,
    *,
    concept_options: list[str] | None = None,
) -> str:
    concept_field = (
        "IdentifiedConcept: <one supplied concept option>\n"
        if concept_options
        else "IdentifiedConcept: <brief concept inferred from item text, or unknown>\n"
    )
    if response_format == "four_tier":
        return (
            "# Structured Response Process Contract #\n"
            "Return exactly these seven fields in order. Use discrete confidence "
            "anchors 0.20, 0.50, or 0.80.\n"
            "The field headers below are literal machine keys: keep their English "
            "spelling and ASCII colon exactly; do not translate or rename them.\n"
            "EvidenceRefs: <comma-separated selected evidence IDs, or none>\n"
            + concept_field
            + "LearnerCorrect: <Yes or No>\n"
            "StudentAnswer: <the learner's submitted option or short answer>\n"
            "AnswerConfidence: <0.20, 0.50, or 0.80>\n"
            "StudentReasoning: <brief observable student-level rationale>\n"
            "ReasoningConfidence: <0.20, 0.50, or 0.80>\n"
            "Do not output Attempt, hidden chain-of-thought, markdown, or extra fields."
        )
    if response_format == "reduced_response":
        return (
            "# Direct Response Generation Contract #\n"
            "Return exactly:\n"
            "The field headers below are literal machine keys: keep their English "
            "spelling and ASCII colon exactly; do not translate or rename them.\n"
            + concept_field
            + "LearnerCorrect: <Yes or No>\n"
            "StudentAnswer: <the learner's submitted option or short answer>\n"
            "Do not output process fields, confidence, reasoning, or markdown."
        )
    if response_format == "answer_only":
        return (
            "# Answer-only Contract #\n"
            "StudentAnswer: <the learner's submitted option or short answer>"
        )
    raise ValueError(f"Unsupported response format: {response_format}")


def _prompt_ncdm_evidence(proficiency: dict[str, Any]) -> dict[str, Any]:
    return {
        "module": "ncdm_state_evidence",
        "concept": proficiency.get("concept"),
        "concept_mastery": proficiency.get("concept_mastery_value"),
        "concept_mastery_level": proficiency.get("level"),
    }


def _prompt_history_state(proficiency: dict[str, Any]) -> dict[str, Any]:
    return {
        "module": "history_based_knowledge_evidence",
        "concept": proficiency.get("concept"),
        "mastery": proficiency.get("value"),
        "level": proficiency.get("level"),
        "source": proficiency.get("source"),
    }


def build_four_tier_response_record(
    action: dict[str, Any],
    four_tier: dict[str, Any],
    irt_evidence: dict[str, Any] | None = None,
    state_item_alignment: dict[str, Any] | None = None,
    process_consistency: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        # Attempt remains removed; confidence uses the original three anchors.
        "module": "structured_response_generation",
        "response_contract": "single_pass_process_verifiable_v6_label_conditioned_answer_realization",
        "generation": {
            "evidence_refs": action.get("evidence_refs", []),
            "identified_concept": action.get("identified_concept"),
            "learner_correct": action.get("learner_correct"),
            "response_decision_source": action.get("response_decision_source"),
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
            "irt_ability_difficulty_evidence": prompt_irt_evidence(irt_evidence),
            "cognitive_state_item_alignment": state_item_alignment,
        },
        "process_consistency": process_consistency,
    }


def _compact(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 2600 else text[:2600] + "...[truncated]"

from __future__ import annotations

import re
from typing import Any, Callable, Optional


ReasonEvaluator = Callable[[str, str], Optional[bool]]


def parse_four_tier_response(raw: str | None) -> dict[str, Any] | None:
    """Parse the learner-facing four-tier response format."""

    if not raw:
        return None
    labels = [
        "Attempt:",
        "IdentifiedConcept:",
        "LearnerCorrect:",
        "StudentAnswer:",
        "AnswerConfidence:",
        "StudentReasoning:",
        "ReasoningConfidence:",
    ]
    required_labels = [
        "Attempt:",
        "IdentifiedConcept:",
        "StudentAnswer:",
        "AnswerConfidence:",
        "StudentReasoning:",
        "ReasoningConfidence:",
    ]
    has_learner_correct = "learnercorrect:" in raw.lower()
    active_labels = labels if has_learner_correct else required_labels
    if not all(label.lower() in raw.lower() for label in required_labels):
        return None

    normalized = raw
    for label in active_labels:
        normalized = re.sub(
            re.escape(label),
            label,
            normalized,
            flags=re.IGNORECASE,
        )

    parsed: dict[str, Any] = {}
    keys = [
        "attempt",
        "identified_concept",
    ]
    if has_learner_correct:
        keys.append("learner_correct")
    keys.extend(
        [
            "student_answer",
            "answer_confidence",
            "student_reasoning",
            "reasoning_confidence",
        ]
    )
    for index, (label, key) in enumerate(zip(active_labels, keys)):
        part = normalized.split(label, 1)[1]
        next_positions = [
            part.find(next_label)
            for next_label in active_labels[index + 1 :]
            if part.find(next_label) >= 0
        ]
        if next_positions:
            part = part[: min(next_positions)]
        parsed[key] = part.strip().strip('"')

    parsed["attempt"] = _normalize_attempt(parsed["attempt"])
    if has_learner_correct:
        parsed["learner_correct"] = _yes_no(parsed.get("learner_correct"))
    parsed["answer_confidence"] = _confidence(parsed["answer_confidence"])
    parsed["reasoning_confidence"] = _confidence(parsed["reasoning_confidence"])
    parsed["solution_process"] = parsed["student_reasoning"]
    return parsed


def parse_answer_only_response(raw: str | None) -> dict[str, Any] | None:
    """Parse the no-four-tier ablation output contract."""

    if not raw:
        return None
    match = re.search(
        r"StudentAnswer\s*:\s*(.+)",
        raw.strip(),
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    answer = match.group(1).strip().strip('"')
    if not answer:
        return None
    return {
        "student_answer": answer,
        "response_format": "answer_only",
    }


def assess_four_tier_response(
    response: dict[str, Any],
    reference_answers: Any,
    reference_reasoning: str = "",
    reason_evaluator: ReasonEvaluator | None = None,
    confidence_threshold: float = 0.6,
) -> dict[str, Any]:
    """Score only tiers supported by available evidence.

    XES3G5M provides reference answers but no observed learner reasoning or
    confidence. Answer correctness is therefore scored deterministically when
    possible. Reason correctness remains unknown unless an explicit evaluator
    is supplied.
    """

    answer_correct = compare_answers(
        response.get("student_answer"),
        reference_answers,
    )
    reasoning_correct = None
    if reason_evaluator is not None and reference_reasoning.strip():
        reasoning_correct = reason_evaluator(
            str(response.get("student_reasoning", "")),
            reference_reasoning,
        )

    answer_confidence = _confidence(response.get("answer_confidence"))
    reasoning_confidence = _confidence(response.get("reasoning_confidence"))
    diagnosis = classify_four_tier(
        answer_correct=answer_correct,
        answer_confidence=answer_confidence,
        reasoning_correct=reasoning_correct,
        reasoning_confidence=reasoning_confidence,
        confidence_threshold=confidence_threshold,
    )
    return {
        "answer_correct": answer_correct,
        "answer_confidence": answer_confidence,
        "reasoning_correct": reasoning_correct,
        "reasoning_confidence": reasoning_confidence,
        "diagnosis": diagnosis,
        "fully_scored": answer_correct is not None and reasoning_correct is not None,
        "scoring_note": (
            "Answer tier scored against metadata; reasoning tier not scored."
            if reasoning_correct is None
            else "Answer and reasoning tiers scored."
        ),
    }


def compare_answers(student_answer: Any, reference_answers: Any) -> bool | None:
    student = _answer_forms(student_answer)
    references = _reference_values(reference_answers)
    if not student or not references:
        return None

    reference_groups = [_answer_forms(answer) for answer in references]
    if any(not group for group in reference_groups):
        return None
    if len(reference_groups) == 1:
        return bool(student & reference_groups[0])
    return all(bool(student & group) for group in reference_groups)


def classify_four_tier(
    answer_correct: bool | None,
    answer_confidence: float,
    reasoning_correct: bool | None,
    reasoning_confidence: float,
    confidence_threshold: float = 0.6,
) -> str:
    answer_confident = answer_confidence >= confidence_threshold
    reasoning_confident = reasoning_confidence >= confidence_threshold

    if answer_correct is None:
        return "unscored_answer"
    if reasoning_correct is None:
        if answer_correct and answer_confident:
            return "confident_correct_answer_reason_unscored"
        if answer_correct:
            return "uncertain_correct_answer_reason_unscored"
        if answer_confident:
            return "confident_incorrect_answer_reason_unscored"
        return "uncertain_incorrect_answer_reason_unscored"

    if answer_correct and reasoning_correct:
        if answer_confident and reasoning_confident:
            return "stable_mastery"
        return "partial_or_uncertain_understanding"
    if not answer_correct and not reasoning_correct:
        if answer_confident and reasoning_confident:
            return "stable_misconception"
        return "lack_of_knowledge_or_guessing"
    if answer_correct and not reasoning_correct:
        if answer_confident and reasoning_confident:
            return "false_positive"
        return "correct_answer_unstable_reasoning"
    if answer_confident and reasoning_confident:
        return "false_negative"
    return "incorrect_answer_with_partial_reasoning"


def _reference_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _answer_forms(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        forms: set[str] = set()
        for item in value:
            forms.update(_answer_forms(item))
        return forms

    text = str(value).strip()
    if not text:
        return set()
    text = re.sub(r"(?i)^final\s+answer\s*:\s*", "", text).strip()
    forms = {_normalize_answer(text)}

    option_match = re.fullmatch(
        r"(?:option\s*)?[\(\[]?\s*([A-Ha-h])\s*[\)\].:]?",
        text,
    )
    if option_match:
        forms.add(option_match.group(1).lower())

    number_matches = re.findall(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)", text)
    forms.update(_normalize_number(number) for number in number_matches)
    return {form for form in forms if form}


def _normalize_answer(value: str) -> str:
    text = value.lower().strip()
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\times", "*").replace("\\cdot", "*")
    text = text.replace("\\,", "").replace("\\;", "")
    text = re.sub(r"\$+", "", text)
    text = re.sub(r"\\(?:mathrm|text)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\s+", "", text)
    text = text.strip("。.,;；:：()[]{}")
    if re.fullmatch(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)", text):
        return _normalize_number(text)
    return text


def _normalize_number(value: str) -> str:
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return str(int(number))
    return format(number, ".12g")


def _confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.5
    return min(1.0, max(0.0, confidence))


def _normalize_attempt(value: Any) -> str:
    text = str(value).strip().lower()
    return "no" if text.startswith(("no", "n")) else "yes"


def _yes_no(value: Any) -> int | None:
    text = str(value or "").strip().lower()
    if text.startswith(("yes", "y", "true", "correct", "1")):
        return 1
    if text.startswith(("no", "n", "false", "incorrect", "0")):
        return 0
    return None

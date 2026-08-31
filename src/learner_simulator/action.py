from __future__ import annotations

import re
import unicodedata
from typing import Any

from learner_simulator.four_tier import parse_four_tier_response


_AUXILIARY_FIELD_ALIASES = {
    "StateAlignment": ("StateAlignment", "状态对齐"),
    "EvidenceRefs": ("EvidenceRefs", "证据参考"),
}


def build_statistical_action(
    cid: int,
    kc_routes: list[str],
    probability: float,
    simulated_response: int,
    answer: Any,
    error_type: str = "simulated_mistake",
) -> dict[str, Any]:
    """Create the action record used by the statistical baseline."""

    return {
        "attempt": "yes" if probability >= 0.35 else "no",
        "identified_concept": kc_routes[0] if kc_routes else str(cid),
        "solution_process": "statistical baseline; no natural-language reasoning generated",
        "student_answer": answer if simulated_response == 1 else "",
        "simulated_correct": int(simulated_response),
        "error_type": "none" if simulated_response == 1 else error_type,
        "confidence": round(max(probability, 1.0 - probability), 4),
    }


def parse_agent_response(raw: str | None) -> dict[str, Any] | None:
    """Parse the process-verifiable response while preserving Four-tier fields."""

    if not raw:
        return None
    alignment = _line_value(raw, "StateAlignment")
    evidence_text = _line_value(raw, "EvidenceRefs")
    stripped = re.sub(
        r"(?im)^\s*(?:StateAlignment|EvidenceRefs|状态对齐|证据参考)\s*[:：]\s*.*(?:\n|$)",
        "",
        raw,
    )
    parsed = parse_four_tier_response(stripped)
    if parsed is None:
        return None
    parsed["state_alignment"] = alignment
    parsed["evidence_refs"] = _evidence_refs(evidence_text)
    parsed["response_format"] = "process_verifiable_four_tier"
    return parsed


def _line_value(raw: str, label: str) -> str:
    aliases = _AUXILIARY_FIELD_ALIASES.get(label, (label,))
    normalized = unicodedata.normalize("NFKC", raw)
    match = re.search(
        rf"(?im)^\s*(?:{'|'.join(re.escape(alias) for alias in aliases)})\s*:\s*(.*?)\s*$",
        normalized,
    )
    return match.group(1).strip().strip('"') if match else ""


def _evidence_refs(value: str) -> list[str]:
    if not value or value.strip().lower() in {"none", "[]", "n/a"}:
        return []
    return [
        item.strip().strip("[]'\"")
        for item in re.split(r"[,;]", value)
        if item.strip().strip("[]'\"")
    ]

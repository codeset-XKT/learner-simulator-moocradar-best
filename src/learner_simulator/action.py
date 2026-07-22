from __future__ import annotations

from typing import Any

from learner_simulator.four_tier import parse_four_tier_response


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
    """Parse the only supported LLM output contract: four-tier response."""

    return parse_four_tier_response(raw)

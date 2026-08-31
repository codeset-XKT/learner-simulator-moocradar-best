from __future__ import annotations

from typing import Any


def verify_process_steps(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit saved process artefacts without making causal claims.

    This verifier performs no regeneration or intervention. It therefore reports
    attribution coverage, citation validity, and contract consistency only; it
    must not be described as counterfactual process verification.
    """
    structured = [
        step
        for step in steps
        if step.get("enabled_modules", {}).get("structured_response_generation")
        and isinstance(step.get("agent_action"), dict)
    ]
    if not structured:
        return {
            "framework": "evidence_grounded_process_audit_v1",
            "structured_step_count": 0,
            "note": "structured response process ablated",
        }

    cited_steps = 0
    citation_supported_steps = 0
    consistent_steps = 0
    total_selected = 0
    total_cited = 0
    for step in structured:
        alignment = step.get("cognitive_state_item_alignment") or {}
        action = step.get("agent_action") or {}
        selected = set(alignment.get("selected_evidence_ids") or [])
        refs = set(action.get("evidence_refs") or [])
        total_selected += len(selected)
        total_cited += len(refs)
        if refs:
            cited_steps += 1
            citation_supported_steps += int(refs.issubset(selected))
        if (step.get("process_consistency") or {}).get("valid"):
            consistent_steps += 1

    return {
        "framework": "evidence_grounded_process_audit_v1",
        "additional_api_calls": 0,
        "structured_step_count": len(structured),
        "evidence_attribution_coverage": round(cited_steps / len(structured), 6),
        # Precision conditions on an actual citation; empty attribution is not
        # silently counted as a supported reference.
        "evidence_reference_precision": (
            round(citation_supported_steps / cited_steps, 6) if cited_steps else None
        ),
        "cited_step_count": cited_steps,
        "citation_supported_step_count": citation_supported_steps,
        "process_consistency_rate": round(consistent_steps / len(structured), 6),
        "mean_selected_evidence_count": round(total_selected / len(structured), 6),
        "mean_cited_evidence_count": round(total_cited / len(structured), 6),
        "mastery_response_monotonicity": None,
        "monotonicity_note": (
            "Not reported: cross-user/cross-item pairwise ordering is confounded "
            "by learner and item differences and is not a process-fidelity test."
        ),
        "audit_checks": {
            "citation_membership": "each cited evidence ID is checked against saved selected_evidence_ids",
            "citation_coverage": "steps with selected evidence but no citation are visible in process_consistency",
            "concept_membership": "identified concept is checked against the full current-item concept set only when alignment is enabled",
            "saved_artifact_only": "no model regeneration, intervention, or causal counterfactual claim is made",
        },
    }

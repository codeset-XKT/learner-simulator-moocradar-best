from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from learner_simulator.data import clean_sequence


DEFAULT_MAX_EVIDENCE = 4
DEFAULT_MAX_FULL_QUESTIONS = 2
DEFAULT_EVIDENCE_CHAR_BUDGET = 2400


def primary_kc_route(question: dict[str, Any], fallback: Any = None) -> list[str]:
    """Return the sole pedagogical concept route used for simulation.

    Dataset metadata can expose several routes for a question. The formal
    method treats only ``kc_routes[0]`` as the question's concept, while
    retaining that *entire* route string so hierarchical path transfer remains
    available during evidence selection. Raw metadata is archived elsewhere;
    it is not an instruction to aggregate all routes into learner state.
    """
    routes = [str(value).strip() for value in (question.get("kc_routes") or [])]
    routes = [route for route in routes if route]
    if routes:
        return [routes[0]]
    return [str(fallback)] if fallback is not None else []


def build_traceable_evidence_repository(
    history_row: dict[str, str] | None,
    questions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build a read-only, provenance-preserving view of observed history.

    The repository is intentionally not an agent memory: it has no reflection,
    reinforcement, forgetting, or generated summaries. Every aggregate retains
    the source event identifiers from which it was computed.
    """

    events: list[dict[str, Any]] = []
    concept_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, step in enumerate(clean_sequence(history_row), start=1):
        qid = int(step["qid"])
        cid = int(step["cid"])
        question = dict(questions.get(str(qid), {}))
        routes = primary_kc_route(question, fallback=cid)
        event = {
            "evidence_id": f"hist_{index:03d}",
            "source": "observed_history",
            "position": int(step.get("position") or index),
            "timestamp": step.get("timestamp"),
            "qid": qid,
            "cid": cid,
            "concepts": routes or [str(cid)],
            "observed_correct": int(step["response"]),
            "content": str(question.get("content") or ""),
            "options": question.get("options"),
        }
        events.append(event)
        for concept in event["concepts"]:
            concept_events[str(concept)].append(event)

    concept_patterns: list[dict[str, Any]] = []
    for concept, records in sorted(concept_events.items()):
        correct = sum(int(item["observed_correct"]) for item in records)
        total = len(records)
        concept_patterns.append(
            {
                "attribute_id": f"concept_pattern_{len(concept_patterns) + 1:03d}",
                "type": "observed_concept_performance",
                "concept": concept,
                "observed_accuracy": round(correct / total, 6),
                "support_count": total,
                "source_event_ids": [item["evidence_id"] for item in records],
                "reliability": round(total / (total + 3.0), 6),
            }
        )

    correct_total = sum(int(item["observed_correct"]) for item in events)
    recent_events = events[-10:]
    recent_correct = sum(int(item["observed_correct"]) for item in recent_events)
    return {
        "module": "traceable_learner_evidence_representation",
        "read_only": True,
        "reflection_used": False,
        "reinforcement_used": False,
        "forgetting_used": False,
        "event_count": len(events),
        "overall_observed_accuracy": (
            round(correct_total / len(events), 6) if events else None
        ),
        "recent_observed_performance": {
            "window_size": len(recent_events),
            "observed_accuracy": (
                round(recent_correct / len(recent_events), 6)
                if recent_events
                else None
            ),
            "observed_error_count": len(recent_events) - recent_correct,
            "source_event_ids": [item["evidence_id"] for item in recent_events],
        },
        "events": events,
        "stable_patterns": concept_patterns,
    }


def build_state_item_alignment(
    repository: dict[str, Any],
    question: dict[str, Any],
    concept_mastery: float,
    irt_evidence: dict[str, Any] | None,
    *,
    include_evidence_representation: bool = True,
    include_alignment: bool = True,
    max_evidence: int = DEFAULT_MAX_EVIDENCE,
    max_full_questions: int = DEFAULT_MAX_FULL_QUESTIONS,
    char_budget: int = DEFAULT_EVIDENCE_CHAR_BUDGET,
) -> dict[str, Any]:
    """Build a bounded, provenance-preserving current-item evidence bundle.

    Full uses a deterministic hierarchy rather than a weighted relevance score:
    same-concept success, same-concept error, same-path transfer, then semantic
    supplementation.  Each selected event therefore has a discrete educational
    role that can be inspected without interpreting arbitrary coefficients.
    """

    # Only the primary route is the current-item concept. Its full hierarchy
    # is retained in ``route_tokens`` for same-path evidence retrieval.
    routes = primary_kc_route(question, fallback=question.get("cid"))
    current_tokens = _tokens(str(question.get("content") or ""))
    route_tokens = _route_tokens(routes)
    events = list(repository.get("events") or []) if include_evidence_representation else []

    if include_alignment:
        selected, selection_trace = _layered_item_evidence(
            events,
            routes=routes,
            route_tokens=route_tokens,
            current_tokens=current_tokens,
            max_evidence=max_evidence,
        )
        selection_policy = "hierarchical_role_based_evidence_selection_v1"
    else:
        # The alignment ablation keeps only a bounded recent-history view;
        # current-item concepts, paths, and text are never consulted.
        selected = _recent_first(events, max_evidence)
        selected = [
            _annotate_selection(
                event,
                tier="recent_history_without_item_alignment",
                rationale="Alignment ablated: retained by recency only.",
                features={},
                rank=index,
            )
            for index, event in enumerate(selected, start=1)
        ]
        selection_trace = [
            {
                "slot": index,
                "tier": "recent_history_without_item_alignment",
                "eligible_count": len(events),
                "selected_evidence_id": event["evidence_id"],
            }
            for index, event in enumerate(selected, start=1)
        ]
        selection_policy = "bounded_recent_evidence_without_item_alignment"

    selected = _bounded_prompt_evidence(
        selected,
        max_full_questions=max_full_questions,
        char_budget=char_budget,
    )
    relevant_patterns = []
    if include_evidence_representation:
        for pattern in repository.get("stable_patterns") or []:
            if str(pattern.get("concept")) in set(routes):
                relevant_patterns.append(pattern)
        relevant_patterns = relevant_patterns[:3]
    difficulty = _float_or_none((irt_evidence or {}).get("item_beta"))
    ability = _float_or_none((irt_evidence or {}).get("learner_theta"))
    selected_correct = sum(int(item["observed_correct"]) for item in selected)
    selected_count = len(selected)
    selected_error_ids = [
        item["evidence_id"] for item in selected if not int(item["observed_correct"])
    ]

    return {
        "module": "cognitive_state_item_alignment",
        "ablated": not include_alignment,
        "current_concepts": routes,
        "concept_mastery": round(float(concept_mastery), 6),
        "irt_ability": ability,
        "irt_item_difficulty": difficulty,
        "cross_model_scale_warning": (
            "NCDM mastery and IRT ability/difficulty are independent evidence; "
            "do not subtract them or apply an invented correctness threshold."
        ),
        "selection_policy": selection_policy,
        "selection_policy_description": (
            "Fixed role order: same-concept success, same-concept error, "
            "same-path transfer, semantic supplement; recency is used only "
            "within a role and as a final no-evidence fallback."
            if include_alignment
            else "Recent historical events only; current-item relevance is unavailable."
        ),
        "max_evidence": int(max_evidence),
        "max_full_questions": int(max_full_questions),
        "evidence_char_budget": int(char_budget),
        "selection_trace": selection_trace,
        "selected_evidence": selected,
        "selected_evidence_ids": [item["evidence_id"] for item in selected],
        "selected_evidence_summary": {
            "support_count": selected_count,
            "observed_correct_count": selected_correct,
            "observed_error_count": selected_count - selected_correct,
            "observed_accuracy": (
                round(selected_correct / selected_count, 6)
                if selected_count
                else None
            ),
            "observed_error_event_ids": selected_error_ids,
        },
        "relevant_stable_patterns": relevant_patterns,
    }


def _layered_item_evidence(
    events: list[dict[str, Any]],
    *,
    routes: list[str],
    route_tokens: set[str],
    current_tokens: set[str],
    max_evidence: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select evidence by fixed educational roles, without weighted fusion."""
    selected: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    used: set[str] = set()
    current_routes = set(routes)

    def available(predicate):
        return [
            event for event in events
            if event["evidence_id"] not in used and predicate(event)
        ]

    def choose(tier: str, rationale: str, candidates: list[dict[str, Any]], *, semantic: bool = False):
        if len(selected) >= max(0, int(max_evidence)) or not candidates:
            return
        if semantic:
            candidates.sort(
                key=lambda event: (
                    -_jaccard(current_tokens, _tokens(str(event.get("content") or ""))),
                    -int(event.get("position") or 0),
                )
            )
        else:
            candidates = _recent_first(candidates, len(candidates))
        event = candidates[0]
        used.add(event["evidence_id"])
        event_routes = [str(value) for value in (event.get("concepts") or [])]
        shared_prefix_depth = _max_shared_route_prefix_depth(routes, event_routes)
        features = {
            "exact_concept_match": bool(current_routes.intersection(event_routes)),
            "shared_route_prefix_depth": shared_prefix_depth,
            "semantic_overlap": (
                round(_jaccard(current_tokens, _tokens(str(event.get("content") or ""))), 6)
                if semantic
                else None
            ),
            "observed_correct": int(event["observed_correct"]),
        }
        annotated = _annotate_selection(
            event,
            tier=tier,
            rationale=rationale,
            features=features,
            rank=len(selected) + 1,
        )
        selected.append(annotated)
        trace.append(
            {
                "slot": len(selected),
                "tier": tier,
                "eligible_count": len(candidates),
                "selected_evidence_id": annotated["evidence_id"],
            }
        )

    same_concept = lambda event: bool(current_routes.intersection(event.get("concepts") or []))
    same_path = lambda event: (
        _max_shared_route_prefix_depth(
            routes,
            [str(value) for value in (event.get("concepts") or [])],
        ) >= 2
        and not same_concept(event)
    )
    semantically_related = lambda event: bool(current_tokens.intersection(_tokens(str(event.get("content") or ""))))

    choose(
        "same_concept_recent_success",
        "Most recent observed success on the current fine-grained concept.",
        available(lambda event: same_concept(event) and int(event["observed_correct"]) == 1),
    )
    choose(
        "same_concept_recent_error",
        "Most recent observed error on the current fine-grained concept.",
        available(lambda event: same_concept(event) and int(event["observed_correct"]) == 0),
    )
    choose(
        "same_path_recent_transfer",
        "Most recent event sharing the current concept path but not its exact concept.",
        available(same_path),
    )
    choose(
        "semantic_supplement",
        "Most textually similar remaining historical item; used only after concept and path roles.",
        available(semantically_related),
        semantic=True,
    )

    # If a learner lacks one of the four roles, fill the bounded context without
    # inventing a score: exact concept, path transfer, semantic relation, then
    # plain recency are considered in that order.
    fill_layers = [
        ("same_concept_recency_fallback", "Additional recent event on the current concept.", same_concept, False),
        ("same_path_recency_fallback", "Additional recent event sharing the current concept path.", same_path, False),
        ("semantic_recency_fallback", "Additional textually related historical event.", semantically_related, True),
        ("recent_history_fallback", "No item-related evidence available; most recent observed history.", lambda event: True, False),
    ]
    for tier, rationale, predicate, semantic in fill_layers:
        while len(selected) < max(0, int(max_evidence)):
            candidates = available(predicate)
            if not candidates:
                break
            choose(tier, rationale, candidates, semantic=semantic)
    return selected, trace


def _recent_first(events: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return sorted(
        events,
        key=lambda event: (
            -int(event.get("position") or 0),
            str(event.get("evidence_id") or ""),
        ),
    )[: max(0, int(limit))]


def _annotate_selection(
    event: dict[str, Any],
    *,
    tier: str,
    rationale: str,
    features: dict[str, Any],
    rank: int,
) -> dict[str, Any]:
    return {
        **event,
        "selection_rank": int(rank),
        "selection_tier": tier,
        "selection_rationale": rationale,
        "selection_features": features,
    }

def build_process_consistency(
    action: dict[str, Any] | None,
    alignment: dict[str, Any],
    true_concepts: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Audit format and evidence attribution, not hidden reasoning quality.

    Concept membership is checked only when the alignment module exposes a
    current-item reference set. The no-alignment ablation must not be penalized
    for lacking a withheld label.
    """
    if not action:
        return {
            "valid": False,
            "violations": ["missing_structured_response"],
            "used_evidence_ids": [],
        }
    allowed = set(alignment.get("selected_evidence_ids") or [])
    used = [str(value) for value in (action.get("evidence_refs") or [])]
    reference_concepts = [str(value) for value in true_concepts if str(value).strip()]
    alignment_active = not bool(alignment.get("ablated")) and bool(
        alignment.get("current_concepts")
    )
    violations: list[str] = []
    unsupported = sorted(set(used) - allowed)
    if unsupported:
        violations.append("unsupported_evidence_reference")
    if allowed and not used:
        violations.append("selected_evidence_not_attributed")
    identified = str(action.get("identified_concept") or "").strip()
    if not identified:
        violations.append("missing_concept_perception")
    elif alignment_active and identified not in set(reference_concepts):
        violations.append("identified_concept_not_current_item_concept")
    return {
        "valid": not violations,
        "violations": violations,
        "allowed_evidence_ids": sorted(allowed),
        "used_evidence_ids": used,
        "unsupported_evidence_ids": unsupported,
        "identified_concept": identified,
        "reference_concepts": reference_concepts,
        "concept_reference_checked": alignment_active,
    }


def _bounded_prompt_evidence(
    events: list[dict[str, Any]],
    *,
    max_full_questions: int,
    char_budget: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_chars = 0
    for index, event in enumerate(events):
        compact = {
            "evidence_id": event["evidence_id"],
            "source": event["source"],
            "position": event["position"],
            "qid": event["qid"],
            "cid": event["cid"],
            "concepts": event["concepts"],
            "observed_correct": event["observed_correct"],
            "selection_rank": event["selection_rank"],
            "selection_tier": event["selection_tier"],
            "selection_rationale": event["selection_rationale"],
            "selection_features": event["selection_features"],
        }
        if index < max(0, int(max_full_questions)):
            compact["question_excerpt"] = str(event.get("content") or "")[:600]
            options = str(event.get("options") or "")
            if options:
                compact["options_excerpt"] = options[:400]
        rendered_length = len(repr(compact))
        if selected and used_chars + rendered_length > max(0, int(char_budget)):
            break
        if not selected and rendered_length > max(0, int(char_budget)):
            compact.pop("options_excerpt", None)
            compact["question_excerpt"] = str(compact.get("question_excerpt") or "")[:200]
            rendered_length = len(repr(compact))
        selected.append(compact)
        used_chars += rendered_length
    return selected


def _route_components(route: str) -> tuple[str, ...]:
    return tuple(
        component.strip()
        for component in str(route).split("----")
        if component.strip()
    )


def _shared_route_prefix_depth(left: str, right: str) -> int:
    depth = 0
    for left_part, right_part in zip(_route_components(left), _route_components(right)):
        if left_part != right_part:
            break
        depth += 1
    return depth


def _max_shared_route_prefix_depth(left_routes: list[str], right_routes: list[str]) -> int:
    return max(
        (
            _shared_route_prefix_depth(left, right)
            for left in left_routes
            for right in right_routes
        ),
        default=0,
    )


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text)
        if len(token.strip()) > 0
    }


def _route_tokens(routes: list[str]) -> set[str]:
    tokens: set[str] = set()
    for route in routes:
        tokens.update(_tokens(str(route).replace("----", " ")))
    return tokens


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left.intersection(right)) / len(left.union(right))


def _float_or_none(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None

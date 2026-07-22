from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from learner_simulator.ability_profile import build_ability_profile
from learner_simulator.cognitive_profile import build_cognitive_profile
from learner_simulator.data import clean_sequence


@dataclass(frozen=True)
class LearnerProfile:
    """Compact learner context for the full simulator.

    The full model uses ``cognitive_profile`` as the single source of
    historical performance, stability, error, transfer, and risk information.
    This wrapper keeps only non-overlapping identity/exposure fields plus an
    optional IRT estimate for compatibility with non-cognitive fallbacks.
    """

    uid: str
    interaction_count: int
    distinct_concept_count: int
    dominant_concept_id: int | None
    dominant_route: str
    cognitive_profile: dict[str, Any]
    ability_profile: dict[str, Any]

    def to_context(self) -> dict[str, Any]:
        context: dict[str, Any] = {
            "uid": self.uid,
            "history_summary": {
                "interaction_count": self.interaction_count,
                "distinct_concept_count": self.distinct_concept_count,
                "dominant_concept_id": self.dominant_concept_id,
                "dominant_route": self.dominant_route,
            },
            "cognitive_profile": self.cognitive_profile,
            "ability_profile": self.ability_profile,
        }
        return context


def build_learner_profiles(
    rows: list[dict[str, str]],
    questions: dict[str, dict[str, Any]] | None = None,
    ability_estimator: Any | None = None,
    normalization_rows: list[dict[str, str]] | None = None,
) -> dict[str, LearnerProfile]:
    sequences = [(row["uid"], clean_sequence(row)) for row in rows]
    reference_rows = normalization_rows or rows

    profiles: dict[str, LearnerProfile] = {}
    for uid, sequence in sequences:
        interaction_count = len(sequence)
        concept_ids = [step["cid"] for step in sequence]
        distinct_concepts = len(set(concept_ids))
        dominant_concept_id = _most_common_or_none(concept_ids)
        dominant_route = _route_for_concept(dominant_concept_id, sequence, questions or {})
        cognitive_profile = build_cognitive_profile(sequence, questions or {})
        ability_profile = build_ability_profile(
            sequence,
            questions=questions or {},
            normalization_rows=reference_rows,
        )

        profiles[uid] = LearnerProfile(
            uid=uid,
            interaction_count=interaction_count,
            distinct_concept_count=distinct_concepts,
            dominant_concept_id=dominant_concept_id,
            dominant_route=dominant_route,
            cognitive_profile=cognitive_profile,
            ability_profile=ability_profile,
        )
    return profiles


def default_profile(uid: str, global_success_rate: float = 0.5) -> LearnerProfile:
    return LearnerProfile(
        uid=uid,
        interaction_count=0,
        distinct_concept_count=0,
        dominant_concept_id=None,
        dominant_route="",
        cognitive_profile=build_cognitive_profile([]),
        ability_profile=build_ability_profile([]),
    )


def _most_common_or_none(values: list[int]) -> int | None:
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def _route_for_concept(
    preference_cid: int | None,
    sequence: list[dict[str, int]],
    questions: dict[str, dict[str, Any]],
) -> str:
    if preference_cid is None:
        return ""
    for step in sequence:
        if step["cid"] != preference_cid:
            continue
        qmeta = questions.get(str(step["qid"]), {})
        routes = qmeta.get("kc_routes") or []
        if routes:
            return str(routes[0])
    return str(preference_cid)

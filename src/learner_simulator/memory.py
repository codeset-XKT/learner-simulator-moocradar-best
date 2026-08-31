from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LearnerMemory:
    """Small factual/short/long memory module for learner simulation."""

    uid: str
    short_window: int = 5
    long_threshold: int = 3
    factual: list[dict[str, Any]] = field(default_factory=list)
    significant_facts: list[dict[str, Any]] = field(default_factory=list)
    practiced_concepts: set[int] = field(default_factory=set)
    learning_status: list[str] = field(default_factory=list)

    def retrieve_short(self) -> list[dict[str, Any]]:
        return [self._compact_record(record) for record in self.factual[-self.short_window :]]

    def retrieve_long(
        self,
        current_cid: int,
        current_routes: list[str] | None = None,
        mastery: float | None = None,
    ) -> dict[str, Any]:
        related = [
            self._compact_record(record)
            for record in self.significant_facts
            if self._is_related(record.get("cid"), record.get("kc_routes", []), current_cid, current_routes or [])
        ]
        return {
            "significant_facts": related[-self.short_window :],
            "practiced_concepts": sorted(self.practiced_concepts),
            "current_mastery": round(mastery, 4) if mastery is not None else None,
            "latest_learning_status": self.learning_status[-1] if self.learning_status else "",
        }

    def observe(self, record: dict[str, Any]) -> None:
        cid = int(record.get("cid", -1))
        routes = record.get("kc_routes") or []
        for old in self.factual:
            if self._is_related(old.get("cid"), old.get("kc_routes", []), cid, routes):
                old["strength"] = int(old.get("strength", 1)) + 1

        stored = dict(record)
        stored["strength"] = 1
        self.factual.append(stored)
        if cid >= 0:
            self.practiced_concepts.add(cid)
        self._promote_significant_facts()
        self._write_status(record)

    def snapshot(
        self,
        current_cid: int,
        current_routes: list[str] | None = None,
        mastery: float | None = None,
    ) -> dict[str, Any]:
        return {
            "short_memory": self.retrieve_short(),
            "long_memory": self.retrieve_long(current_cid, current_routes, mastery),
        }

    def forget(self, time_step: int, forget_lambda: float = 0.99) -> None:
        """Apply Agent4Edu's logistic long-term-memory forgetting rule."""

        kept: list[dict[str, Any]] = []
        for fact in self.significant_facts:
            fact_id = int(fact.get("memory_id", 0))
            probability = 1.0 / (1.0 + math.exp(-(time_step - fact_id)))
            if probability <= forget_lambda:
                kept.append(fact)
                continue
            index = fact_id - 1
            if 0 <= index < len(self.factual):
                self.factual[index]["strength"] = 1
        self.significant_facts = kept

    def _promote_significant_facts(self) -> None:
        existing = {record.get("memory_id") for record in self.significant_facts}
        for index, record in enumerate(self.factual):
            memory_id = index + 1
            if int(record.get("strength", 1)) >= self.long_threshold and memory_id not in existing:
                promoted = self._compact_record(record)
                promoted["memory_id"] = memory_id
                promoted["strength"] = int(record.get("strength", 1))
                self.significant_facts.append(promoted)

    def _write_status(self, record: dict[str, Any]) -> None:
        outcome = "correct" if int(record.get("simulated_response", 0)) == 1 else "wrong"
        source = record.get("source", "simulated")
        status = f"source={source} step={record.get('step_index')} cid={record.get('cid')} outcome={outcome}"
        self.learning_status.append(status)

    @staticmethod
    def _compact_record(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "step_index": record.get("step_index"),
            "qid": record.get("qid"),
            "cid": record.get("cid"),
            "kc_routes": record.get("kc_routes", []),
            "content_preview": record.get("content_preview", ""),
            "simulated_response": record.get("simulated_response"),
            "real_response": record.get("real_response"),
            "source": record.get("source", "simulated"),
            "strength": record.get("strength", 1),
        }

    @staticmethod
    def _is_related(
        left_cid: Any,
        left_routes: list[str],
        right_cid: int,
        right_routes: list[str],
    ) -> bool:
        if left_cid == right_cid:
            return True
        left_tokens = _route_tokens(left_routes)
        right_tokens = _route_tokens(right_routes)
        return bool(left_tokens and right_tokens and left_tokens.intersection(right_tokens))


def _route_tokens(routes: list[str]) -> set[str]:
    tokens: set[str] = set()
    for route in routes:
        for part in str(route).replace("----", "/").split("/"):
            part = part.strip()
            if part:
                tokens.add(part)
    return tokens

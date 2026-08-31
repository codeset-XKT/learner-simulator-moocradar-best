from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

AGENT4EDU_HISTORY_STEPS = 90
AGENT4EDU_TARGET_STEPS = 10

_field_limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(_field_limit)
        break
    except OverflowError:
        _field_limit //= 10


def load_questions(path: str | Path) -> dict[str, dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_int_list(value: str) -> list[int]:
    if value is None or value == "":
        return []
    return [int(x) for x in value.split(",")]


def clean_sequence(row: dict[str, str]) -> list[dict[str, int]]:
    questions = parse_int_list(row["questions"])
    concepts = parse_int_list(row["concepts"])
    responses = parse_int_list(row["responses"])
    timestamps = parse_int_list(row.get("timestamps", ""))
    steps: list[dict[str, int]] = []

    for idx, (qid, cid, response) in enumerate(zip(questions, concepts, responses)):
        if qid < 0 or cid < 0 or response < 0:
            continue
        step = {
            "qid": qid,
            "cid": cid,
            "response": response,
            "position": idx,
        }
        if idx < len(timestamps) and timestamps[idx] >= 0:
            step["timestamp"] = timestamps[idx]
        steps.append(step)
    return steps


def iter_sequence_rows(path: str | Path) -> Iterable[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        yield from reader


def take_sequence_rows(path: str | Path, limit: int | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in iter_sequence_rows(path):
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def split_rows_agent4edu(
    rows: list[dict[str, str]],
    history_steps: int = AGENT4EDU_HISTORY_STEPS,
    target_steps: int = AGENT4EDU_TARGET_STEPS,
    max_users: int | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Build the fixed-history learner-simulation protocol.

    All sequence rows belonging to the same learner are merged in temporal
    order. Eligible learners contribute exactly ``history_steps`` observed
    interactions followed by exactly ``target_steps`` simulation targets.
    """

    if history_steps <= 0 or target_steps <= 0:
        raise ValueError("history_steps and target_steps must be positive")

    history_rows: list[dict[str, str]] = []
    target_rows: list[dict[str, str]] = []
    required_steps = history_steps + target_steps

    for uid, steps in merge_steps_by_uid(rows).items():
        if len(steps) < required_steps:
            continue
        selected = steps[:required_steps]
        history_rows.append(
            sequence_row_from_steps(uid, selected[:history_steps])
        )
        target_rows.append(
            sequence_row_from_steps(uid, selected[history_steps:])
        )
        if max_users is not None and len(target_rows) >= max_users:
            break
    return history_rows, target_rows


def merge_steps_by_uid(
    rows: list[dict[str, str]],
) -> dict[str, list[dict[str, int]]]:
    """Merge fragmented sequence rows into one chronological sequence per UID."""

    grouped: dict[str, list[tuple[int, int, dict[str, int]]]] = defaultdict(list)
    order = 0
    for row in rows:
        uid = str(row["uid"])
        for step in clean_sequence(row):
            timestamp = int(step.get("timestamp", -1))
            grouped[uid].append((timestamp, order, dict(step)))
            order += 1

    merged: dict[str, list[dict[str, int]]] = {}
    for uid, entries in grouped.items():
        has_timestamps = any(timestamp >= 0 for timestamp, _, _ in entries)
        if has_timestamps:
            entries.sort(
                key=lambda item: (
                    item[0] < 0,
                    item[0] if item[0] >= 0 else item[1],
                    item[1],
                )
            )
        else:
            entries.sort(key=lambda item: item[1])

        steps: list[dict[str, int]] = []
        for position, (_, _, step) in enumerate(entries):
            step["position"] = position
            steps.append(step)
        merged[uid] = steps
    return merged


def sequence_row_from_steps(uid: str, steps: list[dict[str, int]]) -> dict[str, str]:
    timestamps = [
        str(step.get("timestamp", -1))
        for step in steps
    ]
    return {
        "uid": str(uid),
        "questions": ",".join(str(step["qid"]) for step in steps),
        "concepts": ",".join(str(step["cid"]) for step in steps),
        "responses": ",".join(str(step["response"]) for step in steps),
        "timestamps": ",".join(timestamps),
    }


def summarize_questions(questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    type_counter = Counter()
    routes = set()
    leaf_kcs = set()
    has_options = 0
    has_image_in_content = 0
    has_image_in_analysis = 0

    for q in questions.values():
        type_counter[q.get("type", "NA")] += 1
        if q.get("options"):
            has_options += 1
        if "image_" in q.get("content", "") or ".png" in q.get("content", ""):
            has_image_in_content += 1
        if "image_" in q.get("analysis", "") or ".png" in q.get("analysis", ""):
            has_image_in_analysis += 1
        for route in q.get("kc_routes", []):
            routes.add(route)
            leaf_kcs.add(str(route).split("----")[-1])

    return {
        "question_count": len(questions),
        "has_content": sum(1 for q in questions.values() if q.get("content")),
        "has_answer": sum(1 for q in questions.values() if q.get("answer")),
        "has_analysis": sum(1 for q in questions.values() if q.get("analysis")),
        "has_options": has_options,
        "type_counts": dict(type_counter),
        "unique_kc_routes": len(routes),
        "unique_leaf_kcs": len(leaf_kcs),
        "image_in_content": has_image_in_content,
        "image_in_analysis": has_image_in_analysis,
    }


def summarize_sequences(rows: list[dict[str, str]]) -> dict[str, Any]:
    lengths = [len(clean_sequence(row)) for row in rows]
    users = {row["uid"] for row in rows}
    if not lengths:
        return {"rows": 0, "users": 0, "min_len": 0, "avg_len": 0, "max_len": 0}
    return {
        "rows": len(rows),
        "users": len(users),
        "min_len": min(lengths),
        "avg_len": round(sum(lengths) / len(lengths), 2),
        "max_len": max(lengths),
    }

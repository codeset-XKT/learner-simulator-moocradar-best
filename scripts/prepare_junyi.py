from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from learner_simulator.data import sequence_row_from_steps  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert JunYi raw CSV files into the learner simulator sequence format."
    )
    parser.add_argument(
        "--dataset-root",
        default="E:/yyx/KT数据集/JunYi/JunYi",
        help="JunYi root containing encode/ and train_test/.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "test"],
        help="Which JunYi split to convert into train_valid_sequences.csv.",
    )
    parser.add_argument(
        "--output-root",
        default="data/junyi",
        help="Output directory with metadata/ and kc_level/ subdirectories.",
    )
    parser.add_argument(
        "--min-interactions",
        type=int,
        default=100,
        help="Keep users with at least this many valid interactions.",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=None,
        help="Optional cap after sorting users by UID.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_root = resolve_output(args.output_root)
    output_metadata = output_root / "metadata"
    output_kc = output_root / "kc_level"
    output_metadata.mkdir(parents=True, exist_ok=True)
    output_kc.mkdir(parents=True, exist_ok=True)

    question_name_to_id = load_dict(dataset_root / "encode" / "question_id_dict.txt")
    skill_name_to_id = load_dict(dataset_root / "encode" / "skill_id_dict.txt")
    area_name_to_id = load_dict(dataset_root / "encode" / "area_id_dict.txt")

    rows_path = dataset_root / "train_test" / f"{args.split}_df.csv"
    user_steps, questions = load_interactions(
        rows_path,
        question_name_to_id=question_name_to_id,
        skill_name_to_id=skill_name_to_id,
        area_name_to_id=area_name_to_id,
    )

    selected = [
        (uid, sorted(steps, key=lambda item: (item.get("timestamp", -1), item["position"])))
        for uid, steps in sorted(user_steps.items(), key=lambda item: str(item[0]))
        if len(steps) >= args.min_interactions
    ]
    if args.max_users is not None:
        selected = selected[: args.max_users]

    sequence_path = output_kc / "train_valid_sequences.csv"
    with sequence_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["uid", "questions", "concepts", "responses", "timestamps"],
        )
        writer.writeheader()
        for uid, steps in selected:
            for position, step in enumerate(steps):
                step["position"] = position
            writer.writerow(sequence_row_from_steps(str(uid), steps))

    questions_path = output_metadata / "questions.json"
    with questions_path.open("w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    summary = {
        "dataset_root": str(dataset_root),
        "split": args.split,
        "output_root": str(output_root),
        "questions": len(questions),
        "users": len(selected),
        "min_interactions": args.min_interactions,
        "sequence_path": str(sequence_path),
        "questions_path": str(questions_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def resolve_output(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def load_dict(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8").strip()
    raw = ast.literal_eval(text)
    return {str(key): int(value) for key, value in raw.items()}


def load_interactions(
    path: Path,
    question_name_to_id: dict[str, int],
    skill_name_to_id: dict[str, int],
    area_name_to_id: dict[str, int],
) -> tuple[dict[str, list[dict[str, int]]], dict[str, dict[str, Any]]]:
    user_steps: dict[str, list[dict[str, int]]] = defaultdict(list)
    questions: dict[str, dict[str, Any]] = {}

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row_index, row in enumerate(reader):
            uid = str(row.get("user_id", "")).strip()
            exercise = str(row.get("exercise", "")).strip()
            topic = str(row.get("topic", "")).strip()
            area = str(row.get("area", "")).strip()
            correct = parse_binary(row.get("correct"))
            if not uid or not exercise or not topic or correct is None:
                continue

            qid = question_name_to_id.get(exercise)
            cid = skill_name_to_id.get(topic)
            if qid is None or cid is None:
                continue

            timestamp = parse_int(row.get("time_done"), default=row_index)
            user_steps[uid].append(
                {
                    "qid": int(qid),
                    "cid": int(cid),
                    "response": int(correct),
                    "timestamp": int(timestamp),
                    "position": row_index,
                }
            )
            key = str(qid)
            if key not in questions:
                area_id = area_name_to_id.get(area)
                route = build_route(area=area, topic=topic)
                questions[key] = {
                    "id": int(qid),
                    "source_id": exercise,
                    "type": "junyi_exercise",
                    "content": build_content(row),
                    "options": [],
                    "answer": [],
                    "analysis": "",
                    "kc_routes": [route],
                    "skill_id": int(cid),
                    "skill_name": topic,
                    "area_id": area_id,
                    "area_name": area,
                }
    return user_steps, questions


def build_route(area: str, topic: str) -> str:
    if area:
        return f"{humanize(area)}----{humanize(topic)}"
    return humanize(topic)


def build_content(row: dict[str, str]) -> str:
    exercise = str(row.get("exercise", "")).strip()
    problem_type = str(row.get("problem_type", "")).strip()
    problem_number = str(row.get("problem_number", "")).strip()
    pieces = [f"Exercise skill: {humanize(exercise)}."]
    if problem_type:
        pieces.append(f"Problem type: {humanize(problem_type)}.")
    if problem_number:
        pieces.append(f"Problem number: {problem_number}.")
    return " ".join(pieces)


def humanize(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").strip()


def parse_binary(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in {"0", "1"}:
        return int(text)
    if text.lower() in {"true", "false"}:
        return int(text.lower() == "true")
    return None


def parse_int(value: Any, default: int) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    main()

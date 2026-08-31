from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from learner_simulator.data import sequence_row_from_steps  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MOOCRadar six-line sequence files into the learner simulator format."
    )
    parser.add_argument(
        "--dataset-root",
        default="E:/yyx/KT数据集/MoocRadar/MOOCRadar",
        help="MOOCRadar root containing problem.json, encode/, graph/, and train_test/.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "test"],
        help="Which split to convert from train_test/{split}_all_feature.txt.",
    )
    parser.add_argument(
        "--output-root",
        default="data/moocradar",
        help="Output directory with metadata/ and kc_level/ subdirectories.",
    )
    parser.add_argument(
        "--min-interactions",
        type=int,
        default=100,
        help="Keep synthetic users with at least this many valid interactions.",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=None,
        help="Optional cap after source order is preserved.",
    )
    parser.add_argument(
        "--skill-policy",
        default="first",
        choices=["first", "min"],
        help="Policy for choosing one primary skill when a question has multiple skills.",
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

    question_name_to_id = load_json_dict(dataset_root / "encode" / "question_id_dict.json")
    skill_name_to_id = load_json_dict(dataset_root / "encode" / "skill_id_dict.json")
    skill_id_to_name = {int(value): str(key) for key, value in skill_name_to_id.items()}
    question_skills = load_question_skills(dataset_root / "graph" / "ques_skill.csv")
    question_dimensions = load_question_dimensions(dataset_root / "graph" / "ques_hierarchy.csv")
    primary_skill = {
        qid: choose_primary_skill(skills, args.skill_policy)
        for qid, skills in question_skills.items()
        if skills
    }

    questions = load_questions(
        dataset_root / "problem.json",
        question_name_to_id=question_name_to_id,
        skill_id_to_name=skill_id_to_name,
        question_skills=question_skills,
        primary_skill=primary_skill,
        question_dimensions=question_dimensions,
    )

    sequence_path = dataset_root / "train_test" / f"{args.split}_all_feature.txt"
    selected_rows, skipped = load_sequences(
        sequence_path,
        split=args.split,
        primary_skill=primary_skill,
        min_interactions=args.min_interactions,
        max_users=args.max_users,
    )

    output_sequence = output_kc / "train_valid_sequences.csv"
    with output_sequence.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["uid", "questions", "concepts", "responses", "timestamps"],
        )
        writer.writeheader()
        for uid, steps in selected_rows:
            writer.writerow(sequence_row_from_steps(uid, steps))

    output_questions = output_metadata / "questions.json"
    with output_questions.open("w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    total_steps = sum(len(steps) for _, steps in selected_rows)
    correct_steps = sum(step["response"] for _, steps in selected_rows for step in steps)
    summary = {
        "dataset_root": str(dataset_root),
        "split": args.split,
        "output_root": str(output_root),
        "questions": len(questions),
        "skills": len(skill_name_to_id),
        "users": len(selected_rows),
        "min_interactions": args.min_interactions,
        "total_steps": total_steps,
        "positive_rate": safe_div(correct_steps, total_steps),
        "skipped_raw_users": skipped,
        "sequence_path": str(output_sequence),
        "questions_path": str(output_questions),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def resolve_output(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def load_json_dict(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(key): int(value) for key, value in raw.items()}


def load_question_skills(path: Path) -> dict[int, list[int]]:
    mapping: dict[int, list[int]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = parse_int(row.get("question_id"))
            sid = parse_int(row.get("skill_id"))
            if qid is None or sid is None:
                continue
            mapping.setdefault(qid, []).append(sid)
    return mapping


def load_question_dimensions(path: Path) -> dict[int, int]:
    mapping: dict[int, int] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = parse_int(row.get("question_id"))
            dimension = parse_int(row.get("cognitive_dimension"))
            if qid is None or dimension is None:
                continue
            mapping[qid] = dimension
    return mapping


def choose_primary_skill(skills: list[int], policy: str) -> int:
    if policy == "min":
        return min(skills)
    return skills[0]


def load_questions(
    path: Path,
    question_name_to_id: dict[str, int],
    skill_id_to_name: dict[int, str],
    question_skills: dict[int, list[int]],
    primary_skill: dict[int, int],
    question_dimensions: dict[int, int],
) -> dict[str, dict[str, Any]]:
    questions: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            source_id = str(item.get("problem_id", "")).strip()
            if not source_id:
                continue
            qid = question_name_to_id.get(source_id)
            if qid is None:
                continue
            detail = parse_detail(item.get("detail"))
            skill_ids = question_skills.get(qid, [])
            skill_names = [skill_id_to_name.get(sid, str(sid)) for sid in skill_ids]
            content = str(detail.get("content") or "").strip()
            if not content:
                content = f"MOOCRadar problem {source_id}."
            answer = parse_answer(detail.get("answer"))
            question_type = str(detail.get("typetext") or detail.get("type") or "unknown")
            questions[str(qid)] = {
                "id": int(qid),
                "source_id": source_id,
                "exercise_id": item.get("exercise_id"),
                "course_id": item.get("course_id"),
                "type": "moocradar_problem",
                "question_type": question_type,
                "content": content,
                "title": str(detail.get("title") or "").strip(),
                "options": normalize_options(detail.get("option")),
                "answer": answer,
                "analysis": "",
                "kc_routes": build_routes(skill_names),
                "skill_id": primary_skill.get(qid),
                "skill_name": skill_id_to_name.get(primary_skill[qid], str(primary_skill[qid]))
                if qid in primary_skill
                else None,
                "all_skill_ids": skill_ids,
                "all_skill_names": skill_names,
                "knowledge_type": item.get("knowledge_type"),
                "cognitive_dimension": item.get("cognitive_dimension", question_dimensions.get(qid)),
                "score": detail.get("score"),
                "language": detail.get("language"),
                "location": detail.get("location"),
            }
    return questions


def parse_detail(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = ast.literal_eval(text)
        return parsed if isinstance(parsed, dict) else {}
    except (SyntaxError, ValueError):
        return {}


def parse_answer(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    return [text]


def normalize_options(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, dict):
        return []
    options = []
    for key in sorted(value):
        options.append({"label": str(key), "text": str(value[key])})
    return options


def build_routes(skill_names: list[str]) -> list[str]:
    return [name for name in skill_names if name]


def load_sequences(
    path: Path,
    split: str,
    primary_skill: dict[int, int],
    min_interactions: int,
    max_users: int | None,
) -> tuple[list[tuple[str, list[dict[str, int]]]], int]:
    selected: list[tuple[str, list[dict[str, int]]]] = []
    skipped = 0
    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    group_size = 6
    for group_index in range(0, len(lines), group_size):
        group = lines[group_index : group_index + group_size]
        if len(group) < group_size:
            skipped += 1
            continue
        uid = f"{split}_{group_index // group_size:06d}"
        steps = parse_sequence_group(group, primary_skill)
        if len(steps) < min_interactions:
            skipped += 1
            continue
        selected.append((uid, steps))
        if max_users is not None and len(selected) >= max_users:
            break
    return selected, skipped


def parse_sequence_group(
    group: list[str],
    primary_skill: dict[int, int],
) -> list[dict[str, int]]:
    expected = parse_int(group[0])
    qids = parse_int_list(group[1])
    timestamps = parse_int_list(group[2])
    responses = parse_int_list(group[5])
    usable = min(len(qids), len(timestamps), len(responses))
    if expected is not None:
        usable = min(usable, expected)
    steps: list[dict[str, int]] = []
    for position in range(usable):
        qid = qids[position]
        cid = primary_skill.get(qid)
        response = responses[position]
        if cid is None or response not in {0, 1}:
            continue
        steps.append(
            {
                "qid": int(qid),
                "cid": int(cid),
                "response": int(response),
                "timestamp": int(timestamps[position]),
                "position": position,
            }
        )
    return steps


def parse_int_list(value: str) -> list[int]:
    result = []
    for item in value.split(","):
        parsed = parse_int(item)
        if parsed is not None:
            result.append(parsed)
    return result


def parse_int(value: Any) -> int | None:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def safe_div(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


if __name__ == "__main__":
    main()

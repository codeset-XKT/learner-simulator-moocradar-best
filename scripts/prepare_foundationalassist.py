from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from learner_simulator.data import sequence_row_from_steps  # noqa: E402


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data:
            self.parts.append(data)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "img":
            attr_map = {key.lower(): value for key, value in attrs if value}
            alt = attr_map.get("alt") or attr_map.get("data-mathml")
            if alt:
                self.parts.append(f" {alt} ")
        elif tag.lower() in {"br", "p", "div", "li", "tr"}:
            self.parts.append(" ")

    def text(self) -> str:
        return normalize_whitespace(" ".join(self.parts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert FOUNDATIONALASSIST raw CSV files into learner simulator format."
    )
    parser.add_argument(
        "--dataset-root",
        default="E:/yyx/KT数据集/FOUNDATIONALASSIST",
        help="Root containing Interactions.csv, Problems.csv, Skills.csv, and Skill_Set.csv.",
    )
    parser.add_argument(
        "--output-root",
        default="data/foundationalassist",
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
        help="Optional cap after deterministic user sorting.",
    )
    parser.add_argument(
        "--skill-policy",
        default="first",
        choices=["first", "min"],
        help="Policy for choosing one primary skill when a problem maps to multiple skills.",
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

    raw_skills = load_problem_skills(dataset_root / "Skills.csv")
    skill_descriptions = load_skill_descriptions(dataset_root / "Skill_Set.csv")
    problem_id_to_qid = build_problem_mapping(dataset_root / "Problems.csv")
    skill_id_to_cid = build_skill_mapping(raw_skills)
    primary_skill = {
        problem_id: choose_primary_skill(skill_ids, args.skill_policy)
        for problem_id, skill_ids in raw_skills.items()
        if skill_ids
    }

    questions = load_questions(
        dataset_root / "Problems.csv",
        problem_id_to_qid=problem_id_to_qid,
        skill_id_to_cid=skill_id_to_cid,
        raw_skills=raw_skills,
        primary_skill=primary_skill,
        skill_descriptions=skill_descriptions,
    )

    user_steps, skipped = load_interactions(
        dataset_root / "Interactions.csv",
        problem_id_to_qid=problem_id_to_qid,
        skill_id_to_cid=skill_id_to_cid,
        primary_skill=primary_skill,
    )

    selected = [
        (
            uid,
            sorted(
                steps,
                key=lambda item: (
                    item.get("timestamp", -1) < 0,
                    item.get("timestamp", -1) if item.get("timestamp", -1) >= 0 else item["position"],
                    item["position"],
                ),
            ),
        )
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
            writer.writerow(sequence_row_from_steps(uid, steps))

    questions_path = output_metadata / "questions.json"
    with questions_path.open("w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    mappings_path = output_metadata / "id_mappings.json"
    mappings_path.write_text(
        json.dumps(
            {
                "problem_id_to_qid": problem_id_to_qid,
                "skill_id_to_cid": skill_id_to_cid,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    total_steps = sum(len(steps) for _, steps in selected)
    correct_steps = sum(step["response"] for _, steps in selected for step in steps)
    summary = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "questions": len(questions),
        "skills": len(skill_id_to_cid),
        "users": len(selected),
        "min_interactions": args.min_interactions,
        "total_steps": total_steps,
        "positive_rate": safe_div(correct_steps, total_steps),
        "skipped_interactions": skipped,
        "sequence_path": str(sequence_path),
        "questions_path": str(questions_path),
        "mappings_path": str(mappings_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def resolve_output(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def build_problem_mapping(path: Path) -> dict[str, int]:
    problem_ids: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            problem_id = clean_id(row.get("problem_id"))
            if problem_id:
                problem_ids.append(problem_id)
    return {problem_id: index for index, problem_id in enumerate(sorted(set(problem_ids)))}


def build_skill_mapping(problem_skills: dict[str, list[str]]) -> dict[str, int]:
    skill_ids = sorted({skill_id for skill_ids in problem_skills.values() for skill_id in skill_ids})
    return {skill_id: index for index, skill_id in enumerate(skill_ids)}


def load_problem_skills(path: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            problem_id = clean_id(row.get("problem_id"))
            skill_id = clean_id(row.get("skill_id"))
            if problem_id and skill_id and skill_id not in result[problem_id]:
                result[problem_id].append(skill_id)
    return dict(result)


def load_skill_descriptions(path: Path) -> dict[str, dict[str, str]]:
    descriptions: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            skill_id = clean_id(row.get("index"))
            if not skill_id:
                continue
            descriptions[skill_id] = {
                "skill_code": str(row.get("skill_code") or "").strip(),
                "full_description": str(row.get("full_description") or "").strip(),
            }
    return descriptions


def choose_primary_skill(skill_ids: list[str], policy: str) -> str:
    if policy == "min":
        return min(skill_ids, key=lambda value: int(value) if value.isdigit() else value)
    return skill_ids[0]


def load_questions(
    path: Path,
    problem_id_to_qid: dict[str, int],
    skill_id_to_cid: dict[str, int],
    raw_skills: dict[str, list[str]],
    primary_skill: dict[str, str],
    skill_descriptions: dict[str, dict[str, str]],
) -> dict[str, dict[str, Any]]:
    questions: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            problem_id = clean_id(row.get("problem_id"))
            if not problem_id or problem_id not in problem_id_to_qid:
                continue
            qid = problem_id_to_qid[problem_id]
            skill_ids = raw_skills.get(problem_id, [])
            cid = skill_id_to_cid.get(primary_skill[problem_id]) if problem_id in primary_skill else None
            skill_names = [skill_label(skill_id, skill_descriptions) for skill_id in skill_ids]
            questions[str(qid)] = {
                "id": int(qid),
                "source_id": problem_id,
                "type": "foundationalassist_problem",
                "problem_set_id": str(row.get("Problem Set Id") or "").strip(),
                "problem_part": str(row.get("Problem Part") or "").strip(),
                "question_type": str(row.get("Problem Type") or "").strip(),
                "answer_type": str(row.get("Answer Types") or "").strip(),
                "content": html_to_text(row.get("Problem Body")),
                "options": parse_options(row),
                "answer": parse_answers(row),
                "analysis": "",
                "kc_routes": build_routes(skill_ids, skill_descriptions),
                "skill_id": cid,
                "skill_name": skill_label(primary_skill[problem_id], skill_descriptions)
                if problem_id in primary_skill
                else None,
                "all_skill_ids": [skill_id_to_cid[skill_id] for skill_id in skill_ids if skill_id in skill_id_to_cid],
                "all_source_skill_ids": skill_ids,
                "all_skill_names": skill_names,
            }
    return questions


def load_interactions(
    path: Path,
    problem_id_to_qid: dict[str, int],
    skill_id_to_cid: dict[str, int],
    primary_skill: dict[str, str],
) -> tuple[dict[str, list[dict[str, int]]], int]:
    user_steps: dict[str, list[dict[str, int]]] = defaultdict(list)
    skipped = 0
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for position, row in enumerate(reader):
            uid = str(row.get("user_id") or "").strip()
            problem_id = clean_id(row.get("problem_id"))
            response = parse_binary_score(row.get("discrete_score"))
            timestamp = parse_timestamp(row.get("end_time"), default=-1)
            if (
                not uid
                or not problem_id
                or problem_id not in problem_id_to_qid
                or problem_id not in primary_skill
                or response is None
            ):
                skipped += 1
                continue
            cid = skill_id_to_cid.get(primary_skill[problem_id])
            if cid is None:
                skipped += 1
                continue
            user_steps[uid].append(
                {
                    "qid": int(problem_id_to_qid[problem_id]),
                    "cid": int(cid),
                    "response": int(response),
                    "timestamp": int(timestamp),
                    "position": position,
                }
            )
    return dict(user_steps), skipped


def html_to_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = html.unescape(text.replace("&laquo;", "<").replace("&raquo;", ">").replace("&uml;", '"'))
    parser = TextExtractor()
    try:
        parser.feed(text)
        return parser.text()
    except Exception:
        return normalize_whitespace(re.sub(r"<[^>]+>", " ", text))


def parse_options(row: dict[str, str]) -> list[dict[str, str]]:
    raw = str(row.get("Multiple Choice Options") or row.get("Fill-in Options") or "").strip()
    if not raw:
        return []
    parts = split_multi_value(raw)
    if len(parts) <= 1 and row.get("Answer Types", "").lower() != "multiple choice":
        return []
    return [
        {"label": chr(ord("A") + index), "text": html_to_text(part)}
        for index, part in enumerate(parts)
        if html_to_text(part)
    ]


def parse_answers(row: dict[str, str]) -> list[str]:
    raw = str(row.get("Multiple Choice Answers") or row.get("Fill-in Answers") or "").strip()
    return [html_to_text(part) for part in split_multi_value(raw) if html_to_text(part)]


def split_multi_value(value: str) -> list[str]:
    if "||" in value:
        return [part.strip() for part in value.split("||") if part.strip()]
    return [value.strip()] if value.strip() else []


def build_routes(skill_ids: list[str], skill_descriptions: dict[str, dict[str, str]]) -> list[str]:
    routes: list[str] = []
    for skill_id in skill_ids:
        detail = skill_descriptions.get(skill_id, {})
        code = detail.get("skill_code") or f"skill_{skill_id}"
        desc = detail.get("full_description") or code
        routes.append(f"{code}----{desc}")
    return routes


def skill_label(skill_id: str, skill_descriptions: dict[str, dict[str, str]]) -> str:
    detail = skill_descriptions.get(skill_id, {})
    code = detail.get("skill_code")
    desc = detail.get("full_description")
    if code and desc:
        return f"{code}: {desc}"
    if code:
        return code
    return f"skill_{skill_id}"


def clean_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def parse_binary_score(value: Any) -> int | None:
    try:
        score = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if score == 0.0:
        return 0
    if score == 1.0:
        return 1
    return None


def parse_timestamp(value: Any, default: int) -> int:
    text = str(value or "").strip()
    if not text:
        return default
    normalized = text.replace("Z", "+00:00")
    if re.search(r"[+-]\d{2}$", normalized):
        normalized = f"{normalized}:00"
    normalized = re.sub(
        r"\.(\d+)([+-]\d{2}:\d{2})$",
        lambda match: f".{match.group(1)[:6].ljust(6, '0')}{match.group(2)}",
        normalized,
    )
    try:
        return int(datetime.fromisoformat(normalized).timestamp())
    except ValueError:
        return default


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def safe_div(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


if __name__ == "__main__":
    main()

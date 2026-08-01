from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from learner_simulator.data import (  # noqa: E402
    clean_sequence,
    load_questions,
    merge_steps_by_uid,
    sequence_row_from_steps,
    take_sequence_rows,
)
from learner_simulator.llm import load_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a fixed-history cohort whose target segment contains only "
            "single-answer items, without deleting interactions from sequences."
        )
    )
    parser.add_argument("--dataset-root", default="data/moocradar")
    parser.add_argument("--source-rows", type=int, default=20000)
    parser.add_argument("--history-steps", type=int, default=90)
    parser.add_argument("--target-steps", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=30)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--min-target-correct-rate",
        type=float,
        default=None,
        help="Optional lower bound for the real target correct rate.",
    )
    parser.add_argument(
        "--max-target-correct-rate",
        type=float,
        default=None,
        help="Optional upper bound for the real target correct rate.",
    )
    parser.add_argument(
        "--prefer-first-window",
        action="store_true",
        help=(
            "Use only each learner's first 90+10 window if it is eligible. "
            "By default, a sliding complete window is allowed."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = ROOT / dataset_root
    questions = load_questions(dataset_root / "metadata" / "questions.json")
    rows = take_sequence_rows(
        dataset_root / "kc_level" / "train_valid_sequences.csv",
        args.source_rows,
    )
    merged = merge_steps_by_uid(rows)

    history_rows: list[dict[str, str]] = []
    target_rows: list[dict[str, str]] = []
    selected_windows: list[dict[str, int | str]] = []
    skipped_no_window = 0
    required = args.history_steps + args.target_steps

    for uid, steps in merged.items():
        if len(steps) < required:
            continue
        starts = [0] if args.prefer_first_window else range(0, len(steps) - required + 1)
        selected_start = None
        for start in starts:
            target = steps[start + args.history_steps : start + required]
            if _all_single_answer_targets(target, questions):
                target_rate = _target_correct_rate(target)
                if (
                    args.min_target_correct_rate is not None
                    and target_rate < args.min_target_correct_rate
                ):
                    continue
                if (
                    args.max_target_correct_rate is not None
                    and target_rate > args.max_target_correct_rate
                ):
                    continue
                selected_start = int(start)
                break
        if selected_start is None:
            skipped_no_window += 1
            continue

        selected = steps[selected_start : selected_start + required]
        history_rows.append(
            sequence_row_from_steps(uid, selected[: args.history_steps])
        )
        target_rows.append(
            sequence_row_from_steps(uid, selected[args.history_steps :])
        )
        selected_windows.append(
            {
                "uid": uid,
                "start_position": selected_start,
                "history_start": selected_start,
                "history_end": selected_start + args.history_steps - 1,
                "target_start": selected_start + args.history_steps,
                "target_end": selected_start + required - 1,
            }
        )
        if len(target_rows) >= args.max_users:
            break

    if len(target_rows) < args.max_users:
        raise RuntimeError(
            f"Only found {len(target_rows)} eligible learners; requested {args.max_users}. "
            "Increase --source-rows or allow sliding windows."
        )

    payload = {
        "protocol": (
            f"fixed_history_{args.history_steps}_{args.target_steps}_"
            "single_answer_targets"
        ),
        "selection_policy": {
            "target_answer_cardinality": "single",
            "sequence_policy": (
                "first_complete_window_only"
                if args.prefer_first_window
                else "sliding_complete_window_no_interaction_deletion"
            ),
            "history_steps": args.history_steps,
            "target_steps": args.target_steps,
            "max_users": args.max_users,
            "min_target_correct_rate": args.min_target_correct_rate,
            "max_target_correct_rate": args.max_target_correct_rate,
        },
        "dataset_root": str(dataset_root),
        "source_rows_scanned": args.source_rows,
        "history_steps": args.history_steps,
        "target_steps": args.target_steps,
        "uids": [row["uid"] for row in target_rows],
        "selected_windows": selected_windows,
        "skipped_no_single_answer_target_window": skipped_no_window,
        "target_answer_cardinality_summary": _target_cardinality_summary(
            target_rows,
            questions,
        ),
        "history_rows": history_rows,
        "target_rows": target_rows,
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "output": str(output), **payload["target_answer_cardinality_summary"]}, ensure_ascii=False, indent=2))


def _all_single_answer_targets(
    target_steps: list[dict[str, int]],
    questions: dict[str, dict],
) -> bool:
    return all(_answer_count(questions.get(str(step["qid"]), {})) == 1 for step in target_steps)


def _target_correct_rate(target_steps: list[dict[str, int]]) -> float:
    if not target_steps:
        return 0.0
    return sum(int(step["response"]) for step in target_steps) / len(target_steps)


def _answer_count(question: dict) -> int:
    answer = question.get("answer")
    if isinstance(answer, list):
        return len([item for item in answer if str(item).strip()])
    if answer is None:
        return 0
    return 1 if str(answer).strip() else 0


def _target_cardinality_summary(
    target_rows: list[dict[str, str]],
    questions: dict[str, dict],
) -> dict[str, int]:
    single = 0
    multi = 0
    missing = 0
    total = 0
    for row in target_rows:
        for step in clean_sequence(row):
            count = _answer_count(questions.get(str(step["qid"]), {}))
            total += 1
            if count == 1:
                single += 1
            elif count > 1:
                multi += 1
            else:
                missing += 1
    return {
        "target_items": total,
        "single_answer_targets": single,
        "multi_answer_targets": multi,
        "missing_answer_targets": missing,
    }


if __name__ == "__main__":
    main()

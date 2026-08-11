from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from learner_simulator.data import (  # noqa: E402
    load_questions,
    merge_steps_by_uid,
    sequence_row_from_steps,
    take_sequence_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build immutable random, non-overlapping 90+10 cohort batches."
    )
    parser.add_argument("--dataset-root", default="data/moocradar")
    parser.add_argument("--source-rows", type=int, default=20000)
    parser.add_argument("--history-steps", type=int, default=90)
    parser.add_argument("--target-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Slug used in plan and cohort filenames; defaults to the dataset directory name.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/cohorts/moocradar_500x10_batches",
    )
    parser.add_argument(
        "--exclude-glob",
        default="experiments/cohorts/**/*moocradar*50x10*.json",
        help="Existing cohort files whose learners must not be selected again.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = _resolve(args.dataset_root)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = _dataset_slug(args.dataset_name or dataset_root.name)
    manifest_path = output_dir / f"{dataset_name}_500x10_plan_seed{args.seed}.json"
    if manifest_path.exists():
        raise FileExistsError(f"Plan already exists and will not be overwritten: {manifest_path}")

    questions = load_questions(dataset_root / "metadata" / "questions.json")
    rows = take_sequence_rows(
        dataset_root / "kc_level" / "train_valid_sequences.csv",
        args.source_rows,
    )
    merged = merge_steps_by_uid(rows)
    required = args.history_steps + args.target_steps
    eligible_windows: dict[str, list[int]] = {}
    for uid, steps in merged.items():
        if len(steps) < required:
            continue
        starts = [
            start
            for start in range(len(steps) - required + 1)
            if _all_single_answer_targets(
                steps[start + args.history_steps : start + required],
                questions,
            )
        ]
        if starts:
            eligible_windows[str(uid)] = starts

    excluded_uids, excluded_files = _load_excluded_uids(args.exclude_glob)
    candidates = sorted(set(eligible_windows) - excluded_uids)
    required_users = args.batch_size * args.batches
    if len(candidates) < required_users:
        raise RuntimeError(
            f"Only {len(candidates)} unused eligible learners; need {required_users}."
        )

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    selected_uids = candidates[:required_users]
    batch_manifests = []
    for batch_index in range(args.batches):
        batch_uids = selected_uids[
            batch_index * args.batch_size : (batch_index + 1) * args.batch_size
        ]
        history_rows = []
        target_rows = []
        selected_windows = []
        target_correct = []
        for uid in batch_uids:
            steps = merged[uid]
            start = rng.choice(eligible_windows[uid])
            selected = steps[start : start + required]
            history = selected[: args.history_steps]
            target = selected[args.history_steps :]
            history_rows.append(sequence_row_from_steps(uid, history))
            target_rows.append(sequence_row_from_steps(uid, target))
            target_correct.extend(int(step["response"]) for step in target)
            selected_windows.append(
                {
                    "uid": uid,
                    "start_position": start,
                    "history_start": start,
                    "history_end": start + args.history_steps - 1,
                    "target_start": start + args.history_steps,
                    "target_end": start + required - 1,
                }
            )

        batch_number = batch_index + 1
        batch_path = output_dir / (
            f"{dataset_name}_500x10_batch{batch_number:02d}_50x10_seed{args.seed}.json"
        )
        payload = {
            "protocol": f"fixed_history_{args.history_steps}_{args.target_steps}",
            "plan_seed": args.seed,
            "batch_index": batch_number,
            "batch_size": args.batch_size,
            "total_planned_batches": args.batches,
            "selection_policy": {
                "learner_sampling": "random_without_replacement",
                "target_answer_cardinality": "single",
                "sequence_policy": "random_complete_sliding_window_no_interaction_deletion",
                "excluded_previous_50x10_cohorts": excluded_files,
            },
            "dataset_root": str(dataset_root),
            "source_rows_scanned": len(rows),
            "history_steps": args.history_steps,
            "target_steps": args.target_steps,
            "uids": batch_uids,
            "selected_windows": selected_windows,
            "real_target_correct_rate": round(sum(target_correct) / len(target_correct), 6),
            "history_rows": history_rows,
            "target_rows": target_rows,
        }
        _write_new_json(batch_path, payload)
        batch_manifests.append(
            {
                "batch_index": batch_number,
                "cohort_file": str(batch_path),
                "uids": batch_uids,
                "real_target_correct_rate": payload["real_target_correct_rate"],
            }
        )

    manifest = {
        "plan": f"{dataset_name}_random_disjoint_500x10",
        "dataset_name": dataset_name,
        "seed": args.seed,
        "total_learners": required_users,
        "targets_per_learner": args.target_steps,
        "eligible_learners_before_exclusion": len(eligible_windows),
        "previously_used_uids_excluded": len(excluded_uids),
        "eligible_learners_after_exclusion": len(candidates),
        "excluded_cohort_files": excluded_files,
        "batches": batch_manifests,
    }
    _write_new_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "ok": True,
                "manifest": str(manifest_path),
                "batches": len(batch_manifests),
                "total_learners": required_users,
                "batch_1": batch_manifests[0],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _load_excluded_uids(pattern: str) -> tuple[set[str], list[str]]:
    files = sorted(ROOT.glob(str(Path(pattern)).replace("\\", "/")))
    excluded: set[str] = set()
    used_files = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        uids = [str(uid) for uid in payload.get("uids", [])]
        if not uids:
            continue
        excluded.update(uids)
        used_files.append(str(path))
    return excluded, used_files


def _all_single_answer_targets(
    target_steps: list[dict[str, int]],
    questions: dict[str, dict[str, Any]],
) -> bool:
    return all(
        _answer_count(questions.get(str(step["qid"]), {})) == 1
        for step in target_steps
    )


def _answer_count(question: dict[str, Any]) -> int:
    answer = question.get("answer")
    if isinstance(answer, list):
        return len([item for item in answer if str(item).strip()])
    if answer is None:
        return 0
    return int(bool(str(answer).strip()))


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _dataset_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not slug:
        raise ValueError("--dataset-name must contain at least one ASCII letter or digit")
    return slug


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

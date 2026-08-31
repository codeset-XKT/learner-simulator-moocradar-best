from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from learner_simulator.data import merge_steps_by_uid, sequence_row_from_steps, take_sequence_rows  # noqa: E402
from learner_simulator.dkt import train_dkt  # noqa: E402
from learner_simulator.llm import load_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a standard DKT model on XES3G5M kc-level sequences.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--source-rows",
        type=int,
        default=0,
        help="Number of source rows to read. Use 0 for the full train_valid_sequences.csv.",
    )
    parser.add_argument("--output-dir", default="outputs/dkt")
    parser.add_argument(
        "--exclude-cohort-file",
        default=None,
        nargs="+",
        help="One or more cohort JSON files whose users must be excluded from DKT training.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=0,
        help="Optionally truncate each merged training sequence to this many interactions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    limit = None if args.source_rows <= 0 else args.source_rows
    source_rows = take_sequence_rows(dataset_root / "kc_level" / "train_valid_sequences.csv", limit)
    cohort_paths: list[Path] = []
    cohort_sha256: dict[str, str] = {}
    excluded_uids: set[str] = set()
    if args.exclude_cohort_file:
        cohort_paths = [Path(value) for value in args.exclude_cohort_file]
        cohort_paths = [path if path.is_absolute() else ROOT / path for path in cohort_paths]
        for cohort_path in cohort_paths:
            cohort_sha256[str(cohort_path)] = hashlib.sha256(cohort_path.read_bytes()).hexdigest()
            cohort = load_json(cohort_path)
            excluded_uids.update(str(uid) for uid in cohort.get("uids", []))
            excluded_uids.update(str(row["uid"]) for row in cohort.get("history_rows", []))
        if not excluded_uids:
            raise ValueError("No learner UIDs found in exclusion cohorts")
    merged_rows = merge_steps_by_uid(source_rows)
    source_uids = {str(uid) for uid in merged_rows}
    excluded_present_uids = source_uids & excluded_uids
    train_rows = [
        sequence_row_from_steps(
            uid,
            steps[: args.max_train_steps] if args.max_train_steps > 0 else steps,
        )
        for uid, steps in merged_rows.items()
        if len(steps) >= 2 and str(uid) not in excluded_uids
    ]
    print(
        f"Training DKT with source_rows={'full' if limit is None else limit}, "
        f"train_sequences={len(train_rows)}, excluded_cohort_users={len(excluded_uids)}, "
        f"excluded_users_present_in_source={len(excluded_present_uids)}",
        flush=True,
    )
    checkpoint = train_dkt(
        train_rows,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        seed=args.seed,
        log_every=args.log_every,
        training_metadata={
            "leakage_safe": bool(cohort_paths),
            "cohort_history_used_for_training": False,
            "excluded_user_ids": sorted(excluded_uids),
            "excluded_users_present_in_source": len(excluded_present_uids),
            "cohort_files": [str(path) for path in cohort_paths],
            "cohort_sha256": cohort_sha256,
            "training_user_count": len(train_rows),
        },
    )
    summary = {
        "ok": True,
        "checkpoint": str(checkpoint),
        "train_users": len(train_rows),
        "source_rows": "full" if limit is None else limit,
        "max_train_steps": args.max_train_steps,
        "leakage_safe": bool(cohort_paths),
        "excluded_cohort_users": len(excluded_uids),
        "excluded_users_present_in_source": len(excluded_present_uids),
        "cohort_history_used_for_training": False,
        "cohort_files": [str(path) for path in cohort_paths],
        "cohort_sha256": cohort_sha256,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

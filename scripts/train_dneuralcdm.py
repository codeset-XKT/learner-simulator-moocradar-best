from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from learner_simulator.data import (  # noqa: E402
    merge_steps_by_uid,
    sequence_row_from_steps,
    take_sequence_rows,
)
from learner_simulator.dneuralcdm import (  # noqa: E402
    export_dneuralcdm_proficiency,
    train_dneuralcdm,
)
from learner_simulator.llm import load_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a DNeuralCDM-style proficiency estimator and export "
            "student knowledge proficiency for a fixed cohort."
        )
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--source-rows", type=int, default=50000)
    parser.add_argument("--cohort-file", default=None)
    parser.add_argument("--output-dir", default="outputs/dneuralcdm")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=0,
        help="Optionally truncate each source training sequence to this many interactions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    source_rows = take_sequence_rows(
        dataset_root / "kc_level" / "train_valid_sequences.csv",
        args.source_rows,
    )
    cohort_path = None
    cohort_history_rows = []
    if args.cohort_file:
        cohort_path = Path(args.cohort_file)
        if not cohort_path.is_absolute():
            cohort_path = ROOT / cohort_path
        cohort = load_json(cohort_path)
        cohort_history_rows = list(cohort["history_rows"])

    cohort_uids = {str(row["uid"]) for row in cohort_history_rows}
    train_rows = [
        sequence_row_from_steps(
            uid,
            steps[: args.max_train_steps] if args.max_train_steps > 0 else steps,
        )
        for uid, steps in merge_steps_by_uid(source_rows).items()
        if len(steps) >= 2 and str(uid) not in cohort_uids
    ]
    print(
        f"Training DNeuralCDM with source_rows={args.source_rows}, "
        f"train_sequences={len(train_rows)}, excluded_cohort_users={len(cohort_uids)}, "
        f"cohort_history_sequences={len(cohort_history_rows)}",
        flush=True,
    )
    checkpoint = train_dneuralcdm(
        train_rows,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        log_every=args.log_every,
        training_metadata={
            "cohort_history_used_for_training": False,
            "excluded_user_ids": sorted(cohort_uids),
            "cohort_file": str(cohort_path) if cohort_path else None,
            "training_user_count": len(train_rows),
        },
    )

    export_rows = train_rows
    if args.cohort_file:
        export_rows = cohort_history_rows

    output_path = output_dir / "stu_know_proficiency.json"
    print(f"Exporting DNeuralCDM proficiency for {len(export_rows)} users...", flush=True)
    export_dneuralcdm_proficiency(export_rows, checkpoint, output_path)
    summary = {
        "ok": True,
        "checkpoint": str(checkpoint),
        "proficiency": str(output_path),
        "train_users": len(train_rows),
        "export_users": len(export_rows),
        "excluded_cohort_users": len(cohort_uids),
        "cohort_history_used_for_training": False,
        "cohort_file": str(cohort_path) if cohort_path else None,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

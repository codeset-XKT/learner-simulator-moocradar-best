from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from learner_simulator.data import merge_steps_by_uid, sequence_row_from_steps, take_sequence_rows  # noqa: E402
from learner_simulator.mikt import train_mikt, torch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MIKT on learner simulator kc-level sequences using the same preprocessed data path as DKT."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--source-rows",
        type=int,
        default=0,
        help="Number of source rows to read. Use 0 for the full train_valid_sequences.csv.",
    )
    parser.add_argument("--output-dir", default="outputs/mikt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--device", default=None, help="Optional torch device, e.g. cuda or cpu.")
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=0,
        help="Optionally truncate each merged training sequence to this many interactions.",
    )
    parser.add_argument(
        "--exclude-cohort-file",
        nargs="+",
        default=[],
        help="Fixed-cohort JSON files whose learner IDs must be excluded before MIKT training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    limit = None if args.source_rows <= 0 else args.source_rows
    source_rows = take_sequence_rows(dataset_root / "kc_level" / "train_valid_sequences.csv", limit)
    excluded_uids: set[str] = set()
    mapping_rows: list[dict[str, str]] = []
    for raw_cohort in args.exclude_cohort_file:
        cohort = json.loads(Path(raw_cohort).read_text(encoding="utf-8-sig"))
        excluded_uids.update(str(uid) for uid in cohort.get("uids", []))
        mapping_rows.extend(cohort.get("history_rows", []))
        mapping_rows.extend(cohort.get("target_rows", []))
    merged_steps = merge_steps_by_uid(source_rows)
    train_rows = [
        sequence_row_from_steps(
            uid,
            steps[: args.max_train_steps] if args.max_train_steps > 0 else steps,
        )
        for uid, steps in merged_steps.items()
        if uid not in excluded_uids and len(steps) >= 2
    ]
    if {str(row["uid"]) for row in train_rows} & excluded_uids:
        raise AssertionError("Cohort users leaked into MIKT training rows")
    print(
        f"Training MIKT with source_rows={'full' if limit is None else limit}, "
        f"train_sequences={len(train_rows)}",
        flush=True,
    )
    checkpoint = train_mikt(
        train_rows,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        seed=args.seed,
        log_every=args.log_every,
        device_name=args.device,
        mapping_rows=mapping_rows,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )
    training_metadata = {
        "leakage_safe": True,
        "excluded_user_ids": sorted(excluded_uids),
        "excluded_cohort_users": len(excluded_uids),
        "cohort_history_used_for_training": False,
        "source_sequence_rows": len(source_rows),
        "source_users_before_exclusion": len(merged_steps),
        "train_users": len(train_rows),
        "official_reference": "third_party/MIKT_official",
        "cohort_question_and_concept_ids_registered_without_labels": True,
        "max_epochs": args.epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
    }
    if torch is None:
        raise RuntimeError("MIKT training completed without a PyTorch runtime")
    payload = torch.load(checkpoint, map_location="cpu")
    payload["training_metadata"] = training_metadata
    torch.save(payload, checkpoint)
    summary = {
        "ok": True,
        "checkpoint": str(checkpoint),
        "train_users": len(train_rows),
        "source_rows": "full" if limit is None else limit,
        "max_train_steps": args.max_train_steps,
        "training_metadata": training_metadata,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

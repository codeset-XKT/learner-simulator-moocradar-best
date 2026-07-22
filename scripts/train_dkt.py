from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from learner_simulator.data import merge_steps_by_uid, sequence_row_from_steps, take_sequence_rows  # noqa: E402
from learner_simulator.dkt import train_dkt  # noqa: E402


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
    train_rows = [
        sequence_row_from_steps(
            uid,
            steps[: args.max_train_steps] if args.max_train_steps > 0 else steps,
        )
        for uid, steps in merge_steps_by_uid(source_rows).items()
        if len(steps) >= 2
    ]
    print(
        f"Training DKT with source_rows={'full' if limit is None else limit}, "
        f"train_sequences={len(train_rows)}",
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
    )
    summary = {
        "ok": True,
        "checkpoint": str(checkpoint),
        "train_users": len(train_rows),
        "source_rows": "full" if limit is None else limit,
        "max_train_steps": args.max_train_steps,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

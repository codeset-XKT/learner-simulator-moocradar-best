#!/usr/bin/env python3
"""Train the IKT (BKT + ability profile + TAN) reference baseline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from learner_simulator.data import merge_steps_by_uid, sequence_row_from_steps, take_sequence_rows
from learner_simulator.ikt import IKTModel


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--exclude-cohort-file", nargs="+", default=[])
    parser.add_argument("--source-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    excluded: set[str] = set()
    for raw in args.exclude_cohort_file:
        excluded.update(str(uid) for uid in json.loads(Path(raw).read_text(encoding="utf-8-sig")).get("uids", []))
    dataset_root = Path(args.dataset_root)
    raw_rows = take_sequence_rows(dataset_root / "kc_level" / "train_valid_sequences.csv", None if args.source_rows <= 0 else args.source_rows)
    merged = merge_steps_by_uid(raw_rows)
    rows = [sequence_row_from_steps(uid, steps) for uid, steps in merged.items() if uid not in excluded and len(steps) >= 2]
    # The source file naturally contains cohort users; leakage is prevented by
    # their exclusion from ``rows`` before IKT sees any interactions.
    if {str(row["uid"]) for row in rows} & excluded:
        raise AssertionError("Cohort users leaked into IKT train rows")
    model = IKTModel()
    fit = model.fit(rows, seed=args.seed)
    metadata = {
        "leakage_safe": True,
        "excluded_user_ids": sorted(excluded),
        "excluded_cohort_users": len(excluded),
        "cohort_history_used_for_training": False,
        "source_sequence_rows": len(raw_rows),
        "source_users_before_exclusion": len(merged),
        "train_users": len(rows),
        "seed": args.seed,
        "model_components": ["per_skill_bkt_mastery", "cross_skill_ability_profile", "item_difficulty_decile", "tree_augmented_naive_bayes"],
    }
    output = Path(args.output_dir)
    checkpoint = model.save(output / "best_model.json", metadata)
    summary = {"ok": True, "method": "ikt", "checkpoint": str(checkpoint), "fit": fit, "training_metadata": metadata}
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

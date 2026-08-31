#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from learner_simulator.data import clean_sequence  # noqa: E402
from learner_simulator.formal_metrics import (  # noqa: E402
    build_ability_difficulty_partition,
    evaluate_formal_response_metrics,
)
from learner_simulator.kes import KnowledgeEvolutionSimulator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a DKT-driven KES on a fixed cohort.")
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--dkt-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--full-report")
    parser.add_argument(
        "--partition-reference",
        help="fixed_full_irt_reference.json whose precomputed global IRT 2x2 partition is reused for ADCDE.",
    )
    parser.add_argument("--feedback-mode", choices=["teacher-forcing", "rollout"], default="teacher-forcing")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cohort = json.loads(Path(args.cohort).read_text(encoding="utf-8-sig"))
    simulator = KnowledgeEvolutionSimulator(args.dkt_checkpoint)
    metadata = simulator.training_metadata
    if not metadata.get("leakage_safe"):
        raise ValueError("KES requires a leakage-safe DKT checkpoint")
    excluded = {str(uid) for uid in metadata.get("excluded_user_ids", [])}
    history = {str(row["uid"]): clean_sequence(row) for row in cohort["history_rows"]}
    target = {str(row["uid"]): clean_sequence(row) for row in cohort["target_rows"]}
    steps: list[dict] = []
    skipped = 0
    for index, uid in enumerate(cohort["uids"]):
        uid = str(uid)
        if uid not in excluded:
            raise ValueError(f"KES target user {uid} was not excluded from DKT training")
        user_steps, user_skipped = simulator.simulate(
            history[uid],
            target[uid],
            feedback_mode=args.feedback_mode,
            threshold=args.threshold,
            seed=args.seed + index,
        )
        for step in user_steps:
            step["uid"] = uid
        steps.extend(user_steps)
        skipped += user_skipped
    expected = len(cohort["uids"]) * int(cohort["target_steps"])
    if len(steps) != expected or skipped:
        raise AssertionError(f"KES target coverage failed: {len(steps)}/{expected}, skipped={skipped}")
    partition = None
    if args.partition_reference:
        reference = json.loads(Path(args.partition_reference).read_text(encoding="utf-8-sig"))
        partition = reference.get("fixed_full_irt_2x2_partition")
        if not isinstance(partition, dict) or not partition.get("available"):
            raise ValueError("partition reference has no usable fixed_full_irt_2x2_partition")
    elif args.full_report:
        full = json.loads(Path(args.full_report).read_text(encoding="utf-8-sig"))
        partition = full["formal_metrics_v6"].get("fixed_full_irt_2x2_partition")
        if not partition:
            partition = build_ability_difficulty_partition(full["reports"]["full"]["all_steps"])
    formal = evaluate_formal_response_metrics(steps, partition)
    result = {
        "method": "KES_DKT_driven_official_paper_reproduction",
        "dkt_checkpoint": str(Path(args.dkt_checkpoint)),
        "cohort": str(Path(args.cohort)),
        "feedback_mode": args.feedback_mode,
        "threshold": args.threshold,
        "seed": args.seed,
        "partition_reference": str(Path(args.partition_reference)) if args.partition_reference else None,
        "training_metadata": metadata,
        "count": len(steps),
        "formal_metrics_v6": formal,
        "steps": steps,
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "count": len(steps), **{key: formal[key] for key in ("baa", "balanced_accuracy", "f1", "adcde")}}, ensure_ascii=False))


if __name__ == "__main__":
    main()

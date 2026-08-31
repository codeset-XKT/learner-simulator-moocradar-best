#!/usr/bin/env python3
"""Teacher-forcing evaluation for the leakage-safe IKT reference baseline."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from learner_simulator.data import clean_sequence
from learner_simulator.formal_metrics import evaluate_formal_response_metrics
from learner_simulator.ikt import IKTStudentState, load_ikt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cohort-file", nargs="+", required=True)
    parser.add_argument("--full-irt-reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, metadata = load_ikt(args.checkpoint)
    if not metadata.get("leakage_safe"):
        raise ValueError("checkpoint is not leakage-safe")
    excluded = set(map(str, metadata.get("excluded_user_ids", [])))
    steps: list[dict] = []
    users: list[str] = []
    for batch_index, path in enumerate(args.cohort_file):
        cohort = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        histories = {str(row["uid"]): clean_sequence(row) for row in cohort["history_rows"]}
        targets = {str(row["uid"]): clean_sequence(row) for row in cohort["target_rows"]}
        for offset, raw_uid in enumerate(cohort["uids"]):
            uid = str(raw_uid)
            if uid not in excluded:
                raise AssertionError(f"{uid} was not excluded during training")
            state = IKTStudentState.empty()
            for item in histories[uid]:
                model.update(state, item, int(item["response"]))
            rng = random.Random(args.seed + batch_index * 10_000 + offset)
            for step_index, item in enumerate(targets[uid]):
                probability = float(model.predict(state, item))
                simulated = int(rng.random() < probability)
                # Formal teacher forcing: only preceding observed target answers
                # enter the state used for later target questions.
                model.update(state, item, int(item["response"]))
                steps.append({
                    "uid": uid, "step_index": step_index, "qid": int(item["qid"]),
                    "real_response": int(item["response"]), "simulated_response": simulated,
                    "probability": probability, "feedback_mode": "teacher_forcing",
                })
            users.append(uid)
    if len(users) != 500 or len(set(users)) != 500 or len(steps) != 5000:
        raise AssertionError(f"coverage invalid: users={len(users)}, steps={len(steps)}")
    ref = json.loads(Path(args.full_irt_reference).read_text(encoding="utf-8-sig"))
    report = {
        "method": "ikt", "checkpoint": args.checkpoint, "cohort_files": args.cohort_file,
        "training_metadata": metadata,
        "audit": {"users": 500, "steps": 5000, "raw_step_pooled": True, "feedback_mode": "teacher_forcing"},
        "formal_metrics_v6": evaluate_formal_response_metrics(steps, ref["fixed_full_irt_2x2_partition"]),
        "all_steps": steps,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = report["formal_metrics_v6"]
    print(json.dumps({**report["audit"], **{key: metrics[key] for key in ("baa", "balanced_accuracy", "f1", "adcde")}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

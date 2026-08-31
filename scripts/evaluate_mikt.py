#!/usr/bin/env python3
"""Teacher-forcing fixed-cohort evaluation for the MIKT reference baseline."""
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
from learner_simulator.mikt import MIKT, MIKTConfig, _build_problem_skill_tensor, torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cohort-file", nargs="+", required=True)
    parser.add_argument("--full-irt-reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if torch is None:
        raise RuntimeError("MIKT evaluation requires PyTorch")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    metadata = checkpoint.get("training_metadata") or {}
    if not metadata.get("leakage_safe"):
        raise ValueError("checkpoint is not leakage-safe")
    config = MIKTConfig(**checkpoint["config"])
    question_map = {str(key): int(value) for key, value in checkpoint["question_id_map"].items()}
    empty_links = _build_problem_skill_tensor({}, config)
    model = MIKT(config, empty_links).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    excluded = set(map(str, metadata.get("excluded_user_ids", [])))
    records: list[dict] = []
    for batch_index, path in enumerate(args.cohort_file):
        cohort = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        histories = {str(row["uid"]): clean_sequence(row) for row in cohort["history_rows"]}
        targets = {str(row["uid"]): clean_sequence(row) for row in cohort["target_rows"]}
        for offset, raw_uid in enumerate(cohort["uids"]):
            uid = str(raw_uid)
            if uid not in excluded:
                raise AssertionError(f"{uid} was not excluded during training")
            history = histories[uid]
            target = targets[uid]
            mapped_history = [question_map.get(str(item["qid"])) for item in history]
            mapped_target = [question_map.get(str(item["qid"])) for item in target]
            if len(history) != 90 or len(target) != 10 or any(value is None for value in mapped_history + mapped_target):
                raise AssertionError(f"MIKT item coverage invalid for {uid}")
            records.append({
                "uid": uid, "q": [int(value) for value in mapped_history],
                "r": [int(item["response"]) for item in history],
                "target": target, "mapped_target": [int(value) for value in mapped_target],
                "rng": random.Random(args.seed + batch_index * 10_000 + offset),
            })
    if len(records) != 500 or len({record["uid"] for record in records}) != 500:
        raise AssertionError("cohort coverage must be exactly 500 unique users")
    steps: list[dict] = []
    for index in range(10):
        # Responses up to index-1 are observed target feedback; the current
        # target response is masked as 0. MIKT emits each position before it
        # writes that position's answer into its dynamic state.
        q = torch.tensor([record["q"] + record["mapped_target"][: index + 1] for record in records], dtype=torch.long, device=device)
        r = torch.tensor([record["r"] + [int(item["response"]) for item in record["target"][:index]] + [0] for record in records], dtype=torch.long, device=device)
        with torch.no_grad():
            probabilities, _ = model(q[:, :-1], r[:, :-1], q[:, 1:], r[:, 1:])
        for record, probability in zip(records, probabilities[:, -1].detach().cpu().tolist()):
            item = record["target"][index]
            probability = float(probability)
            simulated = int(record["rng"].random() < probability)
            steps.append({
                "uid": record["uid"], "step_index": index, "qid": int(item["qid"]),
                "real_response": int(item["response"]), "simulated_response": simulated,
                "probability": probability, "feedback_mode": "teacher_forcing",
            })
    if len(steps) != 5000:
        raise AssertionError(f"coverage invalid: steps={len(steps)}")
    ref = json.loads(Path(args.full_irt_reference).read_text(encoding="utf-8-sig"))
    report = {
        "method": "mikt", "checkpoint": args.checkpoint, "cohort_files": args.cohort_file,
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

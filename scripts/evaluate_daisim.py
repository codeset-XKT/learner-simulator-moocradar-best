#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from learner_simulator.daisim import encode_steps, history_feature, load_generator  # noqa: E402
from learner_simulator.data import clean_sequence  # noqa: E402
from learner_simulator.formal_metrics import (  # noqa: E402
    build_ability_difficulty_partition,
    evaluate_formal_response_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DAISim on an exact fixed cohort.")
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--full-report",
        help="Matching Full report used only to reuse its fixed IRT 2x2 partition.",
    )
    parser.add_argument(
        "--partition-reference",
        help="fixed_full_irt_reference.json whose precomputed global IRT 2x2 partition is reused for ADCDE.",
    )
    parser.add_argument("--mode", choices=["teacher-forcing", "rollout"], default="teacher-forcing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cohort = json.loads(Path(args.cohort).read_text(encoding="utf-8-sig"))
    history_by_uid = {str(row["uid"]): clean_sequence(row) for row in cohort["history_rows"]}
    target_by_uid = {str(row["uid"]): clean_sequence(row) for row in cohort["target_rows"]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    metadata = checkpoint.get("training_metadata") or {}
    if not metadata.get("leakage_safe"):
        raise ValueError("DAISim checkpoint is not marked leakage safe")
    generator, item_index, concept_index = load_generator(checkpoint, device)
    steps = []
    for uid in cohort["uids"]:
        uid = str(uid)
        history = history_by_uid[uid]
        target = target_by_uid[uid]
        if uid not in set(str(value) for value in metadata.get("excluded_user_ids", [])):
            raise ValueError(f"held-out uid {uid} was not excluded from DAISim training")
        item_ids, concept_ids, responses = encode_steps(target, item_index, concept_index)
        feature = history_feature(history, concept_index).unsqueeze(0).to(device)
        items = torch.tensor(item_ids, dtype=torch.long, device=device).unsqueeze(0)
        concepts = torch.tensor(concept_ids, dtype=torch.long, device=device).unsqueeze(0)
        response_tensor = torch.tensor(responses, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            if args.mode == "teacher-forcing":
                probabilities = torch.sigmoid(generator.teacher_forced_logits(feature, items, concepts, response_tensor))[0]
            else:
                probabilities = generator.rollout_probabilities(feature, items, concepts)[0]
        for target_step, probability in zip(target, probabilities.cpu().tolist()):
            steps.append({
                "uid": uid,
                "step_index": target_step.get("position"),
                "qid": target_step.get("qid"),
                "real_response": int(target_step["response"]),
                "simulated_response": int(probability >= 0.5),
                "daisim_probability": float(probability),
                "prediction_source": f"daisim_{args.mode}",
            })
    expected = len(cohort["uids"]) * int(cohort["target_steps"])
    if len(steps) != expected:
        raise AssertionError(f"expected {expected} steps, got {len(steps)}")
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
        "method": "DAISim_official_code_derived_reproduction",
        "official_repository_commit": checkpoint.get("official_repository_commit"),
        "checkpoint": str(Path(args.checkpoint)),
        "cohort": str(Path(args.cohort)),
        "full_report": str(Path(args.full_report)) if args.full_report else None,
        "partition_reference": str(Path(args.partition_reference)) if args.partition_reference else None,
        "mode": args.mode,
        "training_metadata": metadata,
        "count": len(steps),
        "formal_metrics_v6": formal,
        "steps": steps,
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "count": len(steps), **{key: formal[key] for key in ("baa", "balanced_accuracy", "f1", "adcde")}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

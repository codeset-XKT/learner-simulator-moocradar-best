from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from learner_simulator.data import clean_sequence  # noqa: E402
from learner_simulator.formal_metrics import (  # noqa: E402
    build_ability_difficulty_partition,
    evaluate_formal_response_metrics,
)
from learner_simulator.dkt import DKT, require_torch as require_dkt_torch, torch as dkt_torch  # noqa: E402
from learner_simulator.dneuralcdm import (  # noqa: E402
    dneuralcdm_model_from_checkpoint,
    require_torch as require_ncdm_torch,
    torch as ncdm_torch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DKT and DNeuralCDM checkpoints on the target part of a fixed cohort."
    )
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--dkt-checkpoint", required=True)
    parser.add_argument("--ncdm-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--common-support",
        action="store_true",
        help=(
            "When a checkpoint cannot encode every target item, evaluate DKT and NCDM "
            "on their shared (uid, step_index) support and record its coverage."
        ),
    )
    parser.add_argument(
        "--full-report",
        default=None,
        help="Optional simulator report JSON whose NCDM mastery-anchor metrics should be copied for comparison.",
    )
    parser.add_argument(
        "--partition-reference",
        default=None,
        help=(
            "Optional fixed_full_irt_reference.json. When supplied, reuse its "
            "precomputed global IRT 2x2 partition for ADCDE rather than deriving "
            "one from a simulator report."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cohort, history_rows, target_rows = _load_cohort(Path(args.cohort))
    y_dkt, p_dkt, dkt_uids, dkt_cids, dkt_steps, skip_dkt = _eval_dkt(history_rows, target_rows, Path(args.dkt_checkpoint), args.threshold)
    y_ncdm, p_ncdm, ncdm_uids, ncdm_cids, ncdm_steps, skip_ncdm = _eval_ncdm(history_rows, target_rows, Path(args.ncdm_checkpoint), args.threshold)

    common_support = None
    if y_dkt != y_ncdm:
        if not args.common_support:
            raise RuntimeError(
                f"DKT and NCDM evaluated different label sequences: {len(y_dkt)} vs {len(y_ncdm)}"
            )
        dkt_by_key = {
            (str(step["uid"]), int(step["step_index"])): (label, prob, uid, cid, step)
            for label, prob, uid, cid, step in zip(y_dkt, p_dkt, dkt_uids, dkt_cids, dkt_steps)
        }
        ncdm_by_key = {
            (str(step["uid"]), int(step["step_index"])): (label, prob, uid, cid, step)
            for label, prob, uid, cid, step in zip(y_ncdm, p_ncdm, ncdm_uids, ncdm_cids, ncdm_steps)
        }
        shared_keys = sorted(dkt_by_key.keys() & ncdm_by_key.keys(), key=lambda key: (int(key[0]), key[1]))
        if not shared_keys:
            raise RuntimeError("DKT and NCDM have no shared evaluable target steps")
        dkt_values = [dkt_by_key[key] for key in shared_keys]
        ncdm_values = [ncdm_by_key[key] for key in shared_keys]
        if [value[0] for value in dkt_values] != [value[0] for value in ncdm_values]:
            raise RuntimeError("DKT and NCDM labels disagree on their shared target support")
        y_dkt, p_dkt, dkt_uids, dkt_cids, dkt_steps = map(list, zip(*dkt_values))
        y_ncdm, p_ncdm, ncdm_uids, ncdm_cids, ncdm_steps = map(list, zip(*ncdm_values))
        common_support = {
            "enabled": True,
            "shared_target_steps": len(shared_keys),
            "total_target_steps": sum(len(steps) for steps in target_rows.values()),
            "coverage": len(shared_keys) / sum(len(steps) for steps in target_rows.values()),
            "dkt_only_target_steps": len(dkt_by_key.keys() - ncdm_by_key.keys()),
            "ncdm_only_target_steps": len(ncdm_by_key.keys() - dkt_by_key.keys()),
        }

    result = {
        "cohort_file": str(Path(args.cohort)),
        "history_steps": cohort.get("history_steps"),
        "target_steps": cohort.get("target_steps"),
        "users": len(cohort.get("uids", [])),
        "threshold": args.threshold,
        "dkt_checkpoint": str(Path(args.dkt_checkpoint)),
        "ncdm_checkpoint": str(Path(args.ncdm_checkpoint)),
        "dkt_skipped_mapped_steps": skip_dkt,
        "ncdm_skipped_mapped_steps": skip_ncdm,
        "common_support": common_support,
        # Keep step-level outputs so later metrics can be computed offline
        # without rerunning either KT model.
        "dkt_all_steps": dkt_steps,
        "ncdm_all_steps": ncdm_steps,
        "dkt_direct_teacher_forcing": _metrics(
            y_dkt, p_dkt, args.threshold, int(cohort.get("target_steps") or 0), dkt_uids, dkt_cids
        ),
        "ncdm_direct_teacher_forcing": _metrics(
            y_ncdm, p_ncdm, args.threshold, int(cohort.get("target_steps") or 0), ncdm_uids, ncdm_cids
        ),
    }
    if args.partition_reference:
        partition_reference = json.loads(Path(args.partition_reference).read_text(encoding="utf-8-sig"))
        partition = partition_reference.get("fixed_full_irt_2x2_partition")
        if not isinstance(partition, dict) or not partition.get("available"):
            raise RuntimeError("--partition-reference has no usable fixed_full_irt_2x2_partition")
        result["formal_metrics_v6"] = {
            "fair_partition": partition,
            "partition_reference": str(Path(args.partition_reference)),
            "dkt": evaluate_formal_response_metrics(dkt_steps, partition),
            "ncdm": evaluate_formal_response_metrics(ncdm_steps, partition),
        }
    elif args.full_report:
        loaded_report = json.loads(Path(args.full_report).read_text(encoding="utf-8-sig"))
        full_report = (loaded_report.get("reports") or {}).get("full", loaded_report)
        full_steps = full_report.get("all_steps") or []
        partition = build_ability_difficulty_partition(full_steps)
        result["formal_metrics_v6"] = {
            "fair_partition": partition,
            "dkt": evaluate_formal_response_metrics(dkt_steps, partition),
            "ncdm": evaluate_formal_response_metrics(ncdm_steps, partition),
        }
        result["ncdm_mastery_anchor_in_prompt"] = _load_ncdm_anchor_metrics(Path(args.full_report))
    else:
        result["formal_metrics_v6"] = {
            "fair_partition": None,
            "dkt": evaluate_formal_response_metrics(dkt_steps),
            "ncdm": evaluate_formal_response_metrics(ncdm_steps),
            "note": "ADCDE requires --full-report so all methods share full-model IRT groups.",
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def _load_cohort(path: Path) -> tuple[dict, dict[str, list[dict]], dict[str, list[dict]]]:
    cohort = json.loads(path.read_text(encoding="utf-8-sig"))
    history_rows = {str(row["uid"]): clean_sequence(row) for row in cohort["history_rows"]}
    target_rows = {str(row["uid"]): clean_sequence(row) for row in cohort["target_rows"]}
    return cohort, history_rows, target_rows


def _eval_dkt(
    history_rows: dict[str, list[dict]],
    target_rows: dict[str, list[dict]],
    checkpoint_path: Path,
    threshold: float,
) -> tuple[list[int], list[float], list[str], list[str], list[dict], int]:
    require_dkt_torch()
    checkpoint = dkt_torch.load(checkpoint_path, map_location="cpu")
    concept_id_map = {str(key): int(value) for key, value in checkpoint["concept_id_map"].items()}
    model = DKT(
        checkpoint["num_concepts"],
        hidden_dim=int(checkpoint.get("hidden_dim", 100)),
        dropout=float(checkpoint.get("dropout", 0.2)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true: list[int] = []
    y_prob: list[float] = []
    group_uids: list[str] = []
    group_cids: list[str] = []
    formal_steps: list[dict] = []
    skipped = 0
    with dkt_torch.no_grad():
        for uid, history in history_rows.items():
            sequence = history + target_rows[uid]
            encoded: list[tuple[int, int]] = []
            raw_to_encoded: list[int | None] = []
            for step in sequence:
                concept_index = concept_id_map.get(str(step["cid"]))
                if concept_index is None:
                    raw_to_encoded.append(None)
                    skipped += 1
                    continue
                raw_to_encoded.append(len(encoded))
                encoded.append((concept_index, int(step["response"])))
            if len(encoded) < 2:
                continue

            inputs = dkt_torch.zeros(
                (1, len(encoded) - 1, checkpoint["num_concepts"] * 2),
                dtype=dkt_torch.float32,
            )
            for index, (concept_index, response) in enumerate(encoded[:-1]):
                inputs[0, index, (2 * concept_index) + (0 if response == 1 else 1)] = 1.0
            outputs = model(inputs)[0]

            for raw_position in range(len(history), len(sequence)):
                encoded_position = raw_to_encoded[raw_position]
                if encoded_position is None or encoded_position < 1:
                    continue
                concept_index, label = encoded[encoded_position]
                y_prob.append(float(outputs[encoded_position - 1, concept_index].item()))
                y_true.append(int(label))
                group_uids.append(str(uid))
                group_cids.append(str(sequence[raw_position]["cid"]))
                raw_step = sequence[raw_position]
                formal_steps.append({
                    "uid": str(uid),
                    "step_index": raw_step.get("position"),
                    "qid": raw_step.get("qid"),
                    "real_response": int(label),
                    "simulated_response": int(y_prob[-1] >= threshold),
                    "predicted_probability": float(y_prob[-1]),
                })
    return y_true, y_prob, group_uids, group_cids, formal_steps, skipped


def _eval_ncdm(
    history_rows: dict[str, list[dict]],
    target_rows: dict[str, list[dict]],
    checkpoint_path: Path,
    threshold: float,
) -> tuple[list[int], list[float], list[str], list[str], list[dict], int]:
    require_ncdm_torch()
    checkpoint = ncdm_torch.load(checkpoint_path, map_location="cpu")
    exercise_id_map = {str(key): int(value) for key, value in checkpoint["exercise_id_map"].items()}
    concept_id_map = {str(key): int(value) for key, value in checkpoint["concept_id_map"].items()}
    model = dneuralcdm_model_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true: list[int] = []
    y_prob: list[float] = []
    group_uids: list[str] = []
    group_cids: list[str] = []
    formal_steps: list[dict] = []
    skipped = 0
    with ncdm_torch.no_grad():
        for uid, history in history_rows.items():
            sequence = history + target_rows[uid]
            encoded: list[tuple[int, int, int]] = []
            raw_to_encoded: list[int | None] = []
            for step in sequence:
                exercise_index = exercise_id_map.get(str(step["qid"]))
                concept_index = concept_id_map.get(str(step["cid"]))
                if exercise_index is None or concept_index is None:
                    raw_to_encoded.append(None)
                    skipped += 1
                    continue
                raw_to_encoded.append(len(encoded))
                encoded.append((exercise_index, concept_index, int(step["response"])))
            if len(encoded) < 2:
                continue

            length = len(encoded)
            inputs = ncdm_torch.zeros((1, length - 1, checkpoint["num_know"] * 2), dtype=ncdm_torch.float32)
            exercises = ncdm_torch.zeros(
                (1, length - 1, checkpoint["num_exercises"]),
                dtype=ncdm_torch.float32,
            )
            masks = ncdm_torch.zeros((1, length - 1, checkpoint["num_know"]), dtype=ncdm_torch.float32)
            for index, (_, concept_index, response) in enumerate(encoded[:-1]):
                next_exercise, next_concept, _ = encoded[index + 1]
                inputs[0, index, (2 * concept_index) + (0 if response == 1 else 1)] = 1.0
                exercises[0, index, next_exercise] = 1.0
                masks[0, index, next_concept] = 1.0
            outputs, _, _ = model(inputs, exercises, masks)
            outputs = outputs.reshape(-1)

            for raw_position in range(len(history), len(sequence)):
                encoded_position = raw_to_encoded[raw_position]
                if encoded_position is None or encoded_position < 1:
                    continue
                _, _, label = encoded[encoded_position]
                y_prob.append(float(outputs[encoded_position - 1].item()))
                y_true.append(int(label))
                group_uids.append(str(uid))
                group_cids.append(str(sequence[raw_position]["cid"]))
                raw_step = sequence[raw_position]
                formal_steps.append({
                    "uid": str(uid),
                    "step_index": raw_step.get("position"),
                    "qid": raw_step.get("qid"),
                    "real_response": int(label),
                    "simulated_response": int(y_prob[-1] >= threshold),
                    "predicted_probability": float(y_prob[-1]),
                })
    return y_true, y_prob, group_uids, group_cids, formal_steps, skipped


def _metrics(
    y_true: list[int],
    y_prob: list[float],
    threshold: float,
    sequence_length: int = 0,
    group_uids: Optional[list[str]] = None,
    group_cids: Optional[list[str]] = None,
) -> dict:
    y_pred = [int(prob >= threshold) for prob in y_prob]
    tn = sum(1 for label, pred in zip(y_true, y_pred) if label == 0 and pred == 0)
    fp = sum(1 for label, pred in zip(y_true, y_pred) if label == 0 and pred == 1)
    fn = sum(1 for label, pred in zip(y_true, y_pred) if label == 1 and pred == 0)
    tp = sum(1 for label, pred in zip(y_true, y_pred) if label == 1 and pred == 1)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    rouge_3 = _macro_binary_rouge_n(y_true, y_pred, sequence_length, n=3)
    result = {
        "count": len(y_true),
        "real_correct_rate": _safe_div(sum(y_true), len(y_true)),
        "predicted_correct_rate_at_threshold": _safe_div(sum(y_pred), len(y_pred)),
        "mean_probability": _safe_div(sum(y_prob), len(y_prob)),
        "acc": _safe_div(tp + tn, len(y_true)),
        "f1": f1,
        "balanced_acc": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
        "sensitivity": recall,
        "specificity": specificity,
        "mcc": (tp * tn - fp * fn) / denominator if denominator else None,
        "rouge_3": rouge_3["f1"],
        "rouge_3_precision": rouge_3["precision"],
        "rouge_3_recall": rouge_3["recall"],
        "rouge_3_user_count": rouge_3["user_count"],
        "auc": _auc(y_true, y_prob),
        "mae": sum(abs(prob - label) for label, prob in zip(y_true, y_prob)) / len(y_true),
        "rmse": math.sqrt(sum((prob - label) ** 2 for label, prob in zip(y_true, y_prob)) / len(y_true)),
        "brier": sum((prob - label) ** 2 for label, prob in zip(y_true, y_prob)) / len(y_true),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
    if group_uids is not None and len(group_uids) == len(y_true):
        learner_mae, learner_counts = _grouped_rate_stats(y_true, y_pred, group_uids)
        result["learner_distribution_error"] = learner_mae
        result["learner_group_count"] = len(learner_counts)
        result["learner_distribution_group_counts"] = learner_counts
    if group_cids is not None and len(group_cids) == len(y_true):
        concept_mae, concept_counts = _grouped_rate_stats(y_true, y_pred, group_cids)
        result["concept_distribution_error"] = concept_mae
        result["concept_group_count"] = len(concept_counts)
        result["concept_distribution_group_counts"] = concept_counts
    return result


def _grouped_rate_stats(
    y_true: list[int], y_pred: list[int], groups: list[str]
) -> tuple[float, dict[str, list[int]]]:
    totals: dict[str, list[int]] = {}
    for real, predicted, group in zip(y_true, y_pred, groups):
        values = totals.setdefault(str(group), [0, 0, 0])
        values[0] += real
        values[1] += predicted
        values[2] += 1
    mae = sum(abs(real / count - predicted / count) for real, predicted, count in totals.values()) / len(totals)
    return mae, totals


def _macro_binary_rouge_n(
    y_true: list[int],
    y_pred: list[int],
    sequence_length: int,
    n: int,
) -> dict:
    if sequence_length < n or len(y_true) != len(y_pred) or len(y_true) % sequence_length:
        return {"precision": None, "recall": None, "f1": None, "user_count": 0}
    scores = []
    for start in range(0, len(y_true), sequence_length):
        reference = y_true[start:start + sequence_length]
        candidate = y_pred[start:start + sequence_length]
        ref = Counter(tuple(reference[i:i+n]) for i in range(sequence_length-n+1))
        cand = Counter(tuple(candidate[i:i+n]) for i in range(sequence_length-n+1))
        overlap = sum((ref & cand).values())
        precision = overlap / sum(cand.values())
        recall = overlap / sum(ref.values())
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append((precision, recall, f1))
    return {
        "precision": sum(x[0] for x in scores) / len(scores),
        "recall": sum(x[1] for x in scores) / len(scores),
        "f1": sum(x[2] for x in scores) / len(scores),
        "user_count": len(scores),
    }


def _auc(y_true: list[int], y_score: list[float]) -> float | None:
    positive = sum(y_true)
    negative = len(y_true) - positive
    if positive == 0 or negative == 0:
        return None
    order = sorted(range(len(y_true)), key=lambda index: y_score[index])
    ranks = [0.0] * len(y_true)
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and y_score[order[end + 1]] == y_score[order[index]]:
            end += 1
        average_rank = (index + 1 + end + 1) / 2.0
        for rank_index in range(index, end + 1):
            ranks[order[rank_index]] = average_rank
        index = end + 1
    positive_rank_sum = sum(rank for rank, label in zip(ranks, y_true) if label == 1)
    return (positive_rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def _safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _load_ncdm_anchor_metrics(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    report = (payload.get("reports") or {}).get("full", payload)
    metrics = report.get("metrics", {})
    return {
        "count": metrics.get("count"),
        "acc": metrics.get("ncdm_acc_at_threshold"),
        "f1": metrics.get("ncdm_f1_at_threshold"),
        "balanced_acc": metrics.get("ncdm_balanced_accuracy"),
        "specificity": metrics.get("ncdm_specificity"),
        "mcc": metrics.get("ncdm_mcc"),
        "auc": metrics.get("ncdm_auc"),
        "mean_probability": metrics.get("ncdm_predicted_correct_rate"),
        "confusion": metrics.get("ncdm_confusion"),
    }


if __name__ == "__main__":
    main()

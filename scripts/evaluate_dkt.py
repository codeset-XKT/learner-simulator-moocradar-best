from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from learner_simulator.data import clean_sequence, iter_sequence_rows  # noqa: E402
from learner_simulator.dkt import DKT, require_torch, torch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained DKT checkpoint on a full sequence test file.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    require_torch()
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    test_path = Path(args.test_file)
    output_path = Path(args.output)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    concept_id_map = {str(k): int(v) for k, v in checkpoint["concept_id_map"].items()}
    model = DKT(
        checkpoint["num_concepts"],
        hidden_dim=int(checkpoint.get("hidden_dim", 100)),
        dropout=float(checkpoint.get("dropout", 0.2)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true: list[int] = []
    y_score: list[float] = []
    y_pred: list[int] = []
    rows = 0
    total_steps = 0
    raw_candidate_predictions = 0
    valid_mapped_steps = 0
    unmapped_steps = 0
    short_valid_sequences = 0
    started_at = time.time()

    with torch.no_grad():
        for row in iter_sequence_rows(test_path):
            rows += 1
            sequence = clean_sequence(row)
            total_steps += len(sequence)
            raw_candidate_predictions += max(0, len(sequence) - 1)
            encoded: list[tuple[int, int]] = []
            for step in sequence:
                concept_index = concept_id_map.get(str(step["cid"]))
                if concept_index is None:
                    unmapped_steps += 1
                    continue
                encoded.append((concept_index, int(step["response"])))
            valid_mapped_steps += len(encoded)
            if len(encoded) < 2:
                short_valid_sequences += 1
                continue

            n_steps = len(encoded)
            history = torch.zeros((1, n_steps - 1, checkpoint["num_concepts"] * 2), dtype=torch.float32)
            targets: list[int] = []
            target_concepts: list[int] = []
            for index in range(n_steps - 1):
                concept_index, response = encoded[index]
                next_concept_index, next_response = encoded[index + 1]
                history[0, index, (2 * concept_index) + (0 if response == 1 else 1)] = 1.0
                target_concepts.append(next_concept_index)
                targets.append(next_response)

            probabilities = model(history)[0]
            for probability_vector, concept_index, label in zip(probabilities, target_concepts, targets):
                score = float(probability_vector[concept_index].item())
                y_score.append(score)
                y_true.append(int(label))
                y_pred.append(1 if score >= args.threshold else 0)

            if args.log_every > 0 and rows % args.log_every == 0:
                print(
                    f"DKT eval rows={rows} "
                    f"predictions={len(y_true)} "
                    f"elapsed={time.time() - started_at:.1f}s",
                    flush=True,
                )

    result = {
        "dataset": str(test_path),
        "checkpoint": str(checkpoint_path),
        "mode": "teacher_forcing_valid_mapped_steps",
        "threshold": args.threshold,
        "rows": rows,
        "total_steps": total_steps,
        "raw_candidate_predictions": raw_candidate_predictions,
        "valid_mapped_steps": valid_mapped_steps,
        "unmapped_steps": unmapped_steps,
        "short_valid_sequences": short_valid_sequences,
        "evaluated_predictions": len(y_true),
        "coverage_vs_raw_candidate_predictions": _safe_div(len(y_true), raw_candidate_predictions),
        "positive_rate": _safe_div(sum(y_true), len(y_true)),
        "mean_predicted_probability": _safe_div(sum(y_score), len(y_score)),
        "metrics": _metrics(y_true, y_pred, y_score),
        "confusion": _confusion_dict(y_true, y_pred),
        "elapsed_seconds": time.time() - started_at,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def _confusion_dict(y_true: list[int], y_pred: list[int]) -> dict[str, int]:
    tn = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 0)
    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def _metrics(y_true: list[int], y_pred: list[int], y_score: list[float]) -> dict[str, float | None]:
    confusion = _confusion_dict(y_true, y_pred)
    tn, fp, fn, tp = confusion["tn"], confusion["fp"], confusion["fn"], confusion["tp"]
    accuracy = _safe_div(tp + tn, len(y_true))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    balanced_accuracy = None
    if recall is not None and specificity is not None:
        balanced_accuracy = (recall + specificity) / 2.0
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denominator) if denominator else None
    rmse = None
    mae = None
    if y_true:
        rmse = math.sqrt(sum((score - label) ** 2 for score, label in zip(y_score, y_true)) / len(y_true))
        mae = sum(abs(score - label) for score, label in zip(y_score, y_true)) / len(y_true)
    return {
        "acc": accuracy,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "sensitivity": recall,
        "specificity": specificity,
        "mcc": mcc,
        "auc": _auc(y_true, y_score),
        "rmse": rmse,
        "mae": mae,
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


if __name__ == "__main__":
    main()

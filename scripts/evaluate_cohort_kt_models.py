from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from learner_simulator.data import clean_sequence  # noqa: E402
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
        "--full-report",
        default=None,
        help="Optional simulator report JSON whose NCDM mastery-anchor metrics should be copied for comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cohort, history_rows, target_rows = _load_cohort(Path(args.cohort))
    y_dkt, p_dkt, skip_dkt = _eval_dkt(history_rows, target_rows, Path(args.dkt_checkpoint))
    y_ncdm, p_ncdm, skip_ncdm = _eval_ncdm(history_rows, target_rows, Path(args.ncdm_checkpoint))

    if y_dkt != y_ncdm:
        raise RuntimeError(
            f"DKT and NCDM evaluated different label sequences: {len(y_dkt)} vs {len(y_ncdm)}"
        )

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
        "dkt_direct_teacher_forcing": _metrics(y_dkt, p_dkt, args.threshold),
        "ncdm_direct_teacher_forcing": _metrics(y_ncdm, p_ncdm, args.threshold),
    }
    if args.full_report:
        result["ncdm_mastery_anchor_in_prompt"] = _load_ncdm_anchor_metrics(Path(args.full_report))

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
) -> tuple[list[int], list[float], int]:
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
    return y_true, y_prob, skipped


def _eval_ncdm(
    history_rows: dict[str, list[dict]],
    target_rows: dict[str, list[dict]],
    checkpoint_path: Path,
) -> tuple[list[int], list[float], int]:
    require_ncdm_torch()
    checkpoint = ncdm_torch.load(checkpoint_path, map_location="cpu")
    exercise_id_map = {str(key): int(value) for key, value in checkpoint["exercise_id_map"].items()}
    concept_id_map = {str(key): int(value) for key, value in checkpoint["concept_id_map"].items()}
    model = dneuralcdm_model_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true: list[int] = []
    y_prob: list[float] = []
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
    return y_true, y_prob, skipped


def _metrics(y_true: list[int], y_prob: list[float], threshold: float) -> dict:
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
    return {
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
        "auc": _auc(y_true, y_prob),
        "mae": sum(abs(prob - label) for label, prob in zip(y_true, y_prob)) / len(y_true),
        "rmse": math.sqrt(sum((prob - label) ** 2 for label, prob in zip(y_true, y_prob)) / len(y_true)),
        "brier": sum((prob - label) ** 2 for label, prob in zip(y_true, y_prob)) / len(y_true),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
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
    metrics = payload["reports"]["full"]["metrics"]
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

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any


def evaluate_steps(steps: list[dict[str, Any]], threshold: float = 0.5) -> dict[str, Any]:
    if not steps:
        return {
            "count": 0,
            "real_correct_rate": None,
            "sampled_correct_rate": None,
            "sample_match_acc": None,
            "sample_f1": None,
            "sample_confusion": None,
            "response_balanced_accuracy": None,
            "response_specificity": None,
            "response_mcc": None,
            "rouge_3": None,
            "rouge_3_precision": None,
            "rouge_3_recall": None,
            "rouge_3_user_count": 0,
            "learner_distribution_error": None,
            "concept_distribution_error": None,
            "mastery_response_monotonicity": None,
            "mastery_confidence_monotonicity": None,
            "threshold": threshold,
        }
    y_true = [int(step["real_response"]) for step in steps]
    y_prob = _optional_probabilities(steps, "p_correct")
    y_pred = [int(prob >= threshold) for prob in y_prob] if y_prob is not None else None
    y_sample = [int(step["simulated_response"]) for step in steps]
    rouge_3 = response_sequence_rouge_n(steps, n=3)

    metrics = {
        "count": len(steps),
        "real_correct_rate": round(_mean(y_true), 6),
        "sampled_correct_rate": round(_mean(y_sample), 6),
        "sample_match_acc": round(accuracy(y_true, y_sample), 6),
        "sample_f1": round(f1_score(y_true, y_sample), 6),
        "sample_confusion": confusion_matrix(y_true, y_sample),
        "response_balanced_accuracy": _round_or_none(
            balanced_accuracy(y_true, y_sample)
        ),
        "response_specificity": _round_or_none(specificity(y_true, y_sample)),
        "response_mcc": _round_or_none(matthews_corrcoef(y_true, y_sample)),
        "rouge_3": _round_or_none(rouge_3["f1"]),
        "rouge_3_precision": _round_or_none(rouge_3["precision"]),
        "rouge_3_recall": _round_or_none(rouge_3["recall"]),
        "rouge_3_user_count": rouge_3["user_count"],
        "learner_distribution_error": _round_or_none(
            grouped_rate_mae(steps, key="uid")
        ),
        "concept_distribution_error": _round_or_none(
            grouped_rate_mae(steps, key="cid")
        ),
        "mastery_response_monotonicity": mastery_response_monotonicity(steps),
        "mastery_confidence_monotonicity": mastery_confidence_monotonicity(steps),
        "threshold": threshold,
    }
    if y_prob is not None and y_pred is not None:
        metrics.update(
            {
                "prob_predicted_correct_rate": round(_mean(y_prob), 6),
                "prob_auc": _round_or_none(binary_auc(y_true, y_prob)),
                "prob_acc_at_threshold": round(accuracy(y_true, y_pred), 6),
                "prob_f1_at_threshold": round(f1_score(y_true, y_pred), 6),
                "prob_confusion": confusion_matrix(y_true, y_pred),
                "prob_mae": round(mae(y_true, y_prob), 6),
                "prob_rmse": round(rmse(y_true, y_prob), 6),
                "prob_nll": round(nll(y_true, y_prob), 6),
                "prob_brier": round(brier_score(y_true, y_prob), 6),
            }
        )

    dkt_true, dkt_prob = _probability_pairs(steps, "dkt_correct_probability")
    if dkt_prob:
        dkt_pred = [int(prob >= threshold) for prob in dkt_prob]
        metrics.update(
            {
                "dkt_probability_count": len(dkt_prob),
                "dkt_probability_coverage": round(len(dkt_prob) / len(steps), 6),
                "dkt_predicted_correct_rate": round(_mean(dkt_prob), 6),
                "dkt_auc": _round_or_none(binary_auc(dkt_true, dkt_prob)),
                "dkt_acc_at_threshold": round(accuracy(dkt_true, dkt_pred), 6),
                "dkt_f1_at_threshold": round(f1_score(dkt_true, dkt_pred), 6),
                "dkt_balanced_accuracy": _round_or_none(
                    balanced_accuracy(dkt_true, dkt_pred)
                ),
                "dkt_specificity": _round_or_none(specificity(dkt_true, dkt_pred)),
                "dkt_mcc": _round_or_none(matthews_corrcoef(dkt_true, dkt_pred)),
                "dkt_confusion": confusion_matrix(dkt_true, dkt_pred),
                "dkt_mae": round(mae(dkt_true, dkt_prob), 6),
                "dkt_rmse": round(rmse(dkt_true, dkt_prob), 6),
                "dkt_nll": round(nll(dkt_true, dkt_prob), 6),
                "dkt_brier": round(brier_score(dkt_true, dkt_prob), 6),
            }
        )

    ncdm_true, ncdm_prob = _probability_pairs(steps, "ncdm_correct_probability")
    if ncdm_prob:
        ncdm_pred = [int(prob >= threshold) for prob in ncdm_prob]
        metrics.update(
            {
                "ncdm_probability_count": len(ncdm_prob),
                "ncdm_probability_coverage": round(len(ncdm_prob) / len(steps), 6),
                "ncdm_predicted_correct_rate": round(_mean(ncdm_prob), 6),
                "ncdm_auc": _round_or_none(binary_auc(ncdm_true, ncdm_prob)),
                "ncdm_acc_at_threshold": round(accuracy(ncdm_true, ncdm_pred), 6),
                "ncdm_f1_at_threshold": round(f1_score(ncdm_true, ncdm_pred), 6),
                "ncdm_balanced_accuracy": _round_or_none(
                    balanced_accuracy(ncdm_true, ncdm_pred)
                ),
                "ncdm_specificity": _round_or_none(specificity(ncdm_true, ncdm_pred)),
                "ncdm_mcc": _round_or_none(matthews_corrcoef(ncdm_true, ncdm_pred)),
                "ncdm_confusion": confusion_matrix(ncdm_true, ncdm_pred),
                "ncdm_mae": round(mae(ncdm_true, ncdm_prob), 6),
                "ncdm_rmse": round(rmse(ncdm_true, ncdm_prob), 6),
                "ncdm_nll": round(nll(ncdm_true, ncdm_prob), 6),
                "ncdm_brier": round(brier_score(ncdm_true, ncdm_prob), 6),
            }
        )

    llm_predictions = extract_llm_predictions(steps)
    if llm_predictions:
        true_for_llm = [item["real_response"] for item in llm_predictions]
        pred_for_llm = [item["simulated_correct"] for item in llm_predictions]
        metrics["llm_response_count"] = len(llm_predictions)
        metrics["llm_predicted_correct_rate"] = round(_mean(pred_for_llm), 6)
        metrics["llm_response_acc"] = round(accuracy(true_for_llm, pred_for_llm), 6)
        metrics["llm_response_f1"] = round(f1_score(true_for_llm, pred_for_llm), 6)
        metrics["llm_confusion"] = confusion_matrix(true_for_llm, pred_for_llm)
        confidence_predictions = [
            item for item in llm_predictions if item["p_llm_correct"] is not None
        ]
        if confidence_predictions:
            confidence_true = [
                item["real_response"] for item in confidence_predictions
            ]
            prob_for_llm = [
                float(item["p_llm_correct"]) for item in confidence_predictions
            ]
            metrics["llm_confidence_count"] = len(confidence_predictions)
            confidence_auc = _round_or_none(binary_auc(confidence_true, prob_for_llm))
            metrics["answer_confidence_prob_correct_rate"] = round(
                _mean(prob_for_llm),
                6,
            )
            metrics["answer_confidence_auc"] = confidence_auc
            metrics["answer_confidence_mae"] = round(mae(confidence_true, prob_for_llm), 6)
            metrics["answer_confidence_rmse"] = round(rmse(confidence_true, prob_for_llm), 6)
            metrics["answer_confidence_nll"] = round(nll(confidence_true, prob_for_llm), 6)
            metrics["answer_confidence_brier"] = round(
                brier_score(confidence_true, prob_for_llm),
                6,
            )
            # Backward-compatible aliases for earlier experiment reports.
            metrics["llm_prob_correct_rate"] = metrics[
                "answer_confidence_prob_correct_rate"
            ]
            metrics["llm_auc"] = confidence_auc
            metrics["llm_mae"] = metrics["answer_confidence_mae"]
            metrics["llm_rmse"] = metrics["answer_confidence_rmse"]
            metrics["llm_nll"] = metrics["answer_confidence_nll"]
            metrics["llm_brier"] = metrics["answer_confidence_brier"]

    four_tier = extract_four_tier_diagnostics(steps)
    if four_tier:
        metrics["four_tier_response_count"] = len(four_tier)
        metrics["four_tier_answer_scored_count"] = sum(
            item["answer_correct"] is not None for item in four_tier
        )
        metrics["four_tier_fully_scored_count"] = sum(
            item["fully_scored"] for item in four_tier
        )
        metrics["four_tier_mean_answer_confidence"] = round(
            _mean([item["answer_confidence"] for item in four_tier]),
            6,
        )
        metrics["four_tier_mean_reasoning_confidence"] = round(
            _mean([item["reasoning_confidence"] for item in four_tier]),
            6,
        )
        metrics["four_tier_diagnosis_counts"] = _counts(
            [item["diagnosis"] for item in four_tier]
        )
        metrics.update(four_tier_consistency_metrics(four_tier))

    task_diagnostics = extract_simulation_task_diagnostics(steps)
    if task_diagnostics:
        metrics.update(simulation_task_metrics(task_diagnostics, threshold=threshold))
    return metrics


def response_sequence_rouge_n(
    steps: list[dict[str, Any]],
    n: int = 3,
) -> dict[str, Any]:
    """Macro ROUGE-N over each learner's ordered binary response sequence.

    This follows Agent4Edu's distribution-consistency use of ROUGE-3: the
    candidate is the simulated correctness sequence and the reference is the
    real correctness sequence. ``rouge_3`` reports the macro F1; precision and
    recall are retained to make the exact aggregation convention auditable.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for position, step in enumerate(steps):
        if step.get("uid") is None:
            continue
        try:
            real = int(step["real_response"])
            simulated = int(step["simulated_response"])
        except (KeyError, TypeError, ValueError):
            continue
        grouped[str(step["uid"])].append(
            {"position": position, "step_index": step.get("step_index"),
             "real": real, "simulated": simulated}
        )
    scores: list[tuple[float, float, float]] = []
    for items in grouped.values():
        items.sort(key=lambda item: (
            item["step_index"] is None,
            item["step_index"] if item["step_index"] is not None else item["position"],
        ))
        if len(items) < n:
            continue
        reference = [item["real"] for item in items]
        candidate = [item["simulated"] for item in items]
        reference_ngrams = Counter(tuple(reference[i:i+n]) for i in range(len(reference)-n+1))
        candidate_ngrams = Counter(tuple(candidate[i:i+n]) for i in range(len(candidate)-n+1))
        overlap = sum((reference_ngrams & candidate_ngrams).values())
        precision = overlap / sum(candidate_ngrams.values())
        recall = overlap / sum(reference_ngrams.values())
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append((precision, recall, f1))
    if not scores:
        return {"precision": None, "recall": None, "f1": None, "user_count": 0}
    return {
        "precision": _mean([score[0] for score in scores]),
        "recall": _mean([score[1] for score in scores]),
        "f1": _mean([score[2] for score in scores]),
        "user_count": len(scores),
    }


def flatten_simulations(simulations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for simulation in simulations:
        steps.extend(simulation.get("steps", []))
    return steps


def _optional_probabilities(
    steps: list[dict[str, Any]],
    key: str,
) -> list[float] | None:
    values: list[float] = []
    for step in steps:
        if step.get(key) is None:
            return None
        try:
            values.append(float(step[key]))
        except (TypeError, ValueError):
            return None
    return values


def _probability_pairs(
    steps: list[dict[str, Any]],
    key: str,
) -> tuple[list[int], list[float]]:
    labels: list[int] = []
    probabilities: list[float] = []
    for step in steps:
        value = step.get(key)
        if value is None:
            continue
        try:
            probability = float(value)
            label = int(step["real_response"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(probability):
            continue
        labels.append(label)
        probabilities.append(min(1.0, max(0.0, probability)))
    return labels, probabilities


def binary_auc(y_true: list[int], y_score: list[float]) -> float | None:
    positives = sum(y_true)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return None

    pairs = sorted(zip(y_score, y_true), key=lambda x: x[0])
    rank_sum = 0.0
    rank = 1
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        avg_rank = (rank + rank + (end - index) - 1) / 2.0
        positives_in_tie = sum(label for _, label in pairs[index:end])
        rank_sum += positives_in_tie * avg_rank
        rank += end - index
        index = end

    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return 0.0
    return sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)


def f1_score(y_true: list[int], y_pred: list[int]) -> float:
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    denominator = (2 * tp) + fp + fn
    if denominator == 0:
        return 0.0
    return (2 * tp) / denominator


def sensitivity(y_true: list[int], y_pred: list[int]) -> float | None:
    matrix = confusion_matrix(y_true, y_pred)
    denominator = matrix["tp"] + matrix["fn"]
    if denominator == 0:
        return None
    return matrix["tp"] / denominator


def specificity(y_true: list[int], y_pred: list[int]) -> float | None:
    matrix = confusion_matrix(y_true, y_pred)
    denominator = matrix["tn"] + matrix["fp"]
    if denominator == 0:
        return None
    return matrix["tn"] / denominator


def balanced_accuracy(y_true: list[int], y_pred: list[int]) -> float | None:
    tpr = sensitivity(y_true, y_pred)
    tnr = specificity(y_true, y_pred)
    if tpr is None or tnr is None:
        return None
    return (tpr + tnr) / 2


def matthews_corrcoef(y_true: list[int], y_pred: list[int]) -> float | None:
    matrix = confusion_matrix(y_true, y_pred)
    tp = matrix["tp"]
    tn = matrix["tn"]
    fp = matrix["fp"]
    fn = matrix["fn"]
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return None
    return ((tp * tn) - (fp * fn)) / denominator


def confusion_matrix(y_true: list[int], y_pred: list[int]) -> dict[str, int]:
    return {
        "tn": sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0),
        "fp": sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1),
        "fn": sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0),
        "tp": sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1),
    }


def mae(y_true: list[int], y_prob: list[float]) -> float:
    if not y_true:
        return 0.0
    return sum(abs(y - p) for y, p in zip(y_true, y_prob)) / len(y_true)


def rmse(y_true: list[int], y_prob: list[float]) -> float:
    if not y_true:
        return 0.0
    return math.sqrt(sum((y - p) ** 2 for y, p in zip(y_true, y_prob)) / len(y_true))


def brier_score(y_true: list[int], y_prob: list[float]) -> float:
    if not y_true:
        return 0.0
    return sum((y - p) ** 2 for y, p in zip(y_true, y_prob)) / len(y_true)


def nll(y_true: list[int], y_prob: list[float], eps: float = 1e-8) -> float:
    if not y_true:
        return 0.0
    total = 0.0
    for y, p in zip(y_true, y_prob):
        p = min(1.0 - eps, max(eps, p))
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total / len(y_true)


def grouped_rate_mae(steps: list[dict[str, Any]], key: str) -> float | None:
    grouped: dict[Any, list[dict[str, Any]]] = {}
    for step in steps:
        group = step.get(key)
        if group is None:
            continue
        grouped.setdefault(group, []).append(step)
    if not grouped:
        return None
    errors: list[float] = []
    for items in grouped.values():
        true_rate = _mean([int(item["real_response"]) for item in items])
        simulated_rate = _mean([int(item["simulated_response"]) for item in items])
        errors.append(abs(true_rate - simulated_rate))
    return _mean(errors)


def mastery_response_monotonicity(steps: list[dict[str, Any]]) -> dict[str, Any] | None:
    bins = _mastery_bins(steps)
    if not bins:
        return None
    rates = {
        name: _mean([int(item["simulated_response"]) for item in items])
        if items
        else None
        for name, items in bins.items()
    }
    return _monotonicity_result(rates)


def mastery_confidence_monotonicity(
    steps: list[dict[str, Any]],
) -> dict[str, Any] | None:
    bins = _mastery_bins(steps)
    if not bins:
        return None
    rates: dict[str, float | None] = {}
    for name, items in bins.items():
        confidences = [_answer_confidence(item) for item in items]
        valid = [value for value in confidences if value is not None]
        rates[name] = _mean(valid) if valid else None
    return _monotonicity_result(rates) if any(value is not None for value in rates.values()) else None


def extract_llm_predictions(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for step in steps:
        parsed = step.get("llm_parsed_action")
        if not isinstance(parsed, dict) or "simulated_correct" not in parsed:
            continue
        try:
            simulated_correct = int(parsed["simulated_correct"])
            confidence_value = parsed.get("confidence")
            confidence = (
                min(1.0, max(0.0, float(confidence_value)))
                if confidence_value is not None
                else None
            )
            p_llm_correct = (
                confidence if simulated_correct == 1 else 1.0 - confidence
                if confidence is not None
                else None
            )
            predictions.append(
                {
                    "real_response": int(step["real_response"]),
                    "simulated_correct": simulated_correct,
                    "confidence": confidence,
                    "p_llm_correct": p_llm_correct,
                }
            )
        except (TypeError, ValueError):
            continue
    return predictions


def extract_four_tier_diagnostics(
    steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for step in steps:
        assessment = step.get("four_tier_assessment")
        parsed = step.get("llm_parsed_action")
        if not isinstance(assessment, dict):
            if isinstance(parsed, dict):
                assessment = parsed.get("four_tier_assessment")
        if not isinstance(assessment, dict):
            continue
        answer_confidence = assessment.get("answer_confidence")
        reasoning_confidence = assessment.get("reasoning_confidence")
        diagnostics.append(
            {
                "answer_confidence": float(answer_confidence)
                if isinstance(answer_confidence, (int, float))
                else 0.5,
                "reasoning_confidence": float(reasoning_confidence)
                if isinstance(reasoning_confidence, (int, float))
                else 0.5,
                "answer_correct": assessment.get("answer_correct"),
                "reasoning_correct": assessment.get("reasoning_correct"),
                "learner_correct": (
                    parsed.get("learner_correct")
                    if isinstance(parsed, dict)
                    else step.get("task4_learner_correct")
                ),
                "real_response": int(step["real_response"]),
                "ncdm_probability": step.get("ncdm_correct_probability"),
                "fully_scored": bool(assessment.get("fully_scored", False)),
                "diagnosis": str(assessment.get("diagnosis", "unknown")),
            }
        )
    return diagnostics


def extract_simulation_task_diagnostics(
    steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for step in steps:
        tasks = step.get("simulation_tasks")
        if not isinstance(tasks, dict):
            continue
        task1 = (
            tasks.get("module1_traceable_learner_evidence_representation")
            or tasks.get("task1_learner_state_profile")
        )
        # Support current module names as well as frozen archive schemas.
        task2 = (
            tasks.get("module2_cognitive_state_item_alignment")
            or tasks.get("task2_contextual_evidence_and_memory")
        )
        task3 = (
            tasks.get("module3_structured_response_generation")
            or tasks.get("task3_structured_learner_process_response")
            or tasks.get("task3_learner_response_generation")
        )
        task4 = (
            tasks.get("module4_auditable_state_evolution")
            or tasks.get("task4_dynamic_state_evolution")
            or step.get("state_evolution")
        )
        diagnostics.append(
            {
                "real_response": int(step["real_response"]),
                "task1_available": (
                    isinstance(task1, dict) and not bool(task1.get("ablated"))
                ),
                "selected_concept": (
                    task2.get("selected_concept") if isinstance(task2, dict) else None
                ),
                "true_concept": (
                    task2.get("true_concept") if isinstance(task2, dict) else None
                ),
                "concept_match": (
                    task2.get("concept_match") if isinstance(task2, dict) else None
                ),
                "task3_answer_correct": (
                    task3.get("learner_correct") if isinstance(task3, dict) else None
                ),
                "task3_answer_confidence": (
                    task3.get("answer_confidence") if isinstance(task3, dict) else None
                ),
                "state_feedback_mode": (
                    task4.get("feedback_mode") if isinstance(task4, dict) else None
                ),
                "mastery_delta": (
                    task4.get("mastery_delta") if isinstance(task4, dict) else None
                ),
            }
        )
    return diagnostics


def simulation_task_metrics(
    items: list[dict[str, Any]],
    threshold: float = 0.5,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "simulation_task_count": len(items),
        "task1_state_inference_count": sum(
            bool(item["task1_available"]) for item in items
        ),
    }

    concept_items = [
        item for item in items
        if item["selected_concept"] is not None and item["concept_match"] is not None
    ]
    if concept_items:
        metrics["task2_concept_perception_count"] = len(concept_items)
        metrics["task2_concept_accuracy"] = round(
            _mean([int(bool(item["concept_match"])) for item in concept_items]),
            6,
        )

    response_items = [
        item for item in items
        if item["task3_answer_correct"] is not None
    ]
    if response_items:
        y_true = [int(item["real_response"]) for item in response_items]
        y_pred = [int(bool(item["task3_answer_correct"])) for item in response_items]
        metrics.update(
            {
                "task3_response_generation_count": len(response_items),
                "task3_response_acc": round(accuracy(y_true, y_pred), 6),
                "task3_response_f1": round(f1_score(y_true, y_pred), 6),
                "task3_response_balanced_accuracy": _round_or_none(
                    balanced_accuracy(y_true, y_pred)
                ),
                "task3_response_specificity": _round_or_none(
                    specificity(y_true, y_pred)
                ),
                "task3_response_mcc": _round_or_none(
                    matthews_corrcoef(y_true, y_pred)
                ),
                "task3_response_confusion": confusion_matrix(y_true, y_pred),
            }
        )

    deltas = [
        float(item["mastery_delta"])
        for item in items
        if item["mastery_delta"] is not None
    ]
    if deltas:
        metrics["task4_state_evolution_count"] = len(deltas)
        metrics["task4_mean_mastery_delta"] = round(_mean(deltas), 6)
        metrics["task4_mean_abs_mastery_delta"] = round(
            _mean([abs(delta) for delta in deltas]),
            6,
        )
        metrics["task4_positive_update_rate"] = round(
            _mean([int(delta > 0) for delta in deltas]),
            6,
        )
        metrics["task4_negative_update_rate"] = round(
            _mean([int(delta < 0) for delta in deltas]),
            6,
        )
        metrics["task4_feedback_mode_counts"] = _counts(
            [
                str(item["state_feedback_mode"])
                for item in items
                if item["state_feedback_mode"] is not None
            ]
        )
    return metrics


def four_tier_consistency_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    answer_scored = [item for item in items if item["answer_correct"] is not None]
    reasoning_scored = [
        item
        for item in items
        if item["answer_correct"] is not None and item["reasoning_correct"] is not None
    ]
    if answer_scored:
        answer_true = [int(bool(item["answer_correct"])) for item in answer_scored]
        answer_conf = [float(item["answer_confidence"]) for item in answer_scored]
        result["four_tier_answer_confidence_ece"] = _round_or_none(
            calibration_ece(answer_true, answer_conf, bins=5)
        )
        result["four_tier_high_confidence_accuracy"] = _round_or_none(
            _conditional_rate(
                answer_scored,
                lambda item: float(item["answer_confidence"]) >= 0.75,
                lambda item: int(bool(item["answer_correct"])),
            )
        )
        result["four_tier_low_confidence_accuracy"] = _round_or_none(
            _conditional_rate(
                answer_scored,
                lambda item: float(item["answer_confidence"]) < 0.5,
                lambda item: int(bool(item["answer_correct"])),
            )
        )
        result["four_tier_answer_reasoning_confidence_gap"] = round(
            _mean(
                [
                    abs(
                        float(item["answer_confidence"])
                        - float(item["reasoning_confidence"])
                    )
                    for item in answer_scored
                ]
            ),
            6,
        )
        decision_answer_items = [
            item for item in answer_scored
            if item.get("learner_correct") is not None
        ]
        result["four_tier_decision_answer_consistency_count"] = len(
            decision_answer_items
        )
        result["four_tier_decision_answer_consistency"] = _round_or_none(
            _mean(
                [
                    int(
                        int(item["learner_correct"])
                        == int(bool(item["answer_correct"]))
                    )
                    for item in decision_answer_items
                ]
            )
            if decision_answer_items
            else None
        )
        result.update(conditional_confidence_gap(answer_scored))
    if reasoning_scored:
        result["four_tier_answer_reasoning_correctness_consistency"] = round(
            _mean(
                [
                    int(bool(item["answer_correct"]) == bool(item["reasoning_correct"]))
                    for item in reasoning_scored
                ]
            ),
            6,
        )
    else:
        result["four_tier_answer_reasoning_correctness_consistency"] = None
    return result


def conditional_confidence_gap(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Test whether confidence adds reliability within comparable NCDM states.

    Items are grouped by low/mid/high NCDM readiness and the LLM's predicted
    response. Within each comparable group, high-confidence decisions should
    agree with the real learner response more often than lower-confidence ones.
    """

    grouped: dict[tuple[str, int], dict[str, list[int]]] = {}
    for item in items:
        probability = item.get("ncdm_probability")
        learner_correct = item.get("learner_correct")
        if probability is None or learner_correct is None:
            continue
        probability = float(probability)
        if probability < 0.4:
            band = "low"
        elif probability < 0.7:
            band = "medium"
        else:
            band = "high"
        confidence_group = (
            "high" if float(item["answer_confidence"]) >= 0.75 else "lower"
        )
        agreement = int(int(learner_correct) == int(item["real_response"]))
        grouped.setdefault((band, int(learner_correct)), {"high": [], "lower": []})[
            confidence_group
        ].append(agreement)

    comparable = []
    details: dict[str, Any] = {}
    for (band, prediction), groups in sorted(grouped.items()):
        key = f"{band}_pred_{prediction}"
        high_rate = _mean(groups["high"]) if groups["high"] else None
        lower_rate = _mean(groups["lower"]) if groups["lower"] else None
        details[key] = {
            "high_confidence_count": len(groups["high"]),
            "lower_confidence_count": len(groups["lower"]),
            "high_confidence_reliability": _round_or_none(high_rate),
            "lower_confidence_reliability": _round_or_none(lower_rate),
            "gap": _round_or_none(
                high_rate - lower_rate
                if high_rate is not None and lower_rate is not None
                else None
            ),
        }
        if high_rate is not None and lower_rate is not None:
            comparable.append(
                {
                    "gap": high_rate - lower_rate,
                    "weight": min(len(groups["high"]), len(groups["lower"])),
                }
            )

    total_weight = sum(item["weight"] for item in comparable)
    weighted_gap = (
        sum(item["gap"] * item["weight"] for item in comparable) / total_weight
        if total_weight
        else None
    )
    return {
        "four_tier_conditional_confidence_gap": _round_or_none(weighted_gap),
        "four_tier_conditional_confidence_comparable_groups": len(comparable),
        "four_tier_conditional_confidence_details": details,
    }


def calibration_ece(
    y_true: list[int],
    y_prob: list[float],
    bins: int = 5,
) -> float | None:
    if not y_true:
        return None
    total = len(y_true)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        bucket = [
            (truth, prob)
            for truth, prob in zip(y_true, y_prob)
            if (prob >= lower and (prob < upper or index == bins - 1))
        ]
        if not bucket:
            continue
        accuracy_value = _mean([truth for truth, _ in bucket])
        confidence_value = _mean([prob for _, prob in bucket])
        error += (len(bucket) / total) * abs(accuracy_value - confidence_value)
    return error


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mastery_bins(steps: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    bins: dict[str, list[dict[str, Any]]] = {
        "low": [],
        "medium": [],
        "high": [],
    }
    seen = False
    for step in steps:
        mastery = _step_mastery(step)
        if mastery is None:
            continue
        seen = True
        if mastery < 0.4:
            bins["low"].append(step)
        elif mastery < 0.7:
            bins["medium"].append(step)
        else:
            bins["high"].append(step)
    return bins if seen else {}


def _step_mastery(step: dict[str, Any]) -> float | None:
    for key in ["ncdm_concept_mastery", "history_state_probability"]:
        value = step.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    components = step.get("history_state_components") or step.get(
        "probability_components"
    )
    if not isinstance(components, dict):
        return None
    try:
        return float(components["mastery"])
    except (KeyError, TypeError, ValueError):
        return None


def _answer_confidence(step: dict[str, Any]) -> float | None:
    assessment = step.get("four_tier_assessment")
    if isinstance(assessment, dict):
        value = assessment.get("answer_confidence")
    else:
        parsed = step.get("llm_parsed_action")
        value = parsed.get("confidence") if isinstance(parsed, dict) else None
    if value is None:
        return None
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return None


def _monotonicity_result(rates: dict[str, float | None]) -> dict[str, Any]:
    order = ["low", "medium", "high"]
    comparisons: list[bool] = []
    for left, right in zip(order, order[1:]):
        left_value = rates.get(left)
        right_value = rates.get(right)
        if left_value is None or right_value is None:
            continue
        comparisons.append(left_value <= right_value)
    for name in order:
        value = rates.get(name)
        rates[name] = round(value, 6) if value is not None else None
    return {
        "score": round(_mean([1.0 if item else 0.0 for item in comparisons]), 6)
        if comparisons
        else None,
        "rates": rates,
        "valid_comparisons": len(comparisons),
    }


def _conditional_rate(
    items: list[dict[str, Any]],
    predicate: Any,
    value: Any,
) -> float | None:
    selected = [item for item in items if predicate(item)]
    if not selected:
        return None
    return _mean([value(item) for item in selected])


def _mean(values: list[int] | list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _mean_or_none(values: list[int] | list[float]) -> float | None:
    if not values:
        return None
    return round(_mean(values), 6)


def _round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 6)


def _counts(values: list[str]) -> dict[str, int]:
    return {value: values.count(value) for value in sorted(set(values))}

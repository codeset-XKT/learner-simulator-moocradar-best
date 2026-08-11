from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from learner_simulator.evaluation_views import (  # noqa: E402
    compact_table_row,
    extract_named_reports,
    layered_metric_view,
)


HIGHER_IS_BETTER = {
    "acc",
    "f1",
    "balanced_acc",
    "specificity",
    "mcc",
    "auc",
    "task2_concept_accuracy",
    "task3_response_acc",
}

LOWER_IS_BETTER = {
    "lde",
    "cde",
    "four_tier_ece",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize learner-simulation result JSON files into paper-oriented "
            "response, distribution, task, and diagnostic views."
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Result JSON files produced by comparison, ablation, or DKT evaluation.",
    )
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help=(
            "Optional display names. Use one name per extracted report. If omitted, "
            "names are inferred from each JSON wrapper."
        ),
    )
    parser.add_argument(
        "--output",
        default="outputs/evaluation/summary.json",
        help="Path for the structured summary JSON.",
    )
    parser.add_argument(
        "--markdown-output",
        default=None,
        help="Optional path for a Markdown table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    for input_path in args.inputs:
        path = _resolve(input_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.extend(extract_named_reports(payload, source=path))

    if args.names:
        if len(args.names) != len(records):
            raise ValueError(
                f"--names expected {len(records)} values, got {len(args.names)}"
            )
        for record, name in zip(records, args.names):
            record["name"] = name

    layered = [layered_metric_view(record) for record in records]
    table_rows = [compact_table_row(record) for record in records]
    summary = {
        "study": "paper_oriented_evaluation_summary",
        "inputs": [str(_resolve(path)) for path in args.inputs],
        "metric_layers": {
            "response_consistency": (
                "ACC, F1, Balanced Acc, Specificity, MCC, AUC when available."
            ),
            "distribution_consistency": (
                "Learner Distribution Error and Concept Distribution Error."
            ),
            "task_consistency": (
                "Profile availability, knowledge concept selection, "
                "response-generation consistency, and "
                "state-evolution diagnostics when available."
            ),
            "diagnostic_consistency": (
                "Mastery-bucket monotonicity, confidence monotonicity, and "
                "Four-tier confidence calibration when available."
            ),
        },
        "records": layered,
        "table": table_rows,
    }
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown_text = build_markdown_table(table_rows)
    if args.markdown_output:
        markdown_output = _resolve(args.markdown_output)
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(markdown_text, encoding="utf-8")

    print(
        json.dumps(
            {
                "ok": True,
                "output": str(output),
                "markdown_output": (
                    str(_resolve(args.markdown_output))
                    if args.markdown_output
                    else None
                ),
                "methods": [row["method"] for row in table_rows],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(markdown_text)


def build_markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "方法",
        "有效",
        "N",
        "ACC ↑",
        "F1 ↑",
        "Balanced Acc ↑",
        "Specificity ↑",
        "MCC ↑",
        "LDE ↓",
        "CDE ↓",
        "AUC ↑",
        "Task2 概念 ↑",
        "Task3 作答 ↑",
        "Four-tier ECE ↓",
        "预测正确率",
        "混淆矩阵 TP/TN/FP/FN",
        "耗时",
    ]
    metric_keys = [
        "acc",
        "f1",
        "balanced_acc",
        "specificity",
        "mcc",
        "lde",
        "cde",
        "auc",
        "task2_concept_accuracy",
        "task3_response_acc",
        "four_tier_ece",
    ]
    best = _best_values(rows, metric_keys)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [
            str(row.get("method") or ""),
            _format_valid(row.get("valid")),
            _format_int(row.get("n")),
            _format_metric(row, "acc", best),
            _format_metric(row, "f1", best),
            _format_metric(row, "balanced_acc", best),
            _format_metric(row, "specificity", best),
            _format_metric(row, "mcc", best, percent=False),
            _format_metric(row, "lde", best, percent=False),
            _format_metric(row, "cde", best, percent=False),
            _format_metric(row, "auc", best, percent=False),
            _format_metric(row, "task2_concept_accuracy", best),
            _format_metric(row, "task3_response_acc", best),
            _format_metric(row, "four_tier_ece", best, percent=False),
            _format_percent(row.get("predicted_correct_rate")),
            _format_confusion(row.get("confusion")),
            str(row.get("runtime") or ""),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _best_values(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in keys:
        values = [
            row.get(key)
            for row in rows
            if isinstance(row.get(key), (int, float)) and row.get(key) is not None
        ]
        if not values:
            continue
        if key in LOWER_IS_BETTER:
            result[key] = min(values)
        else:
            result[key] = max(values)
    return result


def _format_metric(
    row: dict[str, Any],
    key: str,
    best: dict[str, Any],
    percent: bool = True,
) -> str:
    value = row.get(key)
    if value is None:
        return "None"
    text = _format_percent(value) if percent else _format_float(value)
    if key in best and value == best[key]:
        return f"**{text}**"
    return text


def _format_confusion(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    return f"{value.get('tp')}/{value.get('tn')}/{value.get('fp')}/{value.get('fn')}"


def _format_valid(value: Any) -> str:
    if value is True:
        return "True"
    if value is False:
        return "False"
    return "None"


def _format_int(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def _format_percent(value: Any) -> str:
    if value is None:
        return "None"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return str(value)


def _format_float(value: Any) -> str:
    if value is None:
        return "None"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _resolve(path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else ROOT / resolved


if __name__ == "__main__":
    main()

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from learner_simulator.evaluation_views import (  # noqa: E402
    compact_table_row,
    extract_named_reports,
    layered_metric_view,
)
from experiments.evaluation.summarize_results import build_markdown_table  # noqa: E402


def main() -> None:
    wrapped = {
        "study": "ablation",
        "summary": {
            "full": {
                "valid": True,
                "validity_reason": "ok",
                "count": 2,
                "acc": 0.5,
                "f1": 0.667,
                "balanced_acc": 0.5,
                "specificity": 0.0,
                "mcc": None,
                "learner_distribution_error": 0.5,
                "concept_distribution_error": 0.25,
                "task2_concept_accuracy": 1.0,
                "task3_response_acc": 0.5,
                "four_tier_answer_confidence_ece": 0.2,
                "runtime": "1s",
            }
        },
        "reports": {
            "full": {
                "metrics": {
                    "count": 2,
                    "sample_match_acc": 0.5,
                    "sample_f1": 0.667,
                    "sample_confusion": {"tn": 0, "fp": 1, "fn": 0, "tp": 1},
                    "response_balanced_accuracy": 0.5,
                    "response_specificity": 0.0,
                    "learner_distribution_error": 0.5,
                    "concept_distribution_error": 0.25,
                    "task2_concept_accuracy": 1.0,
                    "task3_response_acc": 0.5,
                    "four_tier_answer_confidence_ece": 0.2,
                },
                "runtime": {"total_human": "1s"},
                "validity": {"valid": True, "reason": "ok"},
            }
        },
    }
    dkt = {
        "checkpoint": "dkt.pt",
        "metrics": {
            "count": 2,
            "sample_match_acc": 0.5,
            "sample_f1": 0.667,
            "sample_confusion": {"tn": 0, "fp": 1, "fn": 0, "tp": 1},
            "response_balanced_accuracy": 0.5,
            "response_specificity": 0.0,
            "response_mcc": None,
            "dkt_auc": 0.75,
            "dkt_predicted_correct_rate": 1.0,
        },
    }

    records = extract_named_reports(wrapped, source="wrapped.json")
    records.extend(extract_named_reports(dkt, source="dkt_target_eval.json"))
    rows = [compact_table_row(record) for record in records]
    assert rows[0]["method"] == "full"
    assert rows[0]["acc"] == 0.5
    assert rows[0]["task2_concept_accuracy"] == 1.0
    assert rows[1]["method"] == "dkt"
    assert rows[1]["auc"] == 0.75
    assert rows[1]["predicted_correct_rate"] == 1.0

    layered = layered_metric_view(records[0])
    assert layered["response_consistency"]["acc"] == 0.5
    assert layered["distribution_consistency"]["learner_distribution_error"] == 0.5
    assert layered["task_consistency"]["task3_response_acc"] == 0.5
    assert layered["diagnostic_consistency"]["four_tier_answer_confidence_ece"] == 0.2

    markdown = build_markdown_table(rows)
    assert "Balanced Acc" in markdown
    assert "**50.00%**" in markdown
    print("evaluation_views_test_ok")


if __name__ == "__main__":
    main()

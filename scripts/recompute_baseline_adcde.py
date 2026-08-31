#!/usr/bin/env python3
"""Attach pooled formal metrics using a saved Full-IRT-only fixed partition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from learner_simulator.formal_metrics import evaluate_formal_response_metrics


def load_steps(report_path: Path) -> list[dict[str, Any]]:
    candidates = sorted(
        p for p in report_path.parent.glob("*.json")
        if p.name not in {report_path.name, "final_report_with_full_irt_adcde.json"}
    )
    steps: list[dict[str, Any]] = []
    for path in candidates:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        raw_steps = payload.get("steps")
        if not isinstance(raw_steps, list):
            raw_steps = payload.get("all_steps")
        if isinstance(raw_steps, list):
            steps.extend(raw_steps)
    if not steps:
        raise RuntimeError(f"no step files beside {report_path}")
    return steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    ref_path = Path(args.reference)
    report_path = Path(args.baseline_report)
    reference = json.loads(ref_path.read_text(encoding="utf-8"))
    steps = load_steps(report_path)
    partition = reference["fixed_full_irt_2x2_partition"]
    metrics = evaluate_formal_response_metrics(steps, partition=partition)
    coverage = metrics.get("adcde_partition_coverage")
    audit = {
        "raw_step_pooled": True,
        "steps": len(steps),
        "unique_users": len({str(s.get("uid")) for s in steps}),
        "partition_coverage": coverage,
        "llm_error": sum(1 for s in steps if s.get("status") == "llm_error"),
    }
    if audit["partition_coverage"] != audit["steps"]:
        raise RuntimeError(f"partition coverage mismatch: {audit}")
    output = Path(args.output) if args.output else report_path.parent / "final_report_with_full_irt_adcde.json"
    result = {
        "type": "pooled_baseline_metrics_with_full_irt_only_reference",
        "version": "v1",
        "baseline_report": str(report_path.resolve()),
        "reference_partition": str(ref_path.resolve()),
        "method": "All metrics are recomputed on pooled raw steps; ADCDE uses the fixed Full-IRT-only global 2x2 reference partition.",
        "audit": audit,
        "pooled_metrics": metrics,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "audit": audit, "metrics": metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()

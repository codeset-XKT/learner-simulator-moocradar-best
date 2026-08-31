#!/usr/bin/env python3
"""Replace failed Agent4Edu learner trajectories with a targeted repair run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.common import valid_metric_steps, validity_summary  # noqa: E402
from learner_simulator.evaluation import evaluate_steps  # noqa: E402
from learner_simulator.formal_metrics import evaluate_formal_response_metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--repair", required=True)
    parser.add_argument("--full-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    original = json.loads(Path(args.original).read_text(encoding="utf-8"))
    repair = json.loads(Path(args.repair).read_text(encoding="utf-8"))
    full = json.loads(Path(args.full_report).read_text(encoding="utf-8"))
    repair_steps = repair.get("all_steps") or []
    repair_uids = {str(step.get("uid")) for step in repair_steps}
    if not repair_uids:
        raise ValueError("repair report has no all_steps")
    if any(step.get("llm_error") for step in repair_steps):
        raise ValueError("repair report still has llm_error")
    counts = {uid: sum(str(s.get("uid")) == uid for s in repair_steps) for uid in repair_uids}
    if any(count != 10 for count in counts.values()):
        raise ValueError(f"repair coverage must be exactly 10 per uid: {counts}")

    replacement = {uid: [s for s in repair_steps if str(s.get("uid")) == uid] for uid in repair_uids}
    merged: list[dict] = []
    emitted: set[str] = set()
    for step in original.get("all_steps") or []:
        uid = str(step.get("uid"))
        if uid in replacement:
            if uid not in emitted:
                merged.extend(replacement[uid])
                emitted.add(uid)
        else:
            merged.append(step)
    if emitted != repair_uids:
        raise ValueError(f"repair uids absent from original: {sorted(repair_uids - emitted)}")
    if len(merged) != 500 or len({str(s.get("uid")) for s in merged}) != 50:
        raise ValueError(f"unexpected merged coverage: steps={len(merged)} users={len({str(s.get('uid')) for s in merged})}")
    if any(step.get("llm_error") for step in merged):
        raise ValueError("merged report still has llm_error")

    metric_steps, excluded_uids = valid_metric_steps("agent4edu", merged)
    if len(metric_steps) != 500 or excluded_uids:
        raise ValueError(f"invalid merged metric population: {len(metric_steps)}, {excluded_uids}")
    merged_report = dict(original)
    merged_report["all_steps"] = merged
    threshold = float(original.get("metrics", {}).get("threshold", 0.5))
    merged_report["metrics"] = evaluate_steps(metric_steps, threshold=threshold)
    merged_report["validity"] = validity_summary("agent4edu", merged)
    merged_report["metric_population"] = {
        "included_steps": len(metric_steps),
        "excluded_steps": 0,
        "excluded_uids": [],
        "policy": "targeted_whole-learner-repair_then_full-cohort-pooled-evaluation",
    }
    partition = full["formal_metrics_v6"]["fixed_full_irt_2x2_partition"]
    formal = evaluate_formal_response_metrics(merged, partition=partition)
    if formal.get("adcde_partition_coverage") != 500:
        raise ValueError(f"incomplete formal partition coverage: {formal.get('adcde_partition_coverage')}")
    merged_report["formal_metrics_v6"] = formal
    merged_report["targeted_repair"] = {
        "repaired_users": sorted(repair_uids),
        "repaired_user_count": len(repair_uids),
        "replacement_steps": len(repair_steps),
        "original_report": str(Path(args.original).resolve()),
        "repair_report": str(Path(args.repair).resolve()),
        "full_partition_report": str(Path(args.full_report).resolve()),
        "preserved_original_successful_steps": 500 - len(repair_steps),
    }
    merged_report["archive"] = dict(merged_report.get("archive") or {})
    merged_report["archive"]["repair_merged"] = True
    out = Path(args.output)
    out.write_text(json.dumps(merged_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "validity": merged_report["validity"], "formal": formal}, ensure_ascii=False))


if __name__ == "__main__":
    main()

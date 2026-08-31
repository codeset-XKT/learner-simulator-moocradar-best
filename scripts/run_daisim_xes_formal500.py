#!/usr/bin/env python3
"""Evaluate one leakage-safe DAISim checkpoint on the fixed XES3G5M 500-user set."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from learner_simulator.formal_metrics import (  # noqa: E402
    build_ability_difficulty_partition,
    evaluate_formal_response_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def spec() -> list[tuple[str, str, str | None]]:
    base = "experiments/cohorts"
    formal = "outputs/formal/v6_primaryroute/xes3g5m"
    result = [
        ("batch01_seed20260803", f"{base}/xes3g5m_500x10_batches/xes3g5m_500x10_batch01_50x10_seed20260803.json", f"{formal}/batch01_50x10_cohort_20260803_20260827_profilecache_p40/final_report.json"),
        ("batch02_seed20260804", f"{base}/xes3g5m_500x10_batches_batch02/xes3g5m_500x10_batch01_50x10_seed20260804.json", f"{formal}/batch02_50x10_cohort_20260804_20260827_profilecache_p40/final_report.json"),
    ]
    for index in range(1, 7):
        result.append((
            f"remaining300_batch{index:02d}",
            f"{base}/xes3g5m_formal_v5_remaining300_seed20260824/xes3g5m_500x10_batch{index:02d}_50x10_seed20260824.json",
            f"{formal}/glm53_flash_cohort_remaining300_batch{index:02d}_50x10_seed20260824_p40/final_report.json",
        ))
    for index in range(1, 3):
        result.append((
            f"v3_batch{index:02d}_seed20260825",
            f"{base}/xes3g5m_500x10_batches_batch03_05_v3/xes3g5m_500x10_batch{index:02d}_50x10_seed20260825.json",
            None,
        ))
    return result


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    all_steps, comparable_steps, full_steps = [], [], []
    for name, cohort, full_report in spec():
        target = output / f"{name}.json"
        command = [
            sys.executable, "scripts/evaluate_daisim.py", "--cohort", cohort,
            "--checkpoint", args.checkpoint, "--output", str(target),
            "--mode", "teacher-forcing",
        ]
        if full_report:
            command += ["--full-report", full_report]
        subprocess.run(command, cwd=ROOT, check=True)
        record = json.loads(target.read_text(encoding="utf-8"))
        steps = record["steps"]
        if len(steps) != 500:
            raise AssertionError(f"{name}: expected 500 predictions, got {len(steps)}")
        records.append({
            "name": name,
            "cohort": cohort,
            "full_report": full_report,
            "step_count": len(steps),
            "metrics": record["formal_metrics_v6"],
        })
        all_steps.extend(steps)
        if full_report:
            comparable_steps.extend(steps)
            full = json.loads((ROOT / full_report).read_text(encoding="utf-8"))
            full_steps.extend(full["reports"]["full"]["all_steps"])
    if len(all_steps) != 5000 or len({str(step["uid"]) for step in all_steps}) != 500:
        raise AssertionError("fixed 500-user audit failed")
    partition = build_ability_difficulty_partition(full_steps)
    report = {
        "method": "DAISim_official_code_derived_reproduction",
        "evaluation_mode": "teacher-forcing",
        "audits": {
            "batches": len(records), "users": len({str(step["uid"]) for step in all_steps}),
            "steps": len(all_steps), "same_unified_checkpoint": str(args.checkpoint),
            "all500_raw_step_pooled": True,
        },
        "per_batch": records,
        "all500_baa_balanced_accuracy_f1": evaluate_formal_response_metrics(all_steps),
        "comparable_existing400_with_fixed_full_global_partition": evaluate_formal_response_metrics(comparable_steps, partition),
        "adcde_note": "The added v3 cohorts lack matching v6 Full reports, so a fixed-Full ADCDE is only defined for the 400 existing formal users.",
    }
    (output / "final_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["audits"], ensure_ascii=False))


if __name__ == "__main__":
    main()

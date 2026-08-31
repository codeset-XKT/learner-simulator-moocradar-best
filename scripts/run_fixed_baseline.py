#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from learner_simulator.formal_metrics import evaluate_formal_response_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one non-LLM baseline on fixed cohorts and pool raw steps.")
    parser.add_argument("--method", choices=["daisim", "kes"], required=True)
    parser.add_argument("--cohort-file", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feedback-mode", choices=["teacher-forcing", "rollout"], default="teacher-forcing")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_steps: list[dict] = []
    records: list[dict] = []
    for index, raw_cohort in enumerate(args.cohort_file, start=1):
        cohort = Path(raw_cohort)
        target = output / f"batch{index:02d}.json"
        command = [sys.executable]
        if args.method == "daisim":
            command += [
                "scripts/evaluate_daisim.py", "--cohort", str(cohort),
                "--checkpoint", args.checkpoint, "--output", str(target),
                "--mode", args.feedback_mode,
            ]
        else:
            command += [
                "scripts/evaluate_kes.py", "--cohort", str(cohort),
                "--dkt-checkpoint", args.checkpoint, "--output", str(target),
                "--feedback-mode", args.feedback_mode,
                "--threshold", str(args.threshold), "--seed", str(args.seed + index - 1),
            ]
        subprocess.run(command, cwd=ROOT, check=True)
        result = json.loads(target.read_text(encoding="utf-8"))
        steps = result["steps"]
        expected = len(json.loads(cohort.read_text(encoding="utf-8-sig"))["uids"]) * 10
        if len(steps) != expected:
            raise AssertionError(f"{cohort}: expected {expected} steps, got {len(steps)}")
        all_steps.extend(steps)
        records.append({
            "cohort": str(cohort),
            "count": len(steps),
            "metrics": result["formal_metrics_v6"],
        })
    users = {str(step["uid"]) for step in all_steps}
    report = {
        "method": args.method,
        "checkpoint": args.checkpoint,
        "feedback_mode": args.feedback_mode,
        "threshold": args.threshold,
        "audits": {
            "cohorts": len(records), "users": len(users), "steps": len(all_steps),
            "raw_step_pooled": True,
        },
        "per_batch": records,
        "pooled_metrics": evaluate_formal_response_metrics(all_steps),
        "adcde_note": "ADCDE is unavailable unless a matching fixed Full IRT partition is supplied for every cohort.",
    }
    (output / "final_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["audits"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

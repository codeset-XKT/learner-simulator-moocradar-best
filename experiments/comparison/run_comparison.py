from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common import (
    add_shared_arguments,
    load_fixed_cohort,
    metric_view,
    run_experiment,
    save_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full-model comparison against Agent4Edu and random simulation."
    )
    add_shared_arguments(parser)
    parser.add_argument(
        "--baselines",
        default="full,agent4edu,random",
        help="Comma-separated subset of: full,multi-role,agent4edu,random",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = [item.strip() for item in args.baselines.split(",") if item.strip()]
    unknown = set(selected) - {"full", "multi-role", "agent4edu", "random"}
    if unknown:
        raise ValueError(f"Unknown baselines: {sorted(unknown)}")

    questions, history_rows, target_rows = load_fixed_cohort(args)
    reports = {
        name: run_experiment(
            name,
            args,
            questions,
            history_rows,
            target_rows,
        )
        for name in selected
    }
    combined = {
        "study": "comparison",
        "baselines": selected,
        "same_fixed_cohort": True,
        "summary": {name: metric_view(report) for name, report in reports.items()},
        "reports": reports,
    }
    output = args.output or "outputs/comparison/comparison.json"
    path = save_report(combined, output)
    print(json.dumps({"ok": True, "output": str(path), **combined["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

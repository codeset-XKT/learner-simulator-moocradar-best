from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common import add_shared_arguments, load_fixed_cohort, metric_view, run_experiment, save_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the isolated Agent4Edu reproduction baseline.")
    add_shared_arguments(parser)
    parser.add_argument("--disable-reflection", action="store_true")
    args = parser.parse_args()
    questions, history, targets = load_fixed_cohort(args)
    report = run_experiment(
        "agent4edu",
        args,
        questions,
        history,
        targets,
        modules={"reflection": not args.disable_reflection},
    )
    path = save_report(report, args.output or "outputs/comparison/agent4edu.json")
    print(json.dumps({"ok": True, "output": str(path), **metric_view(report)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from learner_simulator.llm import load_json  # noqa: E402
from learner_simulator.mikt import export_mikt_proficiency  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export MIKT post-history concept proficiency for a fixed simulation cohort."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cohort-file", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cohort_path = Path(args.cohort_file)
    if not cohort_path.is_absolute():
        cohort_path = ROOT / cohort_path
    cohort = load_json(cohort_path)
    output = export_mikt_proficiency(
        list(cohort["history_rows"]),
        args.checkpoint,
        args.output,
        reference_rows=list(cohort.get("target_rows") or []),
    )
    summary = {
        "ok": True,
        "checkpoint": str(Path(args.checkpoint)),
        "cohort_file": str(cohort_path),
        "history_users": len(cohort["history_rows"]),
        "output": str(output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

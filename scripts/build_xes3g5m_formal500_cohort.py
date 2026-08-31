#!/usr/bin/env python3
"""Build and validate the exact XES3G5M 500-learner formal cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--cohorts", nargs="+", required=True)
    args = parser.parse_args()

    history_rows: list[dict] = []
    target_rows: list[dict] = []
    component_cohorts: list[dict] = []
    seen: set[str] = set()
    for raw_path in args.cohorts:
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        history = list(payload["history_rows"])
        target = list(payload["target_rows"])
        history_uids = {str(row["uid"]) for row in history}
        target_uids = {str(row["uid"]) for row in target}
        if history_uids != target_uids:
            raise ValueError(f"history/target UID mismatch: {path}")
        if len(history_uids) != 50:
            raise ValueError(f"expected 50 users per component: {path}")
        if seen & history_uids:
            raise ValueError(f"overlapping users in component: {path}")
        seen |= history_uids
        history_rows.extend(history)
        target_rows.extend(target)
        component_cohorts.append({
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "users": sorted(history_uids),
        })

    if len(seen) != 500 or len(history_rows) != 500 or len(target_rows) != 500:
        raise ValueError("formal cohort must contain exactly 500 histories and 500 targets")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "protocol": "fixed_history_90_10",
        "plan_manifest": "xes3g5m_formal_v6_exact500",
        "history_steps": 90,
        "target_steps": 10,
        "uids": sorted(seen),
        "history_rows": history_rows,
        "target_rows": target_rows,
        "component_cohorts": component_cohorts,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "users": len(seen), "components": len(component_cohorts)}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a fixed 2x2 ability-difficulty partition from cohort histories only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from learner_simulator.formal_metrics import build_ability_difficulty_partition
from learner_simulator.irt import IRTModel, clean_sequence
from learner_simulator.irt_evidence import build_irt_ability_item_evidence


def load_rows(paths: list[Path]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        history_rows = payload.get("history_rows", payload.get("rows", []))
        target_rows = payload.get("target_rows", payload.get("targets", []))
        target_by_uid = {str(row["uid"]): row for row in target_rows}
        for history in history_rows:
            uid = str(history["uid"])
            if uid in seen:
                raise ValueError(f"duplicate cohort uid: {uid}")
            if uid not in target_by_uid:
                raise ValueError(f"missing target row for {uid} in {path}")
            seen.add(uid)
            rows.append((history, target_by_uid[uid]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-file", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    cohort_paths = [Path(p).resolve() for p in args.cohort_file]
    rows = load_rows(cohort_paths)
    if not rows:
        raise ValueError("no cohort rows")

    # The fit consumes only each learner's observed history.  Target responses
    # are never passed to IRTModel.fit.
    irt = IRTModel()
    history_rows = [history for history, _target in rows]
    irt.fit(history_rows)
    reference_steps: list[dict[str, Any]] = []
    for _history, target in rows:
        uid = str(target["uid"])
        for step_index, target_step in enumerate(clean_sequence(target)):
            qid = int(target_step["qid"])
            response = int(target_step["response"])
            evidence = build_irt_ability_item_evidence(irt, uid, qid)
            # simulated_response merely satisfies the existing partition helper's
            # binary-step schema; it is not a simulator output and is never used
            # when fitting theta/beta or assigning a 2x2 group.
            reference_steps.append({
                "uid": uid,
                "step_index": step_index,
                "qid": qid,
                "real_response": int(response),
                "simulated_response": int(response),
                "irt_ability_difficulty_evidence": evidence,
            })
    partition = build_ability_difficulty_partition(reference_steps)
    audit = {
        "cohort_users": len(rows),
        "reference_steps": len(reference_steps),
        "partition_entries": len(partition.get("groups", {})),
        "unique_uids": len({str(s["uid"]) for s in reference_steps}),
    }
    if audit["partition_entries"] != audit["reference_steps"]:
        raise RuntimeError(f"incomplete partition: {audit}")
    result = {
        "type": "full_irt_only_fixed_partition",
        "version": "v1",
        "dataset": args.dataset,
        "cohort_files": [str(p) for p in cohort_paths],
        "method": {
            "fit_scope": "all-selected-cohort-observed-history-only",
            "fit_uses_target_labels": False,
            "irt_model": "Rasch-1PL",
            "partition": "fixed-global-2x2-median-theta-by-median-item-beta",
            "note": "Reference target responses are schema carriers only; they do not enter IRT fitting or partition assignment.",
        },
        "audit": audit,
        "fixed_full_irt_2x2_partition": partition,
        "reference_steps": reference_steps,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "audit": audit}, ensure_ascii=False))


if __name__ == "__main__":
    main()

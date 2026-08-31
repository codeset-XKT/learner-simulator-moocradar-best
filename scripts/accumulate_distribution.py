from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

VARIANTS = [
    "full",
    "no-evidence-representation",
    "no-state-item-alignment",
    "no-structured-response-process",
    "no-dynamic-state-evolution",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fair-report", required=True)
    parser.add_argument("--state", required=True)
    args = parser.parse_args()
    fair = json.loads(Path(args.fair_report).read_text())
    keep = set(map(str, fair["common_valid_uids"]))
    combined = json.loads(Path(fair["source_combined"]).read_text())
    state_path = Path(args.state)
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    for variant in VARIANTS:
        target = state.setdefault(variant, {"uid": {}, "cid": {}})
        for step in combined["reports"][variant]["all_steps"]:
            if (
                str(step.get("uid")) not in keep
                or step.get("llm_error")
                or not step.get("prediction_valid")
            ):
                continue
            real = int(step["real_response"])
            simulated = int(step["simulated_response"])
            for key in ("uid", "cid"):
                group = str(step[key])
                values = target[key].setdefault(group, [0, 0, 0])
                values[0] += real
                values[1] += simulated
                values[2] += 1
    state_path.write_text(json.dumps(state))
    print(Path(args.fair_report).parents[1].name, len(keep), flush=True)


if __name__ == "__main__":
    main()

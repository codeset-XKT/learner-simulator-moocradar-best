"""Exact pooled LDE/CDE from raw GLM steps and baseline group sufficient statistics."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


ROOT = Path("/ai/KT/LeanerSim")
BATCHES = ("01", "02", "03", "09", "10")
RUNS = {
    batch: ROOT / "outputs/formal/v6_primaryroute/moocradar" / f"glm53_flash_batch{batch}_50x10_20260828_p40"
    for batch in BATCHES
}
OUT = ROOT / "outputs/formal/v6_primaryroute/moocradar/glm53_flash_pooled_legacy_distribution_batch01_02_03_09_10_250x10_20260828.json"


def add_step(groups: dict[str, list[int]], key: object, real: object, simulated: object) -> None:
    values = groups[str(key)]
    values[0] += int(real)
    values[1] += int(simulated)
    values[2] += 1


def merge_stats(target: dict[str, list[int]], source: dict[str, list[int]]) -> None:
    for group, (real, simulated, count) in source.items():
        values = target[str(group)]
        values[0] += int(real)
        values[1] += int(simulated)
        values[2] += int(count)


def mae(groups: dict[str, list[int]]) -> float:
    return sum(abs(real / count - simulated / count) for real, simulated, count in groups.values()) / len(groups)


def result(learner: dict[str, list[int]], concept: dict[str, list[int]]) -> dict:
    lde, cde = mae(learner), mae(concept)
    return {
        "lde": round(lde, 6),
        "cde": round(cde, 6),
        "lde_cde_mean": round((lde + cde) / 2, 6),
        "learner_group_count": len(learner),
        "concept_group_count": len(concept),
    }


def main() -> None:
    first = json.loads((RUNS[BATCHES[0]] / "final_report.json").read_text(encoding="utf-8-sig"))
    variants = tuple(first["reports"])
    del first
    learner = {name: defaultdict(lambda: [0, 0, 0]) for name in variants}
    concept = {name: defaultdict(lambda: [0, 0, 0]) for name in variants}
    baseline_learner = {name: defaultdict(lambda: [0, 0, 0]) for name in ("dkt", "ncdm")}
    baseline_concept = {name: defaultdict(lambda: [0, 0, 0]) for name in ("dkt", "ncdm")}

    for batch in BATCHES:
        report = json.loads((RUNS[batch] / "final_report.json").read_text(encoding="utf-8-sig"))
        for name in variants:
            steps = report["reports"][name]["all_steps"]
            if len(steps) != 500 or any(step.get("llm_error") for step in steps):
                raise RuntimeError(f"unclean report: batch{batch}/{name}")
            for step in steps:
                add_step(learner[name], step["uid"], step["real_response"], step["simulated_response"])
                add_step(concept[name], step["cid"], step["real_response"], step["simulated_response"])
        for name, record_key in (("dkt", "dkt_direct_teacher_forcing"), ("ncdm", "ncdm_direct_teacher_forcing")):
            record = report["kt_baselines"][record_key]
            merge_stats(baseline_learner[name], record["learner_distribution_group_counts"])
            merge_stats(baseline_concept[name], record["concept_distribution_group_counts"])
        del report
        print(f"pooled distribution groups: batch{batch}", flush=True)

    output = {
        "aggregation": "raw_step_pooling_no_batch_average",
        "batches": list(BATCHES),
        "pooled_step_count": 2500,
        "methods": {name: result(learner[name], concept[name]) for name in variants},
        "baselines": {name: result(baseline_learner[name], baseline_concept[name]) for name in ("dkt", "ncdm")},
    }
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

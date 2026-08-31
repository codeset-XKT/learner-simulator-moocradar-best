"""Export small pooled KT prediction records for a later sklearn-only C2ST pass."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path("/ai/KT/LeanerSim")
BATCHES = ("01", "02", "03", "09", "10")
COHORTS = {
    batch: ROOT / "experiments/cohorts/moocradar_500x10_batches" / f"moocradar_500x10_batch{batch}_50x10_seed20260803.json"
    for batch in BATCHES
}
OUT = ROOT / "outputs/formal/v6_primaryroute/moocradar/glm53_flash_pooled_c2st_kt_records_batch01_02_03_09_10_250x10_20260828.json"


def evaluator_module():
    path = ROOT / "scripts/evaluate_cohort_kt_models.py"
    spec = importlib.util.spec_from_file_location("cohort_eval", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def history_features(history: list[dict]) -> list[float | int]:
    responses = [int(step["response"]) for step in history]
    return [len(responses), sum(responses) / len(responses), responses[-1]]


def main() -> None:
    evaluator = evaluator_module()
    output = {"history_features": {}, "dkt": [], "ncdm": []}
    for batch in BATCHES:
        _, histories, targets = evaluator._load_cohort(COHORTS[batch])
        concepts = {}
        for uid, history in histories.items():
            output["history_features"][str(uid)] = history_features(history)
        for uid, sequence in targets.items():
            for step in sequence:
                concepts[(str(uid), int(step["position"]), str(step["qid"]))] = str(step["cid"])
        dkt = evaluator._eval_dkt(histories, targets, ROOT / "outputs/dkt/moocradar_500plan_leakage_safe_e50_h100/best_model.pt", 0.5)
        ncdm = evaluator._eval_ncdm(histories, targets, ROOT / "outputs/dneuralcdm/moocradar_500plan_leakage_safe_v2_posdisc_masked_e30_d32_h64/best_model.pt", 0.5)
        for name, steps in (("dkt", dkt[4]), ("ncdm", ncdm[4])):
            for step in steps:
                output[name].append({
                    "uid": str(step["uid"]), "step_index": int(step["step_index"]), "qid": str(step["qid"]),
                    "cid": concepts[(str(step["uid"]), int(step["step_index"]), str(step["qid"]))],
                    "real_response": int(step["real_response"]), "simulated_response": int(step["simulated_response"]),
                })
        print(f"exported KT records: batch{batch}", flush=True)
    OUT.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    print(str(OUT), flush=True)


if __name__ == "__main__":
    main()

"""Pool GLM-5.3-Flash MOOCradar batches at the step level, never by batch mean."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from learner_simulator.formal_metrics import (
    build_ability_difficulty_partition,
    evaluate_formal_response_metrics,
)


ROOT = Path("/ai/KT/LeanerSim")
BATCHES = ("01", "02", "03", "09", "10")
RUNS = {
    batch: ROOT / "outputs/formal/v6_primaryroute/moocradar" / f"glm53_flash_batch{batch}_50x10_20260828_p40"
    for batch in BATCHES
}
COHORTS = {
    batch: ROOT / "experiments/cohorts/moocradar_500x10_batches" / f"moocradar_500x10_batch{batch}_50x10_seed20260803.json"
    for batch in BATCHES
}
OUT = ROOT / "outputs/formal/v6_primaryroute/moocradar/glm53_flash_pooled_batch01_02_03_09_10_250x10_20260828.json"


def load_evaluator():
    path = ROOT / "scripts/evaluate_cohort_kt_models.py"
    spec = importlib.util.spec_from_file_location("cohort_eval", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric_step(step: dict) -> dict:
    """Discard prompts/traces before pooling: only formal metrics fields are needed."""
    return {
        "uid": step.get("uid"),
        "step_index": step.get("step_index"),
        "qid": step.get("qid"),
        "real_response": step.get("real_response"),
        "simulated_response": step.get("simulated_response"),
        "llm_error": step.get("llm_error"),
        "irt_ability_difficulty_evidence": step.get("irt_ability_difficulty_evidence"),
    }


def main() -> None:
    first = json.loads((RUNS[BATCHES[0]] / "final_report.json").read_text(encoding="utf-8-sig"))
    variants = tuple(first["reports"])
    del first
    pooled = {variant: [] for variant in variants}
    for batch in BATCHES:
        report = json.loads((RUNS[batch] / "final_report.json").read_text(encoding="utf-8-sig"))
        for variant in variants:
            steps = report["reports"][variant]["all_steps"]
            if len(steps) != 500 or any(step.get("llm_error") for step in steps):
                raise RuntimeError(f"{batch}/{variant} is not a clean 500-step report")
            pooled[variant].extend(metric_step(step) for step in steps)
        del report
        print(f"loaded formal steps: batch{batch}", flush=True)

    full_steps = pooled["full"]
    partition = build_ability_difficulty_partition(full_steps)
    methods = {variant: evaluate_formal_response_metrics(steps, partition) for variant, steps in pooled.items()}

    evaluator = load_evaluator()
    dkt_steps, ncdm_steps = [], []
    for batch in BATCHES:
        _, history, target = evaluator._load_cohort(COHORTS[batch])
        dkt = evaluator._eval_dkt(
            history, target,
            ROOT / "outputs/dkt/moocradar_500plan_leakage_safe_e50_h100/best_model.pt", 0.5,
        )
        ncdm = evaluator._eval_ncdm(
            history, target,
            ROOT / "outputs/dneuralcdm/moocradar_500plan_leakage_safe_v2_posdisc_masked_e30_d32_h64/best_model.pt", 0.5,
        )
        dkt_steps.extend(dkt[4])
        ncdm_steps.extend(ncdm[4])
        print(f"re-evaluated KT baselines: batch{batch}", flush=True)

    output = {
        "aggregation": "raw_step_pooling_no_batch_average",
        "batches": list(BATCHES),
        "cohort_count": len(BATCHES),
        "unique_user_count": len({str(step["uid"]) for step in full_steps}),
        "pooled_step_count": len(full_steps),
        "audit": {
            name: {
                "steps": len(steps),
                "users": len({str(step["uid"]) for step in steps}),
                "llm_errors": sum(bool(step.get("llm_error")) for step in steps),
            }
            for name, steps in pooled.items()
        },
        "fixed_full_global_irt_2x2_partition": partition,
        "methods": methods,
        "baselines": {
            "dkt": evaluate_formal_response_metrics(dkt_steps, partition),
            "ncdm": evaluate_formal_response_metrics(ncdm_steps, partition),
        },
    }
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "methods": methods, "baselines": output["baselines"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

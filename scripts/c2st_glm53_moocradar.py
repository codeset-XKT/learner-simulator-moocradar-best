"""Group-CV logistic-regression C2ST for pooled MOOCradar real vs simulated responses."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


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
OUT = ROOT / "outputs/formal/v6_primaryroute/moocradar/glm53_flash_pooled_c2st_batch01_02_03_09_10_250x10_20260828.json"
SEED = 20260828


def evaluator_module():
    path = ROOT / "scripts/evaluate_cohort_kt_models.py"
    spec = importlib.util.spec_from_file_location("cohort_eval", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def history_features(history: list[dict]) -> tuple[int, float, int]:
    responses = [int(step["response"]) for step in history]
    return len(responses), float(sum(responses) / len(responses)), responses[-1]


def row(uid: str, step: dict, response: int, features: dict[str, tuple[int, float, int]]) -> dict:
    history_len, history_rate, last_response = features[str(uid)]
    return {
        "uid": str(uid),
        "history_len": history_len,
        "history_rate": history_rate,
        "history_last_response": last_response,
        "target_position": int(step["step_index"] if "step_index" in step else step["position"]),
        "cid": str(step["cid"]),
        "qid": str(step["qid"]),
        "candidate_response": int(response),
    }


def sourced_row(uid: str, step: dict, response: int, features: dict[str, tuple[int, float, int]], source: int) -> dict:
    value = row(uid, step, response, features)
    value["source"] = source
    return value


def c2st(rows: list[dict]) -> dict:
    # Import sklearn only after the PyTorch KT evaluations have completed.
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    # Each real/simulated pair shares every context feature; source is the label.
    feature_names = [
        "history_len", "history_rate", "history_last_response", "target_position",
        "cid", "qid", "candidate_response",
    ]
    numeric = ["history_len", "history_rate", "history_last_response", "target_position", "candidate_response"]
    categorical = ["cid", "qid"]
    X = [[item[name] for name in feature_names] for item in rows]
    y = np.array([item["source"] for item in rows], dtype=int)
    groups = np.array([item["uid"] for item in rows])
    indexes = {name: feature_names.index(name) for name in feature_names}
    preprocess = ColumnTransformer([
        ("numeric", StandardScaler(), [indexes[name] for name in numeric]),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), [indexes[name] for name in categorical]),
    ])
    splitter = GroupKFold(n_splits=5)
    scores = np.zeros(len(rows), dtype=float)
    fold_auc = []
    for train, test in splitter.split(X, y, groups):
        model = Pipeline([
            ("features", preprocess),
            ("classifier", LogisticRegression(max_iter=2000, C=1.0, solver="liblinear", random_state=SEED)),
        ])
        model.fit([X[i] for i in train], y[train])
        scores[test] = model.predict_proba([X[i] for i in test])[:, 1]
        fold_auc.append(float(roc_auc_score(y[test], scores[test])))
    raw_auc = float(roc_auc_score(y, scores))
    auc = max(raw_auc, 1.0 - raw_auc)
    # Learner-cluster bootstrap on out-of-fold scores gives an uncertainty interval.
    rng = np.random.default_rng(SEED)
    uid_to_indices: dict[str, list[int]] = {}
    for index, uid in enumerate(groups):
        uid_to_indices.setdefault(str(uid), []).append(index)
    units = list(uid_to_indices.values())
    boot = []
    for _ in range(1000):
        sampled = rng.integers(0, len(units), size=len(units))
        selected = [index for unit in sampled for index in units[unit]]
        value = float(roc_auc_score(y[selected], scores[selected]))
        boot.append(max(value, 1.0 - value))
    return {
        "c2st": "grouped_5fold_logistic_regression",
        "interpretation": "AUC closer to 0.5 means real and simulated conditional response distributions are harder to distinguish",
        "features": {
            "history": ["history_len", "history_rate", "history_last_response"],
            "target": ["target_position", "cid", "qid"],
            "candidate": ["candidate_response"],
            "excluded": "all model-internal states, evidence selections, prompts, and explanation text",
        },
        "rows_per_source": len(rows) // 2,
        "learner_count": len(units),
        "raw_auc": round(raw_auc, 6),
        "discriminative_auc": round(auc, 6),
        "fold_auc": [round(value, 6) for value in fold_auc],
        "learner_cluster_bootstrap_95ci": [round(float(np.quantile(boot, 0.025)), 6), round(float(np.quantile(boot, 0.975)), 6)],
    }


def main() -> None:
    evaluator = evaluator_module()
    methods: dict[str, list[dict]] = {}
    features: dict[str, tuple[int, float, int]] = {}
    target_concepts: dict[tuple[str, int, str], str] = {}
    baseline_steps = {"dkt": [], "ncdm": []}
    for batch in BATCHES:
        report = json.loads((RUNS[batch] / "final_report.json").read_text(encoding="utf-8-sig"))
        if not methods:
            methods = {name: [] for name in report["reports"]}
        cohort, histories, targets = evaluator._load_cohort(COHORTS[batch])
        for uid, history in histories.items():
            features[str(uid)] = history_features(history)
        for uid, sequence in targets.items():
            for step in sequence:
                target_concepts[(str(uid), int(step["position"]), str(step["qid"]))] = str(step["cid"])
        for name, method_rows in methods.items():
            steps = report["reports"][name]["all_steps"]
            if len(steps) != 500 or any(step.get("llm_error") for step in steps):
                raise RuntimeError(f"unclean report: batch{batch}/{name}")
            for step in steps:
                method_rows.append(sourced_row(str(step["uid"]), step, int(step["real_response"]), features, 0))
                method_rows.append(sourced_row(str(step["uid"]), step, int(step["simulated_response"]), features, 1))
        dkt = evaluator._eval_dkt(histories, targets, ROOT / "outputs/dkt/moocradar_500plan_leakage_safe_e50_h100/best_model.pt", 0.5)
        ncdm = evaluator._eval_ncdm(histories, targets, ROOT / "outputs/dneuralcdm/moocradar_500plan_leakage_safe_v2_posdisc_masked_e30_d32_h64/best_model.pt", 0.5)
        for name, steps in (("dkt", dkt[4]), ("ncdm", ncdm[4])):
            for step in steps:
                decorated = dict(step)
                key = (str(step["uid"]), int(step["step_index"]), str(step["qid"]))
                decorated["cid"] = target_concepts[key]
                baseline_steps[name].append(sourced_row(str(step["uid"]), decorated, int(step["real_response"]), features, 0))
                baseline_steps[name].append(sourced_row(str(step["uid"]), decorated, int(step["simulated_response"]), features, 1))
        del report
        print(f"prepared batch{batch}", flush=True)
    output = {
        "aggregation": "raw_step_pooling_no_batch_average",
        "batches": list(BATCHES),
        "response_steps_per_method": 2500,
        "methods": {name: c2st(rows) for name, rows in methods.items()},
        "baselines": {name: c2st(rows) for name, rows in baseline_steps.items()},
    }
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

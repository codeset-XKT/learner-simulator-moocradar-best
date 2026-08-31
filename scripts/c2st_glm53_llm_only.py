"""Sklearn-only pooled C2ST for GLM methods; no PyTorch or LLM calls."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path("/ai/KT/LeanerSim")
BATCHES = ("01", "02", "03", "09", "10")
RUNS = {batch: ROOT / "outputs/formal/v6_primaryroute/moocradar" / f"glm53_flash_batch{batch}_50x10_20260828_p40" for batch in BATCHES}
COHORTS = {batch: ROOT / "experiments/cohorts/moocradar_500x10_batches" / f"moocradar_500x10_batch{batch}_50x10_seed20260803.json" for batch in BATCHES}
OUT = ROOT / "outputs/formal/v6_primaryroute/moocradar/glm53_flash_pooled_c2st_llm_batch01_02_03_09_10_250x10_20260828.json"
KT_RECORDS = ROOT / "outputs/formal/v6_primaryroute/moocradar/glm53_flash_pooled_c2st_kt_records_batch01_02_03_09_10_250x10_20260828.json"
SEED = 20260828


def cohort_features(path: Path) -> dict[str, tuple[int, float, int]]:
    cohort = json.loads(path.read_text(encoding="utf-8-sig"))
    values = {}
    for row in cohort["history_rows"]:
        responses = [int(value) for value in str(row["responses"]).split(",") if value != ""]
        values[str(row["uid"])] = (len(responses), sum(responses) / len(responses), responses[-1])
    return values


def add_pair(rows: list[dict], step: dict, features: dict[str, tuple[int, float, int]]) -> None:
    uid = str(step["uid"])
    length, rate, last = features[uid]
    base = {
        "uid": uid, "history_len": length, "history_rate": rate,
        "history_last_response": last, "target_position": int(step["step_index"]),
        "cid": str(step["cid"]), "qid": str(step["qid"]),
    }
    rows.append(dict(base, candidate_response=int(step["real_response"]), source=0))
    rows.append(dict(base, candidate_response=int(step["simulated_response"]), source=1))


def evaluate(rows: list[dict]) -> dict:
    names = ["history_len", "history_rate", "history_last_response", "target_position", "cid", "qid", "candidate_response"]
    numerical = [0, 1, 2, 3, 6]
    categorical = [4, 5]
    X = [[row[name] for name in names] for row in rows]
    y = np.array([row["source"] for row in rows])
    groups = np.array([row["uid"] for row in rows])
    scores = np.zeros(len(rows))
    fold_auc = []
    for train, test in GroupKFold(n_splits=5).split(X, y, groups):
        model = Pipeline([
            ("features", ColumnTransformer([
                ("numeric", StandardScaler(), numerical),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ])),
            ("classifier", LogisticRegression(max_iter=2000, C=1.0, solver="liblinear", random_state=SEED)),
        ])
        model.fit([X[index] for index in train], y[train])
        scores[test] = model.predict_proba([X[index] for index in test])[:, 1]
        fold_auc.append(float(roc_auc_score(y[test], scores[test])))
    raw_auc = float(roc_auc_score(y, scores))
    auc = max(raw_auc, 1 - raw_auc)
    rng = np.random.default_rng(SEED)
    mapping = {}
    for index, uid in enumerate(groups):
        mapping.setdefault(str(uid), []).append(index)
    units = list(mapping.values())
    bootstrap = []
    for _ in range(1000):
        selected = [index for unit in rng.integers(0, len(units), len(units)) for index in units[unit]]
        value = float(roc_auc_score(y[selected], scores[selected]))
        bootstrap.append(max(value, 1 - value))
    return {
        "discriminative_auc": round(auc, 6), "raw_auc": round(raw_auc, 6),
        "fold_auc": [round(value, 6) for value in fold_auc],
        "learner_cluster_bootstrap_95ci": [round(float(np.quantile(bootstrap, .025)), 6), round(float(np.quantile(bootstrap, .975)), 6)],
    }


def main() -> None:
    rows = None
    for batch in BATCHES:
        report = json.loads((RUNS[batch] / "final_report.json").read_text(encoding="utf-8-sig"))
        features = cohort_features(COHORTS[batch])
        if rows is None:
            rows = {name: [] for name in report["reports"]}
        for name, target in rows.items():
            steps = report["reports"][name]["all_steps"]
            if len(steps) != 500 or any(step.get("llm_error") for step in steps):
                raise RuntimeError(f"unclean report: batch{batch}/{name}")
            for step in steps:
                add_pair(target, step, features)
        print(f"loaded batch{batch}", flush=True)
    kt_records = json.loads(KT_RECORDS.read_text(encoding="utf-8-sig"))
    baselines = {"dkt": [], "ncdm": []}
    for name, target in baselines.items():
        for step in kt_records[name]:
            add_pair(target, step, kt_records["history_features"])
    output = {
        "c2st": "grouped_5fold_logistic_regression",
        "aggregation": "raw_step_pooling_no_batch_average",
        "interpretation": "AUC closer to 0.5 is better; it means real and simulated conditional response samples are harder to distinguish.",
        "features": ["history length", "historical correct rate", "last historical response", "target position", "concept ID", "item ID", "candidate response"],
        "excluded": "model-internal states, evidence selections, prompts, explanations, and confidence",
        "batches": list(BATCHES), "learners": 250, "response_steps_per_method": 2500,
        "methods": {name: evaluate(value) for name, value in rows.items()},
        "baselines": {name: evaluate(value) for name, value in baselines.items()},
    }
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

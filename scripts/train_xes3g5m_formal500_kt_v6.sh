#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/ai/KT/LeanerSim
cd "$ROOT"

COHORT=experiments/cohorts/xes3g5m_formal_v6_exact500/xes3g5m_formal500_50x10_fixed.json
DKT_OUT=outputs/dkt/xes3g5m_formal500_v6_exactcohort_e10_h128
NCDM_OUT=outputs/dneuralcdm/xes3g5m_formal500_v6_exactcohort_s20000_e30_d32_h64

mkdir -p "$DKT_OUT" "$NCDM_OUT"

if [[ ! -s "$DKT_OUT/summary.json" ]]; then
  PYTHONPATH=src python scripts/train_dkt.py \
    --dataset-root data/XES3G5M --source-rows 0 --output-dir "$DKT_OUT" \
    --exclude-cohort-file "$COHORT" --epochs 10 --batch-size 32 --hidden-dim 128 \
    --dropout 0.2 --seed 42 --max-train-steps 0 --log-every 100 \
    > "$DKT_OUT/train.log" 2>&1
fi

if [[ ! -s "$NCDM_OUT/summary.json" ]]; then
  PYTHONPATH=src python scripts/train_dneuralcdm.py \
    --dataset-root data/XES3G5M --source-rows 20000 --cohort-file "$COHORT" \
    --output-dir "$NCDM_OUT" --epochs 30 --batch-size 32 --embedding-dim 32 \
    --hidden-dim 64 --seed 42 --max-train-steps 0 --log-every 100 \
    > "$NCDM_OUT/train.log" 2>&1
fi

python - "$DKT_OUT/summary.json" "$NCDM_OUT/summary.json" <<'PY'
import json, sys
for path in sys.argv[1:]:
    data = json.load(open(path, encoding="utf-8"))
    assert data.get("excluded_cohort_users") == 500, (path, data.get("excluded_cohort_users"))
    assert data.get("cohort_history_used_for_training") is False, path
    assert data.get("leakage_safe") is True, path
    print(json.dumps({"path": path, "checkpoint": data.get("checkpoint"), "excluded_cohort_users": 500, "leakage_safe": True}))
PY

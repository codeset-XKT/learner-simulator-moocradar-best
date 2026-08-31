#!/usr/bin/env bash
set -euo pipefail

ROOT=/ai/KT/LeanerSim
DATASET="$ROOT/data/DBE-KT22"
COHORT_DIR="$ROOT/experiments/cohorts/dbe_kt22_formal500_seed20260830"
OUT_ROOT="$ROOT/outputs/baselines/dbe_kt22/formal500_seed20260830_teacher_forcing"
NEURAL_OUT="$ROOT/outputs/baselines/neural_kt/dbe_kt22/formal500_seed20260830_tf_es30"
IKT_OUT="$ROOT/outputs/baselines/ikt/dbe_kt22/formal500_teacher_forcing"
MIKT_OUT="$ROOT/outputs/baselines/mikt/dbe_kt22/formal500_teacher_forcing"
REFERENCE="$OUT_ROOT/fixed_full_irt_reference.json"

cd "$ROOT"
shopt -s nullglob
COHORTS=("$COHORT_DIR"/dbe_kt22_500x10_batch??_50x10_seed20260830.json)
[ "${#COHORTS[@]}" = 10 ] || { echo "expected 10 cohort files, found ${#COHORTS[@]}" >&2; exit 1; }
for path in "$REFERENCE" "$NEURAL_OUT/deepirt/best_model.pt" "$IKT_OUT/best_model.json" "$MIKT_OUT/best_model.pt"; do
  [ ! -e "$path" ] || { echo "refusing to overwrite existing artifact: $path" >&2; exit 1; }
done
mkdir -p "$OUT_ROOT" "$NEURAL_OUT/deepirt" "$IKT_OUT" "$MIKT_OUT"
{
  echo "dataset=$DATASET"
  echo "cohort_dir=$COHORT_DIR"
  printf 'cohorts='; printf '%s ' "${COHORTS[@]}"; echo
  echo "protocol=teacher_forcing"
  echo "users=500"
  echo "steps=5000"
} > "$OUT_ROOT/run_manifest.txt"

PYTHONPATH=src python3 scripts/build_full_irt_reference.py \
  --cohort-file "${COHORTS[@]}" --dataset DBE-KT22 --output "$REFERENCE"

PYTHONPATH=src python3 scripts/train_neural_kt_references.py \
  --method deepirt --dataset-root "$DATASET" --output-dir "$NEURAL_OUT/deepirt" \
  --exclude-cohort-file "${COHORTS[@]}" --source-rows 0 \
  --epochs 30 --early-stopping-patience 5 --early-stopping-min-delta 0.0005 \
  --batch-size 64 --lr 0.001 --embed-dim 64 --memory-size 20 --dropout 0.2 --max-seq-len 200 --seed 42 \
  > "$NEURAL_OUT/deepirt/train.log" 2>&1
PYTHONPATH=src python3 scripts/evaluate_neural_kt_rollout.py \
  --checkpoint "$NEURAL_OUT/deepirt/best_model.pt" --cohort-file "${COHORTS[@]}" \
  --full-irt-reference "$REFERENCE" --feedback-mode teacher_forcing \
  --output "$NEURAL_OUT/deepirt/teacher_forcing_500.json" \
  > "$NEURAL_OUT/deepirt/evaluate_teacher_forcing.log" 2>&1

PYTHONPATH=src python3 scripts/train_ikt.py \
  --dataset-root "$DATASET" --output-dir "$IKT_OUT" --exclude-cohort-file "${COHORTS[@]}" \
  --source-rows 0 --seed 42 > "$IKT_OUT/train.log" 2>&1
PYTHONPATH=src python3 scripts/evaluate_ikt.py \
  --checkpoint "$IKT_OUT/best_model.json" --cohort-file "${COHORTS[@]}" \
  --full-irt-reference "$REFERENCE" --output "$IKT_OUT/teacher_forcing_500.json" \
  > "$IKT_OUT/evaluate_teacher_forcing.log" 2>&1

PYTHONPATH=src python3 scripts/train_mikt.py \
  --dataset-root "$DATASET" --output-dir "$MIKT_OUT" --exclude-cohort-file "${COHORTS[@]}" \
  --source-rows 0 --epochs 30 --early-stopping-patience 5 --early-stopping-min-delta 0.0005 \
  --batch-size 16 --lr 0.001 --hidden-dim 100 --dropout 0.2 --seed 42 \
  > "$MIKT_OUT/train.log" 2>&1
PYTHONPATH=src python3 scripts/evaluate_mikt.py \
  --checkpoint "$MIKT_OUT/best_model.pt" --cohort-file "${COHORTS[@]}" \
  --full-irt-reference "$REFERENCE" --output "$MIKT_OUT/teacher_forcing_500.json" \
  > "$MIKT_OUT/evaluate_teacher_forcing.log" 2>&1

PYTHONPATH=src python3 - "$OUT_ROOT" "$REFERENCE" "$NEURAL_OUT/deepirt/teacher_forcing_500.json" "$IKT_OUT/teacher_forcing_500.json" "$MIKT_OUT/teacher_forcing_500.json" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
reference = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8-sig'))
reports = {}
for method, path in zip(('deepirt', 'ikt', 'mikt'), map(Path, sys.argv[3:])):
    report = json.loads(path.read_text(encoding='utf-8-sig'))
    assert report['audit']['users'] == 500 and report['audit']['steps'] == 5000, (method, report['audit'])
    assert report['audit']['feedback_mode'] == 'teacher_forcing', (method, report['audit'])
    meta = report['training_metadata']
    assert meta.get('leakage_safe') is True and meta.get('excluded_cohort_users') == 500, (method, meta)
    assert meta.get('cohort_history_used_for_training') is False, (method, meta)
    reports[method] = {
        'report': str(path),
        'metrics': {key: report['formal_metrics_v6'][key] for key in ('baa', 'balanced_accuracy', 'f1', 'adcde')},
        'audit': report['audit'],
    }
summary = {
    'dataset': 'DBE-KT22',
    'protocol': 'fixed 90-history / 10-target; teacher_forcing; raw-step pooled',
    'cohort_users': 500,
    'cohort_steps': 5000,
    'partition_reference': str(Path(sys.argv[2])),
    'partition_audit': reference['audit'],
    'reports': reports,
}
(out / 'final_report.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

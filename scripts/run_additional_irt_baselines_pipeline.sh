#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/ai/KT/LeanerSim
cd "$ROOT"

MOOC_COHORT=experiments/cohorts/moocradar_500x10_batches/moocradar_500x10_all_500users_seed20260803.json
MOOC_REF=outputs/baselines/full_irt_reference/moocradar/formal500_seed20260803/reference.json
XES_REF=outputs/baselines/full_irt_reference/xes3g5m/formal500_mixed_existing400_plus_v3b01b02/reference.json
XES_COHORTS=(
  experiments/cohorts/xes3g5m_500x10_batches/xes3g5m_500x10_batch01_50x10_seed20260803.json
  experiments/cohorts/xes3g5m_500x10_batches_batch02/xes3g5m_500x10_batch01_50x10_seed20260804.json
  experiments/cohorts/xes3g5m_formal_v5_remaining300_seed20260824/xes3g5m_500x10_batch01_50x10_seed20260824.json
  experiments/cohorts/xes3g5m_formal_v5_remaining300_seed20260824/xes3g5m_500x10_batch02_50x10_seed20260824.json
  experiments/cohorts/xes3g5m_formal_v5_remaining300_seed20260824/xes3g5m_500x10_batch03_50x10_seed20260824.json
  experiments/cohorts/xes3g5m_formal_v5_remaining300_seed20260824/xes3g5m_500x10_batch04_50x10_seed20260824.json
  experiments/cohorts/xes3g5m_formal_v5_remaining300_seed20260824/xes3g5m_500x10_batch05_50x10_seed20260824.json
  experiments/cohorts/xes3g5m_formal_v5_remaining300_seed20260824/xes3g5m_500x10_batch06_50x10_seed20260824.json
  experiments/cohorts/xes3g5m_500x10_batches_batch03_05_v3/xes3g5m_500x10_batch01_50x10_seed20260825.json
  experiments/cohorts/xes3g5m_500x10_batches_batch03_05_v3/xes3g5m_500x10_batch02_50x10_seed20260825.json
)

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

assert_formal_result() {
  local result="$1" model="$2"
  python - "$result" "$model" <<'PY'
import json, sys
p, model = sys.argv[1:]
with open(p, encoding='utf-8') as f:
    d = json.load(f)
a = d.get('audit', {})
users = a.get('n_users', a.get('users'))
steps = a.get('n_steps', a.get('steps'))
if users != 500 or steps != 5000:
    raise SystemExit(f'{model}: invalid cohort audit {a}')
if a.get('feedback_mode') != 'teacher_forcing':
    raise SystemExit(f'{model}: expected teacher_forcing, got {a.get("feedback_mode")}')
if len(d.get('all_steps', [])) != 5000:
    raise SystemExit(f'{model}: missing step records')
print(f'{model}: formal audit passed')
PY
}

wait_for_deepirt_xes() {
  log 'Waiting for the active Deep-IRT XES3G5M training process.'
  while pgrep -f 'train_neural_kt_references.py.*deepirt.*data/XES3G5M' >/dev/null; do
    sleep 60
  done
  test -s outputs/baselines/neural_kt/xes3g5m/formal500_mixed_existing400_plus_v3b01b02_tf_es30/deepirt/best_model.pt
}

run_deepirt_evaluations() {
  local mooc_out=outputs/baselines/neural_kt/moocradar/formal500_seed20260803_tf_es30/deepirt/teacher_forcing_500.json
  local xes_out=outputs/baselines/neural_kt/xes3g5m/formal500_mixed_existing400_plus_v3b01b02_tf_es30/deepirt/teacher_forcing_500.json
  if [[ ! -s "$mooc_out" ]]; then
    log 'Evaluating Deep-IRT on MOOCradar.'
    python scripts/evaluate_neural_kt_rollout.py --feedback-mode teacher_forcing \
      --checkpoint outputs/baselines/neural_kt/moocradar/formal500_seed20260803_tf_es30/deepirt/best_model.pt \
      --cohort-file "$MOOC_COHORT" --full-irt-reference "$MOOC_REF" --output "$mooc_out"
  fi
  assert_formal_result "$mooc_out" deepirt_moocradar
  if [[ ! -s "$xes_out" ]]; then
    log 'Evaluating Deep-IRT on XES3G5M.'
    python scripts/evaluate_neural_kt_rollout.py --feedback-mode teacher_forcing \
      --checkpoint outputs/baselines/neural_kt/xes3g5m/formal500_mixed_existing400_plus_v3b01b02_tf_es30/deepirt/best_model.pt \
      --cohort-file "${XES_COHORTS[@]}" --full-irt-reference "$XES_REF" --output "$xes_out"
  fi
  assert_formal_result "$xes_out" deepirt_xes3g5m
}

train_and_evaluate_mikt() {
  local dataset="$1" out="$2" ref="$3"; shift 3
  local cohorts=("$@")
  local checkpoint="$out/best_model.pt" result="$out/teacher_forcing_500.json"
  # A checkpoint may have been written during an interrupted epoch.  Only a
  # completed training summary makes it reusable as a formal baseline.
  if [[ ! -s "$checkpoint" || ! -s "$out/summary.json" ]]; then
    log "Training MIKT on $dataset."
    python scripts/train_mikt.py --dataset-root "$dataset" --output-dir "$out" \
      --exclude-cohort-file "${cohorts[@]}" --epochs 30 \
      --early-stopping-patience 5 --early-stopping-min-delta 0.0005 \
      --batch-size 32 --hidden-dim 100 --max-train-steps 200 --log-every 100
  fi
  if [[ ! -s "$result" ]]; then
    log "Evaluating MIKT on $dataset."
    python scripts/evaluate_mikt.py --checkpoint "$checkpoint" --cohort-file "${cohorts[@]}" \
      --full-irt-reference "$ref" --output "$result"
  fi
  assert_formal_result "$result" "mikt_$dataset"
}

wait_for_deepirt_xes
run_deepirt_evaluations
train_and_evaluate_mikt data/moocradar outputs/baselines/mikt/moocradar/formal500_teacher_forcing "$MOOC_REF" "$MOOC_COHORT"
train_and_evaluate_mikt data/XES3G5M outputs/baselines/mikt/xes3g5m/formal500_teacher_forcing "$XES_REF" "${XES_COHORTS[@]}"
log 'All Deep-IRT, IKT and MIKT formal baseline results are complete and audited.'

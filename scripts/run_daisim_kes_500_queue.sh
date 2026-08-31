#!/usr/bin/env bash
set -euo pipefail

cd /ai/KT/LeanerSim

mooc_daisim_out="outputs/baselines/daisim/moocradar/formal500_seed20260803"
while [ ! -f "$mooc_daisim_out/checkpoint/summary.json" ]; do
  sleep 30
done

mooc_cohorts=(
  experiments/cohorts/moocradar_500x10_batches/moocradar_500x10_batch01_50x10_seed20260803.json
  experiments/cohorts/moocradar_500x10_batches/moocradar_500x10_batch02_50x10_seed20260803.json
  experiments/cohorts/moocradar_500x10_batches/moocradar_500x10_batch03_50x10_seed20260803.json
  experiments/cohorts/moocradar_500x10_batches/moocradar_500x10_batch04_50x10_seed20260803.json
  experiments/cohorts/moocradar_500x10_batches/moocradar_500x10_batch05_50x10_seed20260803.json
  experiments/cohorts/moocradar_500x10_batches/moocradar_500x10_batch06_50x10_seed20260803.json
  experiments/cohorts/moocradar_500x10_batches/moocradar_500x10_batch07_50x10_seed20260803.json
  experiments/cohorts/moocradar_500x10_batches/moocradar_500x10_batch08_50x10_seed20260803.json
  experiments/cohorts/moocradar_500x10_batches/moocradar_500x10_batch09_50x10_seed20260803.json
  experiments/cohorts/moocradar_500x10_batches/moocradar_500x10_batch10_50x10_seed20260803.json
)

PYTHONPATH=src python3 scripts/run_fixed_baseline.py \
  --method daisim --cohort-file "${mooc_cohorts[@]}" \
  --checkpoint "$mooc_daisim_out/checkpoint/best_model.pt" \
  --output-dir "$mooc_daisim_out/evaluations" \
  --feedback-mode teacher-forcing

PYTHONPATH=src python3 scripts/run_fixed_baseline.py \
  --method kes --cohort-file "${mooc_cohorts[@]}" \
  --checkpoint outputs/dkt/moocradar_500plan_leakage_safe_e50_h100/best_model.pt \
  --output-dir outputs/baselines/kes/moocradar/formal500_seed20260803_rollout \
  --feedback-mode rollout --seed 42

xes_cohorts=(
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

xes_dkt_out="outputs/dkt/xes3g5m_daisim_kes_formal500_leakage_safe_e10_h128"
PYTHONPATH=src python3 scripts/train_dkt.py \
  --dataset-root data/XES3G5M --exclude-cohort-file "${xes_cohorts[@]}" \
  --output-dir "$xes_dkt_out" --epochs 10 --batch-size 64 \
  --hidden-dim 128 --dropout 0.2 --seed 42 --log-every 20

PYTHONPATH=src python3 scripts/run_fixed_baseline.py \
  --method kes --cohort-file "${xes_cohorts[@]}" \
  --checkpoint "$xes_dkt_out/best_model.pt" \
  --output-dir outputs/baselines/kes/xes3g5m/formal500_mixed_existing400_plus_v3b01b02_rollout \
  --feedback-mode rollout --seed 42

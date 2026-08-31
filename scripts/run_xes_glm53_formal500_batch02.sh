#!/usr/bin/env bash
set -euo pipefail

ROOT=/ai/KT/LeanerSim
RUN_DIR="$ROOT/outputs/formal/v6_primaryroute/xes3g5m/glm53_flash_cohort_initial_batch02_50x10_seed20260804_formal500kt_p40"
COHORT="$ROOT/experiments/cohorts/xes3g5m_500x10_batches_batch02/xes3g5m_500x10_batch01_50x10_seed20260804.json"
MAIN_CONFIG="$ROOT/configs/llm.glm-5.3-flash_v6_single_call_thinking.json"
REPAIR_CONFIG="$ROOT/configs/llm.glm-5.3-flash_v6_repair_lowthinking.json"
DKT="$ROOT/outputs/dkt/xes3g5m_formal500_v6_exactcohort_e10_h128/best_model.pt"
NCDM="$ROOT/outputs/dneuralcdm/xes3g5m_formal500_v6_exactcohort_s20000_e30_d32_h64/best_model.pt"
VARIANTS=full,no-evidence-representation,no-state-item-alignment,no-structured-response-process,no-dynamic-state-evolution

cd "$ROOT"
test ! -e "$RUN_DIR"
test -f "$COHORT"
test -f "$DKT"
test -f "$NCDM"
mkdir -p "$RUN_DIR"
{
  echo "cohort=$COHORT"
  echo "dkt=$DKT"
  echo "ncdm=$NCDM"
  echo "main_config=$MAIN_CONFIG"
  echo "repair_config=$REPAIR_CONFIG"
  echo "variants=$VARIANTS"
  echo "feedback_mode=teacher-forcing"
  echo "parallel_variants=5"
  echo "parallel_learners=8"
} > "$RUN_DIR/run_manifest.txt"

PYTHONPATH=src python3 experiments/ablation/run_ablation.py \
  --dataset-root data/XES3G5M \
  --cohort-file "$COHORT" \
  --variants "$VARIANTS" \
  --parallel-variants 5 \
  --variant-start-stagger-seconds 1.0 \
  --simulator multi-role \
  --feedback-mode teacher-forcing \
  --parallel-learners 8 \
  --llm-config "$MAIN_CONFIG" \
  --dneuralcdm-checkpoint "$NCDM" \
  --checkpoint-dir "$RUN_DIR/checkpoints" \
  --output "$RUN_DIR/combined.json" \
  --progress

FINAL="$RUN_DIR/combined.json"
for ATTEMPT in 1 2 3 4 5; do
  ERRORS=$(PYTHONPATH=src python3 - "$FINAL" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding='utf-8-sig'))
print(sum(1 for variant in report['reports'].values() for step in variant.get('all_steps', []) if step.get('llm_error')))
PY
)
  [ "$ERRORS" = "0" ] && break
  NEXT="$RUN_DIR/combined_repaired_attempt${ATTEMPT}.json"
  set +e
  PYTHONPATH=src python3 scripts/repair_failed_steps.py \
    --base "$FINAL" --config "$REPAIR_CONFIG" --output "$NEXT" --attempt "$ATTEMPT" \
    --max-workers 40 --commit-batch-size 40
  REPAIR_STATUS=$?
  set -e
  [ -f "$NEXT" ] || exit "$REPAIR_STATUS"
  FINAL="$NEXT"
done

ERRORS=$(PYTHONPATH=src python3 - "$FINAL" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding='utf-8-sig'))
print(sum(1 for variant in report['reports'].values() for step in variant.get('all_steps', []) if step.get('llm_error')))
PY
)
[ "$ERRORS" = "0" ] || { echo "Unrepaired llm_error count: $ERRORS" >&2; exit 1; }

cp "$FINAL" "$RUN_DIR/final_combined.json"
PYTHONPATH=src python3 scripts/evaluate_cohort_kt_models.py \
  --cohort "$COHORT" --dkt-checkpoint "$DKT" --ncdm-checkpoint "$NCDM" \
  --full-report "$RUN_DIR/final_combined.json" --output "$RUN_DIR/kt_baselines.json"

PYTHONPATH=src python3 - "$RUN_DIR" "$COHORT" "$DKT" "$NCDM" <<'PY'
import importlib.util
import json
import sys
from pathlib import Path
from learner_simulator.formal_metrics import build_ability_difficulty_partition, evaluate_formal_response_metrics

run, cohort, dkt_path, ncdm_path = map(Path, sys.argv[1:])
report = json.loads((run / 'final_combined.json').read_text(encoding='utf-8-sig'))
for name, variant in report['reports'].items():
    steps = variant['all_steps']
    assert len(steps) == 500 and not any(step.get('llm_error') for step in steps), name
partition = build_ability_difficulty_partition(report['reports']['full']['all_steps'])
script = Path('scripts/evaluate_cohort_kt_models.py')
spec = importlib.util.spec_from_file_location('cohort_eval', script)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)
_, history, target = module._load_cohort(cohort)
dkt = module._eval_dkt(history, target, dkt_path, 0.5)
ncdm = module._eval_ncdm(history, target, ncdm_path, 0.5)
assert len(dkt[4]) == len(ncdm[4]) == 500
report['kt_baselines'] = json.loads((run / 'kt_baselines.json').read_text(encoding='utf-8-sig'))
report['formal_metrics_v6']['fixed_full_irt_2x2_partition'] = partition
report['formal_metrics_v6']['baselines'] = {
    'dkt': evaluate_formal_response_metrics(dkt[4], partition),
    'ncdm': evaluate_formal_response_metrics(ncdm[4], partition),
}
(run / 'kt_baseline_steps.json').write_text(json.dumps({
    'dkt_direct_teacher_forcing_steps': dkt[4],
    'ncdm_direct_teacher_forcing_steps': ncdm[4],
    'fixed_full_irt_2x2_partition': partition,
}, ensure_ascii=False, indent=2), encoding='utf-8')
report['kt_baseline_step_archive'] = str(run / 'kt_baseline_steps.json')
(run / 'final_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
for name, result in report['formal_metrics_v6']['methods'].items():
    print(name, result['baa'], result['balanced_accuracy'], result['f1'], result['adcde'])
for name, result in report['formal_metrics_v6']['baselines'].items():
    print(name, result['baa'], result['balanced_accuracy'], result['f1'], result['adcde'])
PY

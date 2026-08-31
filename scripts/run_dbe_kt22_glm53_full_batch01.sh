#!/usr/bin/env bash
set -euo pipefail

ROOT=/ai/KT/LeanerSim
DATASET="$ROOT/data/DBE-KT22"
COHORT="$ROOT/experiments/cohorts/dbe_kt22_formal500_seed20260830/dbe_kt22_500x10_batch01_50x10_seed20260830.json"
REFERENCE="$ROOT/outputs/baselines/dbe_kt22/formal500_seed20260830_teacher_forcing/fixed_full_irt_reference.json"
NCDM_DIR="$ROOT/outputs/dneuralcdm/dbe_kt22_formal500_leakage_safe_e30_d32_h64"
NCDM="$NCDM_DIR/best_model.pt"
RUN_DIR="$ROOT/outputs/formal/v6_primaryroute/dbe_kt22/glm53_flash_batch01_50x10_seed20260830_full_p40"
MAIN_CONFIG="$ROOT/configs/llm.glm-5.3-flash_v6_single_call_thinking.json"
REPAIR_CONFIG="$ROOT/configs/llm.glm-5.3-flash_v6_repair_lowthinking.json"

cd "$ROOT"
test -f "$COHORT"
test -f "$REFERENCE"
test ! -e "$RUN_DIR"
test -f "$NCDM"
mkdir -p "$RUN_DIR"
exec > "$RUN_DIR/run.log" 2>&1
{
  echo "dataset=$DATASET"
  echo "cohort=$COHORT"
  echo "partition_reference=$REFERENCE"
  echo "ncdm=$NCDM"
  echo "main_config=$MAIN_CONFIG"
  echo "repair_config=$REPAIR_CONFIG"
  echo "variant=full"
  echo "feedback_mode=teacher-forcing"
  echo "parallel_learners=40"
} > "$RUN_DIR/run_manifest.txt"

PYTHONPATH=src python3 - "$NCDM_DIR/summary.json" <<'PY'
import json,sys
from pathlib import Path
summary=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8-sig'))
assert summary.get('leakage_safe') is True, summary
assert summary.get('excluded_cohort_users') == 500, summary
assert summary.get('cohort_history_used_for_training') is False, summary
print(json.dumps({'ncdm_audit':'ok','checkpoint':summary['checkpoint']}))
PY

PYTHONPATH=src python3 experiments/ablation/run_ablation.py \
  --dataset-root "$DATASET" --cohort-file "$COHORT" --variants full \
  --parallel-variants 1 --simulator multi-role --feedback-mode teacher-forcing \
  --parallel-learners 40 --llm-config "$MAIN_CONFIG" --dneuralcdm-checkpoint "$NCDM" \
  --checkpoint-dir "$RUN_DIR/checkpoints" --output "$RUN_DIR/combined.json" --progress

FINAL="$RUN_DIR/combined.json"
for ATTEMPT in 1 2 3 4 5; do
  ERRORS=$(PYTHONPATH=src python3 - "$FINAL" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8-sig'))
print(sum(bool(step.get('llm_error')) for v in d['reports'].values() for step in v.get('all_steps',[])))
PY
)
  [ "$ERRORS" = "0" ] && break
  NEXT="$RUN_DIR/combined_repaired_attempt${ATTEMPT}.json"
  set +e
  PYTHONPATH=src python3 scripts/repair_failed_steps.py \
    --base "$FINAL" --config "$REPAIR_CONFIG" --output "$NEXT" --attempt "$ATTEMPT" \
    --max-workers 40 --commit-batch-size 40
  STATUS=$?
  set -e
  [ -f "$NEXT" ] || exit "$STATUS"
  FINAL="$NEXT"
done

PYTHONPATH=src python3 - "$FINAL" "$RUN_DIR" "$REFERENCE" <<'PY'
import json,sys
from pathlib import Path
from learner_simulator.formal_metrics import evaluate_formal_response_metrics

source, run, reference = map(Path,sys.argv[1:])
report=json.loads(source.read_text(encoding='utf-8-sig'))
full=report['reports']['full']['all_steps']
assert len(full)==500 and not any(step.get('llm_error') for step in full)
partition=json.loads(reference.read_text(encoding='utf-8-sig'))['fixed_full_irt_2x2_partition']
report['formal_metrics_v6']['fixed_full_irt_2x2_partition']=partition
report['formal_metrics_v6']['methods']['full']=evaluate_formal_response_metrics(full,partition)
report['formal_metrics_v6']['methods']['full']['partition_reference']=str(reference)
(run/'final_combined.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
(run/'final_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
metrics=report['formal_metrics_v6']['methods']['full']
print(json.dumps({key:metrics[key] for key in ('baa','balanced_accuracy','f1','adcde')}))
PY

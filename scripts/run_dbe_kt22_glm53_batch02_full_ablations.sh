#!/usr/bin/env bash
set -euo pipefail

ROOT=/ai/KT/LeanerSim
DATASET="$ROOT/data/DBE-KT22"
COHORT="$ROOT/experiments/cohorts/dbe_kt22_formal500_seed20260830/dbe_kt22_500x10_batch02_50x10_seed20260830.json"
REFERENCE="$ROOT/outputs/baselines/dbe_kt22/formal500_seed20260830_teacher_forcing/fixed_full_irt_reference.json"
NCDM="$ROOT/outputs/dneuralcdm/dbe_kt22_formal500_leakage_safe_e30_d32_h64/best_model.pt"
NCDM_SUMMARY="$ROOT/outputs/dneuralcdm/dbe_kt22_formal500_leakage_safe_e30_d32_h64/summary.json"
CONFIG="$ROOT/configs/llm.glm-5.3-flash_v6_single_call_thinking.json"
REPAIR="$ROOT/configs/llm.glm-5.3-flash_v6_repair_lowthinking.json"
RUN="$ROOT/outputs/formal/v6_primaryroute/dbe_kt22/glm53_flash_batch02_50x10_seed20260830_full_ablations_p40"
ARTIFACT="$ROOT/outputs/external_storage/dbe_kt22_formal500_shared_cohort_artifacts_with_glm53_batch01"
ARCHIVE="$ROOT/outputs/external_storage/dbe_kt22_formal500_shared_cohort_artifacts_with_glm53_batch01.tar.gz"

cd "$ROOT"
test -f "$COHORT" && test -f "$REFERENCE" && test -f "$NCDM" && test -f "$NCDM_SUMMARY"
test ! -e "$RUN"
mkdir -p "$RUN"
exec > "$RUN/run.log" 2>&1

printf '%s\n' \
  "cohort=$COHORT" \
  "feedback_mode=teacher-forcing" \
  "variants=full,no-evidence-representation,no-state-item-alignment,no-structured-response-process,no-dynamic-state-evolution" \
  "full_parallel_learners=40" \
  "ablation_parallel_variants=4" \
  "ablation_parallel_learners_per_variant=10" \
  "total_llm_concurrency_per_stage=40" \
  "dkt_ncdm_evaluation=not_run_existing_formal500_results_reused" \
  > "$RUN/run_manifest.txt"

PYTHONPATH=src python3 - "$NCDM_SUMMARY" <<'PY'
import json,sys
m=json.load(open(sys.argv[1],encoding='utf-8-sig'))
assert m.get('leakage_safe') is True
assert m.get('excluded_cohort_users') == 500
assert m.get('cohort_history_used_for_training') is False
PY

run_and_repair () {
  local label="$1"; shift
  local current="$RUN/$label.json"
  PYTHONPATH=src python3 experiments/ablation/run_ablation.py "$@" --output "$current" --progress > "$RUN/${label}_main.log" 2>&1
  for attempt in 1 2 3 4 5; do
    local errors
    errors=$(PYTHONPATH=src python3 - "$current" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8-sig'))
print(sum(bool(s.get('llm_error')) for r in x['reports'].values() for s in r.get('all_steps',[])))
PY
)
    [ "$errors" = 0 ] && break
    local next="$RUN/${label}_repaired_attempt${attempt}.json"
    set +e
    PYTHONPATH=src python3 scripts/repair_failed_steps.py --base "$current" --config "$REPAIR" --output "$next" --attempt "$attempt" --max-workers 40 --commit-batch-size 40 > "$RUN/${label}_repair_attempt${attempt}.log" 2>&1
    status=$?
    set -e
    [ -f "$next" ] || exit "$status"
    current="$next"
  done
  printf '%s' "$current"
}

FULL=$(run_and_repair full \
  --dataset-root "$DATASET" --cohort-file "$COHORT" --variants full \
  --parallel-variants 1 --simulator multi-role --feedback-mode teacher-forcing \
  --parallel-learners 40 --llm-config "$CONFIG" --dneuralcdm-checkpoint "$NCDM" \
  --checkpoint-dir "$RUN/checkpoints/full")

ABLAT=$(run_and_repair ablations \
  --dataset-root "$DATASET" --cohort-file "$COHORT" \
  --variants no-evidence-representation,no-state-item-alignment,no-structured-response-process,no-dynamic-state-evolution \
  --parallel-variants 4 --variant-start-stagger-seconds 3 --simulator multi-role --feedback-mode teacher-forcing \
  --parallel-learners 10 --llm-config "$CONFIG" --dneuralcdm-checkpoint "$NCDM" \
  --checkpoint-dir "$RUN/checkpoints/ablations")

PYTHONPATH=src python3 - "$FULL" "$ABLAT" "$REFERENCE" "$COHORT" "$RUN/merged_final_report.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
from learner_simulator.formal_metrics import evaluate_formal_response_metrics
full_path,abl_path,ref_path,cohort_path,out=map(Path,sys.argv[1:])
full=json.loads(full_path.read_text(encoding='utf-8-sig'))['reports']['full']
abl=json.loads(abl_path.read_text(encoding='utf-8-sig'))['reports']
reports={'full':full,**abl}
expected=['full','no-evidence-representation','no-state-item-alignment','no-structured-response-process','no-dynamic-state-evolution']
assert list(reports)==expected
cohort=json.loads(cohort_path.read_text(encoding='utf-8-sig')); uids=set(map(str,cohort['uids']))
partition=json.loads(ref_path.read_text(encoding='utf-8-sig'))['fixed_full_irt_2x2_partition']
methods={}; audit={}
for name in expected:
  steps=reports[name]['all_steps']; errors=sum(bool(x.get('llm_error')) for x in steps)
  assert len(steps)==500 and {str(x.get('uid')) for x in steps}==uids and errors==0, (name,len(steps),errors)
  methods[name]=evaluate_formal_response_metrics(steps,partition)
  audit[name]={'users':50,'steps':500,'llm_error':0}
result={'study':'dbe_kt22_glm53_flash_batch02_full_ablations','protocol':{'teacher_forcing':True,'history_steps':90,'target_steps':10,'total_llm_concurrency_per_stage':40},'cohort_file':str(cohort_path),'cohort_sha256':hashlib.sha256(cohort_path.read_bytes()).hexdigest(),'partition_reference':str(ref_path),'formal_metrics_v6':{'fixed_full_irt_2x2_partition':partition,'methods':methods},'audit':audit,'reports':reports}
out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'audit':audit,'metrics':methods},ensure_ascii=False))
PY

DEST="$ARTIFACT/results/glm53_flash_batch02_50x10_full_ablations"
mkdir -p "$DEST"
cp "$RUN/run.log" "$RUN/run_manifest.txt" "$FULL" "$ABLAT" "$RUN/merged_final_report.json" "$DEST/"
cp -r "$RUN/checkpoints" "$DEST/"
printf '\n- Added GLM-5.3-Flash DBE-KT22 batch02 (50 users × 10 steps): Full + four ablations, 40 total LLM concurrency per stage.\n' >> "$ARTIFACT/README.md"
TMP="$ARCHIVE.tmp"
tar -czf "$TMP" -C "$(dirname "$ARTIFACT")" "$(basename "$ARTIFACT")"
mv "$TMP" "$ARCHIVE"

#!/usr/bin/env bash
set -euo pipefail

ROOT=/ai/KT/LeanerSim
DATASET="$ROOT/data/DBE-KT22"
FORMAL500="$ROOT/experiments/cohorts/dbe_kt22_formal500_seed20260830/dbe_kt22_formal500_50x10_fixed.json"
COHORT="$ROOT/experiments/cohorts/dbe_kt22_formal500_seed20260830/dbe_kt22_500x10_batch01_50x10_seed20260830.json"
REFERENCE="$ROOT/outputs/baselines/dbe_kt22/formal500_seed20260830_teacher_forcing/fixed_full_irt_reference.json"
FULL_DIR="$ROOT/outputs/formal/v6_primaryroute/dbe_kt22/glm53_flash_batch01_50x10_seed20260830_full_p40"
RUN_DIR="$ROOT/outputs/formal/v6_primaryroute/dbe_kt22/glm53_flash_batch01_50x10_seed20260830_ablations_dkt_ncdm_p40"
DKT_DIR="$ROOT/outputs/dkt/dbe_kt22_formal500_leakage_safe_e30_h128"
DKT="$DKT_DIR/best_model.pt"
NCDM_DIR="$ROOT/outputs/dneuralcdm/dbe_kt22_formal500_leakage_safe_e30_d32_h64"
NCDM="$NCDM_DIR/best_model.pt"
MAIN_CONFIG="$ROOT/configs/llm.glm-5.3-flash_v6_single_call_thinking.json"
REPAIR_CONFIG="$ROOT/configs/llm.glm-5.3-flash_v6_repair_lowthinking.json"
ARTIFACT_DIR="$ROOT/outputs/external_storage/dbe_kt22_glm53_flash_batch01_50x10_full_ablations_dkt_ncdm"
ARCHIVE="$ROOT/outputs/external_storage/dbe_kt22_glm53_flash_batch01_50x10_full_ablations_dkt_ncdm.tar.gz"

cd "$ROOT"
test -f "$COHORT"
test -f "$FORMAL500"
test -f "$REFERENCE"
test -f "$FULL_DIR/final_report.json"
test -f "$NCDM"
test ! -e "$RUN_DIR"
test ! -e "$ARTIFACT_DIR"
test ! -e "$ARCHIVE"
mkdir -p "$RUN_DIR"
exec > "$RUN_DIR/run.log" 2>&1

{
  echo "cohort=$COHORT"
  echo "formal500_exclusion_cohort=$FORMAL500"
  echo "full_report=$FULL_DIR/final_report.json"
  echo "partition_reference=$REFERENCE"
  echo "variants=no-evidence-representation,no-state-item-alignment,no-structured-response-process,no-dynamic-state-evolution"
  echo "feedback_mode=teacher-forcing"
  echo "parallel_variants=4"
  echo "parallel_learners_per_variant=10"
  echo "total_llm_concurrency=40"
  echo "dkt=$DKT"
  echo "ncdm=$NCDM"
} > "$RUN_DIR/run_manifest.txt"

# Train the one shared DKT checkpoint only if it has not already been created.
if [ ! -f "$DKT" ]; then
  test ! -e "$DKT_DIR"
  mkdir -p "$DKT_DIR"
  PYTHONPATH=src python3 scripts/train_dkt.py \
    --dataset-root "$DATASET" --source-rows 0 \
    --exclude-cohort-file "$FORMAL500" \
    --output-dir "$DKT_DIR" --epochs 30 --batch-size 32 --hidden-dim 128 \
    --seed 42 --log-every 20 \
    > "$DKT_DIR/train.log" 2>&1
fi

PYTHONPATH=src python3 - "$DKT_DIR/summary.json" "$NCDM_DIR/summary.json" "$FORMAL500" <<'PY'
import json,sys
from pathlib import Path
dkt,ncdm,cohort=map(Path,sys.argv[1:])
for name,path in [('dkt',dkt),('ncdm',ncdm)]:
    summary=json.loads(path.read_text(encoding='utf-8-sig'))
    assert summary.get('leakage_safe') is True, (name,summary)
    assert summary.get('excluded_cohort_users') == 500, (name,summary)
    assert summary.get('cohort_history_used_for_training') is False, (name,summary)
    files=summary.get('cohort_files') or [summary.get('cohort_file')]
    assert str(cohort) in files, (name,files)
print('{"checkpoint_audit":"ok"}')
PY

PYTHONPATH=src python3 experiments/ablation/run_ablation.py \
  --dataset-root "$DATASET" --cohort-file "$COHORT" \
  --variants no-evidence-representation,no-state-item-alignment,no-structured-response-process,no-dynamic-state-evolution \
  --parallel-variants 4 --variant-start-stagger-seconds 3 \
  --simulator multi-role --feedback-mode teacher-forcing \
  --parallel-learners 10 --llm-config "$MAIN_CONFIG" \
  --dneuralcdm-checkpoint "$NCDM" \
  --checkpoint-dir "$RUN_DIR/checkpoints" --output "$RUN_DIR/ablations_combined.json" --progress

FINAL="$RUN_DIR/ablations_combined.json"
for ATTEMPT in 1 2 3 4 5; do
  ERRORS=$(PYTHONPATH=src python3 - "$FINAL" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8-sig'))
print(sum(bool(step.get('llm_error')) for report in d['reports'].values() for step in report.get('all_steps',[])))
PY
)
  [ "$ERRORS" = "0" ] && break
  NEXT="$RUN_DIR/ablations_repaired_attempt${ATTEMPT}.json"
  set +e
  PYTHONPATH=src python3 scripts/repair_failed_steps.py \
    --base "$FINAL" --config "$REPAIR_CONFIG" --output "$NEXT" --attempt "$ATTEMPT" \
    --max-workers 40 --commit-batch-size 40
  STATUS=$?
  set -e
  [ -f "$NEXT" ] || exit "$STATUS"
  FINAL="$NEXT"
done

PYTHONPATH=src python3 scripts/evaluate_cohort_kt_models.py \
  --cohort "$COHORT" --dkt-checkpoint "$DKT" --ncdm-checkpoint "$NCDM" \
  --full-report "$FULL_DIR/final_report.json" --output "$RUN_DIR/kt_baselines.json"

PYTHONPATH=src python3 - "$FULL_DIR/final_report.json" "$FINAL" "$RUN_DIR/kt_baselines.json" "$REFERENCE" "$COHORT" "$RUN_DIR" "$ARTIFACT_DIR" "$ARCHIVE" <<'PY'
import hashlib,json,shutil,sys,tarfile
from pathlib import Path
from learner_simulator.formal_metrics import evaluate_formal_response_metrics

full_path,abl_path,kt_path,ref_path,cohort_path,run_dir,artifact,archive=map(Path,sys.argv[1:])
full=json.loads(full_path.read_text(encoding='utf-8-sig'))
abl=json.loads(abl_path.read_text(encoding='utf-8-sig'))
kt=json.loads(kt_path.read_text(encoding='utf-8-sig'))
ref=json.loads(ref_path.read_text(encoding='utf-8-sig'))
cohort=json.loads(cohort_path.read_text(encoding='utf-8-sig'))
partition=ref['fixed_full_irt_2x2_partition']
reports={'full':full['reports']['full']}
reports.update(abl['reports'])
expected=['full','no-evidence-representation','no-state-item-alignment','no-structured-response-process','no-dynamic-state-evolution']
assert list(reports)==expected, list(reports)
uids=set(map(str,cohort['uids']))
methods={}
audit={}
for name in expected:
    steps=reports[name].get('all_steps') or []
    errors=sum(bool(x.get('llm_error')) for x in steps)
    got={str(x.get('uid')) for x in steps}
    assert len(steps)==500 and got==uids and errors==0, (name,len(steps),len(got),errors)
    methods[name]=evaluate_formal_response_metrics(steps,partition)
    audit[name]={'steps':len(steps),'users':len(got),'llm_error':errors}
for name,key in [('dkt','dkt_all_steps'),('ncdm','ncdm_all_steps')]:
    steps=kt[key]
    got={str(x.get('uid')) for x in steps}
    assert len(steps)==500 and got==uids, (name,len(steps),len(got))
    methods[name]=evaluate_formal_response_metrics(steps,partition)
    audit[name]={'steps':len(steps),'users':len(got),'llm_error':0}
merged={
    'study':'dbe_kt22_glm53_flash_batch01_full_ablations_dkt_ncdm',
    'protocol':{'feedback_mode':'teacher-forcing','history_steps':90,'target_steps':10,'llm_total_concurrency':40},
    'cohort_file':str(cohort_path),
    'cohort_sha256':hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
    'partition_reference':str(ref_path),
    'formal_metrics_v6':{'fixed_full_irt_2x2_partition':partition,'methods':methods},
    'audit':audit,
    'reports':reports,
    'kt_baselines':kt,
}
(run_dir/'merged_final_report.json').write_text(json.dumps(merged,ensure_ascii=False,indent=2),encoding='utf-8')

root=Path('/ai/KT/LeanerSim')
artifact.mkdir(parents=True)
for source,dest in [
    (root/'data/DBE-KT22',artifact/'source_data/DBE-KT22'),
    (root/'experiments/cohorts/dbe_kt22_formal500_seed20260830',artifact/'cohorts/dbe_kt22_formal500_seed20260830'),
    (full_path.parent,artifact/'results/full'),
    (run_dir,artifact/'results/ablations_dkt_ncdm'),
]:
    shutil.copytree(source,dest)
for source,dest in [
    (root/'scripts/prepare_dbe_kt22.py',artifact/'source_code/prepare_dbe_kt22.py'),
    (root/'scripts/train_dkt.py',artifact/'source_code/train_dkt.py'),
    (root/'scripts/train_dneuralcdm.py',artifact/'source_code/train_dneuralcdm.py'),
    (root/'scripts/evaluate_cohort_kt_models.py',artifact/'source_code/evaluate_cohort_kt_models.py'),
    (root/'experiments/ablation/run_ablation.py',artifact/'source_code/run_ablation.py'),
    (root/'configs/llm.glm-5.3-flash_v6_single_call_thinking.json',artifact/'configs/llm.glm-5.3-flash_v6_single_call_thinking.json'),
    (root/'configs/llm.glm-5.3-flash_v6_repair_lowthinking.json',artifact/'configs/llm.glm-5.3-flash_v6_repair_lowthinking.json'),
    (root/'outputs/dkt/dbe_kt22_formal500_leakage_safe_e30_h128/summary.json',artifact/'models/dkt_summary.json'),
    (root/'outputs/dkt/dbe_kt22_formal500_leakage_safe_e30_h128/best_model.pt',artifact/'models/dkt_best_model.pt'),
    (root/'outputs/dneuralcdm/dbe_kt22_formal500_leakage_safe_e30_d32_h64/summary.json',artifact/'models/ncdm_summary.json'),
    (root/'outputs/dneuralcdm/dbe_kt22_formal500_leakage_safe_e30_d32_h64/best_model.pt',artifact/'models/ncdm_best_model.pt'),
    (ref_path,artifact/'references/fixed_full_irt_reference.json'),
]:
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,dest)
manifest={'archive_kind':'self_contained_dbe_kt22_batch01','metrics':['baa','balanced_accuracy','f1','adcde'],'audit':audit,'cohort_sha256':merged['cohort_sha256'],'source_data_included':True,'results_included':True,'all_steps_included':True,'dkt_ncdm_probability_included':True}
(artifact/'README.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
with tarfile.open(archive,'w:gz') as tar:
    tar.add(artifact,arcname=artifact.name)
print(json.dumps({'merged_report':str(run_dir/'merged_final_report.json'),'artifact_dir':str(artifact),'archive':str(archive),'metrics':{k:{m:round(v[m],6) if isinstance(v.get(m),float) else v.get(m) for m in ['baa','balanced_accuracy','f1','adcde']} for k,v in methods.items()}},ensure_ascii=False))
PY

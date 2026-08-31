#!/usr/bin/env bash
set -euo pipefail

ROOT=/ai/KT/LeanerSim
DATASET="$ROOT/data/DBE-KT22"
COHORT="$ROOT/experiments/cohorts/dbe_kt22_formal500_seed20260830/dbe_kt22_formal500_50x10_fixed.json"
REFERENCE="$ROOT/outputs/baselines/dbe_kt22/formal500_seed20260830_teacher_forcing/fixed_full_irt_reference.json"
OUT_ROOT="$ROOT/outputs/baselines/psychometric/dbe_kt22/formal500_seed20260830_teacher_forcing"
IRT_DIR="$OUT_ROOT/irt_1pl"
MIRT_DIR="$OUT_ROOT/mirt_5d"
SUMMARY="$OUT_ROOT/irt_mirt_teacher_forcing_500_final_report.json"
ARTIFACT="$ROOT/outputs/external_storage/dbe_kt22_glm53_flash_batch01_50x10_full_ablations_dkt_ncdm"
ARCHIVE="$ROOT/outputs/external_storage/dbe_kt22_glm53_flash_batch01_50x10_full_ablations_dkt_ncdm.tar.gz"

cd "$ROOT"
test -f "$COHORT"
test -f "$REFERENCE"
test -d "$ARTIFACT"
test -f "$ARCHIVE"
test ! -e "$OUT_ROOT"
mkdir -p "$IRT_DIR" "$MIRT_DIR"
exec > "$OUT_ROOT/irt_mirt_pipeline.log" 2>&1

PYTHONPATH=src python3 scripts/train_psychometric.py \
  --method irt --dataset-root "$DATASET" --output-dir "$IRT_DIR" \
  --exclude-cohort-file "$COHORT" --source-rows 0 \
  --epochs 30 --batch-size 4096 --lr 0.03 --seed 42 \
  > "$IRT_DIR/train.log" 2>&1
PYTHONPATH=src python3 scripts/evaluate_psychometric_rollout.py \
  --checkpoint "$IRT_DIR/best_model.json" --cohort-file "$COHORT" \
  --full-irt-reference "$REFERENCE" --feedback-mode teacher_forcing \
  --output "$IRT_DIR/teacher_forcing_500.json" \
  > "$IRT_DIR/evaluate_teacher_forcing.log" 2>&1

PYTHONPATH=src python3 scripts/train_psychometric.py \
  --method mirt --dataset-root "$DATASET" --output-dir "$MIRT_DIR" \
  --exclude-cohort-file "$COHORT" --source-rows 0 --dimensions 5 \
  --epochs 30 --batch-size 4096 --lr 0.03 --seed 42 \
  > "$MIRT_DIR/train.log" 2>&1
PYTHONPATH=src python3 scripts/evaluate_psychometric_rollout.py \
  --checkpoint "$MIRT_DIR/best_model.json" --cohort-file "$COHORT" \
  --full-irt-reference "$REFERENCE" --feedback-mode teacher_forcing \
  --output "$MIRT_DIR/teacher_forcing_500.json" \
  > "$MIRT_DIR/evaluate_teacher_forcing.log" 2>&1

PYTHONPATH=src python3 - "$COHORT" "$REFERENCE" "$IRT_DIR" "$MIRT_DIR" "$SUMMARY" "$ARTIFACT" "$ARCHIVE" <<'PY'
import hashlib,json,shutil,sys,tarfile
from pathlib import Path

cohort_path,ref_path,irt_dir,mirt_dir,summary_path,artifact,archive=map(Path,sys.argv[1:])
cohort=json.loads(cohort_path.read_text(encoding='utf-8-sig'))
uids=set(map(str,cohort['uids']))
reports={}
for name,directory,expected_method in [('irt',irt_dir,'irt_1pl'),('mirt',mirt_dir,'mirt_2pl')]:
    training=json.loads((directory/'summary.json').read_text(encoding='utf-8-sig'))
    report=json.loads((directory/'teacher_forcing_500.json').read_text(encoding='utf-8-sig'))
    meta=report['training_metadata']; audit=report['audit']; steps=report['all_steps']
    assert report['method']==expected_method, (name,report['method'])
    assert meta.get('leakage_safe') is True and meta.get('excluded_cohort_users')==500, (name,meta)
    assert meta.get('cohort_history_used_for_training') is False, (name,meta)
    assert audit['users']==500 and audit['steps']==5000 and audit['feedback_mode']=='teacher_forcing', (name,audit)
    assert len(steps)==5000 and {str(x['uid']) for x in steps}==uids and all('probability' in x for x in steps), (name,len(steps))
    metrics=report['formal_metrics_v6']
    reports[name]={'report':str(directory/'teacher_forcing_500.json'),'checkpoint':str(directory/'best_model.json'),'training_summary':training,'audit':audit,'metrics':{k:metrics[k] for k in ('baa','balanced_accuracy','f1','adcde')}}
payload={'dataset':'DBE-KT22','protocol':'fixed 90-history / 10-target; teacher_forcing; raw-step pooled','cohort_file':str(cohort_path),'cohort_sha256':hashlib.sha256(cohort_path.read_bytes()).hexdigest(),'partition_reference':str(ref_path),'reports':reports}
summary_path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
for name,directory in [('irt',irt_dir),('mirt',mirt_dir)]:
    destination=artifact/'results'/name
    if destination.exists(): raise AssertionError(f'artifact destination exists: {destination}')
    shutil.copytree(directory,destination)
for source,name in [
    (Path('/ai/KT/LeanerSim/scripts/train_psychometric.py'),'train_psychometric.py'),
    (Path('/ai/KT/LeanerSim/scripts/evaluate_psychometric_rollout.py'),'evaluate_psychometric_rollout.py'),
]:
    shutil.copy2(source,artifact/'source_code'/name)
shutil.copy2(summary_path,artifact/'results'/'irt_mirt_teacher_forcing_500_final_report.json')
readme_path=artifact/'README.json'; readme=json.loads(readme_path.read_text(encoding='utf-8-sig'))
readme['irt_mirt_added']=True
readme['irt_mirt_protocol']=payload['protocol']
readme['irt_mirt_metrics']={name:entry['metrics'] for name,entry in reports.items()}
readme_path.write_text(json.dumps(readme,ensure_ascii=False,indent=2),encoding='utf-8')
temporary=archive.with_suffix('.tar.gz.tmp')
with tarfile.open(temporary,'w:gz') as tar: tar.add(artifact,arcname=artifact.name)
temporary.replace(archive)
print(json.dumps(payload,ensure_ascii=False))
PY

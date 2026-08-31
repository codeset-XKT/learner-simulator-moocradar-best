#!/usr/bin/env bash
set -euo pipefail

ROOT=/ai/KT/LeanerSim
DATASET="$ROOT/data/DBE-KT22"
COHORT="$ROOT/experiments/cohorts/dbe_kt22_formal500_seed20260830/dbe_kt22_formal500_50x10_fixed.json"
REFERENCE="$ROOT/outputs/baselines/dbe_kt22/formal500_seed20260830_teacher_forcing/fixed_full_irt_reference.json"
OUT_ROOT="$ROOT/outputs/baselines/neural_kt/dbe_kt22/formal500_seed20260830_tf_es30"
SAKT_DIR="$OUT_ROOT/sakt"
AKT_DIR="$OUT_ROOT/akt"
SUMMARY="$OUT_ROOT/sakt_akt_teacher_forcing_500_final_report.json"
ARTIFACT="$ROOT/outputs/external_storage/dbe_kt22_glm53_flash_batch01_50x10_full_ablations_dkt_ncdm"
ARCHIVE="$ROOT/outputs/external_storage/dbe_kt22_glm53_flash_batch01_50x10_full_ablations_dkt_ncdm.tar.gz"

cd "$ROOT"
test -f "$COHORT"
test -f "$REFERENCE"
test -d "$ARTIFACT"
test -f "$ARCHIVE"
test ! -e "$SAKT_DIR"
test ! -e "$AKT_DIR"
test ! -e "$SUMMARY"
mkdir -p "$SAKT_DIR" "$AKT_DIR"
exec > "$OUT_ROOT/sakt_akt_pipeline.log" 2>&1

for METHOD in sakt akt; do
  if [ "$METHOD" = sakt ]; then DIR="$SAKT_DIR"; else DIR="$AKT_DIR"; fi
  PYTHONPATH=src python3 scripts/train_neural_kt_references.py \
    --method "$METHOD" --dataset-root "$DATASET" --output-dir "$DIR" \
    --exclude-cohort-file "$COHORT" --source-rows 0 \
    --epochs 30 --early-stopping-patience 5 --early-stopping-min-delta 0.0005 \
    --batch-size 64 --lr 0.001 --embed-dim 64 --num-heads 4 --dropout 0.2 --max-seq-len 200 --seed 42 \
    > "$DIR/train.log" 2>&1
  PYTHONPATH=src python3 scripts/evaluate_neural_kt_rollout.py \
    --checkpoint "$DIR/best_model.pt" --cohort-file "$COHORT" \
    --full-irt-reference "$REFERENCE" --feedback-mode teacher_forcing \
    --output "$DIR/teacher_forcing_500.json" \
    > "$DIR/evaluate_teacher_forcing.log" 2>&1
done

PYTHONPATH=src python3 - "$COHORT" "$REFERENCE" "$SAKT_DIR" "$AKT_DIR" "$SUMMARY" "$ARTIFACT" "$ARCHIVE" <<'PY'
import hashlib,json,shutil,sys,tarfile
from pathlib import Path

cohort_path,ref_path,sakt_dir,akt_dir,summary_path,artifact,archive=map(Path,sys.argv[1:])
cohort=json.loads(cohort_path.read_text(encoding='utf-8-sig'))
uids=set(map(str,cohort['uids']))
reports={}
for method,directory in [('sakt',sakt_dir),('akt',akt_dir)]:
    summary=json.loads((directory/'summary.json').read_text(encoding='utf-8-sig'))
    report=json.loads((directory/'teacher_forcing_500.json').read_text(encoding='utf-8-sig'))
    meta=report['training_metadata']
    audit=report['audit']
    steps=report['all_steps']
    assert summary['method']==method, (method,summary.get('method'))
    assert meta.get('leakage_safe') is True and meta.get('excluded_cohort_users')==500, (method,meta)
    assert meta.get('cohort_history_used_for_training') is False, (method,meta)
    assert audit['users']==500 and audit['steps']==5000 and audit['feedback_mode']=='teacher_forcing', (method,audit)
    assert len(steps)==5000 and {str(x['uid']) for x in steps}==uids, (method,len(steps),len({str(x['uid']) for x in steps}))
    metrics=report['formal_metrics_v6']
    reports[method]={'report':str(directory/'teacher_forcing_500.json'),'checkpoint':str(directory/'best_model.pt'),'training_summary':summary,'audit':audit,'metrics':{k:metrics[k] for k in ('baa','balanced_accuracy','f1','adcde')}}
payload={'dataset':'DBE-KT22','protocol':'fixed 90-history / 10-target; teacher_forcing; raw-step pooled','cohort_file':str(cohort_path),'cohort_sha256':hashlib.sha256(cohort_path.read_bytes()).hexdigest(),'partition_reference':str(ref_path),'reports':reports}
summary_path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')

for method,directory in [('sakt',sakt_dir),('akt',akt_dir)]:
    destination=artifact/'results'/method
    if destination.exists(): raise AssertionError(f'artifact destination exists: {destination}')
    shutil.copytree(directory,destination)
for source,name in [
    (Path('/ai/KT/LeanerSim/scripts/train_neural_kt_references.py'),'train_neural_kt_references.py'),
    (Path('/ai/KT/LeanerSim/scripts/evaluate_neural_kt_rollout.py'),'evaluate_neural_kt_rollout.py'),
]:
    shutil.copy2(source,artifact/'source_code'/name)
shutil.copy2(summary_path,artifact/'results'/'sakt_akt_teacher_forcing_500_final_report.json')
readme_path=artifact/'README.json'
readme=json.loads(readme_path.read_text(encoding='utf-8-sig'))
readme['sakt_akt_added']=True
readme['sakt_akt_protocol']=payload['protocol']
readme['sakt_akt_metrics']={method:entry['metrics'] for method,entry in reports.items()}
readme_path.write_text(json.dumps(readme,ensure_ascii=False,indent=2),encoding='utf-8')
temporary=archive.with_suffix('.tar.gz.tmp')
with tarfile.open(temporary,'w:gz') as tar:
    tar.add(artifact,arcname=artifact.name)
temporary.replace(archive)
print(json.dumps(payload,ensure_ascii=False))
PY

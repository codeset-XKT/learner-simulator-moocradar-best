#!/usr/bin/env python3
"""Train IRT (1PL) or MIRT (2PL) with cohort-user exclusion."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from learner_simulator.data import clean_sequence, take_sequence_rows
from learner_simulator.psychometric import fit_psychometric

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=("irt","mirt"), required=True); p.add_argument("--dataset-root",required=True); p.add_argument("--output-dir",required=True)
    p.add_argument("--exclude-cohort-file",nargs="+",default=[]); p.add_argument("--source-rows",type=int,default=0); p.add_argument("--dimensions",type=int,default=5); p.add_argument("--epochs",type=int,default=30); p.add_argument("--batch-size",type=int,default=4096); p.add_argument("--lr",type=float,default=.03); p.add_argument("--seed",type=int,default=42)
    a=p.parse_args(); excluded=set()
    for file in a.exclude_cohort_file: excluded.update(str(x) for x in json.loads(Path(file).read_text(encoding="utf-8-sig")).get("uids",[]))
    raw=take_sequence_rows(Path(a.dataset_root)/"kc_level"/"train_valid_sequences.csv", None if a.source_rows<=0 else a.source_rows)
    interactions=[(str(row["uid"]),str(s["qid"]),int(s["response"])) for row in raw if str(row["uid"]) not in excluded for s in clean_sequence(row)]
    if any(uid in excluded for uid,_,_ in interactions): raise AssertionError("Cohort user leakage")
    model, fit=fit_psychometric(interactions, 1 if a.method=="irt" else a.dimensions, a.epochs,a.batch_size,a.lr,a.seed)
    meta={"leakage_safe":True,"excluded_user_ids":sorted(excluded),"excluded_cohort_users":len(excluded),"cohort_history_used_for_training":False,"source_sequence_rows":len(raw),"train_interactions":len(interactions),"seed":a.seed}
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True); payload={**model.to_dict(),"training_metadata":meta,"fit":fit}
    (out/"best_model.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8");(out/"summary.json").write_text(json.dumps({"ok":True,"method":a.method,"checkpoint":str(out/"best_model.json"),"fit":fit,"training_metadata":meta},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"ok":True,"method":a.method,"fit":fit},ensure_ascii=False),flush=True)
if __name__=="__main__": main()

#!/usr/bin/env python3
"""Evaluate a leakage-safe IRT/MIRT checkpoint on fixed 90-to-10 cohorts.

Teacher forcing is the formal default; autoregressive rollout is retained only
as an explicitly selected diagnostic protocol.
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from learner_simulator.data import clean_sequence
from learner_simulator.formal_metrics import evaluate_formal_response_metrics
from learner_simulator.psychometric import IRTReference, MIRTReference

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--checkpoint",required=True);p.add_argument("--cohort-file",nargs="+",required=True);p.add_argument("--full-irt-reference",required=True);p.add_argument("--output",required=True);p.add_argument("--seed",type=int,default=42);p.add_argument("--feedback-mode",choices=("teacher_forcing","rollout"),default="teacher_forcing");a=p.parse_args()
 data=json.load(open(a.checkpoint));meta=data.get("training_metadata",{});
 if not meta.get("leakage_safe"):raise ValueError("checkpoint is not leakage-safe")
 model=IRTReference.from_dict(data) if data.get("method")=="irt_1pl" else MIRTReference.from_dict(data)
 excluded=set(map(str,meta.get("excluded_user_ids",[])));steps=[];cohort_uids=[]
 for batch_index,file in enumerate(a.cohort_file):
  cohort=json.load(open(file,encoding="utf-8-sig"));hist={str(r["uid"]):clean_sequence(r) for r in cohort["history_rows"]};target={str(r["uid"]):clean_sequence(r) for r in cohort["target_rows"]}
  for offset,raw_uid in enumerate(cohort["uids"]):
   uid=str(raw_uid)
   if uid not in excluded:raise AssertionError(f"{uid} was not excluded during training")
   cohort_uids.append(uid);state=model.state_from_history(hist[uid]);rng=random.Random(a.seed+batch_index*10000+offset)
   for index,item in enumerate(target[uid]):
    probability=float(model.predict(state,item));simulated=int(rng.random()<probability)
    next_response=int(item["response"]) if a.feedback_mode=="teacher_forcing" else simulated
    state=model.update(state,item,next_response)
    steps.append({"uid":uid,"step_index":index,"qid":int(item["qid"]),"real_response":int(item["response"]),"simulated_response":simulated,"probability":probability,"feedback_mode":a.feedback_mode})
 reference=json.load(open(a.full_irt_reference,encoding="utf-8-sig"));partition=reference["fixed_full_irt_2x2_partition"]
 expected=len(cohort_uids)*10
 if len(steps)!=expected or len(set(cohort_uids))!=len(cohort_uids):raise AssertionError(f"coverage invalid: steps={len(steps)} users={len(set(cohort_uids))}")
 report={"method":data["method"],"checkpoint":a.checkpoint,"cohort_files":a.cohort_file,"full_irt_reference":a.full_irt_reference,"training_metadata":meta,"audit":{"users":len(cohort_uids),"steps":len(steps),"raw_step_pooled":True,"feedback_mode":a.feedback_mode},"formal_metrics_v6":evaluate_formal_response_metrics(steps,partition),"all_steps":steps}
 Path(a.output).parent.mkdir(parents=True,exist_ok=True);json.dump(report,open(a.output,"w"),ensure_ascii=False,indent=2);print(json.dumps({**report["audit"],**{k:report["formal_metrics_v6"][k] for k in ["baa","balanced_accuracy","f1","adcde"]}},ensure_ascii=False),flush=True)
if __name__=="__main__":main()

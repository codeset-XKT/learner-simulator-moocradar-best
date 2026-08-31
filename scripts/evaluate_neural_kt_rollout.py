#!/usr/bin/env python3
"""Fixed-cohort evaluator for SAKT and official AKT.

Formal baseline comparisons default to teacher forcing: each target response is
generated from the observed history plus the *real* preceding target responses.
Autoregressive rollout remains available only as an explicitly requested
diagnostic mode.
"""
from __future__ import annotations
import argparse, importlib.util, json, random, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from learner_simulator.data import clean_sequence
from learner_simulator.formal_metrics import evaluate_formal_response_metrics
from learner_simulator.neural_kt import DeepIRT, NeuralKTConfig, SAKT, torch

def parse():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--checkpoint",required=True);p.add_argument("--cohort-file",nargs="+",required=True);p.add_argument("--full-irt-reference",required=True);p.add_argument("--output",required=True);p.add_argument("--seed",type=int,default=42);p.add_argument("--device",default=None);p.add_argument("--feedback-mode",choices=("teacher_forcing","rollout"),default="teacher_forcing",help="Formal evaluations use teacher_forcing by default; rollout is diagnostic only.");return p.parse_args()
def make_akt(config,device):
 source=ROOT/"third_party/AKT_official/akt.py";spec=importlib.util.spec_from_file_location("eval_official_akt",source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.device=device
 return mod.AKT(config.num_items,config.num_items,config.embed_dim,1,1,config.dropout,"akt",max(64,config.embed_dim*2),config.num_heads,max(128,config.embed_dim*4),1e-5,False).to(device)
def main():
 a=parse();device=torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"));ckpt=torch.load(a.checkpoint,map_location=device);meta=ckpt.get("training_metadata") or {}
 if not meta.get("leakage_safe"):raise ValueError("checkpoint is not leakage-safe")
 config=NeuralKTConfig(**ckpt["config"]);item_map={str(k):int(v) for k,v in ckpt["item_id_map"].items()};method=ckpt["method"]
 model=(SAKT(config).to(device) if method=="sakt" else DeepIRT(config).to(device) if method=="deepirt" else make_akt(config,device))
 model.load_state_dict(ckpt["state_dict"]);model.eval();excluded=set(map(str,meta.get("excluded_user_ids",[])));records=[]
 for batch,file in enumerate(a.cohort_file):
  cohort=json.load(open(file,encoding="utf-8-sig"));hist={str(x["uid"]):clean_sequence(x) for x in cohort["history_rows"]};tgt={str(x["uid"]):clean_sequence(x) for x in cohort["target_rows"]}
  for offset,raw_uid in enumerate(cohort["uids"]):
   uid=str(raw_uid)
   if uid not in excluded:raise AssertionError(f"target {uid} was not excluded")
   encoded=[(item_map.get(str(x["qid"])),int(x["response"])) for x in hist[uid]]
   if any(item is None for item,_ in encoded) or len(encoded)!=90:raise AssertionError(f"history mapping/length invalid for {uid}")
   target=tgt[uid]
   mapped=[item_map.get(str(x["qid"])) for x in target]
   if any(item is None for item in mapped) or len(target)!=10:raise AssertionError(f"target mapping/length invalid for {uid}")
   records.append({"uid":uid,"q":[x[0] for x in encoded],"r":[x[1] for x in encoded],"target":target,"mapped":mapped,"rng":random.Random(a.seed+batch*10000+offset)})
 if len(records)!=500 or len({x["uid"] for x in records})!=500:raise AssertionError("cohort coverage must be exactly 500 unique users")
 steps=[]
 for index in range(10):
  q=torch.tensor([[*(record["q"][-(config.max_seq_len-1):]), record["mapped"][index]] for record in records],dtype=torch.long,device=device)
  r=torch.tensor([[*(record["r"][-(config.max_seq_len-1):]), 0] for record in records],dtype=torch.long,device=device)
  with torch.no_grad():
   if method=="sakt": probabilities=model(q,r)[:,-1]
   elif method=="deepirt": probabilities=model(q,r)[0][:,-1]
   else:
    qa=q+r*config.num_items;target=torch.zeros_like(q,dtype=torch.float32);_,values,_=model(q,qa,target,q);probabilities=values.reshape_as(q)[:,-1]
  for record,probability in zip(records,probabilities.detach().cpu().tolist()):
   prob=float(probability);sim=int(record["rng"].random()<prob);item=record["target"][index]
   # Teacher forcing is the default formal protocol: prior target responses
   # are observed context, whereas rollout feeds the simulator's own output.
   next_response=int(item["response"]) if a.feedback_mode=="teacher_forcing" else sim
   record["q"].append(record["mapped"][index]);record["r"].append(next_response)
   steps.append({"uid":record["uid"],"step_index":index,"qid":int(item["qid"]),"real_response":int(item["response"]),"simulated_response":sim,"probability":prob,"feedback_mode":a.feedback_mode})
 ref=json.load(open(a.full_irt_reference,encoding="utf-8-sig"));
 if len(steps)!=5000:raise AssertionError(f"coverage steps={len(steps)}")
 result={"method":method,"checkpoint":a.checkpoint,"cohort_files":a.cohort_file,"training_metadata":meta,"audit":{"users":500,"steps":5000,"raw_step_pooled":True,"feedback_mode":a.feedback_mode},"formal_metrics_v6":evaluate_formal_response_metrics(steps,ref["fixed_full_irt_2x2_partition"]),"all_steps":steps}
 Path(a.output).parent.mkdir(parents=True,exist_ok=True);json.dump(result,open(a.output,"w"),ensure_ascii=False,indent=2);print(json.dumps({**result["audit"],**{k:result["formal_metrics_v6"][k] for k in ["baa","balanced_accuracy","f1","adcde"]}},ensure_ascii=False),flush=True)
if __name__=="__main__":main()

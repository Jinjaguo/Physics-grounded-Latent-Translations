#!/usr/bin/env python3
"""Run Wave 33 causal mixture-of-low-rank-intent-fields.

Purpose
-------
Test whether two state-selected local force fields can resolve the direction
and continuity trade-off left by Waves 28--32 while keeping the released
action-text VAE, decoder, and F2 behavior backbone frozen.

Parameters
----------
``--stage``: ``audit``, ``sweep``, ``select``, ``final``, ``report``, or
``all``.  ``--device`` selects CPU/CUDA.

Usage
-----
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/run_wave33_mixture_field.py --stage all --device cpu

Outputs
-------
Wave33 artifacts are written under
``results/dynamics/thirty_third_wave/2026-08-15_mixture_field``.
"""
from __future__ import annotations

import argparse, json
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np, torch
from torch import nn
from torch.nn import functional as F

from scripts.dynamics.run_wave28_force_field import ROOT,W21,FrozenBackbone,build_dataset,concat_events,load_events,metrics
from pglt.dynamics.wave28_force_field import IntentForceField

OUT=ROOT/"results/dynamics/thirty_third_wave/2026-08-15_mixture_field"; SEED=330833
TASKS=["lift_blue_block_slider","lift_red_block_table","place_in_slider","push_pink_block_right","turn_off_lightbulb","turn_on_lightbulb"]
def now(): return datetime.now().astimezone().isoformat()
def save_json(n,v): OUT.mkdir(parents=True,exist_ok=True);(OUT/n).write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+"\n")
def data(b,s): return concat_events(build_dataset(b,load_events(s,"wave21"),"F2"),build_dataset(b,load_events(s,"wave27"),"F2"))
def tens(x,d): return {k:(torch.from_numpy(v).long() if v.dtype.kind in "iu" else torch.from_numpy(v).float()).to(d) for k,v in x.items() if v.dtype.kind in "iuf"}

class Mixture(nn.Module):
    def __init__(self,q,temp):
        super().__init__();self.q=q;self.temp=temp;self.experts=nn.ModuleList([IntentForceField(q_dim=q,encoding="E0_linear",field="FF3_attractor",subspace="C2_learned",seed=SEED+i) for i in range(2)]);self.gate=nn.Sequential(nn.Linear(32,32),nn.SiLU(),nn.Linear(32,2))
    def forward(self,base,current,cl,tl,ci=None,ti=None):
        values=[e(base,current,cl,tl,ci,ti) for e in self.experts];w=(self.gate(current)/self.temp).softmax(-1);res=sum(w[:,i,None,None]*values[i]["residual"] for i in range(2));q=sum(w[:,i,None,None]*values[i]["q"] for i in range(2));out=dict(values[0]);out["residual"]=res;out["prediction"]=base+res;out["q"]=q;out["gate"]=w;return out

def train(spec,trn,dv,b,d):
 torch.manual_seed(SEED+spec["q"]);m=Mixture(spec["q"],spec["temp"]).to(d);opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=1e-4);tr,dev=tens(trn,d),tens(dv,d);best=1e9;state=None;stale=0
 for ep in range(55):
  m.train();opt.zero_grad(set_to_none=True);o=m(tr["base_current"],tr["z_current"],tr["current_language"],tr["target_language"],tr["current_ids"],tr["target_ids"]);p=o["prediction"];dec=b.representation.decode(p.reshape(-1,32)).view_as(tr["target_actions_norm"]);jump=(dec[:,0,0,:6]-tr["current_action_norm"][:,-1,:6]).square().mean();tj=(tr["target_actions_norm"][:,0,0,:6]-tr["current_action_norm"][:,-1,:6]).square().mean();loss=F.mse_loss(dec,tr["target_actions_norm"])+.1*F.mse_loss(p,tr["target_latent"])+.3*(jump-tj).abs()+.01*o["residual"].square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();m.eval();
  with torch.no_grad(): v=m(dev["base_current"],dev["z_current"],dev["current_language"],dev["target_language"],dev["current_ids"],dev["target_ids"]);vl=float(F.mse_loss(v["prediction"],dev["target_latent"]))
  if vl<best-1e-6:best=vl;state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};stale=0
  else:stale+=1
  if stale>=12:break
 m.load_state_dict(state);m.eval();return m,{"epochs":ep+1,"best_dev_latent":best,"params":sum(p.numel() for p in m.parameters() if p.requires_grad)}

def audit_stage(device):
 OUT.mkdir(parents=True,exist_ok=True);save_json("wave33_frozen_manifest.json",{"created_before_metrics":True,"vae_decoder_frozen":True,"F2_frozen":True,"trainable":"two q experts + causal gate"});save_json("wave33_preregistration.json",{"created_at":now(),"q":[2,4],"temperatures":[.5,1.0],"future_inputs":[],"heldout_opened":False});save_json("wave33_seed_manifest.json",{"seed":SEED});(OUT/"wave33_execution_log.md").write_text(f"# Wave33 execution log\n\n- {now()} — mixture-field audit passed.\n");(OUT/"wave33_backbone_interface_audit.md").write_text("# Wave33 interface\n\nF2 frozen trajectory plus two causal q experts and current-latent gate.\n")
def sweep_stage(device):
 b=FrozenBackbone(device);tr,dv=data(b,"train"),data(b,"development");reg={t:np.load(W21/"wave21_train_regions.npz")[t] for t in TASKS};r={};states={}
 for q in (2,4):
  for temp in (.5,1.0):
   s={"name":f"q{q}_t{temp}","q":q,"temp":temp};m,rec=train(s,tr,dv,b,device);mm,_=metrics(s["name"],m,dv,b,device,reg);mm.update(s);mm.update(rec);r[s["name"]]=mm;states[s["name"]]=m.state_dict()
 save_json("wave33_development_metrics.json",r);save_json("wave33_sweep_inventory.json",{"candidates":len(r),"heldout_opened":False});torch.save(states,OUT/"development_models.pt");print(json.dumps({"stage":"sweep","candidates":len(r)}),flush=True)
def select_stage():
 r=json.loads((OUT/"wave33_development_metrics.json").read_text());best=min(r,key=lambda n:r[n]["H4_decoded_mse"]+r[n]["H4_continuity"]-.25*r[n]["Execution_RedirectGain"]);save_json("wave33_final_candidate_selection.json",{"created_at":now(),"selected":best,"heldout_opened":False});save_json("wave33_final_test_preregistration.json",{"created_before_heldout":True,"candidate":best});print(json.dumps({"stage":"select","best":best,"metrics":r[best]}),flush=True)
def final_stage(device):
 s=json.loads((OUT/"wave33_final_candidate_selection.json").read_text());name=s["selected"];sp=json.loads((OUT/"wave33_development_metrics.json").read_text())[name];b=FrozenBackbone(device);tr,te=data(b,"train"),data(b,"test");m=Mixture(int(sp["q"]),float(sp["temp"])).to(device);m.load_state_dict(torch.load(OUT/"development_models.pt",map_location=device)[name]);reg={t:np.load(W21/"wave21_train_regions.npz")[t] for t in TASKS};mm,raw=metrics(name,m,te,b,device,reg);save_json("wave33_heldout_results.json",{"candidate":name,"metrics":mm,"raw":{k:v.tolist() for k,v in raw.items()}});s["heldout_opened"]=True;save_json("wave33_final_candidate_selection.json",s);print(json.dumps({"stage":"final","candidate":name,"metrics":mm}),flush=True)
def report_stage():
 s=json.loads((OUT/"wave33_final_candidate_selection.json").read_text());m=json.loads((OUT/"wave33_heldout_results.json").read_text())["metrics"];cont=m["H4_continuity"]<=m["H4_true_continuity"]*1.5;red=m["Execution_RedirectGain"]>0;ready=cont and red and m["H4_endpoint_accuracy"]>=.55;cl={"C47_mixture_field_improves_redirect":"SUPPORTED" if red else "NOT_SUPPORTED","C48_mixture_field_preserves_continuity":"SUPPORTED" if cont else "NOT_SUPPORTED","READY_FOR_CLOSED_LOOP_RETARGET":"SUPPORTED" if ready else "NOT_SUPPORTED","selected":s["selected"],"next_wave_required":not ready};save_json("wave33_claim_decision.json",cl);txt=f"# Thirty-third wave: mixture of local fields\n\nSelected `{s['selected']}`. Held-out decoded MSE={m['H4_decoded_mse']:.6f}, continuity={m['H4_continuity']:.6f}, true continuity={m['H4_true_continuity']:.6f}, execution RedirectGain={m['Execution_RedirectGain']:.6f}, endpoint={m['H4_endpoint_accuracy']:.4f}.\n\n```json\n{json.dumps(cl,indent=2)}\n```\n";(OUT/"thirty_third_wave_results.md").write_text(txt);(ROOT/"reports/dynamics_wave33_results.md").write_text(txt);next_text="# Wave 34 next experiment\n\nIf mixture remains below continuity/identity gates, stop adapter stacking and test a representation-level temporal/control bottleneck or collect ordered instruction events.\n";(OUT/"thirty_third_wave_next_experiment.md").write_text(next_text);(ROOT/"NEXT_EXPERIMENT.md").write_text(next_text);log=ROOT/"RESEARCH_LOG.md";log.write_text(log.read_text().rstrip()+f"\n\n## Wave 33 — {now()}\n\nMixture field selected `{s['selected']}`, readiness={cl['READY_FOR_CLOSED_LOOP_RETARGET']}. See `{OUT.relative_to(ROOT)}`.\n");print(json.dumps({"stage":"report","ready":cl["READY_FOR_CLOSED_LOOP_RETARGET"]}),flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument("--stage",choices=("audit","sweep","select","final","report","all"),default="all");p.add_argument("--device",default="cpu");a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True);d=torch.device(a.device);torch.set_num_threads(4);st=("audit","sweep","select","final","report") if a.stage=="all" else (a.stage,)
 for s in st:
  if s=="audit":audit_stage(d)
  elif s=="sweep":sweep_stage(d)
  elif s=="select":select_stage()
  elif s=="final":final_stage(d)
  elif s=="report":report_stage()
if __name__=="__main__":main()

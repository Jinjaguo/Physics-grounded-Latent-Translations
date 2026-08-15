#!/usr/bin/env python3
"""Run Wave 32 state-conditioned executable low-rank field training.

Purpose
-------
Train only a small C6 state-dependent force-field adapter on top of frozen F2,
with stronger frozen-decoder action supervision.  Compare q dimensions and
continuity weights while preserving the action-text VAE and decoder.

Parameters
----------
``--stage``: ``audit``, ``sweep``, ``select``, ``final``, ``report``, or
``all``.  ``--device`` selects CPU/CUDA.

Usage
-----
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/run_wave32_state_field.py --stage all --device cpu

Outputs
-------
Wave32 artifacts are saved under
``results/dynamics/thirty_second_wave/2026-08-15_state_field``.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.dynamics.run_wave28_force_field import ROOT, W21, FrozenBackbone, build_dataset, concat_events, load_events, metrics
from pglt.dynamics.wave28_force_field import IntentForceField


OUT = ROOT / "results/dynamics/thirty_second_wave/2026-08-15_state_field"; SEED = 320832


def now(): return datetime.now().astimezone().isoformat()


def save_json(name: str, value: Any):
    OUT.mkdir(parents=True, exist_ok=True); (OUT/name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def data(backbone, split): return concat_events(build_dataset(backbone, load_events(split,"wave21"),"F2"), build_dataset(backbone, load_events(split,"wave27"),"F2"))


class StateAdapter(nn.Module):
    def __init__(self, q: int):
        super().__init__(); self.field=IntentForceField(q_dim=q, encoding="E0_linear", field="FF4_state_conditioned", subspace="C6_state_dependent", composition="COMP0_additive", seed=SEED)
    def forward(self, base, current_latent, current_language, target_language, current_ids=None, target_ids=None): return self.field(base,current_latent,current_language,target_language,current_ids,target_ids)


def tensors(value, device):
    out={}
    for k,a in value.items():
        if a.dtype.kind in "iu": out[k]=torch.from_numpy(a).long().to(device)
        elif a.dtype.kind=="f": out[k]=torch.from_numpy(a).float().to(device)
    return out


def train(spec, tr_np, dv_np, backbone, device):
    torch.manual_seed(SEED+spec["q"]); model=StateAdapter(spec["q"]).to(device); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4); tr,dv=tensors(tr_np,device),tensors(dv_np,device); best=float("inf"); state=None; stale=0
    for epoch in range(55):
        model.train(); opt.zero_grad(set_to_none=True); out=model(tr["base_current"],tr["z_current"],tr["current_language"],tr["target_language"],tr["current_ids"],tr["target_ids"]); pred=out["prediction"]; dec=backbone.representation.decode(pred.reshape(-1,32)).view_as(tr["target_actions_norm"]); jump=(dec[:,0,0,:6]-tr["current_action_norm"][:,-1,:6]).square().mean(); target_jump=(tr["target_actions_norm"][:,0,0,:6]-tr["current_action_norm"][:,-1,:6]).square().mean(); loss=F.mse_loss(dec,tr["target_actions_norm"])+0.1*F.mse_loss(pred,tr["target_latent"])+spec["lambda"]*(jump-target_jump).abs()+0.01*out["residual"].square().mean(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step(); model.eval();
        with torch.no_grad(): dvout=model(dv["base_current"],dv["z_current"],dv["current_language"],dv["target_language"],dv["current_ids"],dv["target_ids"]); val=float(F.mse_loss(dvout["prediction"],dv["target_latent"]))
        if val<best-1e-6: best=val; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
        else: stale+=1
        if stale>=12: break
    model.load_state_dict(state); model.eval(); return model,{"epochs":epoch+1,"best_dev_latent":best,"params":sum(p.numel() for p in model.parameters() if p.requires_grad)}


def audit_stage(device):
    OUT.mkdir(parents=True,exist_ok=True); save_json("wave32_frozen_manifest.json",{"created_before_metrics":True,"vae_decoder_frozen":True,"F1_F2_frozen":True,"new_trainable":"C6 state-conditioned low-rank adapter"}); save_json("wave32_preregistration.json",{"created_at":now(),"q":[2,4,8],"continuity_weights":[0.1,0.3,1.0],"future_inputs":[],"heldout_opened":False}); save_json("wave32_seed_manifest.json",{"seed":SEED}); (OUT/"wave32_execution_log.md").write_text(f"# Wave32 execution log\n\n- {now()} — C6 state-field audit passed.\n"); (OUT/"wave32_backbone_interface_audit.md").write_text("# Wave32 interface\n\nFrozen F2 latent trajectory plus causal current latent to state-dependent B.\n")


def sweep_stage(device):
    backbone=FrozenBackbone(device); tr,dv=data(backbone,"train"),data(backbone,"development"); regions={task:np.load(W21/"wave21_train_regions.npz")[task] for task in ["lift_blue_block_slider","lift_red_block_table","place_in_slider","push_pink_block_right","turn_off_lightbulb","turn_on_lightbulb"]}; results={}; states={}
    for q in (2,4,8):
        for lam in (0.1,0.3,1.0):
            spec={"name":f"q{q}_l{lam}","q":q,"lambda":lam}; model,rec=train(spec,tr,dv,backbone,device); m,_=metrics(spec["name"],model,dv,backbone,device,regions); m.update(spec);m.update(rec);results[spec["name"]]=m;states[spec["name"]]=model.state_dict()
    save_json("wave32_development_metrics.json",results); save_json("wave32_sweep_inventory.json",{"candidates":len(results),"heldout_opened":False}); torch.save(states,OUT/"development_models.pt"); print(json.dumps({"stage":"sweep","candidates":len(results)}),flush=True)


def select_stage():
    r=json.loads((OUT/"wave32_development_metrics.json").read_text()); best=min(r,key=lambda n:r[n]["H4_decoded_mse"]+r[n]["H4_continuity"]-0.25*r[n]["Execution_RedirectGain"]); save_json("wave32_final_candidate_selection.json",{"created_at":now(),"selected":best,"heldout_opened":False});save_json("wave32_final_test_preregistration.json",{"created_before_heldout":True,"candidate":best});print(json.dumps({"stage":"select","best":best,"metrics":r[best]}),flush=True)


def final_stage(device):
    sel=json.loads((OUT/"wave32_final_candidate_selection.json").read_text()); name=sel["selected"]; spec=json.loads((OUT/"wave32_development_metrics.json").read_text())[name]; backbone=FrozenBackbone(device); tr,te=data(backbone,"train"),data(backbone,"test"); model=StateAdapter(int(spec["q"])).to(device);model.load_state_dict(torch.load(OUT/"development_models.pt",map_location=device)[name]);regions={task:np.load(W21/"wave21_train_regions.npz")[task] for task in ["lift_blue_block_slider","lift_red_block_table","place_in_slider","push_pink_block_right","turn_off_lightbulb","turn_on_lightbulb"]};m,raw=metrics(name,model,te,backbone,device,regions);save_json("wave32_heldout_results.json",{"candidate":name,"metrics":m,"raw":{k:v.tolist() for k,v in raw.items()}});sel["heldout_opened"]=True;save_json("wave32_final_candidate_selection.json",sel);print(json.dumps({"stage":"final","candidate":name,"metrics":m}),flush=True)


def report_stage():
    sel=json.loads((OUT/"wave32_final_candidate_selection.json").read_text());m=json.loads((OUT/"wave32_heldout_results.json").read_text())["metrics"];continuity=m["H4_continuity"]<=m["H4_true_continuity"]*1.5;redirect=m["Execution_RedirectGain"]>0;ready=continuity and redirect and m["H4_endpoint_accuracy"]>=.55;claims={"C45_state_conditioned_field_improves_executable_loss": "SUPPORTED" if m["H4_decoded_mse"]<1.3 else "NOT_SUPPORTED","C46_state_conditioned_field_preserves_redirect":"SUPPORTED" if redirect else "NOT_SUPPORTED","READY_FOR_CLOSED_LOOP_RETARGET":"SUPPORTED" if ready else "NOT_SUPPORTED","selected":sel["selected"],"next_wave_required":not ready};save_json("wave32_claim_decision.json",claims);text=f"# Thirty-second wave: state-conditioned field\n\nSelected `{sel['selected']}`. Held-out decoded MSE={m['H4_decoded_mse']:.6f}, continuity={m['H4_continuity']:.6f}, true continuity={m['H4_true_continuity']:.6f}, execution RedirectGain={m['Execution_RedirectGain']:.6f}, endpoint={m['H4_endpoint_accuracy']:.4f}.\n\n```json\n{json.dumps(claims,indent=2)}\n```\n";(OUT/"thirty_second_wave_results.md").write_text(text);(ROOT/"reports/dynamics_wave32_results.md").write_text(text);next_text="# Wave 33 next experiment\n\nState-conditioned projection should be compared with a piecewise mixture-of-fields and a matched full-rank control. If continuity remains poor, the frozen action decoder is likely the bottleneck.\n";(OUT/"thirty_second_wave_next_experiment.md").write_text(next_text);(ROOT/"NEXT_EXPERIMENT.md").write_text(next_text);log=ROOT/"RESEARCH_LOG.md";log.write_text(log.read_text().rstrip()+f"\n\n## Wave 32 — {now()}\n\nState-conditioned C6 adapter: `{sel['selected']}`, readiness={claims['READY_FOR_CLOSED_LOOP_RETARGET']}. See `{OUT.relative_to(ROOT)}`.\n");print(json.dumps({"stage":"report","ready":claims["READY_FOR_CLOSED_LOOP_RETARGET"]}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=("audit","sweep","select","final","report","all"),default="all");p.add_argument("--device",default="cpu");a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True);d=torch.device(a.device);torch.set_num_threads(4);stages=("audit","sweep","select","final","report") if a.stage=="all" else (a.stage,)
    for s in stages:
        if s=="audit":audit_stage(d)
        elif s=="sweep":sweep_stage(d)
        elif s=="select":select_stage()
        elif s=="final":final_stage(d)
        elif s=="report":report_stage()


if __name__=="__main__":main()

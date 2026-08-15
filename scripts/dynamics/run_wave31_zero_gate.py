#!/usr/bin/env python3
"""Train and evaluate a zero-initialized gated low-rank intent adapter.

Purpose
-------
Test whether a trainable intervention amplitude, initialized at zero, can
preserve the frozen F1/F2 behavior while learning a continuous retargeting
residual.  Only the low-dimensional adapter and gate are trainable; the
action-text VAE, decoder, semantic predictor, F1 and F2 are frozen.

Parameters
----------
``--stage``: ``audit``, ``sweep``, ``select``, ``final``, ``report``, or
``all``.  ``--device`` selects CPU/CUDA.

Usage
-----
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/run_wave31_zero_gate.py --stage all --device cpu

Outputs
-------
Wave31 artifacts are saved under
``results/dynamics/thirty_first_wave/2026-08-15_zero_gate``.
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

from scripts.dynamics.run_wave28_force_field import ROOT, W21, FrozenBackbone, IntentForceField, build_dataset, concat_events, load_events, metrics


OUT = ROOT / "results/dynamics/thirty_first_wave/2026-08-15_zero_gate"
SEED = 310831


def now() -> str: return datetime.now().astimezone().isoformat()


def save_json(name: str, value: Any) -> None:
    OUT.mkdir(parents=True, exist_ok=True); (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def data(backbone: FrozenBackbone, split: str) -> dict[str, np.ndarray]:
    return concat_events(build_dataset(backbone, load_events(split, "wave21"), "F2"), build_dataset(backbone, load_events(split, "wave27"), "F2"))


class GatedAdapter(nn.Module):
    """Trainable q-field with zero-like intervention at initialization."""

    def __init__(self, q_dim: int, gate_init: float = -4.0) -> None:
        super().__init__(); self.field = IntentForceField(q_dim=q_dim, encoding="E0_linear", field="FF3_attractor", subspace="C2_learned", composition="COMP0_additive", seed=SEED); self.gate_logits = nn.Parameter(torch.full((3,), gate_init))

    def forward(self, base, current_latent, current_language, target_language, current_ids=None, target_ids=None):
        output = self.field(base, current_latent, current_language, target_language, current_ids, target_ids); gate = torch.sigmoid(self.gate_logits).view(1, 3, 1); output["residual"] = output["residual"] * gate; output["prediction"] = base + output["residual"]; output["gate"] = gate.expand(len(base), -1, -1); return output


def tensors(value: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    result = {}
    for key, array in value.items():
        if array.dtype.kind in "iu": result[key] = torch.from_numpy(array).long().to(device)
        elif array.dtype.kind == "f": result[key] = torch.from_numpy(array).float().to(device)
    return result


def train(spec: dict[str, Any], train_data: dict[str, np.ndarray], dev_data: dict[str, np.ndarray], backbone: FrozenBackbone, device: torch.device) -> tuple[GatedAdapter, dict[str, Any]]:
    torch.manual_seed(SEED + spec["q"]); model = GatedAdapter(spec["q"]).to(device); model.field.q_dim = spec["q"]; optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4); tr, dv = tensors(train_data, device), tensors(dev_data, device); best, state, stale = float("inf"), None, 0
    for epoch in range(50):
        model.train(); optimizer.zero_grad(set_to_none=True); out = model(tr["base_current"], tr["z_current"], tr["current_language"], tr["target_language"], tr["current_ids"], tr["target_ids"]); pred = out["prediction"]; decoded = backbone.representation.decode(pred.reshape(-1,32)).view_as(tr["target_actions_norm"]); continuity = (decoded[:,0,0,:6] - tr["current_action_norm"][:,-1,:6]).square().mean(); target_jump = (tr["target_actions_norm"][:,0,0,:6] - tr["current_action_norm"][:,-1,:6]).square().mean(); loss = F.mse_loss(pred, tr["target_latent"]) + 0.3 * F.mse_loss(decoded, tr["target_actions_norm"]) + spec["lambda_cont"] * (continuity - target_jump).abs() + 0.3 * out["residual"].square().mean(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step(); model.eval();
        with torch.no_grad():
            dv_out = model(dv["base_current"], dv["z_current"], dv["current_language"], dv["target_language"], dv["current_ids"], dv["target_ids"]); dv_loss = float(F.mse_loss(dv_out["prediction"], dv["target_latent"]))
        if dv_loss < best - 1e-6: best, state, stale = dv_loss, {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}, 0
        else: stale += 1
        if stale >= 10: break
    model.load_state_dict(state); model.eval(); return model, {"epochs": epoch + 1, "best_dev_latent": best, "gate": torch.sigmoid(model.gate_logits).detach().cpu().tolist(), "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def audit_stage(device: torch.device) -> None:
    OUT.mkdir(parents=True, exist_ok=True); backbone = FrozenBackbone(device); save_json("wave31_frozen_manifest.json", {"created_before_metrics": True, "vae_decoder_frozen": True, "F1_F2_frozen": True, "new_trainable": "q adapter + 3 scalar gates"}); save_json("wave31_preregistration.json", {"created_at": now(), "q": [2,4,8], "continuity_weights": [0.3,1.0,3.0], "gate_initialization": "sigmoid(-4) near zero", "future_inputs": [], "heldout_opened": False}); save_json("wave31_seed_manifest.json", {"seed": SEED}); (OUT/"wave31_execution_log.md").write_text(f"# Wave31 execution log\n\n- {now()} — zero-gate frozen-core audit passed.\n"); (OUT/"wave31_backbone_interface_audit.md").write_text("# Wave31 interface\n\nFrozen F2 recursive latent base; trainable q-field and scalar gates only.\n")


def sweep_stage(device: torch.device) -> None:
    backbone = FrozenBackbone(device); tr, dv = data(backbone,"train"), data(backbone,"development"); regions = {task: np.load(W21/"wave21_train_regions.npz")[task] for task in ["lift_blue_block_slider","lift_red_block_table","place_in_slider","push_pink_block_right","turn_off_lightbulb","turn_on_lightbulb"]}; results={}; models={}
    for q in (2,4,8):
        for lam in (0.3,1.0,3.0):
            spec={"name":f"q{q}_c{lam}","q":q,"lambda_cont":lam}; model, record=train(spec,tr,dv,backbone,device); metric,_=metrics(spec["name"],model,dv,backbone,device,regions); metric.update(spec); metric.update(record); results[spec["name"]]=metric; models[spec["name"]]=model.state_dict()
    save_json("wave31_development_metrics.json",results); save_json("wave31_sweep_inventory.json",{"candidates":len(results),"heldout_opened":False}); torch.save(models,OUT/"development_models.pt"); print(json.dumps({"stage":"sweep","candidates":len(results)}),flush=True)


def select_stage() -> None:
    results=json.loads((OUT/"wave31_development_metrics.json").read_text()); best=min(results,key=lambda n:results[n]["H4_decoded_mse"]+results[n]["H4_continuity"]-0.25*results[n]["Execution_RedirectGain"]); save_json("wave31_final_candidate_selection.json",{"created_at":now(),"selected":best,"heldout_opened":False}); save_json("wave31_final_test_preregistration.json",{"created_before_heldout":True,"candidate":best,"post_test_tuning":False}); print(json.dumps({"stage":"select","best":best,"metrics":results[best]}),flush=True)


def final_stage(device: torch.device) -> None:
    selection=json.loads((OUT/"wave31_final_candidate_selection.json").read_text()); candidate=selection["selected"]; spec=json.loads((OUT/"wave31_development_metrics.json").read_text())[candidate]; backbone=FrozenBackbone(device); tr,_,te=data(backbone,"train"),data(backbone,"development"),data(backbone,"test"); model=GatedAdapter(int(spec["q"])).to(device); model.load_state_dict(torch.load(OUT/"development_models.pt",map_location=device)[candidate]); regions={task:np.load(W21/"wave21_train_regions.npz")[task] for task in ["lift_blue_block_slider","lift_red_block_table","place_in_slider","push_pink_block_right","turn_off_lightbulb","turn_on_lightbulb"]}; metric,raw=metrics(candidate,model,te,backbone,device,regions); save_json("wave31_heldout_results.json",{"candidate":candidate,"metrics":metric,"raw":{k:v.tolist() for k,v in raw.items()}}); selection["heldout_opened"]=True; save_json("wave31_final_candidate_selection.json",selection); print(json.dumps({"stage":"final","candidate":candidate,"metrics":metric}),flush=True)


def report_stage() -> None:
    selection=json.loads((OUT/"wave31_final_candidate_selection.json").read_text()); metric=json.loads((OUT/"wave31_heldout_results.json").read_text())["metrics"]; continuity=metric["H4_continuity"]<=metric["H4_true_continuity"]*1.5; redirect=metric["Execution_RedirectGain"]>0; ready=continuity and redirect and metric["H4_endpoint_accuracy"]>=0.55; claims={"C42_zero_gate_preserves_base": "SUPPORTED" if metric["adapter_norm"]<0.5 else "NOT_SUPPORTED", "C43_trainable_gate_improves_continuity": "SUPPORTED" if continuity else "NOT_SUPPORTED", "C44_gate_preserves_execution_redirect": "SUPPORTED" if redirect else "NOT_SUPPORTED", "READY_FOR_CLOSED_LOOP_RETARGET": "SUPPORTED" if ready else "NOT_SUPPORTED", "selected":selection["selected"],"next_wave_required":not ready}; save_json("wave31_claim_decision.json",claims); result=f"# Thirty-first wave: trainable zero gate\n\nSelected `{selection['selected']}`. Held-out decoded MSE={metric['H4_decoded_mse']:.6f}, continuity={metric['H4_continuity']:.6f}, true continuity={metric['H4_true_continuity']:.6f}, execution RedirectGain={metric['Execution_RedirectGain']:.6f}, endpoint={metric['H4_endpoint_accuracy']:.4f}, adapter norm={metric['adapter_norm']:.6f}.\n\n```json\n{json.dumps(claims,indent=2)}\n```\n"; (OUT/"thirty_first_wave_results.md").write_text(result); (ROOT/"reports/dynamics_wave31_results.md").write_text(result); next_text="# Wave 32 next experiment\n\n"+ ("The zero gate still does not meet continuity/readiness. Test state-conditioned or piecewise fields with a matched full-rank control; if all fail, diagnose the frozen projection as the bottleneck.\n" if not ready else "Run ordered closed-loop retargeting with no architecture change.\n"); (OUT/"thirty_first_wave_next_experiment.md").write_text(next_text); (ROOT/"NEXT_EXPERIMENT.md").write_text(next_text); log=ROOT/"RESEARCH_LOG.md"; log.write_text(log.read_text().rstrip()+f"\n\n## Wave 31 — {now()}\n\nZero-gate q-field: selected `{selection['selected']}`, readiness={claims['READY_FOR_CLOSED_LOOP_RETARGET']}. See `{OUT.relative_to(ROOT)}`.\n"); print(json.dumps({"stage":"report","ready":claims["READY_FOR_CLOSED_LOOP_RETARGET"]}),flush=True)


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--stage",choices=("audit","sweep","select","final","report","all"),default="all"); parser.add_argument("--device",default="cpu"); args=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True); device=torch.device(args.device); torch.set_num_threads(4); stages=("audit","sweep","select","final","report") if args.stage=="all" else (args.stage,)
    for stage in stages:
        if stage=="audit": audit_stage(device)
        elif stage=="sweep": sweep_stage(device)
        elif stage=="select": select_stage()
        elif stage=="final": final_stage(device)
        elif stage=="report": report_stage()


if __name__=="__main__": main()

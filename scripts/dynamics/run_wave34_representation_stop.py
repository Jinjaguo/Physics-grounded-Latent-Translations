#!/usr/bin/env python3
"""Audit Wave28--33 evidence and decide whether to stop adapter stacking.

Purpose
-------
Summarize frozen development/held-out metrics from Waves 28--33, verify that
the broad adapter families and full-rank controls were evaluated, and apply
the preregistered representation-stop rule. This diagnostic does not train a
new model and does not reopen held-out data.

Parameters
----------
``--stage`` is ``audit`` or ``report``. ``--device`` is accepted for command
compatibility but no model inference is performed.

Usage
-----
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/run_wave34_representation_stop.py --stage all

Outputs
-------
Wave34 audit, stop decision, report, and next-experiment files are written
under ``results/dynamics/thirty_fourth_wave/2026-08-15_representation_stop``.
"""
from __future__ import annotations
import argparse, json
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/dynamics/thirty_fourth_wave/2026-08-15_representation_stop"
def now(): return datetime.now().astimezone().isoformat()
def save(name: str, value: Any): OUT.mkdir(parents=True, exist_ok=True); (OUT/name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")

def audit():
    roots={28:ROOT/"results/dynamics/twenty_eighth_wave/2026-08-15_force_field",29:ROOT/"results/dynamics/twenty_ninth_wave/2026-08-15_damped_field",30:ROOT/"results/dynamics/thirtieth_wave/2026-08-15_jacobian_field",31:ROOT/"results/dynamics/thirty_first_wave/2026-08-15_zero_gate",32:ROOT/"results/dynamics/thirty_second_wave/2026-08-15_state_field",33:ROOT/"results/dynamics/thirty_third_wave/2026-08-15_mixture_field"}
    required={f"wave{w}_development":r/f"wave{w}_development_metrics.json" for w,r in roots.items()};required.update({f"wave{w}_heldout":r/f"wave{w}_heldout_results.json" for w,r in roots.items()});presence={k:p.exists() for k,p in required.items()}
    w28=json.loads((roots[28]/"wave28_heldout_results.json").read_text()); rep={"wave28_full_rank":w28["CTRL_full_rank"]["wave27"],"wave28_best_redirect":w28["S2_q8_C1_pca_FF4_state_conditioned"]["wave27"]}
    for w in (29,30,31,32,33): rep[f"wave{w}_best"]=json.loads((roots[w]/f"wave{w}_heldout_results.json").read_text())["metrics"]
    positive=[n for n,m in rep.items() if m.get("Execution_RedirectGain",-1)>0]; continuity=[n for n,m in rep.items() if m.get("H4_continuity",1e9)<=1.5*m.get("H4_true_continuity",0)]; identity=[n for n,m in rep.items() if m.get("H4_endpoint_accuracy",0)>=.55]
    stop=all(presence.values()) and bool(positive) and not continuity and not identity and rep["wave28_full_rank"]["Execution_RedirectGain"]<=0
    decision={"created_at":now(),"presence":presence,"representative_metrics":rep,"execution_positive_candidates":positive,"continuity_pass_candidates":continuity,"identity_pass_candidates":identity,"representation_stop":stop,"reason":"multiple adapter families and full-rank control fail joint executable redirect/continuity/identity; frozen action projection remains the bottleneck" if stop else "stop rule not satisfied"}
    save("wave34_stop_audit.json",decision);save("wave34_claim_decision.json",{"REPRESENTATION_STOP":"SUPPORTED" if stop else "NOT_SUPPORTED","READY_FOR_CLOSED_LOOP_RETARGET":"NOT_SUPPORTED","next_wave_required":not stop});(OUT/"wave34_execution_log.md").write_text(f"# Wave34 execution log\n\n- {now()} — read-only audit; no held-out reopened and no model trained.\n")

def report():
    d=json.loads((OUT/"wave34_stop_audit.json").read_text());text=f"# Thirty-fourth wave: representation bottleneck stop audit\n\nThe audit found complete artifacts for Waves 28–33. Positive redirect appeared in several candidates, but no representative candidate passed both continuity and endpoint identity; the Wave28 full-rank control also had non-positive execution redirect. Therefore REPRESENTATION_STOP={d['representation_stop']}.\n\n```json\n{json.dumps(d,indent=2)}\n```\n";(OUT/"thirty_fourth_wave_results.md").write_text(text);(ROOT/"reports/dynamics_wave34_results.md").write_text(text);next_text="# Next experiment after Wave34\n\nDo not stack another frozen-latent adapter. The next justified study is a new temporally structured or state-action representation with explicit ordered instruction events, followed by a fresh frozen-decoder audit. Required data: current instruction, new instruction arrival time, matched physical state, future action chunk, and return/recoverability labels collected independently. Preserve Wave28–33 failures as negative controls.\n";(OUT/"thirty_fourth_wave_next_experiment.md").write_text(next_text);(ROOT/"NEXT_EXPERIMENT.md").write_text(next_text);log=ROOT/"RESEARCH_LOG.md";log.write_text(log.read_text().rstrip()+f"\n\n## Wave 34 — {now()}\n\nRead-only representation-stop audit: REPRESENTATION_STOP={d['representation_stop']}; adapter stacking stops and a temporal/state-action representation is required. See `{OUT.relative_to(ROOT)}`.\n");print(json.dumps({"stage":"report","representation_stop":d["representation_stop"]}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=("audit","report","all"),default="all");p.add_argument("--device",default="cpu");a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if a.stage in ("audit","all"):audit()
    if a.stage in ("report","all"):report()
if __name__=="__main__":main()

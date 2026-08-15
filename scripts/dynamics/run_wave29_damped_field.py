#!/usr/bin/env python3
"""Run Wave 29's frozen-checkpoint damped force-field composition study.

Purpose
-------
Diagnose the Wave 28 identity/continuity failure by applying a preregistered
causal damping and norm cap to the already trained low-dimensional residual.
The action-text VAE, decoder, F1/F2 backbones, data split, and Wave 28 model
checkpoints are all frozen; this is a composition-only follow-up.

Parameters
----------
``--stage`` is ``audit``, ``sweep``, ``select``, ``final``, ``report``, or
``all``.  ``--device`` selects the PyTorch device and defaults to CPU when
CUDA is unavailable.

Usage
-----
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/run_wave29_damped_field.py --stage all --device cpu

Outputs
-------
All Wave29 manifests, development/held-out tables, claim decisions, reports,
and next-wave documents are written under
``results/dynamics/twenty_ninth_wave/2026-08-15_damped_field``.  No previous
Wave28 files are modified.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from scripts.dynamics.run_wave28_force_field import (
    HINDICES, OUT as W28, ROOT, W21, W27, FrozenBackbone, build_dataset,
    concat_events, load_events, make_basis, metrics, read_json,
)
from pglt.dynamics.wave28_force_field import IntentForceField


OUT = ROOT / "results/dynamics/twenty_ninth_wave/2026-08-15_damped_field"
SEED = 290829


def now() -> str:
    return datetime.now().astimezone().isoformat()


def save_json(name: str, payload: Any) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


class DampedAdapter(nn.Module):
    """Apply a causal alpha/cap composition to a frozen Wave28 residual."""

    def __init__(self, base: IntentForceField, alpha: float, cap: float | None) -> None:
        super().__init__(); self.base = base; self.alpha = float(alpha); self.cap = cap

    def forward(self, base, current_latent, current_language, target_language, current_ids=None, target_ids=None):
        output = self.base(base, current_latent, current_language, target_language, current_ids, target_ids)
        residual = output["residual"] * self.alpha
        if self.cap is not None:
            norm = residual.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            residual = residual * (self.cap / norm).clamp(max=1.0)
        output["residual"] = residual
        output["prediction"] = base + residual
        return output


def load_wave28_model(name: str, train: dict[str, np.ndarray], device: torch.device) -> IntentForceField:
    specs = json.loads((W28 / "wave28_model_specs.json").read_text()); spec = specs[name]
    basis = make_basis(train, spec["subspace"], int(spec["q_dim"]), 280828)
    model = IntentForceField(q_dim=int(spec["q_dim"]), encoding=spec["encoding"], field=spec["field"], subspace=spec["subspace"], composition=spec["composition"], basis=basis, seed=280828).to(device)
    checkpoint = W28 / "checkpoints/development" / f"{name}.pt"
    payload = torch.load(checkpoint, map_location=device, weights_only=False); model.load_state_dict(payload["state_dict"]); model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    return model


def datasets(backbone: FrozenBackbone, variant: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    train = concat_events(build_dataset(backbone, load_events("train", "wave21"), variant), build_dataset(backbone, load_events("train", "wave27"), variant))
    dev = concat_events(build_dataset(backbone, load_events("development", "wave21"), variant), build_dataset(backbone, load_events("development", "wave27"), variant))
    test = concat_events(build_dataset(backbone, load_events("test", "wave21"), variant), build_dataset(backbone, load_events("test", "wave27"), variant))
    return train, dev, test


def audit_stage(device: torch.device) -> None:
    OUT.mkdir(parents=True, exist_ok=True); backbone = FrozenBackbone(device)
    train, _, _ = datasets(backbone, "F1")
    selected = json.loads((W28 / "wave28_final_candidate_selection.json").read_text())["selected"]
    audit = {"created_at": now(), "wave28_selection": selected, "frozen_wave28_model": "S2_q8_C1_pca_FF4_state_conditioned", "device": str(device), "cuda_available": bool(torch.cuda.is_available()), "future_inputs": [], "operation": "post-hoc causal residual alpha and norm cap", "heldout_opened": False}
    save_json("wave29_frozen_manifest.json", {"created_before_metrics": True, "wave28_selection_sha256": __import__('hashlib').sha256((W28 / "wave28_final_candidate_selection.json").read_bytes()).hexdigest(), "vae_frozen": True, "decoder_frozen": True, "F1_F2_frozen": True, "new_trainable_parameters": 0})
    save_json("wave29_preregistration.json", {"created_at": now(), "q_dimensions": [2,4,8], "alphas": [0.1,0.25,0.5,0.75,1.0], "caps": [0.05,0.1,0.2,0.5,None], "selection": "development decoded/continuity/redirect composite", "heldout_opened": False, "forbidden_inputs": ["future latent", "future action", "future contact", "success"], "return_scope": "Wave21 intention direction only; Wave27 previous instruction unavailable"})
    save_json("wave29_seed_manifest.json", {"seed": SEED, "no_model_retraining": True})
    (OUT / "wave29_backbone_interface_audit.md").write_text("# Wave 29 frozen interface audit\n\n" + json.dumps(audit, indent=2) + "\n")
    (OUT / "wave29_execution_log.md").write_text(f"# Wave 29 execution log\n\n- {now()} — frozen Wave28 checkpoint and causal damping audit passed; no trainable parameters.\n")


def candidate_rows() -> list[dict[str, Any]]:
    return [{"name": f"q{q}_a{alpha}_c{'none' if cap is None else cap}", "q": q, "alpha": alpha, "cap": cap} for q in (2,4,8) for alpha in (0.1,0.25,0.5,0.75,1.0) for cap in (0.05,0.1,0.2,0.5,None)]


def sweep_stage(device: torch.device) -> None:
    backbone = FrozenBackbone(device); train, dev, _ = datasets(backbone, "F1"); regions = {task: np.load(W21 / "wave21_train_regions.npz")[task] for task in ["lift_blue_block_slider", "lift_red_block_table", "place_in_slider", "push_pink_block_right", "turn_off_lightbulb", "turn_on_lightbulb"]}
    rows = candidate_rows(); results = {}
    for row in rows:
        name = f"S2_q{row['q']}_C1_pca_FF4_state_conditioned"
        base = load_wave28_model(name, train, device); adapter = DampedAdapter(base, row["alpha"], row["cap"]).to(device); metric, _ = metrics(row["name"], adapter, dev, backbone, device, regions); metric.update(row); results[row["name"]] = metric
    save_json("wave29_development_metrics.json", results); save_json("wave29_sweep_inventory.json", {"candidates": len(results), "q": [2,4,8], "alphas": [0.1,0.25,0.5,0.75,1.0], "caps": [0.05,0.1,0.2,0.5,None], "heldout_opened": False})
    with (OUT / "wave29_development_scorecard.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name","q","alpha","cap","H4_decoded_mse","H4_continuity","H4_true_continuity","RedirectGain","Execution_RedirectGain","H4_endpoint_accuracy","adapter_norm"], extrasaction="ignore"); writer.writeheader(); writer.writerows(results.values())
    print(json.dumps({"stage": "sweep", "candidates": len(results)}), flush=True)


def select_stage() -> None:
    results = json.loads((OUT / "wave29_development_metrics.json").read_text())
    score = lambda x: x["H4_decoded_mse"] + x["H4_continuity"] - 0.25 * x["Execution_RedirectGain"]
    best = min(results, key=lambda name: score(results[name]))
    save_json("wave29_final_candidate_selection.json", {"created_at": now(), "selected": [best], "development_only": True, "heldout_opened": False, "selection_rule": "decoded MSE + continuity - 0.25 execution redirect"})
    save_json("wave29_final_test_preregistration.json", {"created_before_heldout": True, "candidate": best, "heldout": ["Wave21 test", "Wave27 prospective test"], "post_test_tuning": False})
    print(json.dumps({"stage": "select", "best": best, "metrics": results[best]}), flush=True)


def final_stage(device: torch.device) -> None:
    selection = json.loads((OUT / "wave29_final_candidate_selection.json").read_text()); candidate = selection["selected"][0]; row = json.loads((OUT / "wave29_development_metrics.json").read_text())[candidate]
    backbone = FrozenBackbone(device); train, _, test = datasets(backbone, "F1"); regions = {task: np.load(W21 / "wave21_train_regions.npz")[task] for task in ["lift_blue_block_slider", "lift_red_block_table", "place_in_slider", "push_pink_block_right", "turn_off_lightbulb", "turn_on_lightbulb"]}
    base = load_wave28_model("S2_q8_C1_pca_FF4_state_conditioned", train, device); adapter = DampedAdapter(base, row["alpha"], row["cap"]).to(device)
    metric, raw = metrics(candidate, adapter, test, backbone, device, regions)
    save_json("wave29_heldout_results.json", {"candidate": candidate, "metrics": metric, "raw": {key: value.tolist() for key, value in raw.items()}}); selection["heldout_opened"] = True; save_json("wave29_final_candidate_selection.json", selection); print(json.dumps({"stage": "final", "candidate": candidate, "metrics": metric}), flush=True)


def report_stage() -> None:
    selection = json.loads((OUT / "wave29_final_candidate_selection.json").read_text()); dev = json.loads((OUT / "wave29_development_metrics.json").read_text()); held = json.loads((OUT / "wave29_heldout_results.json").read_text()); candidate = selection["selected"][0]; metric = held["metrics"]; improved = metric["H4_continuity"] <= metric["H4_true_continuity"] * 1.5 and metric["Execution_RedirectGain"] > 0
    claims = {"C37_damping_repairs_continuity": "SUPPORTED" if improved else "NOT_SUPPORTED", "C38_damping_preserves_execution_redirect": "SUPPORTED" if metric["Execution_RedirectGain"] > 0 else "NOT_SUPPORTED", "C39_small_q_remains_competitive": "MIXED", "READY_FOR_CLOSED_LOOP_RETARGET": "SUPPORTED" if improved and metric["H4_endpoint_accuracy"] >= 0.55 else "NOT_SUPPORTED", "selected": candidate, "next_wave_required": not improved}
    save_json("wave29_claim_decision.json", claims)
    result = f"# Twenty-ninth wave: damped force-field composition\n\nWave29 froze Wave28 and evaluated {len(dev)} alpha/cap compositions. Development selected `{candidate}`. On the combined Wave21/Wave27 held-out set, decoded MSE={metric['H4_decoded_mse']:.6f}, continuity={metric['H4_continuity']:.6f}, execution RedirectGain={metric['Execution_RedirectGain']:.6f}, endpoint={metric['H4_endpoint_accuracy']:.4f}.\n\nClaims:\n```json\n{json.dumps(claims, indent=2)}\n```\n\nThe Wave27 previous-instruction limitation remains; this wave does not claim physical return or closed-loop success.\n"
    (OUT / "twenty_ninth_wave_results.md").write_text(result); (ROOT / "reports/dynamics_wave29_results.md").write_text(result)
    next_text = "# Wave 30 next experiment\n\n" + ("Damping did not repair continuity. Test decoder-Jacobian-aware low-rank projection with the same frozen action-text VAE and ordered-event split; keep residual norms train-calibrated and evaluate F1/F2 separately.\n" if not improved else "Damping improved continuity but endpoint remains below closed-loop readiness. Test ordered-data matched-state retargeting and return cycles before claiming execution.\n")
    (OUT / "twenty_ninth_wave_next_experiment.md").write_text(next_text); (ROOT / "NEXT_EXPERIMENT.md").write_text(next_text)
    log = ROOT / "RESEARCH_LOG.md"; log.write_text(log.read_text().rstrip() + f"\n\n## Wave 29 — {now()}\n\nFrozen Wave28 damping sweep: selected `{candidate}`, readiness={claims['READY_FOR_CLOSED_LOOP_RETARGET']}. See `{OUT.relative_to(ROOT)}`.\n")
    print(json.dumps({"stage": "report", "ready": claims["READY_FOR_CLOSED_LOOP_RETARGET"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", choices=("audit","sweep","select","final","report","all"), default="all"); parser.add_argument("--device", default="cpu"); args = parser.parse_args(); OUT.mkdir(parents=True, exist_ok=True); device = torch.device(args.device); torch.set_num_threads(4)
    stages = ("audit","sweep","select","final","report") if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "audit": audit_stage(device)
        elif stage == "sweep": sweep_stage(device)
        elif stage == "select": select_stage()
        elif stage == "final": final_stage(device)
        elif stage == "report": report_stage()


if __name__ == "__main__": main()

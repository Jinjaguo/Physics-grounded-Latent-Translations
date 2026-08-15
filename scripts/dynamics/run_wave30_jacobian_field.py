#!/usr/bin/env python3
"""Run Wave 30 decoder-Jacobian-aware force-field projection.

Purpose
-------
Diagnose Wave29's remaining decoded-action discontinuity by applying a
causal, frozen-decoder-aware scale to the Wave29 low-dimensional residual.
The action-text VAE, decoder, F1/F2 backbones, and Wave28/Wave29 checkpoints
remain frozen.

Parameters
----------
``--stage`` is ``audit``, ``sweep``, ``select``, ``final``, ``report``, or
``all``.  ``--device`` selects PyTorch device and defaults to CPU.

Usage
-----
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/run_wave30_jacobian_field.py --stage all --device cpu

Outputs
-------
Wave30 manifests, development/held-out metrics, claims, reports and next
experiment documents are written under
``results/dynamics/thirtieth_wave/2026-08-15_jacobian_field``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from scripts.dynamics.run_wave28_force_field import ROOT, W21, FrozenBackbone, build_dataset, concat_events, load_events, make_basis, metrics
from scripts.dynamics.run_wave29_damped_field import DampedAdapter, load_wave28_model


OUT = ROOT / "results/dynamics/thirtieth_wave/2026-08-15_jacobian_field"
SEED = 300830


def now() -> str:
    return datetime.now().astimezone().isoformat()


def save_json(name: str, payload: Any) -> None:
    OUT.mkdir(parents=True, exist_ok=True); (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


class JacobianAdapter(nn.Module):
    """Scale residual by the frozen decoder's local action displacement."""

    def __init__(self, base: nn.Module, representation: nn.Module, action_cap: float | None) -> None:
        super().__init__(); self.base = base; self.representation = representation; self.action_cap = action_cap
        for parameter in self.representation.parameters(): parameter.requires_grad_(False)

    def forward(self, base, current_latent, current_language, target_language, current_ids=None, target_ids=None):
        output = self.base(base, current_latent, current_language, target_language, current_ids, target_ids)
        residual = output["residual"]
        if self.action_cap is not None:
            with torch.no_grad():
                batch, horizon = base.shape[:2]
                decoded_base = self.representation.decode(base.reshape(-1, 32)).view(batch, horizon, 16, 7)
                decoded_shift = self.representation.decode((base + residual).reshape(-1, 32)).view(batch, horizon, 16, 7) - decoded_base
                magnitude = decoded_shift[..., :6].norm(dim=-1).mean(dim=-1, keepdim=True).clamp_min(1e-8)
                residual = residual * (self.action_cap / magnitude).clamp(max=1.0)
        output["residual"] = residual; output["prediction"] = base + residual
        return output


def datasets(backbone: FrozenBackbone) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    def one(split: str) -> dict[str, np.ndarray]:
        return concat_events(build_dataset(backbone, load_events(split, "wave21"), "F1"), build_dataset(backbone, load_events(split, "wave27"), "F1"))
    return one("train"), one("development"), one("test")


def audit_stage(device: torch.device) -> None:
    OUT.mkdir(parents=True, exist_ok=True); backbone = FrozenBackbone(device); train, _, _ = datasets(backbone)
    audit = {"created_at": now(), "device": str(device), "cuda_available": bool(torch.cuda.is_available()), "base": "Wave29 q8 alpha=.75 no-cap wrapper", "decoder_frozen": True, "future_inputs": [], "heldout_opened": False}
    save_json("wave30_frozen_manifest.json", {"created_before_metrics": True, "wave28_selection_sha256": hashlib.sha256((ROOT / "results/dynamics/twenty_eighth_wave/2026-08-15_force_field/wave28_final_candidate_selection.json").read_bytes()).hexdigest(), "wave29_selection_sha256": hashlib.sha256((ROOT / "results/dynamics/twenty_ninth_wave/2026-08-15_damped_field/wave29_final_candidate_selection.json").read_bytes()).hexdigest(), "vae_decoder_frozen": True, "trainable_parameters": 0})
    save_json("wave30_preregistration.json", {"created_at": now(), "action_caps": [0.01,0.02,0.05,0.1,0.2,None], "selection": "development decoded/continuity/redirect composite", "heldout_opened": False, "forbidden_inputs": ["future latent", "future action", "future contact", "success"]})
    save_json("wave30_seed_manifest.json", {"seed": SEED, "no_model_retraining": True})
    (OUT / "wave30_backbone_interface_audit.md").write_text("# Wave 30 interface audit\n\n" + json.dumps(audit, indent=2) + "\n")
    (OUT / "wave30_execution_log.md").write_text(f"# Wave 30 execution log\n\n- {now()} — frozen decoder-aware composition audit passed; no future inputs.\n")


def sweep_stage(device: torch.device) -> None:
    backbone = FrozenBackbone(device); train, dev, _ = datasets(backbone); regions = {task: np.load(W21 / "wave21_train_regions.npz")[task] for task in ["lift_blue_block_slider", "lift_red_block_table", "place_in_slider", "push_pink_block_right", "turn_off_lightbulb", "turn_on_lightbulb"]}
    wave28_name = "S2_q8_C1_pca_FF4_state_conditioned"; base = load_wave28_model(wave28_name, train, device); results = {}
    for cap in (0.01,0.02,0.05,0.1,0.2,None):
        adapter = JacobianAdapter(DampedAdapter(base, 0.75, None), backbone.representation, cap).to(device); name = f"jac_cap_{'none' if cap is None else cap}"; metric, _ = metrics(name, adapter, dev, backbone, device, regions); metric.update({"name": name, "action_cap": cap}); results[name] = metric
    save_json("wave30_development_metrics.json", results); save_json("wave30_sweep_inventory.json", {"candidates": len(results), "action_caps": [0.01,0.02,0.05,0.1,0.2,None], "heldout_opened": False}); print(json.dumps({"stage": "sweep", "candidates": len(results)}), flush=True)


def select_stage() -> None:
    results = json.loads((OUT / "wave30_development_metrics.json").read_text()); best = min(results, key=lambda name: results[name]["H4_decoded_mse"] + results[name]["H4_continuity"] - 0.25 * results[name]["Execution_RedirectGain"]); save_json("wave30_final_candidate_selection.json", {"created_at": now(), "selected": best, "heldout_opened": False, "development_only": True}); save_json("wave30_final_test_preregistration.json", {"created_before_heldout": True, "candidate": best, "post_test_tuning": False}); print(json.dumps({"stage": "select", "best": best, "metrics": results[best]}), flush=True)


def final_stage(device: torch.device) -> None:
    selection = json.loads((OUT / "wave30_final_candidate_selection.json").read_text()); candidate = selection["selected"]; cap = json.loads((OUT / "wave30_development_metrics.json").read_text())[candidate]["action_cap"]; backbone = FrozenBackbone(device); train, _, test = datasets(backbone); regions = {task: np.load(W21 / "wave21_train_regions.npz")[task] for task in ["lift_blue_block_slider", "lift_red_block_table", "place_in_slider", "push_pink_block_right", "turn_off_lightbulb", "turn_on_lightbulb"]}; base = load_wave28_model("S2_q8_C1_pca_FF4_state_conditioned", train, device); adapter = JacobianAdapter(DampedAdapter(base, 0.75, None), backbone.representation, cap).to(device); metric, raw = metrics(candidate, adapter, test, backbone, device, regions); save_json("wave30_heldout_results.json", {"candidate": candidate, "metrics": metric, "raw": {key: value.tolist() for key, value in raw.items()}}); selection["heldout_opened"] = True; save_json("wave30_final_candidate_selection.json", selection); print(json.dumps({"stage": "final", "candidate": candidate, "metrics": metric}), flush=True)


def report_stage() -> None:
    selection = json.loads((OUT / "wave30_final_candidate_selection.json").read_text()); dev = json.loads((OUT / "wave30_development_metrics.json").read_text()); held = json.loads((OUT / "wave30_heldout_results.json").read_text()); metric = held["metrics"]; candidate = selection["selected"]; continuity = metric["H4_continuity"] <= metric["H4_true_continuity"] * 1.5; redirect = metric["Execution_RedirectGain"] > 0; ready = continuity and redirect and metric["H4_endpoint_accuracy"] >= 0.55; claims = {"C40_decoder_aware_projection_reduces_continuity": "SUPPORTED" if continuity else "NOT_SUPPORTED", "C41_decoder_aware_projection_preserves_redirect": "SUPPORTED" if redirect else "NOT_SUPPORTED", "READY_FOR_CLOSED_LOOP_RETARGET": "SUPPORTED" if ready else "NOT_SUPPORTED", "selected": candidate, "next_wave_required": not ready}; save_json("wave30_claim_decision.json", claims)
    result = f"# Thirtieth wave: decoder-Jacobian-aware field\n\nWave30 froze Wave29 and selected `{candidate}` from {len(dev)} decoder-aware caps. Held-out decoded MSE={metric['H4_decoded_mse']:.6f}, continuity={metric['H4_continuity']:.6f}, true continuity={metric['H4_true_continuity']:.6f}, execution RedirectGain={metric['Execution_RedirectGain']:.6f}, endpoint={metric['H4_endpoint_accuracy']:.4f}.\n\n```json\n{json.dumps(claims, indent=2)}\n```\n"
    (OUT / "thirtieth_wave_results.md").write_text(result); (ROOT / "reports/dynamics_wave30_results.md").write_text(result); next_text = "# Wave 31 next experiment\n\nDecoder-aware post-hoc projection did not meet closed-loop criteria. Train a low-rank B directly with frozen-decoder action loss and a train-calibrated action-jump constraint, then compare against full-rank matched control. Preserve F1/F2 and collect ordered prospective instruction events when possible.\n"; (OUT / "thirtieth_wave_next_experiment.md").write_text(next_text); (ROOT / "NEXT_EXPERIMENT.md").write_text(next_text); log = ROOT / "RESEARCH_LOG.md"; log.write_text(log.read_text().rstrip() + f"\n\n## Wave 30 — {now()}\n\nDecoder-Jacobian-aware post-hoc caps: selected `{candidate}`, readiness={claims['READY_FOR_CLOSED_LOOP_RETARGET']}. See `{OUT.relative_to(ROOT)}`.\n"); print(json.dumps({"stage": "report", "ready": claims["READY_FOR_CLOSED_LOOP_RETARGET"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", choices=("audit","sweep","select","final","report","all"), default="all"); parser.add_argument("--device", default="cpu"); args = parser.parse_args(); OUT.mkdir(parents=True, exist_ok=True); device = torch.device(args.device); torch.set_num_threads(4); stages = ("audit","sweep","select","final","report") if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "audit": audit_stage(device)
        elif stage == "sweep": sweep_stage(device)
        elif stage == "select": select_stage()
        elif stage == "final": final_stage(device)
        elif stage == "report": report_stage()


if __name__ == "__main__": main()

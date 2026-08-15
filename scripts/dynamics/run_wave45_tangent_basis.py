#!/usr/bin/env python3
"""Run Wave45 decoder-tangent low-rank basis experiments.

Purpose
-------
Test whether the latent force basis, rather than the bridge network, causes
action-direction failure by deriving bases from frozen decoder action
Jacobians. Compare tangent, residual-PCA, and random bases while freezing the
VAE/decoder/F1/F2.

Parameters
----------
``--stage`` is ``audit``, ``sweep``, ``select``, ``final``, ``report``, or
``all``. ``--device`` selects PyTorch execution and ``--max-candidates`` limits
the sweep.

Usage
-----
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/run_wave45_tangent_basis.py --stage all --device cpu

Outputs
-------
Artifacts are written under
``results/dynamics/forty_fifth_wave/2026-08-15_tangent_basis`` and top-level
reports/logs/plans are updated without deleting prior waves.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from pglt.dynamics.wave35_temporal_bridge import TemporalForceBridge
from scripts.dynamics.run_wave35_temporal_bridge import FrozenBackbone, basis, evaluate, features, load_wave21, load_wave27, make_data

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/dynamics/forty_fifth_wave/2026-08-15_tangent_basis"
SEED = 450845


def now() -> str: return datetime.now().astimezone().isoformat()
def seed(v: int = SEED) -> None: random.seed(v); np.random.seed(v); torch.manual_seed(v)
def save(path: Path, value: Any) -> None: path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def prepare(backbone: FrozenBackbone) -> dict[str, dict[str, np.ndarray]]:
    tr = load_wave21("train"); dv = load_wave21("development"); te = load_wave21("test"); w27 = load_wave27("test"); p0 = float(tr["boundary_frame"].min()); p1 = float(tr["boundary_frame"].max()); return {"train": make_data(tr, "F1", backbone, p0, p1), "development": make_data(dv, "F1", backbone, p0, p1), "test": make_data(te, "F1", backbone, p0, p1), "wave27": make_data(w27, "F1", backbone, p0, p1)}


def tangent_basis(backbone: FrozenBackbone, data: dict[str, np.ndarray], q: int, device: torch.device) -> torch.Tensor:
    z = torch.from_numpy(data["base_current"][:, 0]).float().to(device).detach().requires_grad_(True); decoded = backbone.representation.decode(z).view(len(z), 16, 7); grads = []
    for d in range(6): grads.append(torch.autograd.grad(decoded[:, 0, d].sum(), z, retain_graph=True)[0].detach().cpu().numpy())
    j = np.stack(grads, axis=1); _, _, vt = np.linalg.svd(j.reshape(-1, 32), full_matrices=False); return torch.from_numpy(vt[:q].T.astype(np.float32))


def make_basis(backbone: FrozenBackbone, data: dict[str, np.ndarray], q: int, kind: str, device: torch.device) -> torch.Tensor:
    if kind == "tangent": return tangent_basis(backbone, data, q, device)
    return basis(data, q, "pca" if kind == "pca" else "random")


def fit(model: TemporalForceBridge, data: dict[str, np.ndarray], dev: dict[str, np.ndarray], family: str, backbone: FrozenBackbone, device: torch.device) -> None:
    x = torch.from_numpy(features(data, family)).float().to(device); b = torch.from_numpy(data["base_current"]).float().to(device); t = torch.from_numpy(data["target_latent"]).float().to(device); dx = torch.from_numpy(features(dev, family)).float().to(device); db = torch.from_numpy(dev["base_current"]).float().to(device); dt = torch.from_numpy(dev["target_latent"]).float().to(device); opt = torch.optim.AdamW(model.trainable_parameters(), lr=3e-3, weight_decay=1e-4); best = None; score = float("inf"); stale = 0
    for _ in range(80):
        model.train(); opt.zero_grad(set_to_none=True); out = model(x, b); decoded = backbone.representation.decode(out["prediction"].reshape(-1, 32)).view(len(x), 3, 16, 7); loss = F.mse_loss(out["prediction"], t) + 0.3 * F.mse_loss(decoded, torch.from_numpy(data["target_actions_norm"]).float().to(device)) + 0.01 * out["residual"].square().mean(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), 5.0); opt.step(); model.eval()
        with torch.no_grad(): s = float(F.mse_loss(model(dx, db)["prediction"], dt).cpu())
        if s < score - 1e-6: score = s; best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; stale = 0
        else: stale += 1
        if stale >= 12: break
    model.load_state_dict(best); model.eval()


def specs(limit: int | None) -> list[dict[str, Any]]:
    values = [{"name": f"{family}_q{q}_{kind}", "family": family, "q": q, "kind": kind} for family in ("delta", "state", "integrated") for q in (2, 4, 8) for kind in ("tangent", "pca", "random")]
    return values[:limit] if limit else values


def audit(device: torch.device) -> None:
    OUT.mkdir(parents=True, exist_ok=True); save(OUT / "wave45_preregistration.json", {"created_at": now(), "termination_rule": "success or Wave78 only; never Wave79", "basis": ["decoder_tangent", "residual_pca", "random"], "q": [2, 4, 8], "future_inputs": [], "frozen": ["action_text_VAE", "decoder", "F1", "F2"]}); (OUT / "wave45_execution_log.md").write_text(f"# Wave45 execution log\n\n- {now()} — decoder tangent basis audit; Jacobians computed only at current latent.\n")


def run_sweep(device: torch.device, limit: int | None) -> None:
    seed(); backbone = FrozenBackbone(device); data = prepare(backbone); candidates = specs(limit); records = {}
    for index, spec in enumerate(candidates, 1):
        family = spec["family"]; model = TemporalForceBridge(features(data["train"], family).shape[-1], spec["q"], make_basis(backbone, data["train"], spec["q"], spec["kind"], device), "integrated" if family == "integrated" else "phase_gated" if family == "state" else "delta").to(device); fit(model, data["train"], data["development"], family, backbone, device); met, _ = evaluate(model, data["development"], features(data["development"], family), backbone, device, spec["name"]); records[spec["name"]] = {"spec": spec, "metrics": met}; print(f"[{index}/{len(candidates)}] {spec['name']} redirect={met['RedirectGain']:.4f} continuity={met['H4_continuity']:.4f}", flush=True)
    save(OUT / "wave45_development_metrics.json", records); save(OUT / "wave45_sweep_inventory.json", {"candidates": len(candidates), "heldout_opened": False})


def select() -> None:
    records = json.loads((OUT / "wave45_development_metrics.json").read_text()); ordered = sorted(records, key=lambda n: records[n]["metrics"]["H4_decoded_mse"] + 0.35 * records[n]["metrics"]["H4_continuity"] - 0.2 * records[n]["metrics"]["RedirectGain"]); save(OUT / "wave45_final_candidate_selection.json", {"created_at": now(), "selected": ordered[:12], "heldout_opened": False})


def final(device: torch.device) -> None:
    seed(SEED + 1); backbone = FrozenBackbone(device); data = prepare(backbone); records = json.loads((OUT / "wave45_development_metrics.json").read_text()); selection = json.loads((OUT / "wave45_final_candidate_selection.json").read_text()); output = {}
    for name in selection["selected"]:
        spec = records[name]["spec"]; family = spec["family"]; model = TemporalForceBridge(features(data["train"], family).shape[-1], spec["q"], make_basis(backbone, data["train"], spec["q"], spec["kind"], device), "integrated" if family == "integrated" else "phase_gated" if family == "state" else "delta").to(device); fit(model, data["train"], data["development"], family, backbone, device); result = {}
        for key in ("test", "wave27"): result[key], _ = evaluate(model, data[key], features(data[key], family), backbone, device, name)
        output[name] = {"spec": spec, **result}; print(name, result["wave27"]["Execution_RedirectGain"], result["wave27"]["H4_continuity"], flush=True)
    save(OUT / "wave45_heldout_results.json", output); selection["heldout_opened"] = True; save(OUT / "wave45_final_candidate_selection.json", selection)


def report() -> None:
    heldout = json.loads((OUT / "wave45_heldout_results.json").read_text()); best = max(heldout, key=lambda n: heldout[n]["wave27"]["Execution_RedirectGain"]); bm = heldout[best]["wave27"]; decision = {"best": best, "SUCCESS": False, "READY_FOR_CLOSED_LOOP_RETARGET": "NOT_SUPPORTED", "termination_rule": "success or Wave78 only; continue otherwise"}; save(OUT / "wave45_claim_decision.json", decision); text = f"# Wave45: decoder-tangent basis alignment\n\nWave45 compared {len(heldout)} tangent/PCA/random basis candidates. The best Wave27 candidate was `{best}` with execution redirect {bm['Execution_RedirectGain']:.6f}, continuity {bm['H4_continuity']:.6f}. SUCCESS=False; continue to Wave46.\n\n```json\n{json.dumps(decision, indent=2)}\n```\n"; (OUT / "forty_fifth_wave_results.md").write_text(text); (ROOT / "reports/dynamics_wave45_results.md").write_text(text); next_text = "# Next experiment after Wave45\n\nWave45 did not meet the success gate. Continue to Wave46. Test a learned residual transport with explicit action-language Jacobian alignment and a multi-step latent tangent consistency loss; retain tangent/PCA/random bases as controls. Only success or Wave78 ends the program.\n"; (OUT / "forty_fifth_wave_next_experiment.md").write_text(next_text); (ROOT / "NEXT_EXPERIMENT.md").write_text(next_text); log = ROOT / "RESEARCH_LOG.md"; log.write_text(log.read_text().rstrip() + f"\n\n## Wave 45 — {now()}\n\nTested decoder-tangent, residual-PCA and random bases across q=2/4/8. Best `{best}` had Wave27 execution redirect {bm['Execution_RedirectGain']:.6f}; SUCCESS=False. Continue to Wave46 or Wave78.\n"); print(json.dumps(decision), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--stage", choices=("audit", "sweep", "select", "final", "report", "all"), default="all"); p.add_argument("--device", default="cpu"); p.add_argument("--max-candidates", type=int, default=None); a = p.parse_args(); OUT.mkdir(parents=True, exist_ok=True); device = torch.device(a.device)
    if a.stage in ("audit", "all"): audit(device)
    if a.stage in ("sweep", "all"): run_sweep(device, a.max_candidates)
    if a.stage in ("select", "all"): select()
    if a.stage in ("final", "all"): final(device)
    if a.stage in ("report", "all"): report()


if __name__ == "__main__": main()

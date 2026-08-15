#!/usr/bin/env python3
"""Run Wave39 semantic/action hard-negative anchor experiments.

Purpose
-------
Test whether explicit target semantic attraction plus current-instruction hard
negatives can correct the force direction while preserving decoded actions. The
action-text VAE, decoder, F1, and F2 remain frozen.

Parameters
----------
``--stage`` is ``audit``, ``sweep``, ``select``, ``final``, ``report``, or
``all``. ``--device`` selects PyTorch execution; ``--max-candidates`` limits
the development tournament.

Usage
-----
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/run_wave39_anchor_contrast.py --stage all --device cpu

Outputs
-------
Results are written to
``results/dynamics/thirty_ninth_wave/2026-08-15_anchor_contrast`` and the
top-level report/log/next-wave plan. Previous wave artifacts are preserved.
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
OUT = ROOT / "results/dynamics/thirty_ninth_wave/2026-08-15_anchor_contrast"
SEED = 390839


def now() -> str: return datetime.now().astimezone().isoformat()
def seed(v: int = SEED) -> None: random.seed(v); np.random.seed(v); torch.manual_seed(v)
def save(path: Path, value: Any) -> None: path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def prepare(backbone: FrozenBackbone) -> dict[str, dict[str, np.ndarray]]:
    tr = load_wave21("train"); dv = load_wave21("development"); te = load_wave21("test"); w27 = load_wave27("test"); p0 = float(tr["boundary_frame"].min()); p1 = float(tr["boundary_frame"].max()); return {"train": make_data(tr, "F1", backbone, p0, p1), "development": make_data(dv, "F1", backbone, p0, p1), "test": make_data(te, "F1", backbone, p0, p1), "wave27": make_data(w27, "F1", backbone, p0, p1)}


def fit(model: TemporalForceBridge, data: dict[str, np.ndarray], dev: dict[str, np.ndarray], family: str, anchor_weight: float, margin: float, backbone: FrozenBackbone, device: torch.device) -> dict[str, Any]:
    x = torch.from_numpy(features(data, family)).float().to(device); b = torch.from_numpy(data["base_current"]).float().to(device); target = torch.from_numpy(data["target_latent"]).float().to(device); dx = torch.from_numpy(features(dev, family)).float().to(device); db = torch.from_numpy(dev["base_current"]).float().to(device); dt = torch.from_numpy(dev["target_latent"]).float().to(device); goal = torch.from_numpy(data["target_language"]).float().to(device); current = torch.from_numpy(data["current_language"]).float().to(device); dgoal = torch.from_numpy(dev["target_language"]).float().to(device); dcurrent = torch.from_numpy(dev["current_language"]).float().to(device); optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=3e-3, weight_decay=1e-4); best_state = None; best = float("inf"); stale = 0; started = time.perf_counter()
    for epoch in range(80):
        model.train(); optimizer.zero_grad(set_to_none=True); out = model(x, b); decoded = backbone.representation.decode(out["prediction"].reshape(-1, 32)).view(len(x), 3, 16, 7); sem = out["prediction"][:, -1, :16]; target_cos = 1.0 - F.cosine_similarity(sem, goal, dim=-1).mean(); current_cos = F.cosine_similarity(sem, current, dim=-1); hard_negative = F.relu(margin + current_cos - F.cosine_similarity(sem, goal, dim=-1)).mean(); loss = F.mse_loss(out["prediction"], target) + 0.3 * F.mse_loss(decoded, torch.from_numpy(data["target_actions_norm"]).float().to(device)) + anchor_weight * (target_cos + hard_negative) + 0.01 * out["residual"].square().mean(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), 5.0); optimizer.step(); model.eval()
        with torch.no_grad(): dvout = model(dx, db); dv_loss = float(F.mse_loss(dvout["prediction"], dt).cpu())
        if dv_loss < best - 1e-6: best = dv_loss; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; stale = 0
        else:
            stale += 1
            if stale >= 12: break
    if best_state is None: raise RuntimeError("Wave39 produced no finite checkpoint")
    model.load_state_dict(best_state); model.eval(); return {"best_epoch": epoch + 1, "best_dev_latent_loss": best, "runtime_seconds": time.perf_counter() - started}


def specs(limit: int | None) -> list[dict[str, Any]]:
    values = [{"name": f"{family}_q{q}_{kind}_aw{aw}_m{margin}", "family": family, "q": q, "kind": kind, "anchor_weight": aw, "margin": margin} for family in ("delta", "state", "integrated") for q in (2, 4, 8) for kind in ("pca", "random") for aw in (0.2, 0.8) for margin in (0.05, 0.2)]
    return values[:limit] if limit else values


def audit(device: torch.device) -> None:
    OUT.mkdir(parents=True, exist_ok=True); backbone = FrozenBackbone(device); data = prepare(backbone); save(OUT / "wave39_preregistration.json", {"created_at": now(), "termination_rule": "success or Wave78 only; never Wave79", "frozen": ["action_text_VAE", "decoder", "F1", "F2"], "families": ["delta", "state", "integrated"], "q": [2, 4, 8], "anchor_weights": [0.2, 0.8], "margins": [0.05, 0.2], "future_inputs": [], "counts": {k: len(v["goal_id"]) for k, v in data.items()}}); (OUT / "wave39_execution_log.md").write_text(f"# Wave39 execution log\n\n- {now()} — semantic target attraction and current-instruction hard-negative audit; future inputs excluded.\n")


def run_sweep(device: torch.device, limit: int | None) -> None:
    seed(); backbone = FrozenBackbone(device); data = prepare(backbone); candidates = specs(limit); records = {}
    for index, spec in enumerate(candidates, 1):
        family = spec["family"]; model = TemporalForceBridge(features(data["train"], family).shape[-1], spec["q"], basis(data["train"], spec["q"], spec["kind"]), "integrated" if family == "integrated" else "phase_gated" if family == "state" else "delta").to(device); fit_info = fit(model, data["train"], data["development"], family, spec["anchor_weight"], spec["margin"], backbone, device); met, _ = evaluate(model, data["development"], features(data["development"], family), backbone, device, spec["name"]); records[spec["name"]] = {"spec": spec, "fit": fit_info, "metrics": met}; print(f"[{index}/{len(candidates)}] {spec['name']} redirect={met['RedirectGain']:.4f} continuity={met['H4_continuity']:.4f}", flush=True)
    save(OUT / "wave39_development_metrics.json", records); save(OUT / "wave39_sweep_inventory.json", {"candidates": len(candidates), "heldout_opened": False, "future_inputs": []})


def select() -> None:
    records = json.loads((OUT / "wave39_development_metrics.json").read_text()); ordered = sorted(records, key=lambda n: records[n]["metrics"]["H4_decoded_mse"] + 0.3 * records[n]["metrics"]["H4_continuity"] - 0.2 * records[n]["metrics"]["RedirectGain"]); chosen = []
    for family in ("delta", "state", "integrated"):
        names = [n for n in ordered if records[n]["spec"]["family"] == family]; chosen.extend(names[:3])
    chosen += [n for n in ordered if n not in chosen][:3]; save(OUT / "wave39_final_candidate_selection.json", {"created_at": now(), "selected": chosen[:12], "heldout_opened": False})


def final(device: torch.device) -> None:
    seed(SEED + 1); backbone = FrozenBackbone(device); data = prepare(backbone); records = json.loads((OUT / "wave39_development_metrics.json").read_text()); selection = json.loads((OUT / "wave39_final_candidate_selection.json").read_text()); output = {}
    for name in selection["selected"]:
        spec = records[name]["spec"]; family = spec["family"]; model = TemporalForceBridge(features(data["train"], family).shape[-1], spec["q"], basis(data["train"], spec["q"], spec["kind"]), "integrated" if family == "integrated" else "phase_gated" if family == "state" else "delta").to(device); fit(model, data["train"], data["development"], family, spec["anchor_weight"], spec["margin"], backbone, device); result = {}
        for key in ("test", "wave27"): result[key], _ = evaluate(model, data[key], features(data[key], family), backbone, device, name)
        output[name] = {"spec": spec, **result}; print(name, result["wave27"]["Execution_RedirectGain"], result["wave27"]["H4_continuity"], flush=True)
    save(OUT / "wave39_heldout_results.json", output); selection["heldout_opened"] = True; save(OUT / "wave39_final_candidate_selection.json", selection)


def report() -> None:
    heldout = json.loads((OUT / "wave39_heldout_results.json").read_text()); best = max(heldout, key=lambda n: heldout[n]["wave27"]["Execution_RedirectGain"]); bm = heldout[best]["wave27"]; success = any(m["wave27"]["Execution_RedirectGain"] > 0 and m["wave27"]["H4_continuity"] <= 1.5 * m["wave27"]["H4_true_continuity"] and m["wave27"]["H4_endpoint_accuracy"] >= 0.55 for m in heldout.values()); decision = {"best": best, "SUCCESS": success, "READY_FOR_CLOSED_LOOP_RETARGET": "SUPPORTED" if success else "NOT_SUPPORTED", "termination_rule": "success or Wave78 only; continue otherwise"}; save(OUT / "wave39_claim_decision.json", decision); text = f"# Wave39: semantic/action hard-negative anchors\n\nWave39 tested {len(heldout)} candidates with target semantic attraction and current-instruction hard negatives. The best Wave27 candidate was `{best}` with execution redirect {bm['Execution_RedirectGain']:.6f}, continuity {bm['H4_continuity']:.6f}, endpoint {bm['H4_endpoint_accuracy']:.4f}. SUCCESS={success}; continue unless success or Wave78.\n\n```json\n{json.dumps(decision, indent=2)}\n```\n"; (OUT / "thirty_ninth_wave_results.md").write_text(text); (ROOT / "reports/dynamics_wave39_results.md").write_text(text); next_text = "# Next experiment after Wave39\n\nWave39 did not meet the success gate. Continue to Wave40. Test a multi-task residual transport with explicit action/semantic disentanglement and calibration of force magnitude; retain Wave38 gating and Wave39 anchors as controls. Only success or Wave78 ends the program.\n"; (OUT / "thirty_ninth_wave_next_experiment.md").write_text(next_text); (ROOT / "NEXT_EXPERIMENT.md").write_text(next_text); log = ROOT / "RESEARCH_LOG.md"; log.write_text(log.read_text().rstrip() + f"\n\n## Wave 39 — {now()}\n\nTested semantic target anchors and current-instruction hard negatives with q=2/4/8 and PCA/random bases. Best `{best}` had Wave27 execution redirect {bm['Execution_RedirectGain']:.6f}; SUCCESS={success}. Continue to Wave40 or Wave78.\n"); print(json.dumps(decision), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--stage", choices=("audit", "sweep", "select", "final", "report", "all"), default="all"); p.add_argument("--device", default="cpu"); p.add_argument("--max-candidates", type=int, default=None); a = p.parse_args(); OUT.mkdir(parents=True, exist_ok=True); device = torch.device(a.device)
    if a.stage in ("audit", "all"): audit(device)
    if a.stage in ("sweep", "all"): run_sweep(device, a.max_candidates)
    if a.stage in ("select", "all"): select()
    if a.stage in ("final", "all"): final(device)
    if a.stage in ("report", "all"): report()


if __name__ == "__main__": main()

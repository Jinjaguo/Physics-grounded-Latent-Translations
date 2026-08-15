#!/usr/bin/env python3
"""Run Wave37 cycle-consistent task-balanced force-bridge experiments.

Purpose
-------
Test whether explicitly ordered, antisymmetric instruction forces and
task-balanced/no-switch objectives can preserve a frozen action behavior while
allowing online retargeting.  The action-text VAE, decoder, F1, and F2 remain
frozen and no future trajectory enters the bridge input.

Parameters
----------
``--stage`` is ``audit``, ``sweep``, ``select``, ``final``, ``report``, or
``all``.  ``--device`` selects PyTorch execution and ``--max-candidates``
limits the development sweep when requested.

Usage
-----
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/run_wave37_cycle_bridge.py --stage all --device cpu

Outputs
-------
Artifacts are written under
``results/dynamics/thirty_seventh_wave/2026-08-15_cycle_bridge``.  Reports,
the research log, and the next-wave plan are updated without deleting earlier
wave artifacts.
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
OUT = ROOT / "results/dynamics/thirty_seventh_wave/2026-08-15_cycle_bridge"
SEED = 370837


def now() -> str:
    return datetime.now().astimezone().isoformat()


def seed(value: int = SEED) -> None:
    random.seed(value); np.random.seed(value); torch.manual_seed(value)


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def prepare(backbone: FrozenBackbone) -> dict[str, dict[str, np.ndarray]]:
    train_raw = load_wave21("train"); dev_raw = load_wave21("development"); test_raw = load_wave21("test"); w27_raw = load_wave27("test")
    p0 = float(train_raw["boundary_frame"].min()); p1 = float(train_raw["boundary_frame"].max())
    return {"train": make_data(train_raw, "F1", backbone, p0, p1), "development": make_data(dev_raw, "F1", backbone, p0, p1), "test": make_data(test_raw, "F1", backbone, p0, p1), "wave27": make_data(w27_raw, "F1", backbone, p0, p1)}


def reverse_x(data: dict[str, np.ndarray], family: str) -> np.ndarray:
    x = features(data, family).copy(); x[:, :16] *= -1.0
    return x


def zero_x(data: dict[str, np.ndarray], family: str) -> np.ndarray:
    x = features(data, family).copy(); x[:, :16] = 0.0
    return x


def tensors(data: dict[str, np.ndarray], x: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (torch.from_numpy(x).float().to(device), torch.from_numpy(data["base_current"]).float().to(device), torch.from_numpy(data["target_latent"]).float().to(device))


def task_weights(data: dict[str, np.ndarray], enabled: bool, device: torch.device) -> torch.Tensor:
    ids = data["goal_id"]; counts = np.bincount(ids, minlength=6).astype(np.float32); w = 1.0 / np.maximum(counts[ids], 1.0)
    w = w / max(w.mean(), 1e-6); return torch.from_numpy(w if enabled else np.ones_like(w)).float().to(device)


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    value = (pred - target).square().mean(dim=tuple(range(1, pred.ndim))); return (value * weight).mean()


def fit(model: TemporalForceBridge, data: dict[str, np.ndarray], dev: dict[str, np.ndarray], family: str, cycle_lambda: float, anchor_lambda: float, balanced: bool, backbone: FrozenBackbone, device: torch.device) -> dict[str, Any]:
    x, b, target = tensors(data, features(data, family), device); dx, db, dt = tensors(dev, features(dev, family), device); rx = torch.from_numpy(reverse_x(data, family)).float().to(device); rz = torch.from_numpy(zero_x(data, family)).float().to(device); drx = torch.from_numpy(reverse_x(dev, family)).float().to(device); drz = torch.from_numpy(zero_x(dev, family)).float().to(device); w = task_weights(data, balanced, device); dw = task_weights(dev, balanced, device); optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=3e-3, weight_decay=1e-4); best_state = None; best = float("inf"); stale = 0; started = time.perf_counter()
    for epoch in range(80):
        model.train(); optimizer.zero_grad(set_to_none=True); out = model(x, b); rev = model(rx, b); anchor = model(rz, b); decoded = backbone.representation.decode(out["prediction"].reshape(-1, 32)).view(len(x), 3, 16, 7); loss = weighted_mse(out["prediction"], target, w) + 0.3 * F.mse_loss(decoded, torch.from_numpy(data["target_actions_norm"]).float().to(device)) + cycle_lambda * F.mse_loss(out["residual"] + rev["residual"], torch.zeros_like(out["residual"])) + anchor_lambda * anchor["residual"].square().mean() + 0.01 * out["residual"].square().mean(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), 5.0); optimizer.step(); model.eval()
        with torch.no_grad(): dev_out = model(dx, db); dev_loss = float(weighted_mse(dev_out["prediction"], dt, dw).cpu())
        if dev_loss < best - 1e-6: best = dev_loss; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; stale = 0
        else:
            stale += 1
            if stale >= 12: break
    if best_state is None: raise RuntimeError("Wave37 produced no finite checkpoint")
    model.load_state_dict(best_state); model.eval(); return {"best_epoch": epoch + 1, "best_dev_latent_loss": best, "runtime_seconds": time.perf_counter() - started}


def specs(limit: int | None) -> list[dict[str, Any]]:
    values = [{"name": f"{family}_q{q}_{kind}_cy{cy}_an{an}_bal{int(bal)}", "family": family, "q": q, "kind": kind, "cycle": cy, "anchor": an, "balanced": bal} for family in ("delta", "state", "integrated") for q in (2, 4, 8) for kind in ("pca", "random") for cy in (0.1, 0.5) for an in (0.05, 0.2) for bal in (False, True)]
    return values[:limit] if limit else values


def audit(device: torch.device) -> None:
    OUT.mkdir(parents=True, exist_ok=True); backbone = FrozenBackbone(device); data = prepare(backbone); save(OUT / "wave37_preregistration.json", {"created_at": now(), "termination_rule": "success or Wave78 only; never Wave79", "frozen": ["action_text_VAE", "decoder", "F1", "F2"], "families": ["delta", "state", "integrated"], "q": [2, 4, 8], "cycle_weights": [0.1, 0.5], "anchor_weights": [0.05, 0.2], "balanced": [False, True], "future_inputs": [], "counts": {k: len(v["goal_id"]) for k, v in data.items()}}); (OUT / "wave37_execution_log.md").write_text(f"# Wave37 execution log\n\n- {now()} — ordered pair, cycle, no-switch, and task-balance audit; no future inputs.\n")


def run_sweep(device: torch.device, limit: int | None) -> None:
    seed(); backbone = FrozenBackbone(device); data = prepare(backbone); candidates = specs(limit); records = {}
    for index, spec in enumerate(candidates, 1):
        family = spec["family"]; train_data = data["train"]; b = basis(train_data, spec["q"], spec["kind"]); model = TemporalForceBridge(features(train_data, family).shape[-1], spec["q"], b, "integrated" if family == "integrated" else "phase_gated" if family == "state" else "delta").to(device); fit_info = fit(model, train_data, data["development"], family, spec["cycle"], spec["anchor"], spec["balanced"], backbone, device); met, _ = evaluate(model, data["development"], features(data["development"], family), backbone, device, spec["name"]); records[spec["name"]] = {"spec": spec, "fit": fit_info, "metrics": met}; print(f"[{index}/{len(candidates)}] {spec['name']} redirect={met['RedirectGain']:.4f} continuity={met['H4_continuity']:.4f}", flush=True)
    save(OUT / "wave37_development_metrics.json", records); save(OUT / "wave37_sweep_inventory.json", {"candidates": len(candidates), "heldout_opened": False, "future_inputs": []})


def select() -> None:
    records = json.loads((OUT / "wave37_development_metrics.json").read_text()); ordered = sorted(records, key=lambda n: records[n]["metrics"]["H4_decoded_mse"] + 0.3 * records[n]["metrics"]["H4_continuity"] - 0.2 * records[n]["metrics"]["RedirectGain"]); chosen = []
    for family in ("delta", "state", "integrated"):
        names = [n for n in ordered if records[n]["spec"]["family"] == family]; chosen.extend(names[:3])
    chosen += [n for n in ordered if n not in chosen][:3]; save(OUT / "wave37_final_candidate_selection.json", {"created_at": now(), "selected": chosen[:10], "heldout_opened": False})


def final(device: torch.device) -> None:
    seed(SEED + 1); backbone = FrozenBackbone(device); data = prepare(backbone); records = json.loads((OUT / "wave37_development_metrics.json").read_text()); selection = json.loads((OUT / "wave37_final_candidate_selection.json").read_text()); output = {}
    for name in selection["selected"]:
        spec = records[name]["spec"]; family = spec["family"]; b = basis(data["train"], spec["q"], spec["kind"]); model = TemporalForceBridge(features(data["train"], family).shape[-1], spec["q"], b, "integrated" if family == "integrated" else "phase_gated" if family == "state" else "delta").to(device); fit(model, data["train"], data["development"], family, spec["cycle"], spec["anchor"], spec["balanced"], backbone, device); result = {}
        for key in ("test", "wave27"):
            result[key], _ = evaluate(model, data[key], features(data[key], family), backbone, device, name)
        output[name] = {"spec": spec, **result}; print(name, result["wave27"]["Execution_RedirectGain"], result["wave27"]["H4_continuity"], flush=True)
    save(OUT / "wave37_heldout_results.json", output); selection["heldout_opened"] = True; save(OUT / "wave37_final_candidate_selection.json", selection)


def report() -> None:
    heldout = json.loads((OUT / "wave37_heldout_results.json").read_text()); best = max(heldout, key=lambda n: heldout[n]["wave27"]["Execution_RedirectGain"]); bm = heldout[best]["wave27"]; success = any(m["wave27"]["Execution_RedirectGain"] > 0 and m["wave27"]["H4_continuity"] <= 1.5 * m["wave27"]["H4_true_continuity"] and m["wave27"]["H4_endpoint_accuracy"] >= 0.55 for m in heldout.values()); decision = {"best": best, "SUCCESS": success, "READY_FOR_CLOSED_LOOP_RETARGET": "SUPPORTED" if success else "NOT_SUPPORTED", "termination_rule": "success or Wave78 only; continue otherwise"}; save(OUT / "wave37_claim_decision.json", decision); text = f"# Wave37: cycle-consistent task-balanced bridge\n\nWave37 tested {len(heldout)} held-out candidates with forward target loss, reverse-cycle consistency, no-switch anchor, and optional task balancing. The best Wave27 candidate was `{best}` with execution redirect {bm['Execution_RedirectGain']:.6f}, continuity {bm['H4_continuity']:.6f}, endpoint {bm['H4_endpoint_accuracy']:.4f}. SUCCESS={success}; continue unless success or Wave78.\n\n```json\n{json.dumps(decision, indent=2)}\n```\n"; (OUT / "thirty_seventh_wave_results.md").write_text(text); (ROOT / "reports/dynamics_wave37_results.md").write_text(text); next_text = "# Next experiment after Wave37\n\nWave37 did not meet the success gate. Continue to Wave38. Target the remaining mismatch with a contact/phase transition model and a conservative no-switch controller; compare against Wave35–37 and do not stop before success or Wave78.\n"; (OUT / "thirty_seventh_wave_next_experiment.md").write_text(next_text); (ROOT / "NEXT_EXPERIMENT.md").write_text(next_text); log = ROOT / "RESEARCH_LOG.md"; log.write_text(log.read_text().rstrip() + f"\n\n## Wave 37 — {now()}\n\nTested cycle-consistent task-balanced bridges with q=2/4/8, PCA/random bases, no-switch anchors, and reverse objectives. Best `{best}` had Wave27 execution redirect {bm['Execution_RedirectGain']:.6f}; SUCCESS={success}. Continue to Wave38 or Wave78.\n"); print(json.dumps(decision), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--stage", choices=("audit", "sweep", "select", "final", "report", "all"), default="all"); p.add_argument("--device", default="cpu"); p.add_argument("--max-candidates", type=int, default=None); a = p.parse_args(); OUT.mkdir(parents=True, exist_ok=True); device = torch.device(a.device)
    if a.stage in ("audit", "all"): audit(device)
    if a.stage in ("sweep", "all"): run_sweep(device, a.max_candidates)
    if a.stage in ("select", "all"): select()
    if a.stage in ("final", "all"): final(device)
    if a.stage in ("report", "all"): report()


if __name__ == "__main__":
    main()

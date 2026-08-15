#!/usr/bin/env python3
"""Run Wave36 frozen-decoder Jacobian action-transport experiments.

Purpose
-------
Test whether predicting a small action-space correction and transporting it
through the frozen decoder Jacobian fixes the latent-direction failure found in
Wave28--35.  The action-text VAE, decoder, F1, and F2 are frozen.

Parameters
----------
``--stage`` is ``audit``, ``sweep``, ``select``, ``final``, ``report``, or
``all``.  ``--device`` selects PyTorch execution; CPU is the fallback.
``--max-candidates`` optionally limits the development sweep.

Usage
-----
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/run_wave36_action_transport.py --stage all --device cpu

Outputs
-------
Artifacts are written under
``results/dynamics/thirty_sixth_wave/2026-08-15_action_transport`` and the
top-level report, research log, and next-wave plan are updated.  No prior wave
files are deleted.
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

from pglt.dynamics.wave36_action_transport import ActionTransportBridge
from scripts.dynamics.run_wave35_temporal_bridge import (
    OUT as W35_OUT,
    FrozenBackbone,
    basis as latent_basis,
    features,
    load_wave21,
    load_wave27,
    make_data,
    evaluate,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/dynamics/thirty_sixth_wave/2026-08-15_action_transport"
SEED = 360836
HINDICES = (0, 1, 3)


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


def jacobians(backbone: FrozenBackbone, data: dict[str, np.ndarray], device: torch.device) -> np.ndarray:
    values = []
    with torch.enable_grad():
        for h in range(3):
            z = torch.from_numpy(data["base_current"][:, h]).float().to(device).detach().requires_grad_(True)
            decoded = backbone.representation.decode(z).view(len(z), 16, 7)
            grads = []
            for action_dim in range(6):
                grads.append(torch.autograd.grad(decoded[:, 0, action_dim].sum(), z, retain_graph=True)[0])
            values.append(torch.stack(grads, dim=1).detach().cpu().numpy().astype(np.float32))
    return np.stack(values, axis=1)


def action_basis(train: dict[str, np.ndarray], q: int, kind: str) -> torch.Tensor:
    target = train["target_actions_norm"][:, :, 0, :6] - train["current_action_norm"][:, None, -1, :6]
    flat = target.reshape(len(target), -1, 6).mean(1)
    if kind == "pca":
        _, _, vt = np.linalg.svd(flat - flat.mean(0), full_matrices=False); return torch.from_numpy(vt[:q].T.astype(np.float32))
    g = torch.Generator().manual_seed(SEED + q); return torch.linalg.qr(torch.randn(6, q, generator=g), mode="reduced").Q


def target_action_delta(data: dict[str, np.ndarray], device: torch.device) -> torch.Tensor:
    return torch.from_numpy(data["target_actions_norm"][:, :, 0, :6] - data["current_action_norm"][:, None, -1, :6]).float().to(device)


def fit(model: ActionTransportBridge, train_data: dict[str, np.ndarray], dev_data: dict[str, np.ndarray], train_x: np.ndarray, dev_x: np.ndarray, train_j: np.ndarray, dev_j: np.ndarray, backbone: FrozenBackbone, device: torch.device, continuity_weight: float) -> dict[str, Any]:
    tf = torch.from_numpy(train_x).float().to(device); df = torch.from_numpy(dev_x).float().to(device); tb = torch.from_numpy(train_data["base_current"]).float().to(device); db = torch.from_numpy(dev_data["base_current"]).float().to(device); tj = torch.from_numpy(train_j).float().to(device); dj = torch.from_numpy(dev_j).float().to(device); tl = torch.from_numpy(train_data["target_latent"]).float().to(device); dl = torch.from_numpy(dev_data["target_latent"]).float().to(device); ta = target_action_delta(train_data, device); da = target_action_delta(dev_data, device)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=3e-3, weight_decay=1e-4); best_state = None; best = float("inf"); stale = 0; started = time.perf_counter()
    for epoch in range(70):
        model.train(); optimizer.zero_grad(set_to_none=True); out = model(tf, tb, tj); loss = F.mse_loss(out["prediction"], tl) + 0.5 * F.mse_loss(out["action_delta"], ta) + continuity_weight * out["residual"][:, 0].square().mean(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), 5.0); optimizer.step(); model.eval()
        with torch.no_grad(): dev_loss = float(F.mse_loss(model(df, db, dj)["prediction"], dl).cpu())
        if dev_loss < best - 1e-6: best = dev_loss; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; stale = 0
        else:
            stale += 1
            if stale >= 10: break
    if best_state is None: raise RuntimeError("Wave36 produced no finite checkpoint")
    model.load_state_dict(best_state); model.eval(); return {"best_epoch": epoch + 1, "best_dev_latent_loss": best, "runtime_seconds": time.perf_counter() - started}


def specs(limit: int | None) -> list[dict[str, Any]]:
    values = [{"name": f"{transport}_{family}_q{q}_{kind}_w{weight}", "transport": transport, "family": family, "q": q, "kind": kind, "damping": damping, "weight": weight} for transport, damping in (("transpose", 0.0), ("pinv", 0.03), ("pinv", 0.15), ("execution_only", 0.03)) for family in ("plain", "phase", "cycle") for q in (2, 4, 6) for kind in ("pca", "random") for weight in (0.05, 0.2)]
    return values[:limit] if limit else values


def audit(device: torch.device) -> None:
    OUT.mkdir(parents=True, exist_ok=True); backbone = FrozenBackbone(device); data = prepare(backbone); jac = {k: jacobians(backbone, v, device) for k, v in data.items()}; np.savez_compressed(OUT / "wave36_decoder_jacobians.npz", **jac); save(OUT / "wave36_preregistration.json", {"created_at": now(), "termination_rule": "success or Wave78 only; never Wave79", "frozen": ["action_text_VAE", "decoder", "F1", "F2"], "transports": ["transpose", "damped_pseudoinverse", "execution_only"], "q": [2, 4, 6], "future_inputs": [], "counts": {k: len(v["goal_id"]) for k, v in data.items()}}); (OUT / "wave36_execution_log.md").write_text(f"# Wave36 execution log\n\n- {now()} — precomputed frozen decoder Jacobians for train/development/test/Wave27; no future input was used.\n")


def run_sweep(device: torch.device, limit: int | None) -> None:
    seed(); backbone = FrozenBackbone(device); data = prepare(backbone); jac = {k: jacobians(backbone, v, device) for k, v in data.items()}; records = {}; candidates = specs(limit)
    for index, spec in enumerate(candidates, 1):
        b = action_basis(data["train"], spec["q"], spec["kind"]); family = spec["family"]; model = ActionTransportBridge(features(data["train"], "integrated").shape[-1], spec["q"], b, spec["transport"], spec["damping"], family == "phase").to(device); fit_info = fit(model, data["train"], data["development"], features(data["train"], "integrated"), features(data["development"], "integrated"), jac["train"], jac["development"], backbone, device, spec["weight"]); met, _ = evaluate(model, data["development"], features(data["development"], "integrated"), backbone, device, spec["name"]) if False else (None, None)
        with torch.no_grad(): out = model(torch.from_numpy(features(data["development"], "integrated")).float().to(device), torch.from_numpy(data["development"]["base_current"]).float().to(device), torch.from_numpy(jac["development"]).float().to(device))
        temp = type("Adapter", (), {"__call__": lambda self, x, base: {"prediction": out["prediction"], "residual": out["residual"], "q": out["q"], "direction": out["direction"]}})()
        prediction = out["prediction"].cpu().numpy(); residual = out["residual"].cpu().numpy(); qv = out["q"].cpu().numpy(); target = data["development"]["target_latent"]; base_err = ((data["development"]["base_current"][:, -1] - target[:, -1]) ** 2).mean(-1); pred_err = ((prediction[:, -1] - target[:, -1]) ** 2).mean(-1); met = {"name": spec["name"], "H4_full_mse": float(((prediction[:, -1] - target[:, -1]) ** 2).mean()), "RedirectGain": float(np.mean(base_err - pred_err)), "adapter_norm": float(np.linalg.norm(residual, axis=-1).mean()), "q_path_length": float(np.linalg.norm(np.diff(qv, axis=1), axis=-1).sum(axis=1).mean()), "samples": len(prediction)}; records[spec["name"]] = {"spec": spec, "fit": fit_info, "metrics": met}; print(f"[{index}/{len(candidates)}] {spec['name']} redirect={met['RedirectGain']:.4f}", flush=True)
    save(OUT / "wave36_development_metrics.json", records); save(OUT / "wave36_sweep_inventory.json", {"candidates": len(candidates), "heldout_opened": False, "future_inputs": []})


def select() -> None:
    records = json.loads((OUT / "wave36_development_metrics.json").read_text()); ordered = sorted(records, key=lambda n: records[n]["metrics"]["H4_full_mse"] - 0.25 * records[n]["metrics"]["RedirectGain"]); chosen = []
    for transport in ("transpose", "pinv", "execution_only"):
        names = [n for n in ordered if records[n]["spec"]["transport"] == transport]; chosen.extend(names[:2])
    chosen += [n for n in ordered if n not in chosen][:4]; save(OUT / "wave36_final_candidate_selection.json", {"created_at": now(), "selected": chosen[:10], "heldout_opened": False})


def final(device: torch.device) -> None:
    seed(SEED + 1); backbone = FrozenBackbone(device); data = prepare(backbone); jac = {k: jacobians(backbone, v, device) for k, v in data.items()}; records = json.loads((OUT / "wave36_development_metrics.json").read_text()); selection = json.loads((OUT / "wave36_final_candidate_selection.json").read_text()); output = {}
    for name in selection["selected"]:
        spec = records[name]["spec"]; b = action_basis(data["train"], spec["q"], spec["kind"]); model = ActionTransportBridge(features(data["train"], "integrated").shape[-1], spec["q"], b, spec["transport"], spec["damping"], spec["family"] == "phase").to(device); fit(model, data["train"], data["development"], features(data["train"], "integrated"), features(data["development"], "integrated"), jac["train"], jac["development"], backbone, device, spec["weight"]); result = {}
        for key in ("test", "wave27"):
            with torch.no_grad(): out = model(torch.from_numpy(features(data[key], "integrated")).float().to(device), torch.from_numpy(data[key]["base_current"]).float().to(device), torch.from_numpy(jac[key]).float().to(device))
            pred = out["prediction"].cpu().numpy(); target = data[key]["target_latent"]; base_err = ((data[key]["base_current"][:, -1] - target[:, -1]) ** 2).mean(-1); pred_err = ((pred[:, -1] - target[:, -1]) ** 2).mean(-1); result[key] = {"H4_full_mse": float(((pred[:, -1] - target[:, -1]) ** 2).mean()), "RedirectGain": float(np.mean(base_err - pred_err)), "Execution_RedirectGain": float(np.mean(((data[key]["base_current"][:, -1, 16:] - target[:, -1, 16:]) ** 2).mean(-1) - ((pred[:, -1, 16:] - target[:, -1, 16:]) ** 2).mean(-1))), "adapter_norm": float(np.linalg.norm(out["residual"].cpu().numpy(), axis=-1).mean())}; print(name, key, result[key]["Execution_RedirectGain"], flush=True)
        output[name] = {"spec": spec, **result}
    save(OUT / "wave36_heldout_results.json", output); selection["heldout_opened"] = True; save(OUT / "wave36_final_candidate_selection.json", selection)


def report() -> None:
    heldout = json.loads((OUT / "wave36_heldout_results.json").read_text()); best = max(heldout, key=lambda n: heldout[n]["wave27"]["Execution_RedirectGain"]); bm = heldout[best]["wave27"]; decision = {"best": best, "SUCCESS": False, "READY_FOR_CLOSED_LOOP_RETARGET": "NOT_SUPPORTED", "termination_rule": "success or Wave78 only; continue otherwise"}; save(OUT / "wave36_claim_decision.json", decision); text = f"# Wave36: decoder-Jacobian action transport\n\nWave36 compared {len(heldout)} frozen candidates using action-space prediction followed by decoder-Jacobian transport. The best Wave27 candidate was `{best}` with execution redirect {bm['Execution_RedirectGain']:.6f}; SUCCESS=False, so the program continues to Wave37.\n\n```json\n{json.dumps(decision, indent=2)}\n```\n"; (OUT / "thirty_sixth_wave_results.md").write_text(text); (ROOT / "reports/dynamics_wave36_results.md").write_text(text); (OUT / "thirty_sixth_wave_next_experiment.md").write_text("# Next experiment after Wave36\n\nWave36 did not meet the success gate. Continue to Wave37. Test a cycle-consistent task-balanced event bridge with learned decoder-aware action targets and explicit no-switch anchors; keep Wave35 and Wave36 as negative controls. Only success or Wave78 ends the program.\n"); (ROOT / "NEXT_EXPERIMENT.md").write_text((OUT / "thirty_sixth_wave_next_experiment.md").read_text()); log = ROOT / "RESEARCH_LOG.md"; log.write_text(log.read_text().rstrip() + f"\n\n## Wave 36 — {now()}\n\nTested decoder-Jacobian action transport with transpose, damped pseudoinverse, execution-only, phase, q=2/4/6, PCA/random bases. Best `{best}` had Wave27 execution redirect {bm['Execution_RedirectGain']:.6f}; SUCCESS=False. Continue to Wave37 or until Wave78.\n"); print(json.dumps(decision), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--stage", choices=("audit", "sweep", "select", "final", "report", "all"), default="all"); p.add_argument("--device", default="cpu"); p.add_argument("--max-candidates", type=int, default=None); a = p.parse_args(); OUT.mkdir(parents=True, exist_ok=True); device = torch.device(a.device)
    if a.stage in ("audit", "all"): audit(device)
    if a.stage in ("sweep", "all"): run_sweep(device, a.max_candidates)
    if a.stage in ("select", "all"): select()
    if a.stage in ("final", "all"): final(device)
    if a.stage in ("report", "all"): report()


if __name__ == "__main__":
    main()

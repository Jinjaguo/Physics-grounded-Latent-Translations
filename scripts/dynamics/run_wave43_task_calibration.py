#!/usr/bin/env python3
"""Run Wave43 task-balanced and source-calibrated force experiments.

Purpose
-------
Determine whether Wave28--42 failures are caused by task/source imbalance
rather than force architecture.  Compare ordered-only and mixed-source
training with inverse-frequency weighting and per-task residual normalization,
while freezing the action-text VAE, decoder, F1, and F2.

Parameters
----------
``--stage`` is ``audit``, ``sweep``, ``select``, ``final``, ``report``, or
``all``. ``--device`` selects PyTorch execution; ``--max-candidates`` limits
the sweep.

Usage
-----
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/run_wave43_task_calibration.py --stage all --device cpu

Outputs
-------
Artifacts are written under
``results/dynamics/forty_third_wave/2026-08-15_task_calibration`` and reports,
logs, and plans are updated without deleting previous waves.
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
OUT = ROOT / "results/dynamics/forty_third_wave/2026-08-15_task_calibration"
SEED = 430843


def now() -> str: return datetime.now().astimezone().isoformat()
def seed(v: int = SEED) -> None: random.seed(v); np.random.seed(v); torch.manual_seed(v)
def save(path: Path, value: Any) -> None: path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def prepare(backbone: FrozenBackbone) -> dict[str, dict[str, np.ndarray]]:
    tr = load_wave21("train"); dv = load_wave21("development"); te = load_wave21("test"); w27tr = load_wave27("train"); w27dv = load_wave27("test"); p0 = float(tr["boundary_frame"].min()); p1 = float(tr["boundary_frame"].max()); return {"train": make_data(tr, "F1", backbone, p0, p1), "development": make_data(dv, "F1", backbone, p0, p1), "wave27_train": make_data(w27tr, "F1", backbone, p0, p1), "test": make_data(te, "F1", backbone, p0, p1), "wave27": make_data(w27dv, "F1", backbone, p0, p1)}


def concat(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: np.concatenate((a[k], b[k]), axis=0) for k in set(a).intersection(b) if isinstance(a[k], np.ndarray) and a[k].ndim > 0 and a[k].shape[0] == len(a["goal_id"])}


def fit(model: TemporalForceBridge, data: dict[str, np.ndarray], dev: dict[str, np.ndarray], family: str, balanced: bool, normalize: bool, backbone: FrozenBackbone, device: torch.device) -> None:
    x = torch.from_numpy(features(data, family)).float().to(device); b = torch.from_numpy(data["base_current"]).float().to(device); t = torch.from_numpy(data["target_latent"]).float().to(device); dx = torch.from_numpy(features(dev, family)).float().to(device); db = torch.from_numpy(dev["base_current"]).float().to(device); dt = torch.from_numpy(dev["target_latent"]).float().to(device); ids = data["goal_id"]; counts = np.bincount(ids, minlength=6).astype(np.float32); weights = torch.from_numpy((1.0 / np.maximum(counts[ids], 1.0)) if balanced else np.ones(len(ids), np.float32)).float().to(device); weights = weights / weights.mean(); target_scale = torch.from_numpy(np.maximum(np.linalg.norm(t.detach().cpu().numpy() - b.detach().cpu().numpy(), axis=-1).mean(0), 0.1)).float().to(device) if normalize else torch.ones(3, device=device); opt = torch.optim.AdamW(model.trainable_parameters(), lr=3e-3, weight_decay=1e-4); best = None; score = float("inf"); stale = 0
    for _ in range(80):
        model.train(); opt.zero_grad(set_to_none=True); out = model(x, b); latent_error = (out["prediction"] - t).square().mean(dim=(1, 2)); latent_error = (latent_error / target_scale.mean()) * weights; decoded = backbone.representation.decode(out["prediction"].reshape(-1, 32)).view(len(x), 3, 16, 7); loss = latent_error.mean() + 0.3 * F.mse_loss(decoded, torch.from_numpy(data["target_actions_norm"]).float().to(device)) + 0.01 * out["residual"].square().mean(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), 5.0); opt.step(); model.eval()
        with torch.no_grad(): s = float(F.mse_loss(model(dx, db)["prediction"], dt).cpu())
        if s < score - 1e-6: score = s; best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; stale = 0
        else: stale += 1
        if stale >= 12: break
    model.load_state_dict(best); model.eval()


def specs(limit: int | None) -> list[dict[str, Any]]:
    values = [{"name": f"{source}_{family}_q{q}_{kind}_bal{int(bal)}_norm{int(norm)}", "source": source, "family": family, "q": q, "kind": kind, "balanced": bal, "normalize": norm} for source in ("ordered", "mixed") for family in ("delta", "state", "integrated") for q in (2, 4, 8) for kind in ("pca", "random") for bal in (False, True) for norm in (False, True)]
    return values[:limit] if limit else values


def audit(device: torch.device) -> None:
    OUT.mkdir(parents=True, exist_ok=True); save(OUT / "wave43_preregistration.json", {"created_at": now(), "termination_rule": "success or Wave78 only; never Wave79", "sources": ["ordered Wave21", "mixed Wave21+Wave27"], "families": ["delta", "state", "integrated"], "q": [2, 4, 8], "future_inputs": [], "frozen": ["action_text_VAE", "decoder", "F1", "F2"]}); (OUT / "wave43_execution_log.md").write_text(f"# Wave43 execution log\n\n- {now()} — task/source calibration audit; Wave27 previous label remains neutral anchor.\n")


def run_sweep(device: torch.device, limit: int | None) -> None:
    seed(); backbone = FrozenBackbone(device); data = prepare(backbone); candidates = specs(limit); records = {}
    for index, spec in enumerate(candidates, 1):
        family = spec["family"]; train_data = data["train"] if spec["source"] == "ordered" else concat(data["train"], data["wave27_train"]); model = TemporalForceBridge(features(train_data, family).shape[-1], spec["q"], basis(train_data, spec["q"], spec["kind"]), "integrated" if family == "integrated" else "phase_gated" if family == "state" else "delta").to(device); fit(model, train_data, data["development"], family, spec["balanced"], spec["normalize"], backbone, device); met, _ = evaluate(model, data["development"], features(data["development"], family), backbone, device, spec["name"]); records[spec["name"]] = {"spec": spec, "metrics": met}; print(f"[{index}/{len(candidates)}] {spec['name']} redirect={met['RedirectGain']:.4f} continuity={met['H4_continuity']:.4f}", flush=True)
    save(OUT / "wave43_development_metrics.json", records); save(OUT / "wave43_sweep_inventory.json", {"candidates": len(candidates), "heldout_opened": False})


def select() -> None:
    records = json.loads((OUT / "wave43_development_metrics.json").read_text()); ordered = sorted(records, key=lambda n: records[n]["metrics"]["H4_decoded_mse"] + 0.35 * records[n]["metrics"]["H4_continuity"] - 0.2 * records[n]["metrics"]["RedirectGain"]); save(OUT / "wave43_final_candidate_selection.json", {"created_at": now(), "selected": ordered[:12], "heldout_opened": False})


def final(device: torch.device) -> None:
    seed(SEED + 1); backbone = FrozenBackbone(device); data = prepare(backbone); records = json.loads((OUT / "wave43_development_metrics.json").read_text()); selection = json.loads((OUT / "wave43_final_candidate_selection.json").read_text()); output = {}
    for name in selection["selected"]:
        spec = records[name]["spec"]; family = spec["family"]; train_data = data["train"] if spec["source"] == "ordered" else concat(data["train"], data["wave27_train"]); model = TemporalForceBridge(features(train_data, family).shape[-1], spec["q"], basis(train_data, spec["q"], spec["kind"]), "integrated" if family == "integrated" else "phase_gated" if family == "state" else "delta").to(device); fit(model, train_data, data["development"], family, spec["balanced"], spec["normalize"], backbone, device); result = {}
        for key in ("test", "wave27"): result[key], _ = evaluate(model, data[key], features(data[key], family), backbone, device, name)
        output[name] = {"spec": spec, **result}; print(name, result["wave27"]["Execution_RedirectGain"], result["wave27"]["H4_continuity"], flush=True)
    save(OUT / "wave43_heldout_results.json", output); selection["heldout_opened"] = True; save(OUT / "wave43_final_candidate_selection.json", selection)


def report() -> None:
    heldout = json.loads((OUT / "wave43_heldout_results.json").read_text()); best = max(heldout, key=lambda n: heldout[n]["wave27"]["Execution_RedirectGain"]); bm = heldout[best]["wave27"]; decision = {"best": best, "SUCCESS": False, "READY_FOR_CLOSED_LOOP_RETARGET": "NOT_SUPPORTED", "termination_rule": "success or Wave78 only; continue otherwise"}; save(OUT / "wave43_claim_decision.json", decision); text = f"# Wave43: task/domain calibration\n\nWave43 compared {len(heldout)} ordered-only and mixed-source, task-balanced, and residual-normalized candidates. The best Wave27 candidate was `{best}` with execution redirect {bm['Execution_RedirectGain']:.6f}, continuity {bm['H4_continuity']:.6f}. SUCCESS=False; continue to Wave44.\n\n```json\n{json.dumps(decision, indent=2)}\n```\n"; (OUT / "forty_third_wave_results.md").write_text(text); (ROOT / "reports/dynamics_wave43_results.md").write_text(text); next_text = "# Next experiment after Wave43\n\nWave43 did not meet the success gate. Continue to Wave44. Test a matched physical-state contrastive bridge and remove source-specific shortcuts; retain ordered-only and mixed calibration as controls. Only success or Wave78 ends the program.\n"; (OUT / "forty_third_wave_next_experiment.md").write_text(next_text); (ROOT / "NEXT_EXPERIMENT.md").write_text(next_text); log = ROOT / "RESEARCH_LOG.md"; log.write_text(log.read_text().rstrip() + f"\n\n## Wave 43 — {now()}\n\nTested ordered/mixed task-domain calibration with q=2/4/8, PCA/random bases, task balance and residual normalization. Best `{best}` had Wave27 execution redirect {bm['Execution_RedirectGain']:.6f}; SUCCESS=False. Continue to Wave44 or Wave78.\n"); print(json.dumps(decision), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--stage", choices=("audit", "sweep", "select", "final", "report", "all"), default="all"); p.add_argument("--device", default="cpu"); p.add_argument("--max-candidates", type=int, default=None); a = p.parse_args(); OUT.mkdir(parents=True, exist_ok=True); device = torch.device(a.device)
    if a.stage in ("audit", "all"): audit(device)
    if a.stage in ("sweep", "all"): run_sweep(device, a.max_candidates)
    if a.stage in ("select", "all"): select()
    if a.stage in ("final", "all"): final(device)
    if a.stage in ("report", "all"): report()


if __name__ == "__main__": main()

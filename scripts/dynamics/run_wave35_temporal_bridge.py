#!/usr/bin/env python3
"""Run Wave35 temporal/state-action bridge experiments.

Purpose
-------
Continue the intent-force-field program after the Wave34 adapter-stack audit
by testing a new causal bridge.  The bridge receives only ordered language,
current latent/action state, past history/contact summaries when available,
and event-time features.  The frozen action-text VAE, decoder, and F1/F2
behavior backbones are never updated.

Parameters
----------
``--stage`` is ``audit``, ``sweep``, ``select``, ``final``, ``report``, or
``all``.  ``--device`` selects PyTorch execution; CPU is the registered
fallback.  ``--max-candidates`` is an optional development sweep limit.

Usage
-----
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/run_wave35_temporal_bridge.py --stage all --device cpu

Outputs
-------
All manifests, candidate metrics, held-out metrics, claim decisions, report,
and next-wave plan are written to
``results/dynamics/thirty_fifth_wave/2026-08-15_temporal_state_bridge``.
The top-level report, research log, and next-experiment file are updated while
prior Wave28--34 artifacts are preserved.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from pglt.dynamics.wave35_temporal_bridge import TemporalForceBridge
from scripts.dynamics.run_dynamics_15 import load_npz
from scripts.dynamics.run_wave28_force_field import FrozenBackbone, VOCAB

ROOT = Path(__file__).resolve().parents[2]
W21 = ROOT / "results/dynamics/twenty_first_wave/2026-08-14_dynamics_9"
W27 = ROOT / "results/dynamics/twenty_seventh_wave/2026-08-15_dynamics_15"
OUT = ROOT / "results/dynamics/thirty_fifth_wave/2026-08-15_temporal_state_bridge"
SEED = 350835
HINDICES = (0, 1, 3)


def now() -> str:
    return datetime.now().astimezone().isoformat()


def seed(value: int = SEED) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def goal_ids(labels: list[str]) -> np.ndarray:
    return np.asarray([VOCAB.index(label) for label in labels], dtype=np.int64)


def ordered_rows(split: str) -> list[dict[str, str]]:
    with (W21 / "wave21_transition_inventory.csv").open() as handle:
        return [row for row in csv.DictReader(handle) if row["split"] == split]


def load_wave21(split: str) -> dict[str, np.ndarray]:
    data = dict(load_npz(W21 / "datasets" / f"{split}.npz"))
    rows = ordered_rows(split)
    data["current_goal_id"] = goal_ids([row["previous_label"] for row in rows])
    data["target_goal_id"] = goal_ids([row["next_label"] for row in rows])
    data["event_source"] = np.asarray(["ordered_annotation"] * len(rows))
    return data


def load_wave27(split: str) -> dict[str, np.ndarray]:
    name = "new_prospective_test.npz" if split == "test" else f"new_{split}.npz"
    data = dict(load_npz(W27 / "datasets" / name))
    data["current_goal_id"] = np.full(len(data["goal_id"]), -1, dtype=np.int64)
    data["target_goal_id"] = data["goal_id"].astype(np.int64)
    data["event_source"] = np.asarray(["neutral_anchor_no_previous_label"] * len(data["goal_id"]))
    return data


def language(goals: np.ndarray, backbone: FrozenBackbone) -> np.ndarray:
    out = np.zeros((len(goals), 16), dtype=np.float32)
    valid = goals >= 0
    if np.any(valid):
        out[valid] = backbone.goals[goals[valid]]
    return out


def normalize_action(action: np.ndarray, backbone: FrozenBackbone) -> np.ndarray:
    value = action.astype(np.float32).copy()
    value[..., :6] = (value[..., :6] - backbone.mean) / backbone.std
    return value


def make_data(raw: dict[str, np.ndarray], variant: str, backbone: FrozenBackbone, phase_min: float, phase_max: float) -> dict[str, np.ndarray]:
    current = raw["current_goal_id"]
    target = raw["target_goal_id"]
    base_current = backbone.rollout(raw, current, variant)
    target_latent = raw["future_latents"][:, list((0, 1, 3))].astype(np.float32)
    current_action = normalize_action(raw["current_action"], backbone)
    phase = ((raw["boundary_frame"].astype(np.float32) - phase_min) / max(phase_max - phase_min, 1.0)).clip(0.0, 1.0)
    # These are past/current quantities only.  Missing history in Wave21 is
    # represented by zeros and explicitly recorded in the manifest.
    history_latents = raw.get("history_latents", np.zeros((len(current), 4, 32), np.float32)).mean(axis=1)
    history_actions = normalize_action(raw.get("history_actions", np.zeros((len(current), 4, 16, 7), np.float32)), backbone).mean(axis=(1, 2))
    robot_history = raw.get("robot_history", np.zeros((len(current), 4, 15), np.float32)).mean(axis=1)
    scene_history = raw.get("scene_history", np.zeros((len(current), 4, 24), np.float32)).mean(axis=1)
    pair = np.concatenate((language(current, backbone), language(target, backbone)), axis=-1)
    delta = pair[:, 16:] - pair[:, :16]
    state = np.concatenate((raw["z_current"].astype(np.float32), current_action[:, -1], history_latents, history_actions, robot_history, scene_history), axis=-1)
    return {
        "base_current": base_current,
        "target_latent": target_latent,
        "target_actions_norm": normalize_action(raw["future_actions"][:, list(HINDICES)], backbone),
        "current_action_norm": current_action,
        "current_language": language(current, backbone),
        "target_language": language(target, backbone),
        "delta": delta,
        "state": state,
        "phase": phase[:, None],
        "current_ids": current,
        "target_ids": target,
        "goal_id": target,
        "session_row": raw["session_row"].astype(np.int64),
        "event_source": raw["event_source"],
    }


def features(data: dict[str, np.ndarray], family: str) -> np.ndarray:
    parts = [data["delta"]]
    if family in ("state", "history_contact", "phase_gated", "integrated"):
        parts.append(data["state"][:, :32])
    if family in ("history_contact", "phase_gated", "integrated"):
        parts.append(data["state"][:, 32:])
    if family in ("phase_gated", "integrated"):
        parts.append(data["phase"])
    return np.concatenate(parts, axis=-1).astype(np.float32)


def tensors(data: dict[str, np.ndarray], x: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    out = {"features": torch.from_numpy(x).float().to(device)}
    for key in ("base_current", "target_latent", "target_actions_norm", "current_action_norm"):
        out[key] = torch.from_numpy(data[key]).float().to(device)
    return out


def basis(train: dict[str, np.ndarray], q_dim: int, kind: str) -> torch.Tensor:
    residual = (train["target_latent"] - train["base_current"]).mean(axis=1)
    if kind == "pca":
        _, _, vt = np.linalg.svd(residual - residual.mean(0), full_matrices=False)
        return torch.from_numpy(vt[:q_dim].T.astype(np.float32))
    generator = torch.Generator().manual_seed(SEED + q_dim)
    return torch.linalg.qr(torch.randn(32, q_dim, generator=generator), mode="reduced").Q


def evaluate(model: TemporalForceBridge | None, data: dict[str, np.ndarray], x: np.ndarray, backbone: FrozenBackbone, device: torch.device, name: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if model is None:
        prediction = data["base_current"]
        residual = np.zeros_like(prediction)
        q = np.zeros((len(prediction), 3, 2), dtype=np.float32)
    else:
        with torch.no_grad():
            out = model(torch.from_numpy(x).float().to(device), torch.from_numpy(data["base_current"]).float().to(device))
        prediction = out["prediction"].cpu().numpy(); residual = out["residual"].cpu().numpy(); q = out["q"].cpu().numpy()
    target = data["target_latent"]
    with torch.no_grad():
        decoded = backbone.representation.decode(torch.from_numpy(prediction.reshape(-1, 32)).float().to(device)).cpu().numpy().reshape(len(prediction), 3, 16, 7)
    endpoint = ((prediction[:, -1, None, :16] - backbone.goals[None]) ** 2).mean(-1).argmin(-1)
    base_err = ((data["base_current"][:, -1] - target[:, -1]) ** 2).mean(-1)
    pred_err = ((prediction[:, -1] - target[:, -1]) ** 2).mean(-1)
    exec_base = ((data["base_current"][:, -1, 16:] - target[:, -1, 16:]) ** 2).mean(-1)
    exec_pred = ((prediction[:, -1, 16:] - target[:, -1, 16:]) ** 2).mean(-1)
    decoded_mse = ((decoded[..., :6] - data["target_actions_norm"][..., :6]) ** 2).mean(axis=(1, 2, 3))
    continuity = np.linalg.norm(decoded[:, 0, 0, :6] - data["current_action_norm"][:, -1, :6], axis=-1)
    true_continuity = np.linalg.norm(data["target_actions_norm"][:, 0, 0, :6] - data["current_action_norm"][:, -1, :6], axis=-1)
    return ({"name": name, "H2_full_mse": float(((prediction[:, 1] - target[:, 1]) ** 2).mean()), "H4_full_mse": float(((prediction[:, 2] - target[:, 2]) ** 2).mean()), "H4_decoded_mse": float(decoded_mse.mean()), "H4_endpoint_accuracy": float(np.mean(endpoint == data["goal_id"])), "H4_continuity": float(continuity.mean()), "H4_true_continuity": float(true_continuity.mean()), "RedirectGain": float(np.mean(base_err - pred_err)), "Execution_RedirectGain": float(np.mean(exec_base - exec_pred)), "adapter_norm": float(np.linalg.norm(residual, axis=-1).mean()), "q_path_length": float(np.linalg.norm(np.diff(q, axis=1), axis=-1).sum(axis=1).mean()), "samples": int(len(prediction))}, {"prediction": prediction, "continuity": continuity, "endpoint": endpoint})


def train(model: TemporalForceBridge, train_data: dict[str, np.ndarray], train_x: np.ndarray, dev_data: dict[str, np.ndarray], dev_x: np.ndarray, backbone: FrozenBackbone, device: torch.device, weight: float) -> dict[str, Any]:
    train_t = tensors(train_data, train_x, device); dev_t = tensors(dev_data, dev_x, device)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=3e-3, weight_decay=1e-4)
    best_state = None; best = float("inf"); stale = 0; start = time.perf_counter()
    for epoch in range(80):
        model.train(); optimizer.zero_grad(set_to_none=True)
        out = model(train_t["features"], train_t["base_current"])
        decoded = backbone.representation.decode(out["prediction"].reshape(-1, 32)).view_as(train_t["target_actions_norm"])
        loss = F.mse_loss(out["prediction"], train_t["target_latent"]) + 0.3 * F.mse_loss(decoded, train_t["target_actions_norm"]) + weight * F.mse_loss(decoded[:, 0, 0, :6], train_t["current_action_norm"][:, -1, :6]) + 0.01 * out["residual"].square().mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), 5.0); optimizer.step()
        model.eval()
        with torch.no_grad():
            dev_out = model(dev_t["features"], dev_t["base_current"])
            dev_loss = float(F.mse_loss(dev_out["prediction"], dev_t["target_latent"]).cpu())
        if dev_loss < best - 1e-6:
            best = dev_loss; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; stale = 0
        else:
            stale += 1
            if stale >= 12: break
    if best_state is None: raise RuntimeError("no finite Wave35 checkpoint")
    model.load_state_dict(best_state); model.eval()
    return {"best_epoch": epoch + 1, "best_dev_latent_loss": best, "runtime_seconds": time.perf_counter() - start, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def prepare(backbone: FrozenBackbone) -> dict[str, dict[str, np.ndarray]]:
    train_raw = load_wave21("train"); dev_raw = load_wave21("development"); test_raw = load_wave21("test"); w27_raw = load_wave27("test")
    phase_min = float(train_raw["boundary_frame"].min()); phase_max = float(train_raw["boundary_frame"].max())
    return {"train": make_data(train_raw, "F1", backbone, phase_min, phase_max), "development": make_data(dev_raw, "F1", backbone, phase_min, phase_max), "test": make_data(test_raw, "F1", backbone, phase_min, phase_max), "wave27": make_data(w27_raw, "F1", backbone, phase_min, phase_max)}


def audit(device: torch.device) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    backbone = FrozenBackbone(device); data = prepare(backbone)
    write_json(OUT / "wave35_preregistration.json", {"created_at": now(), "termination_rule": "success or Wave78 only; never Wave79", "frozen": ["action_text_VAE", "decoder", "F1", "F2"], "families": ["delta", "state", "history_contact", "phase_gated", "integrated"], "q_dimensions": [2, 4, 8], "basis": ["pca", "random"], "loss_continuity_weights": [0.05, 0.15, 0.3], "future_inputs": [], "wave27_previous_label": "unavailable; neutral anchor only", "counts": {k: len(v["goal_id"]) for k, v in data.items()}})
    (OUT / "wave35_execution_log.md").write_text(f"# Wave35 execution log\n\n- {now()} — temporal/state-action bridge audit; train={len(data['train']['goal_id'])}, development={len(data['development']['goal_id'])}, test={len(data['test']['goal_id'])}, Wave27={len(data['wave27']['goal_id'])}.\n")


def candidate_specs() -> list[dict[str, Any]]:
    return [{"name": f"{family}_q{q}_{kind}_w{weight}", "family": family, "q": q, "kind": kind, "weight": weight} for family in ("delta", "state", "history_contact", "phase_gated", "integrated") for q in (2, 4, 8) for kind in ("pca", "random") for weight in (0.05, 0.15, 0.3)]


def run_sweep(device: torch.device, limit: int | None) -> None:
    seed(); backbone = FrozenBackbone(device); data = prepare(backbone); specs = candidate_specs()[:limit] if limit else candidate_specs(); records = {}
    for index, spec in enumerate(specs, 1):
        train_data, dev_data = data["train"], data["development"]; b = basis(train_data, spec["q"], spec["kind"]); model = TemporalForceBridge(features(train_data, spec["family"]).shape[-1], spec["q"], b, spec["family"]).to(device)
        fit = train(model, train_data, features(train_data, spec["family"]), dev_data, features(dev_data, spec["family"]), backbone, device, spec["weight"])
        met, _ = evaluate(model, dev_data, features(dev_data, spec["family"]), backbone, device, spec["name"]); records[spec["name"]] = {"spec": spec, "fit": fit, "metrics": met}
        print(f"[{index}/{len(specs)}] {spec['name']} redirect={met['RedirectGain']:.4f} continuity={met['H4_continuity']:.4f}", flush=True)
    write_json(OUT / "wave35_development_metrics.json", records); write_json(OUT / "wave35_sweep_inventory.json", {"candidates": len(specs), "future_inputs": [], "heldout_opened": False})


def select() -> None:
    records = json.loads((OUT / "wave35_development_metrics.json").read_text()); ordered = sorted(records, key=lambda n: (records[n]["metrics"]["H4_decoded_mse"] + 0.25 * records[n]["metrics"]["H4_continuity"] - 0.25 * records[n]["metrics"]["RedirectGain"]))
    chosen = []
    for family in ("delta", "state", "history_contact", "phase_gated", "integrated"):
        candidates = [n for n in ordered if records[n]["spec"]["family"] == family]
        if candidates: chosen.append(candidates[0])
    chosen += [n for n in ordered if n not in chosen][:3]; chosen = chosen[:8]
    write_json(OUT / "wave35_final_candidate_selection.json", {"created_at": now(), "selected": chosen, "heldout_opened": False, "selection_rule": "development decoded error + continuity - redirect, preserving families"})


def final(device: torch.device) -> None:
    seed(SEED + 1); backbone = FrozenBackbone(device); data = prepare(backbone); selection = json.loads((OUT / "wave35_final_candidate_selection.json").read_text()); output = {}
    for name in selection["selected"]:
        spec = json.loads((OUT / "wave35_development_metrics.json").read_text())[name]["spec"]; b = basis(data["train"], spec["q"], spec["kind"]); model = TemporalForceBridge(features(data["train"], spec["family"]).shape[-1], spec["q"], b, spec["family"]).to(device); train(model, data["train"], features(data["train"], spec["family"]), data["development"], features(data["development"], spec["family"]), backbone, device, spec["weight"])
        metrics = {}
        for key in ("test", "wave27"):
            metrics[key], _ = evaluate(model, data[key], features(data[key], spec["family"]), backbone, device, name)
        output[name] = {"spec": spec, **metrics}; print(name, metrics["wave27"]["Execution_RedirectGain"], metrics["wave27"]["H4_continuity"], flush=True)
    write_json(OUT / "wave35_heldout_results.json", output); selection["heldout_opened"] = True; write_json(OUT / "wave35_final_candidate_selection.json", selection)


def report() -> None:
    heldout = json.loads((OUT / "wave35_heldout_results.json").read_text()); best = max(heldout, key=lambda n: heldout[n]["wave27"]["Execution_RedirectGain"]); bm = heldout[best]["wave27"]; any_success = any(m["wave27"]["Execution_RedirectGain"] > 0 and m["wave27"]["H4_continuity"] <= 1.5 * m["wave27"]["H4_true_continuity"] and m["wave27"]["H4_endpoint_accuracy"] >= 0.55 for m in heldout.values()); decision = {"best": best, "SUCCESS": any_success, "READY_FOR_CLOSED_LOOP_RETARGET": "SUPPORTED" if any_success else "NOT_SUPPORTED", "representation_stop": False, "termination_rule": "only success or Wave78; continue otherwise"}; write_json(OUT / "wave35_claim_decision.json", decision)
    text = f"# Wave35: temporal/state-action bridge\n\nWave35 tested {len(heldout)} frozen held-out candidates across five causal bridge families, q=2/4/8, PCA/random bases, and three continuity weights. The best Wave27 prospective candidate was `{best}` with execution redirect {bm['Execution_RedirectGain']:.6f}, decoded continuity {bm['H4_continuity']:.6f}, and endpoint accuracy {bm['H4_endpoint_accuracy']:.4f}. SUCCESS={any_success}; the program must continue unless SUCCESS is true or Wave78 is completed.\n\n```json\n{json.dumps(decision, indent=2)}\n```\n"; (OUT / "thirty_fifth_wave_results.md").write_text(text); (ROOT / "reports/dynamics_wave35_results.md").write_text(text); next_text = "# Next experiment after Wave35\n\nWave35 did not end the program. Continue to Wave36 unless the success gate is satisfied. Target the strongest remaining bottleneck: train a bridge with explicit event timing and matched current physical state while testing task-balanced and return-symmetric ordered transitions. Keep VAE/F1/F2 frozen, compare Wave35 as a negative control, and do not stop before Wave78 unless the success gate is met.\n"; (OUT / "thirty_fifth_wave_next_experiment.md").write_text(next_text); (ROOT / "NEXT_EXPERIMENT.md").write_text(next_text); log = ROOT / "RESEARCH_LOG.md"; log.write_text(log.read_text().rstrip() + f"\n\n## Wave 35 — {now()}\n\nTested five temporal/state-action bridge families with q=2/4/8 and PCA/random projections. Best `{best}` had Wave27 execution redirect {bm['Execution_RedirectGain']:.6f}; SUCCESS={any_success}. Under the user termination rule, continue to Wave36 unless success or Wave78.\n"); print(json.dumps(decision), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", choices=("audit", "sweep", "select", "final", "report", "all"), default="all"); parser.add_argument("--device", default="cpu"); parser.add_argument("--max-candidates", type=int, default=None); args = parser.parse_args(); OUT.mkdir(parents=True, exist_ok=True); device = torch.device(args.device)
    if args.stage in ("audit", "all"): audit(device)
    if args.stage in ("sweep", "all"): run_sweep(device, args.max_candidates)
    if args.stage in ("select", "all"): select()
    if args.stage in ("final", "all"): final(device)
    if args.stage in ("report", "all"): report()


if __name__ == "__main__":
    main()

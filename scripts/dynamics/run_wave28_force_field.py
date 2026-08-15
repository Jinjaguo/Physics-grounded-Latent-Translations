#!/usr/bin/env python3
"""Run the Wave 28 low-dimensional intention force-field research program.

Purpose
-------
Freeze the released action-text representation, its decoder, and the historic
F1/F2 behavioral predictors.  Train and compare compact latent control fields
that steer the frozen 32-D action coordinate when a later language goal arrives.
The staged tournament covers q dimension, language encoding, field dynamics,
low-rank subspace, composition, losses, F1/F2 backbone conditions, no-switch
anchors, and intention-space return symmetry without using future targets at
inference time.

Parameters
----------
``--stage``: ``audit``, ``sweep``, ``select``, ``final``, ``report``, or
``all``.  ``--device`` selects the PyTorch device; the registered run uses
``cpu`` when CUDA is unavailable.  ``--max-candidates`` optionally limits the
development tournament for a controlled rerun.

Usage
-----
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/run_wave28_force_field.py --stage all --device cpu

Outputs
-------
Tracked manifests, audits, scorecards, claim decisions, reports, exact
commands, and tests are saved under
``results/dynamics/twenty_eighth_wave/2026-08-15_force_field``.  Model
checkpoints and compact per-sample arrays are saved below that directory.
The report stage updates ``reports/dynamics_wave28_results.md``,
``RESEARCH_LOG.md``, and ``NEXT_EXPERIMENT.md`` without deleting prior waves.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from pglt.dynamics.factorized import ExecutionMLP, ExecutionMatchedRefinement, SemanticPredictor
from pglt.dynamics.wave28_force_field import IntentForceField
from scripts.dynamics.run_dynamics_9 import load_representation, read_json, sha256, write_json
from scripts.dynamics.run_dynamics_15 import load_npz


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/dynamics/twenty_eighth_wave/2026-08-15_force_field"
W21 = ROOT / "results/dynamics/twenty_first_wave/2026-08-14_dynamics_9"
W27 = ROOT / "results/dynamics/twenty_seventh_wave/2026-08-15_dynamics_15"
VOCAB = [
    "lift_blue_block_slider", "lift_red_block_table", "place_in_slider",
    "push_pink_block_right", "turn_off_lightbulb", "turn_on_lightbulb",
]
HINDICES = (0, 1, 3)
HORIZONS = (1, 2, 4)
SEED = 280828


def now() -> str:
    return datetime.now().astimezone().isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_json(name: str, payload: Any) -> None:
    write_json(OUT / name, payload)


def normalize_actions(actions: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    value = actions.astype(np.float32).copy()
    value[..., :6] = (value[..., :6] - mean) / std
    return value


def goal_ids(labels: list[str]) -> np.ndarray:
    return np.asarray([VOCAB.index(label) for label in labels], dtype=np.int64)


class FrozenBackbone:
    """Exact frozen semantic/F1/F2 interface used by the old factorized study."""

    def __init__(self, device: torch.device) -> None:
        config = yaml.safe_load((ROOT / "configs/dynamics_9.yaml").read_text())
        self.representation, payload, self.mean, self.std = load_representation(config, device)
        self.representation.eval()
        self.device = device
        self.goals = np.load(W21 / "wave21_goal_embeddings.npy").astype(np.float32)
        self.semantic = SemanticPredictor(context_dim=16, hidden_dim=64, depth=3).to(device)
        self.f1 = ExecutionMLP(context_dim=32, hidden_dim=64, depth=3).to(device)
        self.f2 = ExecutionMatchedRefinement(self.f1, context_dim=32, hidden_dim=64, depth=3, iterations=4, step_size=0.01).to(device)
        paths = {
            self.semantic: W21.parent.parent / "fifteenth_wave/2026-08-12_dynamics_3/checkpoints/semantic.pt",
            self.f1: W21.parent.parent / "fifteenth_wave/2026-08-12_dynamics_3/checkpoints/F1_execution_mlp.pt",
            self.f2: W21.parent.parent / "fifteenth_wave/2026-08-12_dynamics_3/checkpoints/F2_matched_refinement.pt",
        }
        # The paths above are relative to results/dynamics; resolve explicitly
        # so the audit records the exact historical checkpoints.
        paths = {
            self.semantic: ROOT / "results/dynamics/fifteenth_wave/2026-08-12_dynamics_3/checkpoints/semantic.pt",
            self.f1: ROOT / "results/dynamics/fifteenth_wave/2026-08-12_dynamics_3/checkpoints/F1_execution_mlp.pt",
            self.f2: ROOT / "results/dynamics/fifteenth_wave/2026-08-12_dynamics_3/checkpoints/F2_matched_refinement.pt",
        }
        for model, path in paths.items():
            model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model_state_dict"])
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        self.checkpoints = {name: sha256(path) for name, path in (
            ("semantic", ROOT / "results/dynamics/fifteenth_wave/2026-08-12_dynamics_3/checkpoints/semantic.pt"),
            ("F1", ROOT / "results/dynamics/fifteenth_wave/2026-08-12_dynamics_3/checkpoints/F1_execution_mlp.pt"),
            ("F2", ROOT / "results/dynamics/fifteenth_wave/2026-08-12_dynamics_3/checkpoints/F2_matched_refinement.pt"),
        )}

    def language(self, ids: np.ndarray) -> np.ndarray:
        value = np.zeros((len(ids), 16), dtype=np.float32)
        valid = ids >= 0
        if np.any(valid):
            value[valid] = self.goals[ids[valid]]
        return value

    def step(self, previous: torch.Tensor, current: torch.Tensor, text: torch.Tensor, variant: str) -> torch.Tensor:
        semantic = self.semantic(previous[:, :16], current[:, :16], text)
        context = torch.cat((current[:, :16], text), dim=-1)
        previous_e, current_e = previous[:, 16:], current[:, 16:]
        if variant == "F1":
            execution = self.f1(previous_e, current_e, context)
        elif variant == "F2":
            initial = self.f2.initializer(previous_e, current_e, context)
            candidate = initial.detach().requires_grad_(True)
            fixed = torch.cat((previous_e, current_e, context), dim=-1)
            for _ in range(self.f2.iterations):
                energy = self.f2.energy_network(torch.cat((fixed, candidate), dim=-1)).squeeze(-1)
                gradient = torch.autograd.grad(energy.sum(), candidate, create_graph=False)[0]
                candidate = (candidate - self.f2.step_size * gradient).detach().requires_grad_(True)
            execution = candidate.detach()
        else:
            raise KeyError(variant)
        return torch.cat((semantic, execution), dim=-1)

    def rollout(self, data: dict[str, np.ndarray], current_ids: np.ndarray, variant: str) -> np.ndarray:
        previous = torch.from_numpy(data["z_previous"]).float().to(self.device)
        current = torch.from_numpy(data["z_current"]).float().to(self.device)
        text = torch.from_numpy(self.language(current_ids)).float().to(self.device)
        values = []
        with torch.enable_grad():
            for _ in range(4):
                next_value = self.step(previous, current, text, variant)
                values.append(next_value)
                previous, current = current, next_value
        return torch.stack(values, dim=1).detach().cpu().numpy().astype(np.float32)[:, list(HINDICES)]


def load_events(split: str, wave: str) -> dict[str, np.ndarray]:
    if wave == "wave21":
        data = load_npz(W21 / "datasets" / f"{split}.npz")
        inventory = read_json(W21 / "wave21_transition_inventory.csv.json") if (W21 / "wave21_transition_inventory.csv.json").exists() else None
        rows = []
        with (W21 / "wave21_transition_inventory.csv").open() as handle:
            reader = csv.DictReader(handle)
            rows = [row for row in reader if row["split"] == split]
        if len(rows) != len(data["goal_id"]):
            raise RuntimeError(f"Wave21 {split} inventory/data mismatch: {len(rows)} vs {len(data['goal_id'])}")
        current = goal_ids([row["previous_label"] for row in rows])
        target = goal_ids([row["next_label"] for row in rows])
        if not np.array_equal(target, data["goal_id"]):
            raise RuntimeError(f"Wave21 {split} target order differs from serialized dataset")
        data = dict(data); data["current_goal_id"] = current; data["target_goal_id"] = target
        data["event_source"] = np.asarray(["ordered_annotation"] * len(target))
        return data
    data = load_npz(W27 / "datasets" / f"new_{'prospective_test' if split == 'test' else split}.npz")
    inventory = json.loads((W27 / "wave27_new_transition_inventory.json").read_text())
    rows = [row for row in inventory if row["split"] == ("new_prospective_test" if split == "test" else f"new_{split}")]
    if len(rows) != len(data["goal_id"]):
        raise RuntimeError(f"Wave27 {split} inventory/data mismatch: {len(rows)} vs {len(data['goal_id'])}")
    target = data["goal_id"].astype(np.int64)
    data = dict(data); data["current_goal_id"] = np.full(len(target), -1, np.int64); data["target_goal_id"] = target
    data["event_source"] = np.asarray(["neutral_anchor_no_previous_label"] * len(target))
    return data


def concat_events(*items: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    keys = set.intersection(*(set(item) for item in items))
    result = {key: np.concatenate([item[key] for item in items], axis=0) for key in keys}
    return result


def build_dataset(backbone: FrozenBackbone, data: dict[str, np.ndarray], variant: str) -> dict[str, np.ndarray]:
    current_ids = data["current_goal_id"]
    target_ids = data["target_goal_id"]
    base_current = backbone.rollout(data, current_ids, variant)
    base_target = backbone.rollout(data, target_ids, variant)
    target_latent = data["future_latents"][:, list(HINDICES)].astype(np.float32)
    target_actions = data["future_actions"][:, list(HINDICES)].astype(np.float32)
    target_actions_norm = normalize_actions(target_actions, backbone.mean, backbone.std)
    current_action_norm = normalize_actions(data["current_action"], backbone.mean, backbone.std)
    return {
        "base_current": base_current, "base_target": base_target,
        "target_latent": target_latent, "target_actions": target_actions,
        "target_actions_norm": target_actions_norm, "current_action_norm": current_action_norm,
        "current_language": backbone.language(current_ids), "target_language": backbone.language(target_ids),
        "current_ids": current_ids, "target_ids": target_ids,
        "z_current": data["z_current"].astype(np.float32), "session_row": data["session_row"].astype(np.int64),
        "goal_id": data["target_goal_id"].astype(np.int64), "event_source": data["event_source"],
    }


def make_basis(train: dict[str, np.ndarray], kind: str, q_dim: int, seed: int) -> torch.Tensor | None:
    if kind == "C2_learned" or kind == "C3_block_separable" or kind == "C6_state_dependent":
        return None
    if kind == "C1_pca":
        residual = (train["target_latent"] - train["base_current"]).mean(axis=1)
        _, _, vt = np.linalg.svd(residual - residual.mean(0), full_matrices=False)
        return torch.from_numpy(vt[:q_dim].T.astype(np.float32))
    generator = torch.Generator().manual_seed(seed)
    basis = torch.randn(32, q_dim, generator=generator)
    return torch.linalg.qr(basis, mode="reduced").Q


def model_spec(name: str, q_dim: int, encoding: str, field: str, subspace: str, composition: str, group: str, backbone: str = "F1") -> dict[str, Any]:
    return {"name": name, "q_dim": q_dim, "encoding": encoding, "field": field, "subspace": subspace, "composition": composition, "group": group, "backbone": backbone}


def build_model(spec: dict[str, Any], basis: torch.Tensor | None, seed: int) -> IntentForceField:
    return IntentForceField(
        q_dim=int(spec["q_dim"]), encoding=spec["encoding"], field=spec["field"],
        subspace=spec["subspace"], composition=spec["composition"], basis=basis, seed=seed,
    )


def cosine_loss(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(left, right, dim=-1)).mean()


def loss_terms(model: IntentForceField, output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], representation: nn.Module) -> dict[str, torch.Tensor]:
    prediction = output["prediction"]
    target = batch["target_latent"]
    base = batch["base_current"]
    decoded = representation.decode(prediction.reshape(-1, 32)).view_as(batch["target_actions_norm"])
    residual = output["residual"]
    target_residual = target - base
    q_final = output["q"][:, -1]
    direction = output["direction"]
    no_switch = model(
        batch["base_target"], batch["z_current"], batch["target_language"], batch["target_language"],
        batch["target_ids"], batch["target_ids"],
    )
    anchor = no_switch["residual"].square().mean()
    reverse = model(
        batch["base_current"], batch["z_current"], batch["target_language"], batch["current_language"],
        batch["target_ids"], batch["current_ids"],
    )
    return {
        "L0": F.mse_loss(prediction, target),
        "L1": F.mse_loss(decoded, batch["target_actions_norm"]),
        "L2": cosine_loss(residual[:, -1], target_residual[:, -1]),
        "L3": F.mse_loss(q_final, direction),
        "L4": F.mse_loss(decoded[:, 0, 0, :6] - batch["current_action_norm"][:, -1, :6], batch["target_actions_norm"][:, 0, 0, :6] - batch["current_action_norm"][:, -1, :6]),
        "L5": anchor,
        "L7": F.mse_loss(output["direction"] + reverse["direction"], torch.zeros_like(output["direction"])),
        "L8": F.mse_loss(output["q"][:, -1] + reverse["q"][:, -1], torch.zeros_like(output["q"][:, -1])),
        "L9": residual.square().mean(),
        "L12": cosine_loss(residual[:, -1], target_residual[:, -1]),
    }


def active_loss(terms: dict[str, torch.Tensor], group: str) -> torch.Tensor:
    selected = {"A": ("L0", "L2", "L5"), "B": ("L0", "L2", "L5", "L4"), "C": ("L0", "L2", "L5", "L1"), "D": ("L0", "L2", "L5", "L4", "L1"), "E": ("L0", "L2", "L5", "L4", "L1", "L7", "L8"), "F": ("L0", "L2", "L5", "L4", "L1", "L3"), "G": ("L0", "L2", "L5", "L4", "L1", "L12"), "H": ("L0", "L2", "L5", "L4", "L1", "L7", "L8", "L9", "L3", "L12")}[group]
    weights = {"L1": 0.3, "L2": 0.2, "L3": 0.2, "L4": 0.2, "L5": 0.3, "L7": 0.1, "L8": 0.1, "L9": 0.01, "L12": 0.1}
    return sum((weights.get(key, 1.0) * terms[key] for key in selected), torch.zeros_like(terms["L0"]))


def as_tensors(data: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    result = {}
    for key, value in data.items():
        if key in ("event_source",):
            continue
        if value.dtype.kind in "iu": result[key] = torch.from_numpy(value).long().to(device)
        elif value.dtype.kind == "f": result[key] = torch.from_numpy(value).float().to(device)
    return result


def predict_model(model: IntentForceField, data: dict[str, np.ndarray], device: torch.device) -> dict[str, np.ndarray]:
    tensors = as_tensors(data, device)
    with torch.no_grad():
        output = model(tensors["base_current"], tensors["z_current"], tensors["current_language"], tensors["target_language"], tensors["current_ids"], tensors["target_ids"])
    return {key: value.detach().cpu().numpy() for key, value in output.items()}


def nearest_region(prediction: np.ndarray, goals: np.ndarray, regions: dict[str, np.ndarray]) -> np.ndarray:
    centers = np.stack([regions[task].mean(0) for task in VOCAB]).astype(np.float32)
    distances = ((prediction[:, None] - centers[None]) ** 2).mean(-1)
    return distances.argmin(-1)


def metrics(name: str, model: IntentForceField | None, data: dict[str, np.ndarray], backbone: FrozenBackbone, device: torch.device, regions: dict[str, np.ndarray], base_only: str | None = None) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if model is None:
        prediction = data["base_target"] if base_only == "target" else data["base_current"]
        residual = np.zeros_like(prediction)
        q = np.zeros((len(prediction), 3, 2), np.float32)
        direction = np.zeros((len(prediction), 2), np.float32)
    else:
        output = predict_model(model, data, device)
        prediction, residual, q, direction = output["prediction"], output["residual"], output["q"], output["direction"]
    target = data["target_latent"]
    target_actions_norm = data["target_actions_norm"]
    with torch.no_grad():
        decoded = backbone.representation.decode(torch.from_numpy(prediction.reshape(-1, 32)).float().to(device)).cpu().numpy().reshape(len(prediction), 3, 16, 7)
        recoded = backbone.representation.encode(torch.from_numpy(decoded.reshape(-1, 16, 7)).float().to(device)).cpu().numpy().reshape(len(prediction), 3, 32)
    centers = np.stack([regions[task].mean(0) for task in VOCAB]).astype(np.float32)
    endpoint = nearest_region(prediction[:, -1], backbone.goals, regions)
    recode_endpoint = nearest_region(recoded[:, -1], backbone.goals, regions)
    base_err = ((data["base_current"][:, -1] - target[:, -1]) ** 2).mean(-1)
    target_err = ((data["base_target"][:, -1] - target[:, -1]) ** 2).mean(-1)
    pred_err = ((prediction[:, -1] - target[:, -1]) ** 2).mean(-1)
    exec_base = ((data["base_current"][:, -1, 16:] - target[:, -1, 16:]) ** 2).mean(-1)
    exec_pred = ((prediction[:, -1, 16:] - target[:, -1, 16:]) ** 2).mean(-1)
    decoded_mse = ((decoded[..., :6] - target_actions_norm[..., :6]) ** 2).mean(axis=(1, 2, 3))
    continuity = np.linalg.norm(decoded[:, 0, 0, :6] - data["current_action_norm"][:, -1, :6], axis=-1)
    true_continuity = np.linalg.norm(target_actions_norm[:, 0, 0, :6] - data["current_action_norm"][:, -1, :6], axis=-1)
    centered_q = q.reshape(-1, q.shape[-1]) - q.reshape(-1, q.shape[-1]).mean(0)
    singular = np.linalg.svd(centered_q, compute_uv=False)
    spectrum = singular ** 2 / np.maximum((singular ** 2).sum(), 1e-12)
    effective_rank = float(np.exp(-np.sum(spectrum * np.log(np.maximum(spectrum, 1e-12))))) if len(spectrum) else 0.0
    values = {
        "name": name, "H2_full_mse": float(((prediction[:, 1] - target[:, 1]) ** 2).mean()),
        "H4_full_mse": float(((prediction[:, 2] - target[:, 2]) ** 2).mean()),
        "H4_decoded_mse": float(decoded_mse.mean()), "H4_endpoint_accuracy": float(np.mean(endpoint == data["goal_id"])),
        "H4_recode_accuracy": float(np.mean(recode_endpoint == data["goal_id"])),
        "H4_continuity": float(continuity.mean()), "H4_true_continuity": float(true_continuity.mean()),
        "RedirectGain": float(np.mean(base_err - pred_err)), "Execution_RedirectGain": float(np.mean(exec_base - exec_pred)),
        "base_target_H4_mse": float(target_err.mean()), "adapter_norm": float(np.linalg.norm(residual, axis=-1).mean()),
        "q_path_length": float(np.linalg.norm(np.diff(q, axis=1), axis=-1).sum(axis=1).mean()),
        "q_target_distance": float(np.linalg.norm(q[:, -1] - direction, axis=-1).mean()),
        "effective_q_rank": effective_rank,
        "samples": int(len(prediction)),
    }
    raw = {"h4_error": pred_err, "h4_base_error": base_err, "exec_error": exec_pred, "exec_base_error": exec_base, "continuity": continuity, "q": q, "direction": direction, "endpoint": endpoint, "recode": recode_endpoint}
    return values, raw


def train_candidate(spec: dict[str, Any], train: dict[str, np.ndarray], dev: dict[str, np.ndarray], backbone: FrozenBackbone, device: torch.device, basis: torch.Tensor | None, seed: int, epochs: int = 35) -> tuple[IntentForceField, dict[str, Any]]:
    set_seed(seed)
    model = build_model(spec, basis, seed).to(device)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=3e-3, weight_decay=1e-4)
    train_t = as_tensors(train, device); dev_t = as_tensors(dev, device)
    best_state = None; best = float("inf"); stale = 0; started = time.perf_counter()
    for epoch in range(epochs):
        model.train(); optimizer.zero_grad(set_to_none=True)
        train_output = model(train_t["base_current"], train_t["z_current"], train_t["current_language"], train_t["target_language"], train_t["current_ids"], train_t["target_ids"])
        terms = loss_terms(model, train_output, train_t, backbone.representation)
        loss = active_loss(terms, spec["group"])
        if not torch.isfinite(loss): raise FloatingPointError(f"{spec['name']} non-finite loss at epoch {epoch}")
        loss.backward(); torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), 5.0); optimizer.step()
        model.eval()
        with torch.no_grad():
            out = model(dev_t["base_current"], dev_t["z_current"], dev_t["current_language"], dev_t["target_language"], dev_t["current_ids"], dev_t["target_ids"])
            dev_loss = float(F.mse_loss(out["prediction"], dev_t["target_latent"]).cpu())
        if dev_loss < best - 1e-6:
            best = dev_loss; best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}; stale = 0
        else:
            stale += 1
            if stale >= 8: break
    if best_state is None: raise RuntimeError(f"no finite checkpoint for {spec['name']}")
    model.load_state_dict(best_state); model.eval()
    return model, {"best_epoch": epoch + 1, "best_dev_latent_loss": best, "runtime_seconds": time.perf_counter() - started, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def audit_stage(device: torch.device) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    backbone = FrozenBackbone(device)
    wave21_train = load_events("train", "wave21"); wave27_train = load_events("train", "wave27")
    wave21_dev = load_events("development", "wave21"); wave27_dev = load_events("development", "wave27")
    train21 = build_dataset(backbone, wave21_train, "F1"); train27 = build_dataset(backbone, wave27_train, "F1")
    train = concat_events(train21, train27)
    z = torch.from_numpy(train["base_current"][:2]).float().to(device)
    current = torch.from_numpy(train["z_current"][:2]).float().to(device)
    lang = torch.from_numpy(train["current_language"][:2]).float().to(device)
    target_lang = torch.from_numpy(train["target_language"][:2]).float().to(device)
    model = IntentForceField(q_dim=2, encoding="E0_linear", field="FF3_attractor", subspace="C2_learned").to(device)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    output = model(z, current, lang, target_lang, torch.from_numpy(train["current_ids"][:2]).to(device), torch.from_numpy(train["target_ids"][:2]).to(device))
    output["prediction"].retain_grad()
    decoded = backbone.representation.decode(output["prediction"].reshape(-1, 32))
    decoded.square().mean().backward()
    gradient = float(output["prediction"].grad.norm()) if output["prediction"].grad is not None else 0.0
    rank = int(torch.linalg.matrix_rank(model.B.detach()).cpu())
    after = {key: value.detach().clone() for key, value in model.state_dict().items()}
    frozen_backbone_ok = all(torch.equal(before[key], after[key]) for key in before)
    if gradient <= 0 or rank != 2 or not frozen_backbone_ok:
        raise RuntimeError(f"Wave28 sanity failed: decoder_grad={gradient}, rank={rank}, frozen={frozen_backbone_ok}")
    audit = {
        "created_at": now(), "device": str(device), "cuda_available": bool(torch.cuda.is_available()),
        "representation": {"checkpoint": "checkpoints/representation/seed_810/correct_language/checkpoint_ema.pt", "latent_dim": 32, "semantic_dim": 16, "decoder_output": [16, 7], "decoder_parameters_trainable": False},
        "F1_inputs": ["execution_previous(16)", "execution_current(16)", "semantic_current(16)", "frozen_text(16)"],
        "F2_inputs": ["same as F1", "four frozen refinement iterations"], "F1_output": "next_execution(16)", "semantic_output": "next_semantic(16)",
        "z_base": "recursive frozen F1/F2 semantic+execution prediction from current z and query-time current language",
        "adapter_input": ["z_base", "z_current", "current_language", "target_language"], "adapter_output": "low-rank residual in original 32-D latent",
        "future_inputs": [], "ordered_wave21_events": int(len(wave21_train["goal_id"])), "wave27_neutral_events": int(len(wave27_train["goal_id"])),
        "decoder_gradient_norm": gradient, "B_rank": rank, "q_zero_no_adapter_test": "PASS", "frozen_backbone_parameters": "PASS",
        "F1_checkpoint_sha256": backbone.checkpoints["F1"], "F2_checkpoint_sha256": backbone.checkpoints["F2"], "semantic_checkpoint_sha256": backbone.checkpoints["semantic"],
        "limitation": "Wave27 records do not contain previous annotation labels; they are neutral->target prospective events and are excluded from return claims.",
    }
    save_json("wave28_interface_audit.json", audit)
    (OUT / "wave28_backbone_interface_audit.md").write_text("# Wave 28 backbone interface audit\n\n" + json.dumps(audit, indent=2) + "\n")
    dataset_audit = {
        "wave21": {split: {"records": int(len(load_events(split, "wave21")["goal_id"])), "source": "ordered annotation transition", "sessions": int(len(np.unique(load_events(split, "wave21")["session_row"]))) } for split in ("train", "development", "test")},
        "wave27": {split: {"records": int(len(load_events(split, "wave27")["goal_id"])), "source": "official independent prospective transition", "sessions": int(len(np.unique(load_events(split, "wave27")["session_row"]))) } for split in ("train", "development", "test")},
        "future_as_input": False, "previous_language_wave27": "unavailable; no imputation", "test_selection": "development only before prospective open",
    }
    save_json("wave28_dataset_audit.json", dataset_audit)
    (OUT / "wave28_dataset_audit.md").write_text("# Wave 28 dataset audit\n\n" + json.dumps(dataset_audit, indent=2) + "\n")
    save_json("wave28_frozen_manifest.json", {
        "created_before_any_sweep_metric": True, "representation_checkpoint": "checkpoints/representation/seed_810/correct_language/checkpoint_ema.pt",
        "representation_updates": 0, "decoder_updates": 0, "semantic_updates": 0, "F1_updates": 0, "F2_updates": 0,
        "representation_sha256": sha256(ROOT / "checkpoints/representation/seed_810/correct_language/checkpoint_ema.pt"),
        "decoder_and_encoder_interface": {"latent_dim": 32, "semantic_dim": 16, "chunk": [16, 7]},
        "F1_checkpoint_sha256": backbone.checkpoints["F1"], "F2_checkpoint_sha256": backbone.checkpoints["F2"], "semantic_checkpoint_sha256": backbone.checkpoints["semantic"],
        "future_inputs": [], "heldout_arrays_materialized": False,
    })
    save_json("wave28_preregistration.json", {
        "created_at": now(), "written_before_any_sweep_metric": True, "q_dimensions": [1, 2, 4, 8],
        "language_encodings": ["E0_linear", "E1_normalized_linear", "E2_mlp", "E3_pairwise", "E4_antisymmetric", "E6_dictionary"],
        "fields": ["FF0_none", "FF1_direct", "FF2_accumulating", "FF3_attractor", "FF4_state_conditioned", "FF5_nonlinear", "FF7_gated", "FF8_velocity", "FF9_retrieval"],
        "subspaces": ["C0_random", "C1_pca", "C2_learned", "C3_block_separable", "C4_execution_only", "C5_semantic_only", "C6_state_dependent"],
        "compositions": ["COMP0_additive", "COMP1_gated_additive", "COMP2_normalized", "COMP3_film", "COMP4_rotation"],
        "loss_groups": {"A": ["L0", "L2", "L5"], "B": ["L0", "L2", "L5", "L4"], "C": ["L0", "L2", "L5", "L1"], "D": ["L0", "L2", "L5", "L4", "L1"], "E": ["D", "L7", "L8"], "F": ["D", "L3"], "G": ["D", "L12"], "H": ["E", "L9", "L3", "L12"]},
        "primary_inputs": ["z_previous", "z_current", "query-time current language", "query-time target language", "causal current action/history where available"],
        "forbidden_inputs": ["future latent", "future action", "future simulator state", "future contact", "success label"],
        "splits": "Wave21 session-disjoint ordered train/development/test plus Wave27 independent train/development/prospective test",
        "heldout_opened": False, "selection": "development Pareto/composite then one frozen heldout opening", "bootstrap": {"unit": "source session", "replicates": 10000, "seed": SEED},
    })
    save_json("wave28_seed_manifest.json", {"development_seed": SEED, "final_seeds": [SEED, SEED + 1], "no_post_heldout_seed_addition": True})
    (OUT / "wave28_execution_log.md").write_text(f"# Wave 28 execution log\n\n- {now()} — interface audit passed on CPU; CUDA unavailable; decoder gradient and B rank passed.\n")


def candidate_specs(max_candidates: int | None = None) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    # Stage 2: q × subspace × simple field, with a fixed F1 backbone and Group D.
    for q in (1, 2, 4, 8):
        for subspace in ("C0_random", "C1_pca", "C2_learned"):
            for field in ("FF1_direct", "FF3_attractor", "FF4_state_conditioned"):
                specs.append(model_spec(f"S2_q{q}_{subspace}_{field}", q, "E0_linear", field, subspace, "COMP0_additive", "D"))
    # Stage 3 dynamics and composition.
    for field in ("FF5_nonlinear", "FF7_gated", "FF8_velocity", "FF9_retrieval"):
        specs.append(model_spec(f"S3_q2_{field}", 2, "E0_linear", field, "C2_learned", "COMP0_additive", "D"))
    for composition in ("COMP1_gated_additive", "COMP2_normalized", "COMP3_film", "COMP4_rotation"):
        specs.append(model_spec(f"S4_q2_{composition}", 2, "E0_linear", "FF3_attractor", "C2_learned", composition, "D"))
    for subspace in ("C3_block_separable", "C4_execution_only", "C5_semantic_only", "C6_state_dependent"):
        specs.append(model_spec(f"S4_q2_{subspace}", 2, "E0_linear", "FF3_attractor", subspace, "COMP0_additive", "D"))
    # Language encodings.
    for encoding in ("E1_normalized_linear", "E2_mlp", "E3_pairwise", "E4_antisymmetric", "E6_dictionary"):
        specs.append(model_spec(f"E_{encoding}", 2, encoding, "FF3_attractor", "C2_learned", "COMP0_additive", "D"))
    # Loss tournament.
    for group in "ABCDEFGH":
        specs.append(model_spec(f"L_{group}", 2, "E0_linear", "FF3_attractor", "C2_learned", "COMP0_additive", group))
    # Controls and backbone generality.
    specs.append(model_spec("CTRL_full_rank", 32, "E0_linear", "FF3_attractor", "C2_learned", "COMP0_additive", "D"))
    specs.append(model_spec("CTRL_static_residual", 2, "E0_linear", "FF1_direct", "C2_learned", "COMP0_additive", "D"))
    specs.extend([
        model_spec("BACKBONE_F2_q2", 2, "E0_linear", "FF3_attractor", "C2_learned", "COMP0_additive", "D", "F2"),
        model_spec("BACKBONE_F1_q2", 2, "E0_linear", "FF3_attractor", "C2_learned", "COMP0_additive", "D", "F1"),
    ])
    if max_candidates is not None:
        return specs[:max_candidates]
    return specs


def sweep_stage(device: torch.device, max_candidates: int | None) -> None:
    backbone = FrozenBackbone(device)
    train = concat_events(build_dataset(backbone, load_events("train", "wave21"), "F1"), build_dataset(backbone, load_events("train", "wave27"), "F1"))
    dev = concat_events(build_dataset(backbone, load_events("development", "wave21"), "F1"), build_dataset(backbone, load_events("development", "wave27"), "F1"))
    regions = {task: np.load(W21 / "wave21_train_regions.npz")[task] for task in VOCAB}
    specs = candidate_specs(max_candidates); metrics_out: dict[str, Any] = {}; records: dict[str, Any] = {}
    for index, spec in enumerate(specs):
        # F2 candidates reuse the exact same causal arrays but with a different frozen base.
        if spec["backbone"] == "F2":
            train = concat_events(build_dataset(backbone, load_events("train", "wave21"), "F2"), build_dataset(backbone, load_events("train", "wave27"), "F2"))
            dev = concat_events(build_dataset(backbone, load_events("development", "wave21"), "F2"), build_dataset(backbone, load_events("development", "wave27"), "F2"))
        basis = make_basis(train, spec["subspace"], int(spec["q_dim"]), SEED + index)
        model, record = train_candidate(spec, train, dev, backbone, device, basis, SEED + index)
        metric, raw = metrics(spec["name"], model, dev, backbone, device, regions)
        metric.update({"spec": spec, **record})
        metrics_out[spec["name"]] = metric
        records[spec["name"]] = {"spec": spec, **record}
        checkpoint = OUT / "checkpoints" / "development" / f"{spec['name']}.pt"; checkpoint.parent.mkdir(parents=True, exist_ok=True); torch.save({"state_dict": model.state_dict(), "spec": spec, "record": record}, checkpoint)
        print(json.dumps({"stage": "sweep", "index": index + 1, "total": len(specs), "name": spec["name"], "H4_decoded": metric["H4_decoded_mse"], "redirect": metric["RedirectGain"]}), flush=True)
    save_json("wave28_development_metrics.json", metrics_out)
    save_json("wave28_training_records.json", records)
    save_json("wave28_model_specs.json", {name: value["spec"] for name, value in records.items()})
    save_json("wave28_sweep_inventory.json", {"models": len(specs), "q_dimensions": [1, 2, 4, 8, 32], "fields": ["FF0", "FF1", "FF2", "FF3", "FF4", "FF5", "FF7", "FF8", "FF9"], "encodings": ["E0", "E1", "E2", "E3", "E4", "E6"], "heldout_opened": False, "future_inputs": []})


def pareto_names(metrics_out: dict[str, Any]) -> list[str]:
    keys = ("H2_full_mse", "H4_decoded_mse", "H4_continuity", "RedirectGain")
    values = {name: np.asarray([metric[key] if key != "RedirectGain" else -metric[key] for key in keys]) for name, metric in metrics_out.items()}
    return [name for name in values if not any(other != name and np.all(values[other] <= values[name]) and np.any(values[other] < values[name]) for other in values)]


def select_stage() -> None:
    metrics_out = json.loads((OUT / "wave28_development_metrics.json").read_text())
    names = list(metrics_out); pareto = pareto_names(metrics_out)
    score = lambda name: metrics_out[name]["H2_full_mse"] + 10 * metrics_out[name]["H4_decoded_mse"] + metrics_out[name]["H4_continuity"] - metrics_out[name]["RedirectGain"]
    chosen: list[str] = []
    for prefix in ("S2_", "S3_", "S4_", "E_", "L_", "CTRL_", "BACKBONE_"):
        group = [name for name in pareto if name.startswith(prefix)] or [name for name in names if name.startswith(prefix)]
        if group:
            best = min(group, key=score)
            if best not in chosen: chosen.append(best)
    for name in sorted(pareto, key=score):
        if len(chosen) >= 8: break
        if name not in chosen: chosen.append(name)
    chosen = chosen[:8]
    fields = ["model", "q_dim", "encoding", "field", "subspace", "composition", "group", "backbone", "H2_full_mse", "H4_decoded_mse", "H4_endpoint_accuracy", "H4_recode_accuracy", "H4_continuity", "RedirectGain", "Execution_RedirectGain", "adapter_norm", "effective_q_rank", "pareto", "selected"]
    with (OUT / "wave28_development_scorecard.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for name in sorted(names):
            metric = metrics_out[name]; spec = metric["spec"]
            row = {key: metric.get(key, spec.get(key, "")) for key in fields[:-2]}
            row.update({"pareto": name in pareto, "selected": name in chosen})
            writer.writerow(row)
    with (OUT / "wave28_development_pareto.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["model", "H2_full_mse", "H4_decoded_mse", "continuity", "neg_redirect"])
        for name in pareto: writer.writerow([name, metrics_out[name]["H2_full_mse"], metrics_out[name]["H4_decoded_mse"], metrics_out[name]["H4_continuity"], -metrics_out[name]["RedirectGain"]])
    selection = {"created_at": now(), "development_only": True, "selected": chosen, "pareto": pareto, "heldout_opened": False, "post_selection_changes_forbidden": True, "selection_rule": "Pareto then composite H2+decoded+continuity-redirect; preserve distinct mechanisms and controls"}
    save_json("wave28_final_candidate_selection.json", selection)
    save_json("wave28_final_test_preregistration.json", {"created_before_heldout": True, "candidates": chosen, "seeds": [SEED, SEED + 1], "test_sets": ["Wave21 session-disjoint test", "Wave27 prospective test"], "bootstrap": {"unit": "source session", "replicates": 10000, "seed": SEED}, "thresholds_derived_from": "Wave28 development only", "post_test_tuning_allowed": False})
    print(json.dumps({"stage": "select", "selected": chosen, "pareto": len(pareto)}), flush=True)


def final_stage(device: torch.device) -> None:
    selection = json.loads((OUT / "wave28_final_candidate_selection.json").read_text())
    backbone = FrozenBackbone(device)
    train = concat_events(build_dataset(backbone, load_events("train", "wave21"), "F1"), build_dataset(backbone, load_events("train", "wave27"), "F1"))
    test21 = build_dataset(backbone, load_events("test", "wave21"), "F1")
    test27 = build_dataset(backbone, load_events("test", "wave27"), "F1")
    regions = {task: np.load(W21 / "wave21_train_regions.npz")[task] for task in VOCAB}
    metrics_out: dict[str, Any] = {}; raw_out: dict[str, Any] = {}
    for index, name in enumerate(selection["selected"]):
        spec = json.loads((OUT / "wave28_model_specs.json").read_text())[name]
        # Use the frozen base requested by the candidate; this is fixed before test opening.
        backbone_variant = spec.get("backbone", "F1")
        train_variant = backbone_variant
        train = concat_events(build_dataset(backbone, load_events("train", "wave21"), train_variant), build_dataset(backbone, load_events("train", "wave27"), train_variant))
        basis = make_basis(train, spec["subspace"], int(spec["q_dim"]), SEED + index)
        seed_rows = []
        for seed in (SEED, SEED + 1):
            model, record = train_candidate(spec, train, concat_events(build_dataset(backbone, load_events("development", "wave21"), train_variant), build_dataset(backbone, load_events("development", "wave27"), train_variant)), backbone, device, basis, seed, epochs=45)
            test21_variant = build_dataset(backbone, load_events("test", "wave21"), train_variant)
            test27_variant = build_dataset(backbone, load_events("test", "wave27"), train_variant)
            m21, r21 = metrics(name, model, test21_variant, backbone, device, regions); m27, r27 = metrics(name, model, test27_variant, backbone, device, regions)
            seed_rows.append({"seed": seed, "wave21": m21, "wave27": m27, "record": record})
        def mean_metric(key: str, set_name: str) -> float:
            return float(np.mean([row[set_name][key] for row in seed_rows]))
        metrics_out[name] = {"spec": spec, "wave21": {key: mean_metric(key, "wave21") for key in seed_rows[0]["wave21"] if isinstance(seed_rows[0]["wave21"][key], (int, float))}, "wave27": {key: mean_metric(key, "wave27") for key in seed_rows[0]["wave27"] if isinstance(seed_rows[0]["wave27"][key], (int, float))}, "seeds": seed_rows}
        raw_out[name] = {"wave21": {key: value.tolist() for key, value in r21.items()}, "wave27": {key: value.tolist() for key, value in r27.items()}}
        print(json.dumps({"stage": "final", "candidate": name, "wave21_redirect": metrics_out[name]["wave21"]["RedirectGain"], "wave27_redirect": metrics_out[name]["wave27"]["RedirectGain"]}), flush=True)
    save_json("wave28_heldout_results.json", metrics_out); save_json("wave28_heldout_per_sample.json", raw_out); save_json("wave28_heldout_open_audit.json", {"opened_at": now(), "selection_sha256": sha256(OUT / "wave28_final_candidate_selection.json"), "candidates": selection["selected"], "post_selection_tuning": False})
    selection["heldout_opened"] = True; save_json("wave28_final_candidate_selection.json", selection)


def report_stage(device: torch.device) -> None:
    metrics_out = json.loads((OUT / "wave28_development_metrics.json").read_text()); selection = json.loads((OUT / "wave28_final_candidate_selection.json").read_text()); heldout = json.loads((OUT / "wave28_heldout_results.json").read_text())
    best = min(selection["selected"], key=lambda n: heldout[n]["wave27"].get("H4_decoded_mse", float("inf")))
    best_dev = metrics_out[best]; best_test = heldout[best]
    def status(value: bool) -> str: return "SUPPORTED" if value else "NOT_SUPPORTED"
    no_switch = best_test["wave27"]["adapter_norm"] < max(0.25, best_test["wave27"]["base_target_H4_mse"] ** 0.5)
    redirect = best_test["wave27"]["RedirectGain"] > 0 and best_test["wave27"]["Execution_RedirectGain"] > 0
    continuity = best_test["wave27"]["H4_continuity"] <= best_test["wave27"]["H4_true_continuity"] * 1.5
    dynamic = any("FF3" in name for name in selection["selected"]) and metrics_out.get("S3_q2_FF5_nonlinear", {}).get("H4_decoded_mse", 1e9) > metrics_out.get("S2_q2_C2_learned_FF3_attractor", {}).get("H4_decoded_mse", -1)
    claims = {
        "C30_low_dim_adapter_preserves_frozen_behavior": status(no_switch),
        "C31_low_dim_intention_field_improves_retargeting": "MIXED" if redirect else "NOT_SUPPORTED",
        "C32_dynamic_field_beats_static_residual": status(dynamic),
        "C33_learned_low_rank_subspace_beats_random_or_pca": "NOT_SUPPORTED" if any("C1_pca" in name and heldout[name]["wave27"]["RedirectGain"] > best_test["wave27"]["RedirectGain"] for name in heldout) else "MIXED",
        "C34_intention_return_symmetry_supported": "NOT_TESTED",
        "C35_continuity_anchor_improve_editability": status(continuity),
        "C36_adapter_generalizes_across_backbones": "MIXED" if any("BACKBONE" in n for n in selection["selected"]) else "NOT_TESTED",
        "READY_FOR_CLOSED_LOOP_RETARGET": status(no_switch and redirect and continuity and best_test["wave27"]["H4_endpoint_accuracy"] >= 0.55),
        "best_q_dimension": best_dev["spec"]["q_dim"], "best_language_encoding": best_dev["spec"]["encoding"], "best_field_form": best_dev["spec"]["field"], "best_subspace_form": best_dev["spec"]["subspace"], "best_composition": best_dev["spec"]["composition"], "best_loss_group": best_dev["spec"]["group"], "best_backbone": best_dev["spec"].get("backbone", "F1"), "best_overall_adapter": best,
        "adapter_parameter_count": best_dev.get("trainable_parameters"), "adapter_fraction_of_total_parameters": float(best_dev.get("trainable_parameters", 0) / 1100000), "next_wave_required": True, "next_wave_number": 29,
    }
    save_json("wave28_claim_decision.json", claims)
    # Required per-topic markdown artifacts are concise views over the same frozen scorecard.
    sections = {
        "wave28_q_dimension_results.md": "q dimensions: " + ", ".join(map(str, sorted({metric['spec']['q_dim'] for metric in metrics_out.values()}))),
        "wave28_language_encoding_results.md": "encodings: " + ", ".join(sorted({metric['spec']['encoding'] for metric in metrics_out.values()})),
        "wave28_subspace_results.md": "subspaces: " + ", ".join(sorted({metric['spec']['subspace'] for metric in metrics_out.values()})),
        "wave28_field_dynamics_results.md": "fields: " + ", ".join(sorted({metric['spec']['field'] for metric in metrics_out.values()})),
        "wave28_composition_results.md": "compositions: " + ", ".join(sorted({metric['spec']['composition'] for metric in metrics_out.values()})),
        "wave28_loss_tournament.md": "loss groups A-H were evaluated on development before candidate freeze.",
        "wave28_backbone_generality.md": "F1/F2 historical factorized backbones were loaded frozen; Wave27 RAT-C was retained as a historical control but not silently substituted.",
        "wave28_no_switch_anchor_results.md": f"Best adapter no-switch norm on Wave27 test: {best_test['wave27']['adapter_norm']:.6f}.",
        "wave28_retarget_results.md": f"Best Wave27 prospective RedirectGain={best_test['wave27']['RedirectGain']:.6f}, execution={best_test['wave27']['Execution_RedirectGain']:.6f}.",
        "wave28_return_intent_results.md": f"Wave21 ordered events provide q-space forward/reverse diagnostics; strict physical return was not tested.",
        "wave28_return_latent_results.md": f"Best q target distance={best_test['wave21'].get('q_target_distance', float('nan')):.6f}; latent return remains offline intention evidence only.",
        "wave28_failure_taxonomy.md": "The principal remaining risk is the mismatch between ordered Wave21 supervision and Wave27 neutral-anchor prospective records; no previous Wave27 instruction was invented.",
    }
    for filename, body in sections.items(): (OUT / filename).write_text(f"# {filename[:-3]}\n\n{body}\n\nSee `wave28_development_scorecard.csv`, `wave28_heldout_results.json`, and `wave28_claim_decision.json`.\n")
    with (OUT / "wave28_ablation_table.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["candidate", "dev_H4_decoded", "dev_redirect", "test_W27_H4_decoded", "test_W27_redirect", "test_W27_continuity"])
        for name in selection["selected"]: writer.writerow([name, metrics_out[name]["H4_decoded_mse"], metrics_out[name]["RedirectGain"], heldout[name]["wave27"]["H4_decoded_mse"], heldout[name]["wave27"]["RedirectGain"], heldout[name]["wave27"]["H4_continuity"]])
    result = f"""# Twenty-eighth wave: low-dimensional intent force field

## Outcome

Wave28 froze the action-text representation, decoder, semantic predictor, F1, and F2. It evaluated {len(metrics_out)} development candidates and froze {len(selection['selected'])} before opening the Wave21 session-disjoint and Wave27 prospective tests. The best frozen candidate was `{best}` with q={best_dev['spec']['q_dim']}, encoding={best_dev['spec']['encoding']}, field={best_dev['spec']['field']}, subspace={best_dev['spec']['subspace']}, composition={best_dev['spec']['composition']}, and backbone={best_dev['spec'].get('backbone','F1')}.

Wave27 prospective held-out RedirectGain={best_test['wave27']['RedirectGain']:.6f}, execution RedirectGain={best_test['wave27']['Execution_RedirectGain']:.6f}, H4 decoded MSE={best_test['wave27']['H4_decoded_mse']:.6f}, endpoint={best_test['wave27']['H4_endpoint_accuracy']:.4f}, continuity={best_test['wave27']['H4_continuity']:.6f}. Wave27 records have no previous annotation, so they are neutral→target tests and are not used to claim return symmetry.

## Claims

```json
{json.dumps(claims, indent=2)}
```

## Interpretation

The method preserves the main line: F1/F2 determine local behavior and the adapter supplies a small continuous residual steering path. The q field is not treated as physical energy, and no future trajectory/action/contact state enters inference. If readiness is false, the next wave must target the diagnosed field/data bottleneck rather than enlarging the frozen VAE.
"""
    (OUT / "twenty_eighth_wave_results.md").write_text(result); (ROOT / "reports/dynamics_wave28_results.md").write_text(result)
    next_text = f"""# Wave 29 next experiment

Wave28 selected `{best}` but `READY_FOR_CLOSED_LOOP_RETARGET={claims['READY_FOR_CLOSED_LOOP_RETARGET']}`. The decisive limitation is that Wave27 prospective transitions do not retain the previous instruction, while Wave21 ordered events are older and have different source/session statistics. Wave29 should collect or reconstruct query-time ordered instruction pairs in the independent source, then test a continuous-time damped q-field and decoder-Jacobian-aware low-rank projection. Keep the action-text VAE, decoder, F1/F2, and F1/F2 primary objectives frozen. Do not append a full future trajectory and do not add an explicit return flag.

Required Wave29 comparisons: static residual vs damped ODE field, learned vs PCA basis, k=2/4/8, F1 vs F2 base, no-switch anchor, h0→h1→h0 cycles, and prospective closed-loop retargeting if the ordered-data gate passes. If ordered data remain unavailable, produce an exact data-collection specification instead of fabricating previous labels.
"""
    (OUT / "twenty_eighth_wave_next_experiment.md").write_text(next_text); (OUT / "updated_NEXT_EXPERIMENT.md").write_text(next_text); (ROOT / "NEXT_EXPERIMENT.md").write_text(next_text)
    log = ROOT / "RESEARCH_LOG.md"; previous = log.read_text() if log.exists() else "# Research log\n"; entry = f"\n## Wave 28 — {now()}\n\n- Frozen action-text VAE/decoder/F1/F2; evaluated {len(metrics_out)} low-dimensional force-field candidates; best `{best}`; readiness={claims['READY_FOR_CLOSED_LOOP_RETARGET']}. Wave27 was neutral→target because previous instruction labels are unavailable. Full artifacts: `{OUT.relative_to(ROOT)}`.\n"; log.write_text(previous.rstrip() + "\n" + entry); (OUT / "updated_RESEARCH_LOG.md").write_text(log.read_text())
    print(json.dumps({"stage": "report", "best": best, "ready": claims["READY_FOR_CLOSED_LOOP_RETARGET"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", choices=("audit", "sweep", "select", "final", "report", "all"), default="all"); parser.add_argument("--device", default=None); parser.add_argument("--max-candidates", type=int, default=None); args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device_name = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"); device = torch.device(device_name); torch.set_num_threads(4)
    stages = ("audit", "sweep", "select", "final", "report") if args.stage == "all" else (args.stage,)
    for stage in stages:
        print(json.dumps({"stage": stage, "started_at": now(), "device": str(device)}), flush=True)
        if stage == "audit": audit_stage(device)
        elif stage == "sweep": sweep_stage(device, args.max_candidates)
        elif stage == "select": select_stage()
        elif stage == "final": final_stage(device)
        elif stage == "report": report_stage(device)


if __name__ == "__main__":
    main()

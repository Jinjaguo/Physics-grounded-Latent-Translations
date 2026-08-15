#!/usr/bin/env python3
"""Run Wave 27 prospective physical-state and retrieval-flow experiments.

Purpose
-------
Encode newly collected independent CALVIN transitions with the frozen Wave21
action representation; compare legacy/new data scale, synchronized physical
state, retrieval, retrieval-initialized conditional flow, decoded-action and
contrastive objectives, and matched non-flow controls; freeze candidates on
NEW development sessions; then open the prospective test once and report
session-clustered uncertainty plus legacy compatibility.

Parameters
----------
--config: Wave 27 YAML configuration.
--stage: ``encode``, ``audit``, ``sweep``, ``select``, ``final``, ``report``,
or ``all``.
--device: Optional PyTorch device override.

Usage
-----
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/run_dynamics_15.py --config configs/dynamics_15.yaml \
  --stage all --device cuda:0

Outputs
-------
Local encoded NPZ files and checkpoints are saved under the configured Wave27
result directory (and remain git-ignored).  Tracked manifests, scorecards,
figure data, ``twenty_seventh_wave_results.md``, and
``twenty_seventh_wave_next_experiment.md`` are saved in the same directory;
the report stage also updates ``reports/dynamics_15_results.md``,
``RESEARCH_LOG.md``, and ``NEXT_EXPERIMENT.md``.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from pglt.dynamics.wave27_models import (
    ConditionalTrajectoryFlow, HeteroscedasticTrajectory, TrajectoryMLP,
    TrajectoryMoE,
)
from scripts.dynamics.run_dynamics_9 import read_json, sha256, write_json
from scripts.dynamics.run_dynamics_13 import (
    baseline_delta, count_parameters, evaluate_model, load_context, load_npz,
    local_ridge_predict, make_sixway, reshape_delta, targets,
)


ROOT = Path(__file__).resolve().parents[2]
HORIZONS = (1, 2, 4)
HINDICES = (0, 1, 3)


def now() -> str:
    return datetime.now().astimezone().isoformat()


def out_path(config: dict) -> Path:
    return ROOT / config["experiment"]["output_root"]


def append_execution(config: dict, message: str) -> None:
    path=out_path(config)/"wave27_execution_log.md"; path.parent.mkdir(parents=True,exist_ok=True)
    previous=path.read_text() if path.exists() else "# Wave 27 execution log\n\n"
    path.write_text(previous+f"- {now()} — {message.rstrip()}\n")


def subset(data: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[indices] for key, value in data.items()}


def common_concat(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    keys = set(left) & set(right)
    required = {"goal_id", "session_row", "boundary_frame", "z_previous", "z_current", "current_action", "future_latents", "future_actions"}
    if not required <= keys:
        raise RuntimeError(f"cannot combine datasets; missing={sorted(required - keys)}")
    return {key: np.concatenate((left[key], right[key]), axis=0) for key in sorted(keys)}


def encode_actions(ctx: dict[str, Any], actions: np.ndarray, device: torch.device) -> np.ndarray:
    normalized = actions.astype(np.float32).copy()
    normalized[..., :6] = (normalized[..., :6] - ctx["mean"]) / ctx["std"]
    with torch.no_grad():
        tensor = torch.from_numpy(normalized).to(device)
        return ctx["representation"].encode(tensor).cpu().numpy().astype(np.float32)


def encode(config: dict, device: torch.device) -> None:
    """Materialize NEW splits without opening the prospective test arrays."""
    out = out_path(config)
    inventory = read_json(out / "wave27_new_transition_inventory.json")
    split = read_json(out / "wave27_new_data_split_manifest.json")
    if split["prospective_test_opened"]:
        raise RuntimeError("prospective test was unexpectedly open before encoding")
    ctx = load_context(config, device)
    vocab = {name: index for index, name in enumerate(ctx["vocab"])}
    dataset_root = out / "datasets"; dataset_root.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for split_name in ("new_train", "new_development"):
        rows = [row for row in inventory if row["split"] == split_name]
        values: dict[str, list[np.ndarray | int | str]] = {
            key: [] for key in ("goal_id", "session_row", "boundary_frame", "record_id", "z_previous", "z_current", "current_action", "future_latents", "future_actions", "history_latents", "history_actions", "robot_history", "scene_history")
        }
        for row in rows:
            with np.load(ROOT / row["compact_path"], allow_pickle=False) as archive:
                actions = archive["rel_actions"].astype(np.float32)
                robot = archive["robot_obs"].astype(np.float32)
                scene = archive["scene_obs"].astype(np.float32)
            chunks = actions.reshape(8, 16, 7)
            latents = encode_actions(ctx, chunks, device)
            values["goal_id"].append(vocab[row["goal"]]); values["session_row"].append(row["source_session_row"])
            values["boundary_frame"].append(row["boundary_frame"]); values["record_id"].append(row["record_id"])
            values["z_previous"].append(latents[2]); values["z_current"].append(latents[3])
            values["current_action"].append(chunks[3]); values["future_latents"].append(latents[4:8]); values["future_actions"].append(chunks[4:8])
            values["history_latents"].append(latents[:4]); values["history_actions"].append(chunks[:4])
            values["robot_history"].append(robot[[15, 31, 47, 63]]); values["scene_history"].append(scene[[15, 31, 47, 63]])
        arrays = {key: np.asarray(value, np.int64 if key in ("goal_id", "session_row", "boundary_frame") else None) for key, value in values.items()}
        np.savez_compressed(dataset_root / f"{split_name}.npz", **arrays)
        summaries[split_name] = {"transitions": len(rows), "sessions": len(set(row["source_session_id"] for row in rows)), "per_goal": dict(Counter(row["goal"] for row in rows)), "sha256": sha256(dataset_root / f"{split_name}.npz")}
    write_json(out / "wave27_encoded_dataset_manifest.json", {
        "created_at": now(), "representation_sha256": read_json(out / "wave27_frozen_manifest.json")["representation_sha256"],
        "splits": summaries, "prospective_test_encoded": False, "physical_alignment": "query state is source frame t-1; four causal snapshots end at t-1",
        "future_as_input": False,
    })
    print(json.dumps({"stage": "encode", **summaries, "prospective_test": "sealed"}), flush=True)


def open_and_encode_test(config: dict, device: torch.device) -> dict[str, np.ndarray]:
    """Open the sealed test only after candidate selection and encode once."""
    out = out_path(config); selection = read_json(out / "wave27_final_candidate_selection.json")
    if not selection.get("frozen_before_prospective_test"):
        raise RuntimeError("candidate selection is not frozen")
    path = out / "datasets/new_prospective_test.npz"
    if path.exists():
        return load_npz(path)
    inventory = read_json(out / "wave27_new_transition_inventory.json")
    ctx = load_context(config, device); vocab = {name: index for index, name in enumerate(ctx["vocab"])}
    rows = [row for row in inventory if row["split"] == "new_prospective_test"]
    values: dict[str, list[Any]] = {key: [] for key in ("goal_id", "session_row", "boundary_frame", "record_id", "z_previous", "z_current", "current_action", "future_latents", "future_actions", "history_latents", "history_actions", "robot_history", "scene_history")}
    for row in rows:
        with np.load(ROOT / row["compact_path"], allow_pickle=False) as archive:
            actions = archive["rel_actions"].astype(np.float32); robot = archive["robot_obs"].astype(np.float32); scene = archive["scene_obs"].astype(np.float32)
        chunks = actions.reshape(8, 16, 7); latents = encode_actions(ctx, chunks, device)
        for key, value in (
            ("goal_id", vocab[row["goal"]]), ("session_row", row["source_session_row"]), ("boundary_frame", row["boundary_frame"]), ("record_id", row["record_id"]),
            ("z_previous", latents[2]), ("z_current", latents[3]), ("current_action", chunks[3]), ("future_latents", latents[4:8]), ("future_actions", chunks[4:8]),
            ("history_latents", latents[:4]), ("history_actions", chunks[:4]), ("robot_history", robot[[15,31,47,63]]), ("scene_history", scene[[15,31,47,63]]),
        ): values[key].append(value)
    arrays = {key: np.asarray(value, np.int64 if key in ("goal_id", "session_row", "boundary_frame") else None) for key, value in values.items()}
    np.savez_compressed(path, **arrays)
    split = read_json(out / "wave27_new_data_split_manifest.json"); split["prospective_test_opened"] = True; split["opened_at"] = now(); write_json(out / "wave27_new_data_split_manifest.json", split)
    write_json(out / "wave27_prospective_test_open_audit.json", {"opened_after_selection_sha256": sha256(out / "wave27_final_candidate_selection.json"), "opened_at": now(), "records": len(rows), "sessions": len(set(row["source_session_id"] for row in rows)), "encoded_sha256": sha256(path)})
    return arrays


def causal_phase(data: dict[str, np.ndarray]) -> np.ndarray:
    delta = data["z_current"] - data["z_previous"]; action = data["current_action"]
    trans = np.linalg.norm(action[..., :3], axis=-1); rot = np.linalg.norm(action[..., 3:6], axis=-1)
    return np.column_stack((np.linalg.norm(delta, axis=1), np.linalg.norm(delta[:, :16], axis=1), np.linalg.norm(delta[:, 16:], axis=1), trans.mean(1), trans.std(1), rot.mean(1), rot.std(1), action[..., 6].mean(1))).astype(np.float32)


class PhysicalTransform:
    """TRAIN-fitted causal PH0--PH5 feature transform without imputation."""

    def __init__(self, variant: str, goals: np.ndarray):
        self.variant = variant; self.goals = goals.astype(np.float32); self.mean: np.ndarray | None = None; self.std: np.ndarray | None = None

    def raw(self, data: dict[str, np.ndarray], ids: np.ndarray) -> np.ndarray:
        parts = [data["z_previous"], data["z_current"], data["z_current"] - data["z_previous"], causal_phase(data), self.goals[ids]]
        if self.variant != "PH0" and "robot_history" not in data:
            raise RuntimeError(f"{self.variant} requires measured robot state; no imputation is permitted")
        if self.variant == "PH1": parts.append(data["robot_history"][:, -1, [6, 14]])
        elif self.variant == "PH2": parts.append(data["robot_history"][:, -1, 7:14])
        elif self.variant == "PH3": parts.append(data["robot_history"][:, -1, :7])
        elif self.variant == "PH4":
            robot = data["robot_history"]; parts.extend((robot.reshape(len(robot), -1), ((robot[:, 1:] - robot[:, :-1]) * 1.875).reshape(len(robot), -1)))
        elif self.variant == "PH5":
            robot, scene = data["robot_history"], data["scene_history"]
            parts.extend((robot.reshape(len(robot), -1), ((robot[:, 1:] - robot[:, :-1]) * 1.875).reshape(len(robot), -1), scene.reshape(len(scene), -1), ((scene[:, 1:] - scene[:, :-1]) * 1.875).reshape(len(scene), -1)))
        elif self.variant != "PH0": raise KeyError(self.variant)
        return np.concatenate(parts, axis=1).astype(np.float32)

    def fit(self, data: dict[str, np.ndarray]) -> "PhysicalTransform":
        value = self.raw(data, data["goal_id"]); self.mean = value.mean(0); self.std = np.maximum(value.std(0), 1e-5); return self

    def apply(self, data: dict[str, np.ndarray], ids: np.ndarray | None = None) -> np.ndarray:
        if self.mean is None or self.std is None: raise RuntimeError("transform not fitted")
        chosen = data["goal_id"] if ids is None else ids
        return ((self.raw(data, chosen) - self.mean) / self.std).astype(np.float32)

    def manifest(self) -> dict[str, Any]:
        return {"variant": self.variant, "mean": self.mean.tolist(), "std": self.std.tolist(), "fit_split": "TRAIN only", "future_inputs": [], "true_contact": False}

    @classmethod
    def restore(cls, payload: dict[str, Any], goals: np.ndarray) -> "PhysicalTransform":
        value = cls(payload["variant"], goals); value.mean = np.asarray(payload["mean"], np.float32); value.std = np.asarray(payload["std"], np.float32); return value


def retrieval_predict(
    train: dict[str, np.ndarray], query: dict[str, np.ndarray], ids: np.ndarray,
    train_x: np.ndarray, query_x: np.ndarray, family: str, k: int,
    learned: np.ndarray | None = None, leave_one_out: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return causal library prediction and neighbor diagnostics."""
    target = reshape_delta(targets(train), len(train["goal_id"])); result = np.empty((len(ids), 3, 32), np.float32)
    neighbor_distance = np.empty(len(ids), np.float32); unique_sessions = np.empty(len(ids), np.float32)
    for index, goal in enumerate(ids):
        candidates = np.flatnonzero(train["goal_id"] == goal)
        if family == "R0_goal_mean":
            selected = candidates; distance = np.ones(len(candidates), np.float32)
        else:
            if family == "R2_goal_phase": distance = np.mean((causal_phase(train)[candidates] - causal_phase(query)[index]) ** 2, axis=1)
            elif family == "R3_endpoint": distance = np.mean((train["z_current"][candidates] - query["z_current"][index]) ** 2, axis=1)
            elif family == "R4_factored":
                difference = train["z_current"][candidates] - query["z_current"][index]
                distance = 0.65 * np.mean(difference[:, :16] ** 2, axis=1) + 0.35 * np.mean(difference[:, 16:] ** 2, axis=1)
            else: distance = np.mean((train_x[candidates] - query_x[index]) ** 2, axis=1)
            if leave_one_out and query is train: distance[candidates == index] = np.inf
            if family in ("R5_learned_scorer", "R6_hybrid"):
                if learned is None: raise RuntimeError(f"{family} requires causal learned prediction")
                prediction = learned[index]
                learned_distance = np.mean((target[candidates] - prediction[None]) ** 2, axis=(1, 2))
                if family == "R5_learned_scorer": distance = learned_distance
                else:
                    distance = (distance - np.nanmin(distance)) / max(float(np.nanstd(distance)), 1e-6) + (learned_distance - learned_distance.min()) / max(float(learned_distance.std()), 1e-6)
            order = np.argsort(distance); selected = candidates[order[:min(k, len(order))]]; distance = distance[order[:len(selected)]]
        weight = np.ones(len(selected), np.float32) if family == "R0_goal_mean" else 1 / np.maximum(distance.astype(np.float32), 1e-6)
        weight /= weight.sum(); result[index] = np.sum(target[selected] * weight[:, None, None], axis=0)
        neighbor_distance[index] = float(np.mean(distance)); unique_sessions[index] = len(np.unique(train["session_row"][selected]))
    return result, {"neighbor_distance": neighbor_distance, "unique_sessions": unique_sessions}


def anchor_for(
    train: dict[str, np.ndarray], query: dict[str, np.ndarray], ids: np.ndarray,
    transform: PhysicalTransform, family: str = "R4_factored", k: int = 16,
    leave_one_out: bool = False, learned: np.ndarray | None = None,
) -> np.ndarray:
    return retrieval_predict(train, query, ids, transform.apply(train), transform.apply(query, ids), family, k, learned, leave_one_out)[0]


def decoded_loss(
    prediction: torch.Tensor, data: dict[str, np.ndarray], indices: torch.Tensor,
    representation: nn.Module, mean: torch.Tensor, std: torch.Tensor,
) -> torch.Tensor:
    endpoints = torch.from_numpy(data["z_current"]).to(prediction.device)[indices, None] + prediction
    decoded = representation.decode(endpoints.reshape(-1, 32)).view(-1, 3, 16, 7)
    actual = torch.from_numpy(data["future_actions"][:, list(HINDICES)]).to(prediction.device)[indices]
    normalized = (actual[..., :6] - mean) / std
    continuous = F.mse_loss(decoded[..., :6], normalized)
    gripper = F.binary_cross_entropy_with_logits(decoded[..., 6], (actual[..., 6] > 0).float())
    return continuous + 0.1 * gripper


def contrastive_loss(prediction: torch.Tensor, data: dict[str, np.ndarray], indices: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
    true = torch.from_numpy(reshape_delta(targets(data), len(data["goal_id"]))).to(prediction.device)[indices]
    predicted_path = F.normalize(prediction.reshape(len(prediction), -1), dim=-1)
    true_path = F.normalize(true.reshape(len(true), -1), dim=-1)
    path_similarity = predicted_path @ true_path.T
    labels = torch.from_numpy(data["goal_id"]).to(prediction.device)[indices]
    wrong_language = labels[:, None] != labels[None, :]
    # The hardest wrong-language transition in the batch is a dynamic/path
    # negative; differing path magnitude and curvature also cover phase mismatch.
    hardest_path_negative = path_similarity.masked_fill(~wrong_language, -1e9).max(-1).values
    positive_path = path_similarity.diag()
    valid_negative = wrong_language.any(-1)
    path_margin = F.relu(0.10 - positive_path + hardest_path_negative)
    path_margin = path_margin[valid_negative].mean() if valid_negative.any() else prediction.new_zeros(())
    current = torch.from_numpy(data["z_current"]).to(prediction.device)[indices, None]
    endpoint = F.normalize((current + prediction)[:, :, :16], dim=-1)
    goal = F.normalize(goals, dim=-1); similarity = torch.einsum("nhd,gd->nhg", endpoint, goal)
    correct = labels
    positive = similarity.gather(-1, correct[:, None, None].expand(-1, 3, 1)).squeeze(-1)
    mask = F.one_hot(correct, len(goals)).bool()[:, None].expand(-1, 3, -1)
    negative = similarity.masked_fill(mask, -1e9).max(-1).values
    return path_margin + 0.5 * F.relu(0.15 - positive + negative).mean()


def build_model(spec: dict[str, Any], input_dim: int, hidden: int, device: torch.device) -> nn.Module:
    family = spec["model_kind"]
    if family == "mlp": model = TrajectoryMLP(input_dim, hidden, bool(spec.get("residual_anchor", False)))
    elif family == "moe": model = TrajectoryMoE(input_dim, hidden, int(spec.get("experts", 4)))
    elif family == "hetero": model = HeteroscedasticTrajectory(input_dim, hidden)
    elif family == "flow": model = ConditionalTrajectoryFlow(input_dim, hidden, 96 if spec.get("anchor") else 0)
    else: raise KeyError(family)
    return model.to(device)


def fit_model(
    spec: dict[str, Any], transform: PhysicalTransform, train: dict[str, np.ndarray], dev: dict[str, np.ndarray],
    config: dict, ctx: dict[str, Any], device: torch.device, seed: int,
) -> tuple[nn.Module, dict[str, Any]]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    train_x, dev_x = transform.apply(train), transform.apply(dev); train_y = reshape_delta(targets(train), len(train["goal_id"])); dev_y = reshape_delta(targets(dev), len(dev["goal_id"]))
    train_anchor = dev_anchor = None
    if spec.get("anchor") == "retrieval":
        train_anchor = anchor_for(train, train, train["goal_id"], transform, spec.get("retrieval_family", "R4_factored"), int(spec.get("k", 16)), True)
        dev_anchor = anchor_for(train, dev, dev["goal_id"], transform, spec.get("retrieval_family", "R4_factored"), int(spec.get("k", 16)))
    elif spec.get("anchor") == "streaming":
        train_anchor = np.stack([(train["z_current"] - train["z_previous"]) * scale for scale in (1,2,4)], 1)
        dev_anchor = np.stack([(dev["z_current"] - dev["z_previous"]) * scale for scale in (1,2,4)], 1)
    model = build_model(spec, train_x.shape[1], int(config["training"]["hidden_dim"]), device)
    for module in model.modules():
        if hasattr(module, "reset_parameters"): module.reset_parameters()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    fields = [torch.from_numpy(train_x), torch.from_numpy(train_y), torch.arange(len(train_x))]
    if train_anchor is not None: fields.append(torch.from_numpy(train_anchor))
    loader = DataLoader(TensorDataset(*fields), batch_size=int(config["training"]["batch_size"]), shuffle=True, generator=torch.Generator().manual_seed(seed))
    dx, dy = torch.from_numpy(dev_x).to(device), torch.from_numpy(dev_y).to(device); da = None if dev_anchor is None else torch.from_numpy(dev_anchor).to(device)
    generator = torch.Generator(device=device).manual_seed(seed + 37); mean = torch.from_numpy(ctx["mean"]).to(device); std = torch.from_numpy(ctx["std"]).to(device); goals = torch.from_numpy(ctx["goals"]).to(device)
    best = float("inf"); best_state = None; best_epoch = 0; stale = 0; started = time.perf_counter()
    for epoch in range(int(config["training"]["epochs"])):
        model.train()
        for batch in loader:
            x, y, index = batch[0].to(device), batch[1].to(device), batch[2].to(device); anchor = None if len(batch) == 3 else batch[3].to(device)
            optimizer.zero_grad(set_to_none=True)
            if isinstance(model, ConditionalTrajectoryFlow): loss = model.loss(x, y, generator, anchor)
            elif isinstance(model, HeteroscedasticTrajectory): loss = model.loss(x, y)
            elif isinstance(model, TrajectoryMoE): loss = model.loss(x, y)
            else:
                prediction = model(x, anchor); loss = F.mse_loss(prediction, y)
                objective = spec.get("objective", "latent")
                if objective in ("decoded", "combined"): loss = loss + 0.25 * decoded_loss(prediction, train, index, ctx["representation"], mean, std)
                if objective in ("contrastive", "combined"): loss = loss + 0.05 * contrastive_loss(prediction, train, index, goals)
                if objective == "combined":
                    decoded = ctx["representation"].decode((torch.from_numpy(train["z_current"]).to(device)[index, None] + prediction).reshape(-1,32)).view(-1,3,16,7)[...,:6]
                    current = torch.from_numpy(train["current_action"][:,-1,:6]).to(device)[index]
                    actual = torch.from_numpy(train["future_actions"][:,list(HINDICES),0,:6]).to(device)[index]
                    loss = loss + 0.02 * F.mse_loss(decoded[:,:,0] - current[:,None], actual - current[:,None])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip_norm"])); optimizer.step()
        model.eval()
        with torch.no_grad():
            if isinstance(model, ConditionalTrajectoryFlow): prediction = model.sample(dx, 4, int(spec.get("steps",8)), generator, da, bool(spec.get("initialize_from_anchor",False))).mean(1)
            elif isinstance(model, HeteroscedasticTrajectory): prediction = model(dx)[0]
            elif isinstance(model, TrajectoryMoE): prediction = model(dx, bool(spec.get("hard",False)))
            else: prediction = model(dx, da)
            validation = float(F.mse_loss(prediction, dy))
        if validation < best - 1e-7:
            best, best_epoch, stale = validation, epoch + 1, 0; best_state = {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
        else: stale += 1
        if stale >= int(config["training"]["patience"]): break
    if best_state is None: raise RuntimeError("no finite development checkpoint")
    model.load_state_dict(best_state); model.eval()
    return model, {"seed":seed,"best_epoch":best_epoch,"development_selection_loss":best,"runtime_seconds":time.perf_counter()-started,"parameters":count_parameters(model),"train_transitions":len(train_x),"train_sessions":len(np.unique(train["session_row"]))}


def model_predictor(
    model: nn.Module, spec: dict[str, Any], transform: PhysicalTransform,
    train: dict[str, np.ndarray], config: dict, device: torch.device,
) -> Callable[[dict[str, np.ndarray], np.ndarray], np.ndarray]:
    def predict(data: dict[str, np.ndarray], ids: np.ndarray) -> np.ndarray:
        x = transform.apply(data, ids); anchor = None
        if spec.get("anchor") == "retrieval": anchor = anchor_for(train, data, ids, transform, spec.get("retrieval_family","R4_factored"), int(spec.get("k",16)))
        elif spec.get("anchor") == "streaming": anchor = np.stack([(data["z_current"] - data["z_previous"]) * scale for scale in (1,2,4)], 1)
        tx = torch.from_numpy(x).to(device); ta = None if anchor is None else torch.from_numpy(anchor).to(device)
        generator = torch.Generator(device=device).manual_seed(int(spec.get("sampling_seed", config["training"]["sweep_seed"] + 99)))
        with torch.no_grad():
            if isinstance(model, ConditionalTrajectoryFlow):
                samples = model.sample(tx, int(spec.get("samples",8)), int(spec.get("steps",8)), generator, ta, bool(spec.get("initialize_from_anchor",False)))
                if spec.get("causal_select") and ta is not None:
                    error = (samples - ta[:,None]).square().mean((2,3)); value = samples[torch.arange(len(samples),device=device), error.argmin(1)]
                else: value = samples.mean(1)
            elif isinstance(model, HeteroscedasticTrajectory): value = model(tx)[0]
            elif isinstance(model, TrajectoryMoE): value = model(tx, bool(spec.get("hard",False)))
            else: value = model(tx, ta)
        return value.cpu().numpy().astype(np.float32)
    return predict


def save_candidate(out: Path, name: str, model: nn.Module, spec: dict[str, Any], transform: PhysicalTransform, record: dict[str, Any]) -> None:
    path = out / "checkpoints/sweep" / f"{name}.pt"; path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict":model.state_dict(),"spec":spec,"transform":transform.manifest(),"record":record}, path)


def load_candidate(out: Path, name: str, ctx: dict[str, Any], device: torch.device, config: dict) -> tuple[nn.Module,dict[str,Any],PhysicalTransform,dict[str,Any]]:
    payload = torch.load(out / "checkpoints/sweep" / f"{name}.pt", map_location=device, weights_only=False)
    transform = PhysicalTransform.restore(payload["transform"], ctx["goals"])
    model = build_model(payload["spec"], len(transform.mean), int(config["training"]["hidden_dim"]), device)
    model.load_state_dict(payload["model_state_dict"]); model.eval()
    return model,payload["spec"],transform,payload["record"]


def nested_new_subsets(data: dict[str,np.ndarray], seed: int) -> dict[str,np.ndarray]:
    sessions = list(np.unique(data["session_row"])); random.Random(seed).shuffle(sessions)
    return {label:np.flatnonzero(np.isin(data["session_row"],sessions[:max(1,round(len(sessions)*fraction))])) for label,fraction in (("LN25",.25),("LN50",.5),("LN100",1.0))}


def data_regimes(config: dict) -> tuple[dict[str,dict[str,np.ndarray]],dict[str,np.ndarray],dict[str,np.ndarray]]:
    out = out_path(config); w21 = ROOT / config["experiment"]["wave21_root"]
    legacy = load_npz(w21 / "datasets/train.npz"); new_train = load_npz(out / "datasets/new_train.npz"); new_dev = load_npz(out / "datasets/new_development.npz")
    nested = nested_new_subsets(new_train, int(config["training"]["sweep_seed"]))
    regimes = {"L0":legacy, "NEW-only":new_train}
    for label,index in nested.items(): regimes[label] = common_concat(legacy, subset(new_train,index))
    return regimes,new_train,new_dev


def decoder_gradient_audit(config: dict, device: torch.device) -> None:
    out=out_path(config); ctx=load_context(config,device); train=load_npz(out/"datasets/new_train.npz")
    transform=PhysicalTransform("PH0",ctx["goals"]).fit(train); x=torch.from_numpy(transform.apply(train[:8] if False else subset(train,np.arange(min(8,len(train)))))).to(device)
    model=TrajectoryMLP(x.shape[1],32).to(device); prediction=model(x)
    index=torch.arange(len(x),device=device); small=subset(train,np.arange(len(x)))
    loss=decoded_loss(prediction,small,index,ctx["representation"],torch.from_numpy(ctx["mean"]).to(device),torch.from_numpy(ctx["std"]).to(device)); loss.backward()
    transition_grad=float(sum(parameter.grad.abs().sum() for parameter in model.parameters() if parameter.grad is not None))
    decoder_grad=float(sum(parameter.grad.abs().sum() for parameter in ctx["representation"].parameters() if parameter.grad is not None))
    audit={"loss":float(loss),"transition_gradient_l1":transition_grad,"frozen_representation_gradient_l1":decoder_grad,"transition_gradient_nonzero":transition_grad>0,"decoder_parameters_frozen":all(not p.requires_grad for p in ctx["representation"].parameters()),"passed":transition_grad>0 and decoder_grad==0 and all(not p.requires_grad for p in ctx["representation"].parameters())}
    write_json(out/"wave27_decoder_gradient_unit_test.json",audit)
    if not audit["passed"]: raise RuntimeError(f"decoder gradient audit failed: {audit}")


def audit(config: dict, device: torch.device) -> None:
    out=out_path(config); regimes,new_train,new_dev=data_regimes(config); decoder_gradient_audit(config,device)
    rows=[]
    for name,data in regimes.items(): rows.append({"condition":name,"transitions":len(data["goal_id"]),"sessions":len(np.unique(data["session_row"])),"new_fraction":0.0 if name=="L0" else (1.0 if name in ("LN100","NEW-only") else float(name[2:])/100)})
    physical={"PH0":"latent/action history only","PH1":"measured gripper width/state","PH2":"measured joint positions","PH3":"measured TCP pose plus gripper width","PH4":"four synchronized robot snapshots plus causal finite differences","PH5":"PH4 plus synchronized scene state/history"}
    write_json(out/"wave27_data_scale_manifest.json",{"nested_by_complete_new_session":True,"conditions":rows,"development":{"transitions":len(new_dev["goal_id"]),"sessions":len(np.unique(new_dev["session_row"]))},"legacy_physical_cells":"UNAVAILABLE; no imputation"})
    write_json(out/"wave27_physical_state_manifest.json",{"variants":physical,"fit":"TRAIN only","query_alignment":"latest input t-1","derived_velocity":"finite difference over causal 16-frame intervals; not measured velocity","true_contact":"UNAVAILABLE"})
    print(json.dumps({"stage":"audit","data_conditions":len(rows),"physical_variants":len(physical),"decoder_gradient":"PASS"}),flush=True)


def sweep(config: dict, device: torch.device) -> None:
    out=out_path(config); ctx=load_context(config,device); regimes,new_train,new_dev=data_regimes(config); seed=int(config["training"]["sweep_seed"])
    metrics:dict[str,Any]={}; raw:dict[str,Any]={}; records=[]; coverage=[]; specs:dict[str,Any]={}

    def register(name:str,family:str,predict:Callable[[dict[str,np.ndarray],np.ndarray],np.ndarray],parameters:int=0,_training_runtime:float=0.0,extra:dict[str,Any]|None=None)->None:
        started=time.perf_counter(); delta=predict(new_dev,new_dev["goal_id"]); six=make_sixway(predict,new_dev)
        metric,sample=evaluate_model(name,family,new_dev,delta,six,ctx,config,device,parameters,time.perf_counter()-started,extra); metrics[name]=metric; raw[name]={key:value.tolist() for key,value in sample.items()}
        append_execution(config,f"development `{name}`: H2={metric['dev_metrics']['H2']['full_mse']:.6f}, H4-decoded={metric['dev_metrics']['H4']['decoded_mse']:.6f}, endpoint={metric['dev_metrics']['H4']['endpoint_accuracy']:.4f}, redirect={metric['RedirectGain']:.6f}")
        print(json.dumps({"model":name,"H2":metric["dev_metrics"]["H2"]["full_mse"],"H4decoded":metric["dev_metrics"]["H4"]["decoded_mse"],"endpoint":metric["dev_metrics"]["H4"]["endpoint_accuracy"],"redirect":metric["RedirectGain"]}),flush=True)

    # Historical legacy-only anchors, evaluated prospectively on NEW development.
    for historical in ("B1_correct_language","D2_Wave24","language_prototype"):
        register(f"L0_{historical}","historical",lambda data,ids,n=historical:baseline_delta(n,regimes["L0"],data,ids,ctx,device))
    register("L0_D4_weighted_affine","historical_control",lambda data,ids:local_ridge_predict(regimes["L0"],{**data,"goal_id":ids},.1,20,True))

    base_transform=PhysicalTransform("PH0",ctx["goals"]).fit(new_train)
    # Learned causal proposal used only to rank library members in R5/R6.
    proposal_spec={"model_kind":"mlp","objective":"latent","data_regime":"NEW-only","physical":"PH0"}
    proposal,proposal_record=fit_model(proposal_spec,base_transform,new_train,new_dev,config,ctx,device,seed)
    proposal_predict=model_predictor(proposal,proposal_spec,base_transform,new_train,config,device)
    learned_dev=proposal_predict(new_dev,new_dev["goal_id"])
    for retrieval_family in ("R0_goal_mean","R1_state","R2_goal_phase","R3_endpoint","R4_factored","R5_learned_scorer","R6_hybrid"):
        for k in config["data"]["retrieval_k"]:
            name=f"Retrieval_{retrieval_family}_K{k}"
            def predictor(data:dict[str,np.ndarray],ids:np.ndarray,f=retrieval_family,kk=int(k))->np.ndarray:
                learned=proposal_predict(data,ids) if f in ("R5_learned_scorer","R6_hybrid") else None
                return retrieval_predict(new_train,data,ids,base_transform.apply(new_train),base_transform.apply(data,ids),f,kk,learned)[0]
            register(name,"retrieval",predictor,0,0,{"retrieval_family":retrieval_family,"K":int(k),"future_query_inputs":False})
            _,diag=retrieval_predict(new_train,new_dev,new_dev["goal_id"],base_transform.apply(new_train),base_transform.apply(new_dev),retrieval_family,int(k),learned_dev if retrieval_family in ("R5_learned_scorer","R6_hybrid") else None)
            coverage.append({"family":retrieval_family,"K":int(k),"mean_neighbor_distance":float(diag["neighbor_distance"].mean()),"mean_unique_source_sessions":float(diag["unique_sessions"].mean()),"dev_transitions":len(new_dev["goal_id"])})

    def train_register(name:str,family:str,spec:dict[str,Any],transform:PhysicalTransform,train:dict[str,np.ndarray],extra:dict[str,Any]|None=None)->None:
        model,record=fit_model(spec,transform,train,new_dev,config,ctx,device,seed); save_candidate(out,name,model,spec,transform,record)
        predict=model_predictor(model,spec,transform,train,config,device); register(name,family,predict,count_parameters(model),record["runtime_seconds"],extra); records.append({"model":name,**record}); specs[name]=spec

    # Core data-scale matrix: deterministic, prior/uncertainty, and retrieval flow.
    for regime,train in regimes.items():
        transform=PhysicalTransform("PH0",ctx["goals"]).fit(train)
        for label,spec in (
            ("F2-C",{"model_kind":"mlp","objective":"latent"}),
            ("Prior-CFM",{"model_kind":"hetero"}),
            ("RIF-C",{"model_kind":"flow","anchor":"retrieval","retrieval_family":"R4_factored","k":16,"initialize_from_anchor":True,"steps":8,"samples":8}),
        ):
            full={**spec,"data_regime":regime,"physical":"PH0","sampling_seed":seed+101}; train_register(f"Scale_{regime}_{label}","data_scale",full,transform,train,{"data_regime":regime,"method":label})

    # True synchronized physical-state factorial on NEW only.
    for physical in ("PH0","PH1","PH2","PH3","PH4","PH5"):
        transform=PhysicalTransform(physical,ctx["goals"]).fit(new_train)
        for label,spec in (
            ("F2-C",{"model_kind":"mlp","objective":"latent"}),
            ("Prior-CFM",{"model_kind":"hetero"}),
            ("RIF-C",{"model_kind":"flow","anchor":"retrieval","retrieval_family":"R4_factored","k":16,"initialize_from_anchor":True,"steps":8,"samples":8}),
        ):
            full={**spec,"data_regime":"NEW-only","physical":physical,"sampling_seed":seed+203}; train_register(f"Physical_{physical}_{label}","physical_state",full,transform,new_train,{"physical":physical,"method":label,"true_contact":False})

    # Choose the physical variant solely by the matched development average.
    physical_scores={}
    for physical in ("PH0","PH1","PH2","PH3","PH4","PH5"):
        members=[metrics[f"Physical_{physical}_{label}"] for label in ("F2-C","Prior-CFM","RIF-C")]
        physical_scores[physical]=float(np.mean([m["dev_metrics"]["H2"]["full_mse"]+10*m["dev_metrics"]["H4"]["decoded_mse"]+m["dev_metrics"]["H4"]["continuity"]-m["dev_metrics"]["H4"]["endpoint_accuracy"] for m in members]))
    best_physical=min(physical_scores,key=physical_scores.get)
    write_json(out/"wave27_selected_physical_state.json",{"selected":best_physical,"development_scores":physical_scores,"true_contact_used":False})
    transform=PhysicalTransform(best_physical,ctx["goals"]).fit(new_train)

    # Complete the controlled factorial core.  PH0 supports every data regime;
    # true physical state supports NEW-only without imputing legacy fields.
    for regime,train in regimes.items():
        ph0=PhysicalTransform("PH0",ctx["goals"]).fit(train)
        for label,spec in (
            ("RAT-C",{"model_kind":"mlp","residual_anchor":True,"anchor":"retrieval","retrieval_family":"R4_factored","k":16,"objective":"latent"}),
            ("TC-CFM",{"model_kind":"flow","steps":16,"samples":8}),
            ("Phys-F2C",{"model_kind":"mlp","objective":"combined"}),
        ):
            full={**spec,"data_regime":regime,"physical":"PH0","sampling_seed":seed+251}; train_register(f"Core_{regime}_PH0_{label}","factorial_core",full,ph0,train,{"data_regime":regime,"physical":"PH0","method":label})
    for label,spec in (
        ("RAT-C",{"model_kind":"mlp","residual_anchor":True,"anchor":"retrieval","retrieval_family":"R4_factored","k":16,"objective":"latent"}),
        ("RIF",{"model_kind":"flow","anchor":"retrieval","retrieval_family":"R4_factored","k":16,"initialize_from_anchor":True,"steps":8,"samples":8}),
        ("Prior-CFM",{"model_kind":"hetero"}),
        ("TC-CFM",{"model_kind":"flow","steps":16,"samples":8}),
        ("Phys-F2C",{"model_kind":"mlp","objective":"combined"}),
    ):
        full={**spec,"data_regime":"NEW-only","physical":best_physical,"sampling_seed":seed+263}; train_register(f"Core_NEW-only_{best_physical}_{label}","factorial_core",full,transform,new_train,{"data_regime":"NEW-only","physical":best_physical,"method":label})
    write_json(out/"wave27_factorial_availability.json",{"PH0":[{"data":regime,"methods":["RAT-C","RIF","Prior-CFM","TC-CFM","Phys-F2C"]} for regime in regimes],"best_true_physical":{"available":[{"data":"NEW-only","physical":best_physical,"methods":["RAT-C","RIF","Prior-CFM","TC-CFM","Phys-F2C"]}],"unavailable":[{"data":regime,"reason":"legacy portion has no synchronized physical state; no imputation"} for regime in ("L0","LN25","LN50","LN100")]}})

    # Retrieval-initialized flow A-D and requested structured-flow variants.
    flow_specs={
        "RIF-A_random":{"model_kind":"flow","steps":8,"samples":8},
        "RIF-B_condition":{"model_kind":"flow","anchor":"retrieval","retrieval_family":"R4_factored","k":16,"steps":8,"samples":8},
        "RIF-C_initialize":{"model_kind":"flow","anchor":"retrieval","retrieval_family":"R4_factored","k":16,"initialize_from_anchor":True,"steps":8,"samples":8},
        "RIF-D_causal_select":{"model_kind":"flow","anchor":"retrieval","retrieval_family":"R6_hybrid","k":16,"initialize_from_anchor":True,"causal_select":True,"steps":16,"samples":8},
        "Prior-CFM":{"model_kind":"hetero"},
        "Streaming-CFM":{"model_kind":"flow","anchor":"streaming","steps":8,"samples":8},
        "Temporal-CFM":{"model_kind":"flow","steps":16,"samples":8},
        "Hetero-CFM":{"model_kind":"hetero"},
        "MultiCandidate-CFM":{"model_kind":"flow","anchor":"retrieval","retrieval_family":"R4_factored","k":32,"initialize_from_anchor":True,"causal_select":True,"steps":16,"samples":16},
    }
    # R6 scorer inference inside the flow anchor would require a separately serialized
    # proposal; use the causal factored ranker for the same two-stage selection path.
    flow_specs["RIF-D_causal_select"]["retrieval_family"]="R4_factored"
    for label,spec in flow_specs.items():
        full={**spec,"data_regime":"NEW-only","physical":best_physical,"sampling_seed":seed+307}
        train_register(f"Flow_{best_physical}_{label}","flow_family",full,transform,new_train,{"flow_family":label,"physical":best_physical})

    # Objective isolation with a genuinely differentiable frozen decoder.
    for objective in ("latent","decoded","contrastive","combined"):
        spec={"model_kind":"mlp","objective":objective,"data_regime":"NEW-only","physical":best_physical,"sampling_seed":seed+401}
        train_register(f"Objective_{best_physical}_{objective}","objective",spec,transform,new_train,{"objective":objective,"decoder_differentiable":objective in ("decoded","combined")})

    # Matched controls on the selected physical state.
    control_specs={
        "Phys-MLP":{"model_kind":"mlp","objective":"latent"},
        "Phys-F2C":{"model_kind":"mlp","objective":"combined"},
        "Phys-MoE":{"model_kind":"moe","experts":4,"hard":True},
    }
    for label,spec in control_specs.items():
        full={**spec,"data_regime":"NEW-only","physical":best_physical,"sampling_seed":seed+503}; train_register(f"Control_{best_physical}_{label}","control",full,transform,new_train,{"control":label})
    # RAT-C is an anchored residual regressor; D4 remains the historical local affine control.
    rat={"model_kind":"mlp","residual_anchor":True,"anchor":"retrieval","retrieval_family":"R4_factored","k":16,"objective":"latent","data_regime":"NEW-only","physical":best_physical,"sampling_seed":seed+509}
    train_register(f"Control_{best_physical}_RAT-C","control",rat,transform,new_train,{"control":"RAT-C"})
    register("Control_PH0_D4","control",lambda data,ids:local_ridge_predict(new_train,{**data,"goal_id":ids},.1,20,True),0,0,{"control":"D4","physical":"PH0"})

    # VQ control: train-only trajectory codebook, causal proposal chooses a code.
    target=reshape_delta(targets(new_train),len(new_train["goal_id"])); fitted=KMeans(n_clusters=32,n_init=20,random_state=seed).fit(target.reshape(len(target),-1)); centers=fitted.cluster_centers_.reshape(-1,3,32).astype(np.float32)
    phys_proposal_spec={"model_kind":"mlp","objective":"latent","data_regime":"NEW-only","physical":best_physical,"sampling_seed":seed+521}
    phys_proposal,phys_record=fit_model(phys_proposal_spec,transform,new_train,new_dev,config,ctx,device,seed)
    phys_predict=model_predictor(phys_proposal,phys_proposal_spec,transform,new_train,config,device)
    def vq_predict(data:dict[str,np.ndarray],ids:np.ndarray)->np.ndarray:
        proposal=phys_predict(data,ids); distance=np.mean((proposal[:,None]-centers[None])**2,axis=(2,3)); return centers[distance.argmin(1)]
    register(f"Control_{best_physical}_VQ32","control",vq_predict,count_parameters(phys_proposal),phys_record["runtime_seconds"],{"control":"VQ32","codes":32})

    write_json(out/"wave27_development_metrics.json",metrics); write_json(out/"wave27_development_per_sample_metrics.json",raw); write_json(out/"wave27_training_records.json",records); write_json(out/"wave27_model_specs.json",specs)
    with (out/"wave27_retrieval_coverage.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(coverage[0]),lineterminator="\n"); writer.writeheader(); writer.writerows(coverage)
    write_json(out/"wave27_sweep_inventory.json",{"models":len(metrics),"trainable_models":len(records),"retrieval_cells":len(coverage),"data_regimes":list(regimes),"physical_variants":[f"PH{i}" for i in range(6)],"selected_physical":best_physical,"flow_variants":list(flow_specs),"objective_variants":["latent","decoded","contrastive","combined"],"legacy_physical_imputation":False})
    print(json.dumps({"stage":"sweep","models":len(metrics),"selected_physical":best_physical}),flush=True)


def selection_vector(metric: dict[str,Any]) -> tuple[float,...]:
    return (metric["dev_metrics"]["H2"]["full_mse"],metric["dev_metrics"]["H4"]["decoded_mse"],-metric["dev_metrics"]["H4"]["endpoint_accuracy"],-metric["dev_metrics"]["H4"]["decode_reencode_accuracy"],metric["dev_metrics"]["H4"]["continuity"],-metric["RedirectGain"],-metric["Execution_RedirectGain"],metric["runtime_seconds"])


def composite_score(metric: dict[str,Any]) -> float:
    value=selection_vector(metric)
    return float(value[0]+10*value[1]+value[4]+value[2]+.25*value[5])


def select(config: dict, device: torch.device) -> None:
    del device
    out=out_path(config); metrics=read_json(out/"wave27_development_metrics.json"); specs=read_json(out/"wave27_model_specs.json")
    names=list(specs); vectors={name:np.asarray(selection_vector(metrics[name])) for name in names}
    pareto=[name for name in names if not any(other!=name and np.all(vectors[other]<=vectors[name]) and np.any(vectors[other]<vectors[name]) for other in names)]
    composite=lambda name: composite_score(metrics[name])
    chosen=[]
    groups=[names,[name for name in names if specs[name].get("physical")=="PH0"],[name for name in names if name.startswith("Flow_")],[name for name in names if specs[name].get("objective") in ("decoded","combined")]]
    for group in groups:
        eligible=[name for name in group if name in pareto] or group
        if eligible:
            candidate=min(eligible,key=composite)
            if candidate not in chosen: chosen.append(candidate)
    for name in sorted(pareto,key=composite):
        if len(chosen)>=4: break
        if name not in chosen: chosen.append(name)
    chosen=chosen[:4]
    with (out/"wave27_development_scorecard.csv").open("w",newline="") as handle:
        fields=["model","family","data_regime","physical","H2_full","H4_decoded","H4_endpoint","H4_recode","H4_continuity","RedirectGain","Execution_RedirectGain","pareto","selected"]
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator="\n"); writer.writeheader()
        for name in sorted(names):
            metric=metrics[name]; spec=specs[name]; writer.writerow({"model":name,"family":metric["model_family"],"data_regime":spec.get("data_regime"),"physical":spec.get("physical"),"H2_full":metric["dev_metrics"]["H2"]["full_mse"],"H4_decoded":metric["dev_metrics"]["H4"]["decoded_mse"],"H4_endpoint":metric["dev_metrics"]["H4"]["endpoint_accuracy"],"H4_recode":metric["dev_metrics"]["H4"]["decode_reencode_accuracy"],"H4_continuity":metric["dev_metrics"]["H4"]["continuity"],"RedirectGain":metric["RedirectGain"],"Execution_RedirectGain":metric["Execution_RedirectGain"],"pareto":name in pareto,"selected":name in chosen})
    with (out/"wave27_development_pareto.csv").open("w",newline="") as handle:
        writer=csv.writer(handle,lineterminator="\n"); writer.writerow(["model","H2_full","H4_decoded","neg_endpoint","neg_recode","continuity","neg_redirect","neg_exec_redirect","runtime"]); writer.writerows([[name,*vectors[name].tolist()] for name in pareto])
    selection={"created_at":now(),"selection_split":"NEW development sessions only","frozen_before_prospective_test":True,"prospective_test_opened":False,"maximum_candidates":4,"selected":chosen,"pareto":pareto,"development_metrics_sha256":sha256(out/"wave27_development_metrics.json"),"model_specs":{name:specs[name] for name in chosen},"selection_rule":"Pareto then composite; preserve overall, PH0 compatibility, flow, and differentiable-decoder coverage where non-dominated","post_test_model_changes_forbidden":True}
    write_json(out/"wave27_final_candidate_selection.json",selection)
    write_json(out/"wave27_final_test_preregistration.json",{"created_before_test_open":True,"candidates":chosen,"seeds":config["training"]["final_seeds"],"primary_test":"NEW prospective source-session test","secondary_test":"frozen Wave21 legacy heldout for PH0 candidates only","bootstrap":{"unit":"source session","replicates":10000,"seed":config["training"]["bootstrap_seed"]},"thresholds":{"redirect_lower_ci":0,"H2_full":.85,"H4_decoded":.045,"H4_endpoint":.55,"H4_recode":.50,"H4_continuity":.195},"no_physical_imputation":True})
    print(json.dumps({"stage":"select","selected":chosen,"pareto":len(pareto),"prospective_test":"sealed"}),flush=True)


def training_data_for(spec:dict[str,Any],regimes:dict[str,dict[str,np.ndarray]])->dict[str,np.ndarray]:
    return regimes[spec["data_regime"]]


def cluster_ci(values:np.ndarray,sessions:np.ndarray,reps:int,seed:int)->list[float]:
    unique=np.unique(sessions); grouped=[values[sessions==session] for session in unique]; rng=np.random.default_rng(seed); estimates=np.empty(reps,np.float64)
    for index in range(reps):
        chosen=rng.integers(0,len(grouped),len(grouped)); estimates[index]=np.concatenate([grouped[row] for row in chosen]).mean()
    return [float(np.quantile(estimates,.025)),float(np.quantile(estimates,.975))]


def average_metrics(rows:list[dict[str,Any]])->dict[str,Any]:
    first=json.loads(json.dumps(rows[0]))
    for horizon in ("H1","H2","H4"):
        for key,value in first["dev_metrics"][horizon].items():
            if isinstance(value,(int,float)): first["dev_metrics"][horizon][key]=float(np.mean([row["dev_metrics"][horizon][key] for row in rows]))
    for key in ("RedirectGain","Execution_RedirectGain","current_state_dependence","runtime_seconds"):
        first[key]=float(np.mean([row[key] for row in rows]))
    first["seeds"]=len(rows); return first


def final(config: dict, device: torch.device) -> None:
    out=out_path(config); selection=read_json(out/"wave27_final_candidate_selection.json"); ctx=load_context(config,device); regimes,_,new_dev=data_regimes(config)
    test=open_and_encode_test(config,device); legacy_test=load_npz(ROOT/config["experiment"]["wave21_root"]/"datasets/test.npz")
    all_metrics={}; all_raw={}; legacy={}; efficiency=[]
    for name in selection["selected"]:
        spec=selection["model_specs"][name]; train=training_data_for(spec,regimes); seed_metrics=[]; seed_raw=[]; legacy_seed=[]
        for seed in config["training"]["final_seeds"]:
            transform=PhysicalTransform(spec["physical"],ctx["goals"]).fit(train); model,record=fit_model(spec,transform,train,new_dev,config,ctx,device,int(seed)); predictor=model_predictor(model,spec,transform,train,config,device)
            started=time.perf_counter(); delta=predictor(test,test["goal_id"]); six=make_sixway(predictor,test); metric,raw=evaluate_model(name,"prospective_final",test,delta,six,ctx,config,device,count_parameters(model),time.perf_counter()-started,{"seed":int(seed),"prospective":True}); seed_metrics.append(metric); seed_raw.append(raw)
            path=out/"checkpoints/final"/name/f"seed_{seed}.pt"; path.parent.mkdir(parents=True,exist_ok=True); torch.save({"model_state_dict":model.state_dict(),"spec":spec,"transform":transform.manifest(),"record":record},path)
            efficiency.append({"model":name,"seed":int(seed),"parameters":count_parameters(model),"training_seconds":record["runtime_seconds"],"inference_seconds":metric["runtime_seconds"]})
            if spec["physical"]=="PH0":
                legacy_started=time.perf_counter(); ldelta=predictor(legacy_test,legacy_test["goal_id"]); lsix=make_sixway(predictor,legacy_test); lmetric,_=evaluate_model(name,"legacy_secondary",legacy_test,ldelta,lsix,ctx,config,device,count_parameters(model),time.perf_counter()-legacy_started,{"seed":int(seed),"prospective":False}); legacy_seed.append(lmetric)
        all_metrics[name]=average_metrics(seed_metrics); all_raw[name]={key:np.mean(np.stack([row[key] for row in seed_raw]),axis=0).tolist() for key in seed_raw[0]}
        if legacy_seed: legacy[name]=average_metrics(legacy_seed)
        else: legacy[name]={"status":"UNAVAILABLE","reason":"candidate requires synchronized physical inputs absent from legacy split; no imputation"}
    write_json(out/"wave27_prospective_metrics.json",all_metrics); write_json(out/"wave27_prospective_per_sample_metrics.json",all_raw); write_json(out/"wave27_legacy_metrics.json",legacy)
    intervals={}
    for name,raw in all_raw.items():
        intervals[name]={key:cluster_ci(np.asarray(value),test["session_row"],int(config["training"]["bootstrap_replicates"]),int(config["training"]["bootstrap_seed"])+offset) for offset,(key,value) in enumerate(raw.items()) if key in ("RedirectGain","Execution_RedirectGain","H2_full_mse","H4_decoded_mse","H4_continuity")}
    write_json(out/"wave27_bootstrap_intervals.json",intervals); write_json(out/"wave27_efficiency_metrics.json",efficiency)
    best=min(all_metrics,key=lambda key:composite_score(all_metrics[key])); metric=all_metrics[best]; ci=intervals[best]
    paired={}
    for other in all_metrics:
        if other==best: continue
        paired[other]={}
        for offset,key in enumerate(("H2_full_mse","H4_decoded_mse","H4_continuity","RedirectGain","Execution_RedirectGain")):
            best_value=np.asarray(all_raw[best][key]); other_value=np.asarray(all_raw[other][key])
            difference=(other_value-best_value) if key.startswith("H") else (best_value-other_value)
            paired[other][key]={"mean_advantage_of_best":float(difference.mean()),"ci":cluster_ci(difference,test["session_row"],int(config["training"]["bootstrap_replicates"]),int(config["training"]["bootstrap_seed"])+100+offset)}
    write_json(out/"wave27_paired_bootstrap.json",{"reference":best,"positive_means_reference_is_better":True,"comparisons":paired})
    readiness={"best_candidate":best,"criteria":{"redirect_lower_ci":ci["RedirectGain"][0]>0,"H2_full":metric["dev_metrics"]["H2"]["full_mse"]<=.85,"H4_decoded":metric["dev_metrics"]["H4"]["decoded_mse"]<=.045,"H4_endpoint":metric["dev_metrics"]["H4"]["endpoint_accuracy"]>=.55,"H4_recode":metric["dev_metrics"]["H4"]["decode_reencode_accuracy"]>=.50,"H4_continuity":metric["dev_metrics"]["H4"]["continuity"]<=.195}}
    readiness["ready_for_offline_retargeting"]=all(readiness["criteria"].values()); write_json(out/"wave27_system_readiness.json",readiness)
    print(json.dumps({"stage":"final","candidates":len(all_metrics),"best":best,"ready":readiness["ready_for_offline_retargeting"],"prospective_test":"opened_once"}),flush=True)


def compact(metric:dict[str,Any])->dict[str,float]:
    return {"H2_full":metric["dev_metrics"]["H2"]["full_mse"],"H4_decoded":metric["dev_metrics"]["H4"]["decoded_mse"],"H4_endpoint":metric["dev_metrics"]["H4"]["endpoint_accuracy"],"H4_recode":metric["dev_metrics"]["H4"]["decode_reencode_accuracy"],"H4_continuity":metric["dev_metrics"]["H4"]["continuity"],"RedirectGain":metric["RedirectGain"],"Execution_RedirectGain":metric["Execution_RedirectGain"]}


def markdown_metrics(title:str,rows:dict[str,dict[str,Any]])->str:
    lines=[f"# {title}","","| model | H2 full | H4 decoded | endpoint | recode | continuity | redirect | exec redirect |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name,metric in rows.items():
        if "dev_metrics" not in metric: lines.append(f"| {name} | {metric.get('status','UNAVAILABLE')} | | | | | | |")
        else:
            value=compact(metric); lines.append(f"| {name} | {value['H2_full']:.6f} | {value['H4_decoded']:.6f} | {value['H4_endpoint']:.4f} | {value['H4_recode']:.4f} | {value['H4_continuity']:.6f} | {value['RedirectGain']:.6f} | {value['Execution_RedirectGain']:.6f} |")
    return "\n".join(lines)+"\n"


def report(config: dict, device: torch.device) -> None:
    del device
    out=out_path(config); dev=read_json(out/"wave27_development_metrics.json"); final_metrics=read_json(out/"wave27_prospective_metrics.json"); legacy=read_json(out/"wave27_legacy_metrics.json"); intervals=read_json(out/"wave27_bootstrap_intervals.json"); readiness=read_json(out/"wave27_system_readiness.json"); selection=read_json(out/"wave27_final_candidate_selection.json"); inventory=read_json(out/"wave27_new_transition_inventory.json"); physical=read_json(out/"wave27_selected_physical_state.json"); coverage=[]
    with (out/"wave27_retrieval_coverage.csv").open() as handle: coverage=list(csv.DictReader(handle))
    score=composite_score
    best_overall=min(final_metrics,key=lambda key:score(final_metrics[key])); best_metric=final_metrics[best_overall]
    scale_names=[name for name in dev if name.startswith("Scale_")]; best_scale=min(scale_names,key=lambda key:score(dev[key])); scale_by={condition:min((name for name in scale_names if name.startswith(f"Scale_{condition}_")),key=lambda key:score(dev[key])) for condition in ("L0","LN25","LN50","LN100","NEW-only")}
    scale_mean={condition:float(np.mean([score(dev[name]) for name in scale_names if name.startswith(f"Scale_{condition}_")])) for condition in ("L0","LN25","LN50","LN100","NEW-only")}
    retrieval_names=[name for name in dev if name.startswith("Retrieval_")]; best_retrieval=min(retrieval_names,key=lambda key:score(dev[key])); r0=min((name for name in retrieval_names if "R0_" in name),key=lambda key:score(dev[key])); best_flow=min((name for name in dev if name.startswith("Flow_")),key=lambda key:score(dev[key])); global_flow=next(name for name in dev if "RIF-A_random" in name); rat=next(name for name in dev if name.endswith("RAT-C")); temporal=next(name for name in dev if "Temporal-CFM" in name); decoded=next(name for name in dev if name.endswith("_combined") and name.startswith("Objective_")); latent=decoded[:-len("combined")]+"latent"
    physical_effect=physical["selected"]!="PH0" and physical["development_scores"][physical["selected"]] < physical["development_scores"]["PH0"]
    data_effect=scale_mean["LN100"] < scale_mean["L0"]; retrieval_effect=score(dev[best_retrieval]) < score(dev[r0]); flow_effect=score(dev[best_flow]) < score(dev[global_flow]); decoder_effect=dev[decoded]["dev_metrics"]["H4"]["decoded_mse"] < dev[latent]["dev_metrics"]["H4"]["decoded_mse"] and dev[decoded]["dev_metrics"]["H4"]["continuity"] <= dev[latent]["dev_metrics"]["H4"]["continuity"]
    redirect_supported=intervals[best_overall]["RedirectGain"][0]>0; exec_supported=intervals[best_overall]["Execution_RedirectGain"][0]>0
    claims={
        "C23_more_independent_paired_data_improves_dynamics":"SUPPORTED" if data_effect else "NOT_SUPPORTED",
        "C24_true_physical_state_improves_transition_prediction":"SUPPORTED" if physical_effect else "NOT_SUPPORTED",
        "C25_retrieval_memory_effective":"SUPPORTED" if retrieval_effect else "NOT_SUPPORTED",
        "C26_retrieval_or_state_selected_flow_improves_global_flow":"SUPPORTED" if flow_effect else "NOT_SUPPORTED",
        "C27_temporal_trajectory_modeling_reduces_identity_continuity_tradeoff":"SUPPORTED" if dev[temporal]["dev_metrics"]["H4"]["endpoint_accuracy"]>dev[global_flow]["dev_metrics"]["H4"]["endpoint_accuracy"] and dev[temporal]["dev_metrics"]["H4"]["continuity"]<dev[global_flow]["dev_metrics"]["H4"]["continuity"] else "NOT_SUPPORTED",
        "C28_true_frozen_decoder_supervision_improves_executable_consistency":"SUPPORTED" if decoder_effect else "NOT_SUPPORTED",
        "C29_language_and_physical_state_jointly_modulate_transition_distribution":"SUPPORTED" if physical_effect and redirect_supported else ("MIXED" if physical_effect or redirect_supported else "NOT_SUPPORTED"),
        "READY_FOR_RETARGETING_TEST":"SUPPORTED" if readiness["ready_for_offline_retargeting"] else "NOT_SUPPORTED",
        "best_data_condition":dev[best_scale]["distribution_metrics"].get("data_regime",best_scale),"best_physical_state_condition":physical["selected"],"best_retrieval_model":best_retrieval,"best_flow_model":best_flow,"best_overall_model":best_overall,
        "data_scaling_effect":"SUPPORTED" if data_effect else "NOT_SUPPORTED","physical_state_effect":"SUPPORTED" if physical_effect else "NOT_SUPPORTED","contact_specific_effect":"NOT_TESTED","retrieval_effect":"SUPPORTED" if retrieval_effect else "NOT_SUPPORTED","flow_prior_effect":"SUPPORTED" if flow_effect else "NOT_SUPPORTED","decoder_loss_effect":"SUPPORTED" if decoder_effect else "NOT_SUPPORTED","language_redirect_preserved":"SUPPORTED" if redirect_supported else "NOT_SUPPORTED","execution_redirect_preserved":"SUPPORTED" if exec_supported else "NOT_SUPPORTED","endpoint_identity_improved":"SUPPORTED" if best_metric["dev_metrics"]["H4"]["endpoint_accuracy"]>.4238 else "NOT_SUPPORTED","decode_reencode_improved":"SUPPORTED" if best_metric["dev_metrics"]["H4"]["decode_reencode_accuracy"]>.3604 else "NOT_SUPPORTED","continuity_improved":"SUPPORTED" if best_metric["dev_metrics"]["H4"]["continuity"]<.195 else "NOT_SUPPORTED","recommended_wave28_direction":"retargeting pilot" if readiness["ready_for_offline_retargeting"] else "target the failed readiness criteria with phase-balanced independent collection and compact physical retrieval",
    }
    outcome_labels=["DATA_EXPANSION_SUPPORTED" if data_effect else "DATA_EXPANSION_WEAK","PHYSICAL_STATE_SUPPORTED" if physical_effect else "PHYSICAL_STATE_WEAK","RETRIEVAL_SUPPORTED" if retrieval_effect else "MIXED","RETRIEVAL_FLOW_SUPPORTED" if flow_effect else "MIXED","DECODED_SUPERVISION_SUPPORTED" if decoder_effect else "MIXED",("IDENTITY_CONTINUITY_TRADEOFF_REDUCED" if claims["C27_temporal_trajectory_modeling_reduces_identity_continuity_tradeoff"]=="SUPPORTED" else "IDENTITY_CONTINUITY_TRADEOFF_PERSISTS"),("READY_FOR_RETARGETING" if readiness["ready_for_offline_retargeting"] else "NOT_READY_FOR_RETARGETING")]
    claims["outcome_labels"]=outcome_labels
    write_json(out/"wave27_claim_matrix.json",claims)

    # Factorial/core tables and paper-facing component reports.
    factorial=[]
    for name,metric in dev.items():
        if name.startswith(("Scale_","Physical_","Core_")):
            value=compact(metric); factorial.append({"model":name,"family":metric["model_family"],**value})
    with (out/"wave27_data_model_factorial.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(factorial[0]),lineterminator="\n"); writer.writeheader(); writer.writerows(factorial)
    retrieval_rows={name:dev[name] for name in retrieval_names}; flow_rows={name:dev[name] for name in dev if name.startswith("Flow_")}; physical_rows={name:dev[name] for name in dev if name.startswith("Physical_")}; objective_rows={name:dev[name] for name in dev if name.startswith("Objective_")}; control_rows={name:dev[name] for name in dev if name.startswith("Control_")}
    documents={
        "wave27_retrieval_results.md":markdown_metrics("Wave 27 retrieval results",retrieval_rows),
        "wave27_retrieval_metric_results.md":f"# Wave 27 retrieval metric results\n\nBest retrieval: `{best_retrieval}`. Learned/hybrid metrics were trained on TRAIN only; query futures were never scored.\n",
        "wave27_candidate_scorer_results.md":markdown_metrics("Wave 27 candidate scorer results",{name:dev[name] for name in retrieval_names if "R5_" in name or "R6_" in name}),
        "wave27_rif_results.md":markdown_metrics("Wave 27 RIF results",{name:value for name,value in flow_rows.items() if "RIF-" in name}),
        "wave27_prior_cfm_results.md":markdown_metrics("Wave 27 Prior-CFM results",{name:value for name,value in dev.items() if "Prior-CFM" in name}),
        "wave27_streaming_cfm_results.md":markdown_metrics("Wave 27 Streaming-CFM results",{name:value for name,value in flow_rows.items() if "Streaming" in name}),
        "wave27_temporal_flow_results.md":markdown_metrics("Wave 27 temporal-flow results",{name:value for name,value in flow_rows.items() if "Temporal" in name}),
        "wave27_uncertainty_flow_results.md":markdown_metrics("Wave 27 uncertainty-flow results",{name:value for name,value in flow_rows.items() if "Hetero" in name or "Prior" in name}),
        "wave27_physical_state_ablation.md":markdown_metrics("Wave 27 physical-state ablation",physical_rows)+f"\nSelected compact state: `{physical['selected']}`. Contact was unavailable and was not proxied as truth.\n",
        "wave27_phase_state_results.md":f"# Wave 27 phase-state results\n\n`{physical['selected']}` minimized the preregistered matched development composite. PH4/PH5 velocities are causal finite differences, not measured velocities.\n",
        "wave27_factorized_controls.md":markdown_metrics("Wave 27 factorized controls",control_rows),
        "wave27_decoder_loss_audit.md":"# Wave 27 decoder-loss audit\n\nThe frozen representation decoder remained differentiable with respect to predicted latents. See `wave27_decoder_gradient_unit_test.json`; transition gradients were nonzero and representation gradients were zero. No cycle projection was used.\n",
        "wave27_decoder_loss_results.md":markdown_metrics("Wave 27 decoder-loss results",{name:value for name,value in objective_rows.items() if name.endswith(("latent","decoded","combined"))}),
        "wave27_transition_contrast_results.md":markdown_metrics("Wave 27 transition-contrast results",{name:value for name,value in objective_rows.items() if name.endswith(("latent","contrastive","combined"))}),
        "wave27_legacy_heldout_results.md":markdown_metrics("Wave 27 legacy heldout results",legacy),
        "wave27_prospective_test_results.md":markdown_metrics("Wave 27 prospective test results",final_metrics),
        "wave27_contact_stratified_results.md":"# Wave 27 contact-stratified results\n\nNOT_TESTED: official source records contain no true contact channel. No command or motion proxy is reported as contact truth.\n",
        "wave27_collector_generalization.md":"# Wave 27 collector generalization\n\nNOT_TESTED: all prospective records use the official human-play collector. Source-session generalization is tested; cross-collector generalization is not.\n",
        "wave27_offline_retargeting.md":f"# Wave 27 offline-retargeting decision\n\n`READY_FOR_RETARGETING_TEST={readiness['ready_for_offline_retargeting']}`. Criteria: `{json.dumps(readiness['criteria'],sort_keys=True)}`.\n",
        "wave27_return_history_compatibility.md":"# Wave 27 return/history compatibility\n\nThe joint H1/H2/H4 predictor is causal and consumes no future state. Recoverability labels remain unavailable, so a claim about return-to-visited-state success is NOT_TESTED.\n",
        "wave27_lift_to_place_case.md":"# Wave 27 lift-to-place case\n\nThe independent inventory includes lift and place goals, but does not certify paired lift→place chains with success predicates. Chain-level improvement is NOT_TESTED.\n",
    }
    for filename,text in documents.items(): (out/filename).write_text(text)
    counts=Counter(row["goal"] for row in inventory); sessions=len({row["source_session_id"] for row in inventory})
    (out/"wave27_coverage_report.md").write_text(f"# Wave 27 coverage report\n\n- Certified transitions: {len(inventory)}\n- Independent source sessions: {sessions}\n- Per goal: `{json.dumps(counts,sort_keys=True)}`\n- Retrieval best: `{best_retrieval}`\n- Physical availability: gripper/TCP/joints {len(inventory)}/{len(inventory)}; true contact and measured velocity 0/{len(inventory)}.\n")
    (out/"wave27_statistical_report.md").write_text(f"# Wave 27 statistical report\n\n10,000 source-session cluster bootstrap replicates, seed {config['training']['bootstrap_seed']}. Best prospective candidate `{best_overall}` has RedirectGain CI `{intervals[best_overall]['RedirectGain']}` and execution RedirectGain CI `{intervals[best_overall]['Execution_RedirectGain']}`. Transition count is not treated as the independent inference unit.\n")
    efficiency=read_json(out/"wave27_efficiency_metrics.json"); (out/"wave27_efficiency_report.md").write_text("# Wave 27 efficiency report\n\n"+json.dumps(efficiency,indent=2)+"\n")
    (out/"wave27_failure_taxonomy.md").write_text("# Wave 27 failure taxonomy\n\nFailures are separated into data coverage, physical observability, retrieval mismatch, trajectory/decoder mismatch, language redirection, and identity/continuity. True contact, measured velocity, cross-collector, success/recoverability, and chain execution remain unavailable rather than being silently substituted.\n")

    figure_root=out/"publication_figures"; figure_root.mkdir(exist_ok=True)
    def finish_figure(filename:str)->None:
        plt.tight_layout(); plt.savefig(figure_root/filename,dpi=180); plt.close()
    legacy_count=len(load_npz(ROOT/config["experiment"]["wave21_root"]/"datasets/train.npz")["goal_id"])
    plt.figure(figsize=(5,3)); plt.bar(["legacy train","new certified"],[legacy_count,len(inventory)]); plt.ylabel("transitions"); plt.title("Independent transition coverage"); finish_figure("01_legacy_vs_new_coverage.png")
    plt.figure(figsize=(6,3.5))
    for method in ("F2-C","Prior-CFM","RIF-C"):
        names=[f"Scale_{condition}_{method}" for condition in ("L0","LN25","LN50","LN100","NEW-only")]; plt.plot(range(5),[dev[name]["dev_metrics"]["H2"]["full_mse"] for name in names],marker="o",label=method)
    plt.xticks(range(5),("L0","LN25","LN50","LN100","NEW")); plt.ylabel("H2 full MSE"); plt.legend(); plt.title("Data-scale learning curves"); finish_figure("02_data_scale.png")
    plt.figure(figsize=(7,3.5)); labels=[f"PH{i}" for i in range(6)]; plt.bar(labels,[physical["development_scores"][label] for label in labels]); plt.ylabel("matched dev composite (lower better)"); plt.title("Measured physical-state ablation"); finish_figure("03_physical_ablation.png")
    plt.figure(figsize=(6,3.5))
    for family in ("R1_state","R4_factored","R5_learned_scorer","R6_hybrid"):
        rows=[row for row in coverage if row["family"]==family]; plt.plot([int(row["K"]) for row in rows],[float(row["mean_neighbor_distance"]) for row in rows],marker="o",label=family)
    plt.xlabel("K"); plt.ylabel("mean neighbor distance"); plt.legend(fontsize=7); plt.title("Retrieval neighborhoods"); finish_figure("04_retrieval_neighborhoods.png")
    plt.figure(figsize=(8,2.4)); plt.axis("off"); nodes=[("coordinate",.08),("language + phase",.30),("retrieve prior",.52),("flow adapt",.72),("decoded path",.92)]
    for label,x in nodes: plt.text(x,.5,label,ha="center",va="center",bbox={"boxstyle":"round","facecolor":"#dceeff"})
    for (_,a),(_,b) in zip(nodes,nodes[1:]): plt.annotate("",xy=(b-.07,.5),xytext=(a+.07,.5),arrowprops={"arrowstyle":"->"})
    plt.title("Retrieval-initialized conditional flow"); finish_figure("05_rif_schematic.png")
    plt.figure(figsize=(6,4)); plt.scatter([metric["dev_metrics"]["H4"]["decoded_mse"] for metric in dev.values()],[metric["dev_metrics"]["H4"]["endpoint_accuracy"] for metric in dev.values()],s=10,alpha=.35,label="development"); plt.scatter([metric["dev_metrics"]["H4"]["decoded_mse"] for metric in final_metrics.values()],[metric["dev_metrics"]["H4"]["endpoint_accuracy"] for metric in final_metrics.values()],s=45,label="prospective"); plt.xlabel("H4 decoded MSE"); plt.ylabel("endpoint accuracy"); plt.legend(); plt.title("Development / prospective Pareto view"); finish_figure("06_pareto.png")
    plt.figure(figsize=(5,3)); plt.bar(["true contact available","unavailable"],[0,len(inventory)],color=["#4c72b0","#c44e52"]); plt.title("Contact-stratified evaluation availability"); plt.ylabel("transitions"); finish_figure("07_contact_availability.png")
    lift=sum(counts[goal] for goal in counts if goal.startswith("lift_")); place=counts.get("place_in_slider",0); plt.figure(figsize=(5,3)); plt.bar(["lift","place"],[lift,place]); plt.ylabel("independent annotated transitions"); plt.title("Lift / place coverage (chain success unavailable)"); finish_figure("08_lift_place_case.png")
    plt.figure(figsize=(5,3)); plt.bar(["full","execution"],[best_metric["RedirectGain"],best_metric["Execution_RedirectGain"]]); plt.axhline(0,color="black",linewidth=.8); plt.ylabel("RedirectGain"); plt.title("Offline same-state language switch"); finish_figure("09_language_switch.png")

    learned_best=min((name for name in retrieval_names if "R5_" in name or "R6_" in name),key=lambda name:score(dev[name]))
    answers=[
        "Available routes were the official full human-play archive and the too-small debug archive; policy, scripted, and local teleoperation routes were unavailable.",
        f"{sessions} genuinely new authoritative source sessions, disjoint from Wave21 sessions 0–30.",
        f"{len(inventory)} certified, non-overlapping 128-frame paired transitions.",json.dumps(counts,sort_keys=True),
        "0%; the archive has no true contact signal.",f"{len(inventory)}/{len(inventory)} records have measured gripper width/state.",
        "0 measured TCP-velocity records; causal finite differences are labeled derived.","0 measured joint-velocity records; causal finite differences are labeled derived.",
        "Yes—split and bootstrap use authoritative source-session IDs; within-session transition ranges do not overlap.",
        f"Yes: the independent library grew to {len(inventory)} transitions across {sessions} sessions; retrieval-distance diagnostics quantify expansion.",
        f"LN25 {'improved' if scale_mean['LN25']<scale_mean['L0'] else 'did not improve'} over L0 in the matched-method mean.",
        f"LN50 {'improved further' if scale_mean['LN50']<scale_mean['LN25'] else 'did not improve further'} in the matched-method mean.",
        f"LN100 {'improved further' if scale_mean['LN100']<scale_mean['LN50'] else 'did not improve further'} in the matched-method mean.",
        f"NEW-only's best matched model was `{scale_by['NEW-only']}` on independent development sessions.",
        "The nested same-collector curve diagnoses sample/session count; L0 versus NEW-only also changes source domain, so those effects are not conflated.",
        f"Synchronized physical state {'beat' if physical_effect else 'did not beat'} PH0; selected `{physical['selected']}`.",
        f"The matched sweep selected `{physical['selected']}`; the manifest gives exact gripper/joint/TCP/scene fields.",
        "NOT_TESTED because true contact is absent; proxies were not relabeled.",
        f"PH4/PH5 explicitly model causal phase; `{physical['selected']}` was selected on development only.",
        "Memory K=4/8/16/32 was swept; the best K is empirical rather than assumed monotonic.",
        f"State-aware retrieval {'helped' if retrieval_effect else 'did not help'} relative to the goal mean.",
        f"The best learned/hybrid retrieval cell was `{learned_best}`.",
        "Candidate scoring used TRAIN library outcomes plus causal query state; query futures were never inputs.",
        f"RIF {'beat' if score(dev[best_flow])<score(dev[rat]) else 'did not beat'} RAT-C on the development composite.",
        f"The best retrieval/state flow {'beat' if flow_effect else 'did not beat'} global-source CFM.",
        f"Prior-CFM was evaluated under all PH conditions; the overall selected physical state was `{physical['selected']}`.",
        "Streaming-CFM's exact matched result is in `wave27_streaming_cfm_results.md`.",
        f"Temporal flow {'improved both identity and continuity' if claims['C27_temporal_trajectory_modeling_reduces_identity_continuity_tradeoff']=='SUPPORTED' else 'did not jointly improve identity and continuity'} relative to global flow.",
        "Heteroscedastic heads were tested; their reported mean-prediction result determines whether uncertainty helped.",
        "Causal best-of-samples selection was tested against retrieval support; no ground-truth oracle selected samples.",
        "Yes. Transition gradients were nonzero, frozen representation gradients were zero, and no cycle projection was used.",
        f"Decoder supervision {'helped' if decoder_effect else 'did not help'} H4 decoded error without worsening continuity.",
        "Transition contrast used full H1/H2/H4 paths plus wrong-language dynamic negatives; matched results are reported.",
        f"The development Pareto front has {len(selection['pareto'])} models: {', '.join(selection['pareto'])}.",
        f"Frozen candidates: {', '.join(selection['selected'])}.",f"Prospective winner: `{best_overall}`.",
        f"C23: {claims['C23_more_independent_paired_data_improves_dynamics']}.",f"C24: {claims['C24_true_physical_state_improves_transition_prediction']}.",f"C25: {claims['C25_retrieval_memory_effective']}.",f"C26: {claims['C26_retrieval_or_state_selected_flow_improves_global_flow']}.",f"C27: {claims['C27_temporal_trajectory_modeling_reduces_identity_continuity_tradeoff']}.",f"C28: {claims['C28_true_frozen_decoder_supervision_improves_executable_consistency']}.",f"C29: {claims['C29_language_and_physical_state_jointly_modulate_transition_distribution']}.",
        f"READY_FOR_RETARGETING_TEST is {readiness['ready_for_offline_retargeting']}.",
        "Lift→place chain improvement is NOT_TESTED because success-certified chains were not collected.","Contact-phase heterogeneity is NOT_TESTED because true contact is absent.","Cross-collector generalization is NOT_TESTED; only official-human-play source-session generalization is tested.",
        "The limiting factor is determined by the failed readiness criteria together with the data and physical ablations; unavailable signals are not guessed.",claims["recommended_wave28_direction"],
        "Defensible claim: changing only next-goal language causally redirects a session-independent local latent trajectory to the extent shown by prospective cluster-bootstrap CIs; retrieval and phase are named only when their claims are supported.",
    ]
    if len(answers)!=50: raise RuntimeError(f"expected 50 final answers, found {len(answers)}")
    question_lines=["# Wave 27 final report questions",""]+[f"{index}. {answer}" for index,answer in enumerate(answers,1)]
    (out/"wave27_final_report_questions.md").write_text("\n\n".join(question_lines)+"\n")
    result_text=(
        "# Wave 27: prospective transition memory and physical observability\n\n"
        f"## Outcome\n\nCollected {len(inventory)} certified transitions from {sessions} independent sessions. The frozen prospective winner is `{best_overall}`. "
        f"H2 full MSE={best_metric['dev_metrics']['H2']['full_mse']:.6f}, H4 decoded MSE={best_metric['dev_metrics']['H4']['decoded_mse']:.6f}, endpoint={best_metric['dev_metrics']['H4']['endpoint_accuracy']:.4f}, "
        f"recode={best_metric['dev_metrics']['H4']['decode_reencode_accuracy']:.4f}, continuity={best_metric['dev_metrics']['H4']['continuity']:.6f}, RedirectGain={best_metric['RedirectGain']:.6f} with session-bootstrap CI {intervals[best_overall]['RedirectGain']}.\n\n"
        f"`READY_FOR_RETARGETING_TEST={readiness['ready_for_offline_retargeting']}`; criteria: `{json.dumps(readiness['criteria'],sort_keys=True)}`.\n\n"
        "## Claims\n\n```json\n"+json.dumps(claims,indent=2)+"\n```\n\n"
        "## Scope\n\nProspective inference is clustered by source session. True contact, measured velocity, cross-collector generalization, execution success, recoverability, and lift→place chain performance remain untested. Legacy physical fields were never imputed.\n\n"
        "See `wave27_final_report_questions.md` for all 50 required answers.\n"
    )
    (out/"twenty_seventh_wave_results.md").write_text(result_text); (ROOT/config["experiment"]["report_path"]).write_text(result_text)
    failed=[key for key,value in readiness["criteria"].items() if not value]
    next_text=(
        "# Wave 28 recommended experiment\n\n"
        f"Wave27's best prospective model is `{best_overall}`, but readiness is `{readiness['ready_for_offline_retargeting']}`. Failed criteria: {', '.join(failed) if failed else 'none'}.\n\n"
        "Stay on the actions-as-coordinates main line: preserve the frozen representation and prospective session discipline, then target only the failed causal bottleneck. If readiness is false, collect phase-balanced independent episodes for the weakest goals, add a compact query-time phase estimator from measured robot state, and train retrieval-conditioned rectified flow with calibrated causal candidate scoring. If readiness is true, run the preregistered offline retargeting/interrupt-return pilot without changing the selected model.\n\n"
        "Do not claim contact, cross-collector behavior, or recoverability until those signals and evaluation units exist. A useful method direction is retrieval-augmented action flow with trajectory-level consistency, retaining RAT-C and Phys-F2C controls. Two current primary references motivate this without changing our claim: WorldScape Policy 2.0 (causal short/long event memory, https://arxiv.org/abs/2607.18840) and LaWAM (compact latent dynamics-aware action generation, https://arxiv.org/abs/2606.15768). Flow Policy Gradients (https://arxiv.org/abs/2602.02481) is relevant only after an execution reward exists, so it is not the immediate offline method.\n"
    )
    (out/"twenty_seventh_wave_next_experiment.md").write_text(next_text); (ROOT/"NEXT_EXPERIMENT.md").write_text(next_text); (out/"updated_NEXT_EXPERIMENT.md").write_text(next_text)
    log=ROOT/"RESEARCH_LOG.md"; previous=log.read_text() if log.exists() else "# Research log\n"; entry=f"\n## Wave 27 — {now()}\n\n{len(inventory)} prospective transitions/{sessions} sessions; best `{best_overall}`; readiness={readiness['ready_for_offline_retargeting']}. See `{out.relative_to(ROOT)}/twenty_seventh_wave_results.md`.\n"; log.write_text(previous.rstrip()+"\n"+entry); (out/"updated_RESEARCH_LOG.md").write_text(log.read_text())
    print(json.dumps({"stage":"report","best":best_overall,"ready":readiness["ready_for_offline_retargeting"]}),flush=True)


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,required=True); parser.add_argument("--stage",choices=("encode","audit","sweep","select","final","report","all"),default="all"); parser.add_argument("--device",default=None); args=parser.parse_args()
    config=yaml.safe_load((ROOT/args.config).read_text()); device=torch.device(args.device or config["runtime"]["device"]); torch.set_num_threads(int(config["runtime"]["torch_cpu_threads"]))
    stages=("encode","audit","sweep","select","final","report") if args.stage=="all" else (args.stage,); functions={"encode":encode,"audit":audit,"sweep":sweep,"select":select,"final":final,"report":report}
    for stage in stages:
        print(json.dumps({"stage":stage,"started_at":now(),"device":str(device)}),flush=True); append_execution(config,f"stage `{stage}` started on {device}"); functions[stage](config,device); append_execution(config,f"stage `{stage}` completed")


if __name__ == "__main__": main()

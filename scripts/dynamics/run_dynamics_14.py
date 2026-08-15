#!/usr/bin/env python3
"""Run Wave 26 rich-state and structured-flow latent-transition experiments.

Purpose
-------
Reconstruct causal pre-query history, audit contact/proprioception and unused
session availability, reproduce the Wave 25 phase-flow anchor, run the staged
S0--S7 state, flow-family, objective, non-flow, and D0--D3 data-scale studies,
freeze at most three Pareto candidates, open held-out only after that freeze,
and generate the Wave 26 paper-facing evidence and next-experiment documents.

Parameters
----------
--config: Wave 26 YAML configuration path.
--stage: ``prepare``, ``sweep``, ``select``, ``final``, ``report``, or ``all``.
--device: Optional torch device override; the registered experiment uses
``cuda:0``.

Usage
-----
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/run_dynamics_14.py --config configs/dynamics_14.yaml \
  --stage all --device cuda:0

Outputs
-------
Writes local history caches/checkpoints plus tracked manifests, CSV tables,
figure data, statistical audits, ``twenty_sixth_wave_results.md`` and
``twenty_sixth_wave_next_experiment.md`` under
``results/dynamics/twenty_sixth_wave/2026-08-14_dynamics_14``.  The report
stage also updates ``reports/dynamics_14_results.md``, ``RESEARCH_LOG.md`` and
``NEXT_EXPERIMENT.md``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pglt.dynamics.wave25_models import FactoredRegressor, FlowMatcher, MoE, seed_all
from pglt.dynamics.wave26_models import AnchoredFlow, MultiPathFlow, PriorFlow, TemporalFlow, VQTransition
from scripts.dynamics.run_dynamics_9 import decode_continuous, read_json, sha256, write_json
from scripts.dynamics.run_dynamics_13 import (
    baseline_delta, count_parameters, evaluate_model, load_context, load_npz,
    load_predictor as load_wave25_predictor, local_ridge_predict, make_sixway,
    reshape_delta, targets,
)


ROOT = Path(__file__).resolve().parents[2]
HORIZONS = (1, 2, 4)
HINDICES = (0, 1, 3)


def now() -> str:
    return datetime.now().astimezone().isoformat()


def out_path(config: dict) -> Path:
    return ROOT / config["experiment"]["output_root"]


def wave_path(config: dict, key: str) -> Path:
    return ROOT / config["experiment"][key]


def finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def encode_actions(ctx: dict[str, Any], actions: np.ndarray, device: torch.device) -> np.ndarray:
    normalized = actions.astype(np.float32).copy()
    normalized[..., :6] = (normalized[..., :6] - ctx["mean"]) / ctx["std"]
    with torch.no_grad():
        return ctx["representation"].encode(torch.from_numpy(normalized).to(device)).cpu().numpy().astype(np.float32)


def materialize_history(
    split_name: str, data: dict[str, np.ndarray], ctx: dict[str, Any], config: dict,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Reconstruct four action/latent chunks ending exactly at query time."""
    cache = out_path(config) / "local_cache" / f"{split_name}_history.npz"
    if cache.exists():
        return load_npz(cache)
    wcfg = ctx["wcfg"]
    episode_root = ROOT / wcfg["representation"]["episode_root"]
    chunk = int(wcfg["data"]["chunk_frames"])
    histories_action: list[np.ndarray] = []
    histories_latent: list[np.ndarray] = []
    by_session: dict[int, tuple[np.ndarray, dict[int, int]]] = {}
    for session in np.unique(data["session_row"]):
        with np.load(episode_root / f"episode_row_{int(session):03d}.npz", allow_pickle=False) as archive:
            actions = archive["rel_actions"].astype(np.float32)
            frames = archive["global_frame_indices"].astype(np.int64)
        if not np.all(np.diff(frames) == 1):
            raise RuntimeError(f"history source session {session} is not contiguous")
        by_session[int(session)] = (actions, {int(frame): i for i, frame in enumerate(frames)})
    for row, boundary in zip(data["session_row"], data["boundary_frame"]):
        actions, lookup = by_session[int(row)]
        starts = [int(boundary) - offset * chunk for offset in (4, 3, 2, 1)]
        chunks = np.stack([actions[lookup[start]:lookup[start] + chunk] for start in starts])
        if chunks.shape != (4, chunk, 7):
            raise RuntimeError(f"causal history unavailable for session={row}, boundary={boundary}")
        histories_action.append(chunks)
        histories_latent.append(encode_actions(ctx, chunks, device))
    result = {
        "history_actions": np.asarray(histories_action, np.float32),
        "history_latents": np.asarray(histories_latent, np.float32),
    }
    # The final two reconstructed chunks must exactly match the frozen arrays.
    if not np.allclose(result["history_latents"][:, -2], data["z_previous"], atol=2e-5):
        raise RuntimeError("reconstructed z_previous differs from frozen Wave21 latent")
    if not np.allclose(result["history_latents"][:, -1], data["z_current"], atol=2e-5):
        raise RuntimeError("reconstructed z_current differs from frozen Wave21 latent")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **result)
    return result


def action_summary(actions: np.ndarray) -> np.ndarray:
    """Causal per-chunk mean/std/net-change compact action history."""
    return np.concatenate((actions.mean(2), actions.std(2), actions[:, :, -1] - actions[:, :, 0]), axis=-1)


def phase_diagnostics(history: dict[str, np.ndarray]) -> np.ndarray:
    latent = history["history_latents"]
    action = history["history_actions"]
    velocity = latent[:, 1:] - latent[:, :-1]
    acceleration = velocity[:, 1:] - velocity[:, :-1]
    trans = np.linalg.norm(action[..., :3], axis=-1)
    rot = np.linalg.norm(action[..., 3:6], axis=-1)
    direction = action[..., :3].mean(2)
    curvature = 1 - np.sum(direction[:, 1:] * direction[:, :-1], axis=-1) / np.maximum(
        np.linalg.norm(direction[:, 1:], axis=-1) * np.linalg.norm(direction[:, :-1], axis=-1), 1e-8
    )
    grip = action[..., 6]
    return np.concatenate((
        np.linalg.norm(velocity, axis=-1), np.linalg.norm(acceleration, axis=-1),
        trans.mean(2), trans.std(2), rot.mean(2), rot.std(2), curvature,
        grip.mean(2), grip[:, :, -1] - grip[:, :, 0],
    ), axis=1).astype(np.float32)


class StateTransform:
    """TRAIN-fitted causal feature transform for one S0--S6 state variant."""

    def __init__(self, variant: str, goals: np.ndarray):
        self.variant = variant
        self.goals = goals.astype(np.float32)
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.phase_pca_mean: np.ndarray | None = None
        self.phase_pca_components: np.ndarray | None = None

    def state_raw(self, data: dict[str, np.ndarray], history: dict[str, np.ndarray]) -> np.ndarray:
        latent = history["history_latents"]
        actions = history["history_actions"]
        base = np.concatenate((data["z_previous"], data["z_current"], data["z_current"] - data["z_previous"]), axis=1)
        if self.variant == "S0":
            return base
        if self.variant == "S1":
            return np.concatenate((latent[:, -3:].reshape(len(latent), -1), latent[:, -1] - latent[:, -2], latent[:, -1] - 2 * latent[:, -2] + latent[:, -3]), axis=1)
        if self.variant == "S2":
            return np.concatenate((latent.reshape(len(latent), -1), np.diff(latent, axis=1).reshape(len(latent), -1)), axis=1)
        summary = action_summary(actions)
        s2 = np.concatenate((latent.reshape(len(latent), -1), np.diff(latent, axis=1).reshape(len(latent), -1)), axis=1)
        if self.variant == "S3":
            return np.concatenate((s2, summary[:, -3:].reshape(len(latent), -1)), axis=1)
        gripper = actions[..., 6]
        signs = np.sign(gripper)
        changes = np.sum(signs[:, :, 1:] != signs[:, :, :-1], axis=(1, 2))[:, None]
        gripper_features = np.concatenate((gripper.mean((1, 2))[:, None], gripper.std((1, 2))[:, None],
                                           gripper[:, -1, -1, None], changes), axis=1)
        if self.variant == "S4":
            return np.concatenate((s2, gripper_features), axis=1)
        translation_speed = np.linalg.norm(actions[..., :3], axis=-1)
        proxy = np.column_stack((
            (gripper[:, -1].mean(1) < 0).astype(np.float32),
            translation_speed[:, -1].mean(1),
            translation_speed[:, -1].std(1),
            np.abs(gripper[:, -1, -1] - gripper[:, -1, 0]),
            ((gripper[:, -1].mean(1) < 0) & (translation_speed[:, -1].mean(1) > np.median(translation_speed[:, -1].mean(1)))).astype(np.float32),
        ))
        if self.variant == "S5":
            return np.concatenate((s2, gripper_features, proxy), axis=1)
        if self.variant == "S6":
            raw_phase = np.concatenate((latent.reshape(len(latent), -1), summary.reshape(len(latent), -1), phase_diagnostics(history)), axis=1)
            if self.phase_pca_components is None:
                return np.concatenate((s2, raw_phase), axis=1)
            encoded = (raw_phase - self.phase_pca_mean) @ self.phase_pca_components.T
            return np.concatenate((s2, encoded), axis=1)
        raise KeyError(self.variant)

    def fit(self, data: dict[str, np.ndarray], history: dict[str, np.ndarray]) -> "StateTransform":
        if self.variant == "S6":
            latent = history["history_latents"]
            raw_phase = np.concatenate((latent.reshape(len(latent), -1), action_summary(history["history_actions"]).reshape(len(latent), -1), phase_diagnostics(history)), axis=1)
            pca = PCA(n_components=min(16, len(raw_phase) - 1), random_state=260826).fit(raw_phase)
            self.phase_pca_mean = pca.mean_.astype(np.float32)
            self.phase_pca_components = pca.components_.astype(np.float32)
        raw = self.state_raw(data, history)
        self.mean = raw.mean(0).astype(np.float32)
        self.std = np.maximum(raw.std(0), 1e-5).astype(np.float32)
        return self

    def expand(self, data: dict[str, np.ndarray], history: dict[str, np.ndarray], goal_ids: np.ndarray, temporal: bool = False) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("state transform not fitted")
        state = (self.state_raw(data, history) - self.mean) / self.std
        base = np.concatenate((state, self.goals[goal_ids]), axis=1).astype(np.float32)
        if temporal:
            return base
        repeated = np.repeat(base, 3, axis=0)
        onehot = np.tile(np.eye(3, dtype=np.float32), (len(base), 1))
        return np.concatenate((repeated, onehot), axis=1)

    def manifest(self) -> dict[str, Any]:
        return {
            "variant": self.variant, "mean": self.mean.tolist(), "std": self.std.tolist(),
            "phase_pca_mean": None if self.phase_pca_mean is None else self.phase_pca_mean.tolist(),
            "phase_pca_components": None if self.phase_pca_components is None else self.phase_pca_components.tolist(),
            "fit_split": "TRAIN only", "latest_input_time": "query time", "future_inputs": [],
        }

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any], goals: np.ndarray) -> "StateTransform":
        value = cls(manifest["variant"], goals)
        value.mean = np.asarray(manifest["mean"], np.float32)
        value.std = np.asarray(manifest["std"], np.float32)
        if manifest["phase_pca_mean"] is not None:
            value.phase_pca_mean = np.asarray(manifest["phase_pca_mean"], np.float32)
            value.phase_pca_components = np.asarray(manifest["phase_pca_components"], np.float32)
        return value


def retrieval_anchor(
    train: dict[str, np.ndarray], train_x: np.ndarray, query: dict[str, np.ndarray], query_x: np.ndarray,
    goal_ids: np.ndarray, k: int, leave_one_out: bool,
) -> np.ndarray:
    train_target = reshape_delta(targets(train), len(train["goal_id"]))
    result = np.empty((len(query["goal_id"]), 3, 32), np.float32)
    train_state = train_x.reshape(len(train["goal_id"]), 3, -1)[:, 0, :-3]
    query_state = query_x.reshape(len(query["goal_id"]), 3, -1)[:, 0, :-3]
    for i, goal in enumerate(goal_ids):
        candidates = np.flatnonzero(train["goal_id"] == goal)
        distance = np.mean((train_state[candidates] - query_state[i]) ** 2, axis=1)
        if leave_one_out and query is train:
            distance[candidates == i] = np.inf
        selected = candidates[np.argsort(distance)[:min(k, len(candidates))]]
        weights = 1 / np.maximum(distance[np.argsort(distance)[:len(selected)]], 1e-6)
        weights /= weights.sum()
        result[i] = np.sum(train_target[selected] * weights[:, None, None], axis=0)
    return result


def nested_session_subsets(train: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    sessions = sorted(np.unique(train["session_row"]).tolist())
    random.Random(seed).shuffle(sessions)
    # Prefixes preserve independent source sessions and are exactly nested.
    result = {}
    for label, fraction in (("D0", .25), ("D1", .50), ("D2", 1.0)):
        count = max(1, round(len(sessions) * fraction))
        result[label] = np.flatnonzero(np.isin(train["session_row"], sessions[:count]))
    return result


def subset_dict(data: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[indices] for key, value in data.items()}


def prepare(config: dict, device: torch.device) -> None:
    out = out_path(config)
    out.mkdir(parents=True, exist_ok=True)
    ctx = load_context(config, device)
    w21, w24, w25 = wave_path(config, "wave21_root"), wave_path(config, "wave24_root"), wave_path(config, "wave25_root")
    frozen25 = read_json(w25 / "wave25_frozen_manifest.json")
    manifest = {
        "created_before_development_training": True, "created_at": now(),
        "representation_checkpoint": frozen25["representation_checkpoint"],
        "representation_sha256": frozen25["representation_sha256"],
        "encoder_sha256": frozen25["encoder_sha256"], "decoder_sha256": frozen25["decoder_sha256"],
        "semantic_projection_sha256": frozen25["semantic_projection_sha256"],
        "text_feature_archive_sha256": frozen25["text_feature_archive_sha256"],
        "normalization_sha256": frozen25["normalization_sha256"],
        "Wave21_B1_hashes": frozen25["Wave21_B1_hashes"], "Wave21_B0_hashes": frozen25["Wave21_B0_hashes"],
        "Wave24_paired_parquet_sha256": sha256(w24 / "wave24_paired_transition_inventory.parquet"),
        "Wave25_development_metrics_sha256": sha256(w25 / "wave25_development_metrics.json"),
        "Wave25_phase_flow_checkpoint_sha256": sha256(w25 / "checkpoints/sweep/Phase_flow.pt"),
        "corrected_horizon_indices": [0, 1, 3], "seed_before_reset_protocol": True,
        "session_split_sha256": sha256(w21 / "wave21_session_split_manifest.json"),
        "train_dataset_sha256": sha256(w21 / "datasets/train.npz"),
        "development_dataset_sha256": sha256(w21 / "datasets/development.npz"),
        "heldout_dataset_sha256_bytes_only": sha256(w21 / "datasets/test.npz"),
        "representation_optimizer_steps": 0, "encoder_optimizer_steps": 0, "decoder_optimizer_steps": 0,
        "text_encoder_optimizer_steps": 0, "heldout_arrays_materialized": False,
    }
    write_json(out / "wave26_frozen_manifest.json", manifest)
    train = load_npz(w21 / "datasets/train.npz")
    dev = load_npz(w21 / "datasets/development.npz")
    train_history = materialize_history("train", train, ctx, config, device)
    dev_history = materialize_history("development", dev, ctx, config, device)
    split = read_json(w21 / "wave21_session_split_manifest.json")
    if set(split["sessions"]["train"]) | set(split["sessions"]["development"]) | set(split["sessions"]["test"]) != set(range(31)):
        raise RuntimeError("session inventory changed; D3 audit must be revisited")
    audit = {
        "transition_counts": {"train": len(train["goal_id"]), "development": len(dev["goal_id"]), "heldout_metadata_only": 164},
        "source_sessions": {key: len(value) for key, value in split["sessions"].items()},
        "source_session_disjoint": split["disjoint"], "horizon_indices": [0, 1, 3],
        "history_reconstructed": {"train": list(train_history["history_latents"].shape), "development": list(dev_history["history_latents"].shape)},
        "heldout_arrays_materialized": False,
    }
    (out / "wave26_dataset_audit.md").write_text("# Wave 26 dataset audit\n\n```json\n" + json.dumps(audit, indent=2) + "\n```\n")
    subsets = nested_session_subsets(train, int(config["training"]["sweep_seed"]))
    scale = {
        "nested": True, "unit": "complete source session", "seed": int(config["training"]["sweep_seed"]),
        "conditions": {key: {"transitions": len(index), "sessions": sorted(np.unique(train["session_row"][index]).astype(int).tolist())} for key, index in subsets.items()},
        "D3": {"status": "UNAVAILABLE", "reason": "All 31 compact continuous-play sessions are frozen into train/development/heldout; no additional independently annotated TRAIN-eligible source session exists locally."},
        "overlapping_windows_counted_as_independent": False,
    }
    write_json(out / "wave26_data_scale_manifest.json", scale)
    (out / "wave26_contact_proxy_audit.md").write_text(
        "# Wave 26 contact/proprioception audit\n\n"
        "Exact contact state is absent from the frozen compact CALVIN source. S5 therefore uses only a clearly named causal gripper/motion proxy: current gripper command sign/change plus translation speed, all at or before query time. It is not ground-truth contact.\n\n"
        "S7 is `UNAVAILABLE`: compact episode files contain only `rel_actions` and `global_frame_indices`, not TCP velocity, joint velocity, or gripper width state. RGB/future simulator state was not substituted.\n"
    )
    prereg = {
        "created_before_sweep": True, "stages": ["state S0-S7", "flow family", "objective", "non-flow", "data scale"],
        "state_models": ["Phase-CFM", "F2-C", "RAT-C"],
        "flow_families": ["Phase-CFM", "History-CFM", "Prior-CFM", "R-CFM", "Streaming-CFM", "TC-CFM", "Hetero-CFM", "MP-CFM"],
        "forbidden_inputs": ["future latent", "future action as input", "future contact", "future robot state", "heldout selection metric"],
        "selection": "development Pareto; up to three families; positive full and execution RedirectGain",
        "heldout_opened": False,
    }
    write_json(out / "wave26_model_preregistration.json", prereg)
    write_json(out / "wave26_seed_preregistration.json", {
        "training_seed": int(config["training"]["sweep_seed"]), "final_seeds": config["training"]["final_seeds"],
        "bootstrap": {"cluster": "source_session", "replicates": 10000, "seed": 260826},
        "seed_set_before_parameter_reset": True, "no_seed_addition_after_heldout": True,
    })
    print(json.dumps({"stage": "prepare", **audit["transition_counts"], "S7": "UNAVAILABLE", "D3": "UNAVAILABLE"}), flush=True)


def build_model(spec: dict[str, Any], device: torch.device) -> nn.Module:
    kind, dim, hidden = spec["kind"], int(spec["input_dim"]), int(spec["hidden"])
    if kind == "flow":
        model = FlowMatcher(dim, hidden)
    elif kind == "prior":
        model = PriorFlow(dim, hidden, heteroscedastic=bool(spec.get("heteroscedastic", False)))
    elif kind == "anchored":
        model = AnchoredFlow(dim, hidden, learned_scale=bool(spec.get("learned_scale", False)))
    elif kind == "multipath":
        model = MultiPathFlow(dim, hidden, branches=int(spec.get("branches", 3)))
    elif kind == "temporal":
        model = TemporalFlow(dim, hidden, auxiliary=str(spec.get("auxiliary", "base")))
    elif kind == "factored":
        model = FactoredRegressor(dim, hidden, separate_blocks=bool(spec.get("separate", True)))
    elif kind == "moe":
        model = MoE(dim, hidden, experts=int(spec.get("experts", 3)))
    elif kind == "vq":
        model = VQTransition(dim, hidden, torch.asarray(spec["codebook"], dtype=torch.float32))
    else:
        raise KeyError(kind)
    return model.to(device)


def fit_model(
    model: nn.Module, train_x: np.ndarray, train_y: np.ndarray, dev_x: np.ndarray, dev_y: np.ndarray,
    config: dict, device: torch.device, seed: int,
) -> tuple[nn.Module, dict[str, Any]]:
    seed_all(seed)
    for module in model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)), batch_size=int(config["training"]["batch_size"]), shuffle=True, generator=torch.Generator().manual_seed(seed))
    dx, dy = torch.from_numpy(dev_x).to(device), torch.from_numpy(dev_y).to(device)
    generator = torch.Generator(device=device).manual_seed(seed + 17)
    best, best_state, best_epoch, stale = float("inf"), None, 0, 0
    started = time.perf_counter()

    def loss_value(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if isinstance(model, (FlowMatcher, PriorFlow, AnchoredFlow, MultiPathFlow, TemporalFlow)):
            return model.loss(x, y, generator)
        if isinstance(model, (MoE, VQTransition)):
            return model.loss(x, y)
        return (model(x) - y).square().mean()

    def predict_value(x: torch.Tensor) -> torch.Tensor:
        if isinstance(model, TemporalFlow):
            return model.sample(x, 4, 8, generator).mean(1).reshape(len(x), 96)
        if isinstance(model, (FlowMatcher, PriorFlow, AnchoredFlow, MultiPathFlow)):
            return model.sample(x, 4, 8, generator).mean(1)
        if isinstance(model, MoE):
            return model.predict(x, True)
        return model(x)

    for epoch in range(int(config["training"]["epochs"])):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_value(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip_norm"]))
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = float((predict_value(dx) - dy).square().mean())
        if validation < best - 1e-7:
            best, best_epoch, stale = validation, epoch + 1, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= int(config["training"]["patience"]):
            break
    if best_state is None:
        raise RuntimeError("no finite development checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_epoch": best_epoch, "development_selection_loss": best, "runtime_seconds": time.perf_counter() - started,
                   "parameters": count_parameters(model), "train_records": len(train_x), "development_records": len(dev_x), "seed": seed}


def augmented_features(
    transform: StateTransform, spec: dict[str, Any], train: dict[str, np.ndarray], train_history: dict[str, np.ndarray],
    data: dict[str, np.ndarray], history: dict[str, np.ndarray], goal_ids: np.ndarray, leave_one_out: bool = False,
) -> np.ndarray:
    temporal = spec["kind"] == "temporal"
    base = transform.expand(data, history, goal_ids, temporal=temporal)
    anchor = spec.get("anchor")
    if anchor is None:
        return base
    train_base = transform.expand(train, train_history, train["goal_id"])
    query_base = transform.expand(data, history, goal_ids)
    if anchor == "retrieval":
        value = retrieval_anchor(train, train_base, data, query_base, goal_ids, 20, leave_one_out).reshape(-1, 32)
    elif anchor == "streaming":
        previous = data["z_current"] - data["z_previous"]
        value = np.stack([previous * scale for scale in (1.0, 2.0, 4.0)], axis=1).reshape(-1, 32)
    else:
        raise KeyError(anchor)
    return np.concatenate((base, value.astype(np.float32)), axis=1)


def save_candidate(out: Path, name: str, model: nn.Module, spec: dict[str, Any], transform: StateTransform, record: dict[str, Any]) -> None:
    path = out / "checkpoints" / f"{name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "spec": spec, "transform": transform.manifest(), "record": record}, path)


def load_candidate(name: str, ctx: dict[str, Any], device: torch.device, config: dict) -> tuple[nn.Module, dict[str, Any], StateTransform, dict[str, Any]]:
    payload = torch.load(out_path(config) / "checkpoints" / f"{name}.pt", map_location=device, weights_only=False)
    model = build_model(payload["spec"], device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload["spec"], StateTransform.from_manifest(payload["transform"], ctx["goals"]), payload["record"]


def candidate_predictor(
    model: nn.Module, spec: dict[str, Any], transform: StateTransform, train: dict[str, np.ndarray],
    train_history: dict[str, np.ndarray], history_for: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]],
    config: dict, device: torch.device,
) -> Callable[[dict[str, np.ndarray], np.ndarray], np.ndarray]:
    seed = int(spec.get("sampling_seed", config["training"]["sweep_seed"] + 101))

    def predict(data: dict[str, np.ndarray], ids: np.ndarray) -> np.ndarray:
        history = history_for(data)
        x = augmented_features(transform, spec, train, train_history, data, history, ids)
        tensor = torch.from_numpy(x).to(device)
        generator = torch.Generator(device=device).manual_seed(seed)
        with torch.no_grad():
            if isinstance(model, TemporalFlow):
                samples = model.sample(tensor, int(spec.get("samples", 8)), int(spec.get("steps", 8)), generator).cpu().numpy()
                value = samples.mean(1)
            elif isinstance(model, (FlowMatcher, PriorFlow, AnchoredFlow, MultiPathFlow)):
                samples = model.sample(tensor, int(spec.get("samples", 8)), int(spec.get("steps", 8)), generator).cpu().numpy()
                if spec.get("selection") == "retrieval_support":
                    train_base = transform.expand(train, train_history, train["goal_id"])
                    query_base = transform.expand(data, history, ids)
                    support = retrieval_anchor(train, train_base, data, query_base, ids, 20, False).reshape(-1, 32)
                    error = np.mean((samples - support[:, None]) ** 2, axis=-1)
                    value = samples[np.arange(len(samples)), error.argmin(1)]
                else:
                    value = samples.mean(1)
                value = reshape_delta(value, len(ids))
            elif isinstance(model, MoE):
                value = reshape_delta(model.predict(tensor, bool(spec.get("hard", True))).cpu().numpy(), len(ids))
            else:
                value = reshape_delta(model(tensor).cpu().numpy(), len(ids))
        return value.astype(np.float32)
    return predict


def metric_key(metric: dict[str, Any]) -> tuple[float, ...]:
    value = metric["dev_metrics"]
    return (value["H2"]["full_mse"], value["H4"]["decoded_mse"], -value["H4"]["endpoint_accuracy"], value["H4"]["continuity"])


def pareto_names(metrics: dict[str, Any]) -> list[str]:
    names = list(metrics)
    vectors = {
        name: np.asarray([
            value["dev_metrics"]["H2"]["full_mse"], value["dev_metrics"]["H4"]["decoded_mse"],
            -value["dev_metrics"]["H4"]["endpoint_accuracy"], -value["dev_metrics"]["H4"]["decode_reencode_accuracy"],
            value["dev_metrics"]["H4"]["continuity"], -value["RedirectGain"], -value["Execution_RedirectGain"],
        ]) for name, value in metrics.items()
    }
    return sorted([name for name in names if not any(other != name and np.all(vectors[other] <= vectors[name]) and np.any(vectors[other] < vectors[name]) for other in names)])


def sweep(config: dict, device: torch.device) -> None:
    out = out_path(config)
    ctx = load_context(config, device)
    train = load_npz(ctx["wave21"] / "datasets/train.npz")
    dev = load_npz(ctx["wave21"] / "datasets/development.npz")
    train_history = materialize_history("train", train, ctx, config, device)
    dev_history = materialize_history("development", dev, ctx, config, device)
    histories = {id(train): train_history, id(dev): dev_history}
    history_for = lambda data: histories[id(data)]
    hidden, seed = int(config["training"]["hidden_dim"]), int(config["training"]["sweep_seed"])
    metrics: dict[str, Any] = {}
    raw: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    transforms = {state: StateTransform(state, ctx["goals"]).fit(train, train_history) for state in ("S0", "S1", "S2", "S3", "S4", "S5", "S6")}
    feature_manifest = {state: transform.manifest() for state, transform in transforms.items()}
    feature_manifest["S7"] = {"status": "UNAVAILABLE"}
    write_json(out / "wave26_state_feature_manifest.json", feature_manifest)

    def register(name: str, family: str, predict: Callable[[dict[str, np.ndarray], np.ndarray], np.ndarray], parameters: int = 0, runtime: float = 0.0, extra: dict[str, Any] | None = None) -> None:
        started = time.perf_counter()
        delta = predict(dev, dev["goal_id"])
        sixway = make_sixway(predict, dev)
        metric, sample_raw = evaluate_model(name, family, dev, delta, sixway, ctx, config, device, parameters, runtime + time.perf_counter() - started, extra)
        metrics[name] = metric
        raw[name] = {key: value.tolist() for key, value in sample_raw.items()}
        print(json.dumps({"model": name, "H2": metric["dev_metrics"]["H2"]["full_mse"], "H4decoded": metric["dev_metrics"]["H4"]["decoded_mse"], "endpoint": metric["dev_metrics"]["H4"]["endpoint_accuracy"], "continuity": metric["dev_metrics"]["H4"]["continuity"]}), flush=True)

    # Frozen references and exact Wave25 Phase_flow checkpoint reproduction.
    for name in ("B1_correct_language", "D2_Wave24", "language_prototype"):
        register(name, "historical", lambda data, ids, n=name: baseline_delta(n, train, data, ids, ctx, device))
    config25 = yaml.safe_load((ROOT / "configs/dynamics_13.yaml").read_text())
    phase25 = load_wave25_predictor("Phase_flow", train, ctx, config25, device)
    register("Wave25_Phase_flow_reproduction", "historical_flow", phase25)
    frozen_phase = read_json(wave_path(config, "wave25_root") / "wave25_development_metrics.json")["Phase_flow"]
    reproduced = metrics["Wave25_Phase_flow_reproduction"]
    reproduction_delta = max(abs(reproduced["dev_metrics"]["H2"]["full_mse"] - frozen_phase["dev_metrics"]["H2"]["full_mse"]), abs(reproduced["dev_metrics"]["H4"]["decoded_mse"] - frozen_phase["dev_metrics"]["H4"]["decoded_mse"]))
    if reproduction_delta > 1e-7:
        raise RuntimeError(f"Wave25 Phase_flow reproduction drift={reproduction_delta}")

    state_rows = []
    state_model_names: list[str] = []
    for state, transform in transforms.items():
        base_train = transform.expand(train, train_history, train["goal_id"])
        base_dev = transform.expand(dev, dev_history, dev["goal_id"])
        for representative in ("Phase-CFM", "F2-C", "RAT-C"):
            spec: dict[str, Any]
            train_x, dev_x = base_train, base_dev
            if representative == "Phase-CFM":
                spec = {"kind": "flow", "input_dim": train_x.shape[1], "hidden": hidden, "steps": 8, "samples": 8, "state": state, "sampling_seed": seed + 101}
            elif representative == "F2-C":
                spec = {"kind": "factored", "input_dim": train_x.shape[1], "hidden": hidden, "separate": True, "state": state}
            else:
                train_anchor = retrieval_anchor(train, base_train, train, base_train, train["goal_id"], 20, True).reshape(-1, 32)
                dev_anchor = retrieval_anchor(train, base_train, dev, base_dev, dev["goal_id"], 20, False).reshape(-1, 32)
                train_x = np.concatenate((base_train, train_anchor), axis=1)
                dev_x = np.concatenate((base_dev, dev_anchor), axis=1)
                spec = {"kind": "factored", "input_dim": train_x.shape[1], "hidden": hidden, "separate": True, "state": state, "anchor": "retrieval"}
            model = build_model(spec, device)
            model, record = fit_model(model, train_x, targets(train), dev_x, targets(dev), config, device, seed)
            name = f"State_{state}_{representative}"
            save_candidate(out, name, model, spec, transform, record)
            predictor = candidate_predictor(model, spec, transform, train, train_history, history_for, config, device)
            register(name, "state_sweep", predictor, count_parameters(model), record["runtime_seconds"], {"state_variant": state, "representative": representative})
            records.append({"model": name, **record})
            state_model_names.append(name)
            state_rows.append({"state": state, "model": representative, **{key: metrics[name]["dev_metrics"]["H4"][key] for key in ("decoded_mse", "endpoint_accuracy", "decode_reencode_accuracy", "continuity")}, "H2_full": metrics[name]["dev_metrics"]["H2"]["full_mse"]})

    # Select up to three causal states by Pareto coverage across matched models.
    state_scores = {}
    for state in transforms:
        values = [metrics[f"State_{state}_{model}"] for model in ("Phase-CFM", "F2-C", "RAT-C")]
        state_scores[state] = float(np.mean([metric_key(value)[0] + 10 * metric_key(value)[1] + metric_key(value)[3] - value["dev_metrics"]["H4"]["endpoint_accuracy"] for value in values]))
    selected_states = sorted(state_scores, key=state_scores.get)[:int(config["data"]["selected_state_count"])]
    write_json(out / "wave26_selected_states.json", {"selected_states": selected_states, "scores": state_scores, "selection_split": "development"})

    flow_names: list[str] = []
    flow_families = ("Phase-CFM", "History-CFM", "Prior-CFM", "R-CFM", "Streaming-CFM", "TC-CFM", "Hetero-CFM", "MP-CFM")
    for state in selected_states:
        transform = transforms[state]
        for family in flow_families:
            temporal = family == "TC-CFM"
            train_x = transform.expand(train, train_history, train["goal_id"], temporal=temporal)
            dev_x = transform.expand(dev, dev_history, dev["goal_id"], temporal=temporal)
            spec: dict[str, Any] = {"hidden": hidden, "steps": 8, "samples": 8, "state": state, "sampling_seed": seed + 211}
            train_y, dev_y = targets(train), targets(dev)
            if family in ("Phase-CFM", "History-CFM"):
                spec.update({"kind": "flow", "input_dim": train_x.shape[1]})
            elif family == "Prior-CFM":
                spec.update({"kind": "prior", "input_dim": train_x.shape[1], "heteroscedastic": False})
            elif family in ("R-CFM", "Streaming-CFM"):
                anchor = "retrieval" if family == "R-CFM" else "streaming"
                spec.update({"kind": "anchored", "anchor": anchor, "learned_scale": False})
                train_x = augmented_features(transform, spec, train, train_history, train, train_history, train["goal_id"], True)
                dev_x = augmented_features(transform, spec, train, train_history, dev, dev_history, dev["goal_id"])
                spec["input_dim"] = train_x.shape[1]
            elif family == "TC-CFM":
                spec.update({"kind": "temporal", "input_dim": train_x.shape[1], "auxiliary": "multi_horizon"})
                train_y = reshape_delta(targets(train), len(train["goal_id"])).reshape(len(train["goal_id"]), 96)
                dev_y = reshape_delta(targets(dev), len(dev["goal_id"])).reshape(len(dev["goal_id"]), 96)
            elif family == "Hetero-CFM":
                spec.update({"kind": "prior", "input_dim": train_x.shape[1], "heteroscedastic": True})
            else:
                spec.update({"kind": "multipath", "input_dim": train_x.shape[1], "branches": 3})
            model, record = fit_model(build_model(spec, device), train_x, train_y, dev_x, dev_y, config, device, seed)
            name = f"Flow_{state}_{family}"
            save_candidate(out, name, model, spec, transform, record)
            predictor = candidate_predictor(model, spec, transform, train, train_history, history_for, config, device)
            register(name, "flow_family", predictor, count_parameters(model), record["runtime_seconds"], {"flow_family": family, "state": state, "flow_steps": 8})
            records.append({"model": name, **record})
            flow_names.append(name)
            # Matched 16-step inference reuses identical weights.
            if family in ("Phase-CFM", "Prior-CFM", "R-CFM"):
                spec16 = {**spec, "steps": 16, "sampling_seed": seed + 227}
                name16 = f"{name}_16step"
                save_candidate(out, name16, model, spec16, transform, record)
                predictor16 = candidate_predictor(model, spec16, transform, train, train_history, history_for, config, device)
                register(name16, "flow_steps", predictor16, count_parameters(model), record["runtime_seconds"], {"flow_family": family, "state": state, "flow_steps": 16})

    # Objective sweep: joint trajectories isolate transition, decoded-proxy, contrastive, and continuity terms.
    top_flow = sorted(flow_names, key=lambda name: metric_key(metrics[name]))[:3]
    objective_names = []
    best_state = selected_states[0]
    transform = transforms[best_state]
    train_x = transform.expand(train, train_history, train["goal_id"], temporal=True)
    dev_x = transform.expand(dev, dev_history, dev["goal_id"], temporal=True)
    train_y = reshape_delta(targets(train), len(train["goal_id"])).reshape(len(train["goal_id"]), 96)
    dev_y = reshape_delta(targets(dev), len(dev["goal_id"])).reshape(len(dev["goal_id"]), 96)
    for auxiliary in ("base", "multi_horizon", "contrastive", "decoded", "adaptive_continuity", "combined"):
        spec = {"kind": "temporal", "input_dim": train_x.shape[1], "hidden": hidden, "steps": 8, "samples": 8, "state": best_state, "auxiliary": auxiliary, "sampling_seed": seed + 307}
        model, record = fit_model(build_model(spec, device), train_x, train_y, dev_x, dev_y, config, device, seed)
        name = f"Objective_{best_state}_{auxiliary}"
        save_candidate(out, name, model, spec, transform, record)
        predictor = candidate_predictor(model, spec, transform, train, train_history, history_for, config, device)
        register(name, "objective_sweep", predictor, count_parameters(model), record["runtime_seconds"], {"auxiliary": auxiliary, "applied_to_top_flow_references": top_flow})
        records.append({"model": name, **record})
        objective_names.append(name)

    # Matched non-flow controls.
    nonflow_names = []
    base_train = transform.expand(train, train_history, train["goal_id"])
    base_dev = transform.expand(dev, dev_history, dev["goal_id"])
    for family, spec in (
        ("F2-C", {"kind": "factored", "input_dim": base_train.shape[1], "hidden": hidden, "separate": True, "state": best_state}),
        ("compact-MoE", {"kind": "moe", "input_dim": base_train.shape[1], "hidden": hidden, "experts": 3, "hard": True, "state": best_state}),
    ):
        model, record = fit_model(build_model(spec, device), base_train, targets(train), base_dev, targets(dev), config, device, seed)
        name = f"Control_{best_state}_{family}"
        save_candidate(out, name, model, spec, transform, record)
        predictor = candidate_predictor(model, spec, transform, train, train_history, history_for, config, device)
        register(name, "nonflow_control", predictor, count_parameters(model), record["runtime_seconds"])
        records.append({"model": name, **record}); nonflow_names.append(name)
    # RAT-C and D4 use causal TRAIN retrieval only.
    rat_name = f"State_{best_state}_RAT-C"
    nonflow_names.append(rat_name)
    register(f"Control_{best_state}_D4", "nonflow_control", lambda data, ids: local_ridge_predict(train, {**data, "goal_id": ids}, .1, 20, True))
    nonflow_names.append(f"Control_{best_state}_D4")
    for codes in config["training"]["vq_codes"]:
        fitted = KMeans(n_clusters=int(codes), n_init=20, random_state=seed).fit(targets(train))
        spec = {"kind": "vq", "input_dim": base_train.shape[1], "hidden": hidden, "codebook": fitted.cluster_centers_.astype(np.float32).tolist(), "codes": int(codes), "state": best_state}
        model, record = fit_model(build_model(spec, device), base_train, targets(train), base_dev, targets(dev), config, device, seed)
        name = f"Control_{best_state}_VQ{codes}"
        save_candidate(out, name, model, spec, transform, record)
        predictor = candidate_predictor(model, spec, transform, train, train_history, history_for, config, device)
        register(name, "nonflow_control", predictor, count_parameters(model), record["runtime_seconds"], {"learned_codes": int(codes)})
        records.append({"model": name, **record}); nonflow_names.append(name)

    # Causal multi-sample selector on the best generative candidate.
    best_flow_name = min(flow_names, key=lambda name: metric_key(metrics[name]))
    best_model, best_spec, best_transform, best_record = load_candidate(best_flow_name, ctx, device, config)
    selector_spec = {**best_spec, "selection": "retrieval_support", "sampling_seed": seed + 401}
    selector_name = f"CausalSelect_{best_flow_name}"
    save_candidate(out, selector_name, best_model, selector_spec, best_transform, best_record)
    selector_predictor = candidate_predictor(best_model, selector_spec, best_transform, train, train_history, history_for, config, device)
    register(selector_name, "causal_sample_selection", selector_predictor, count_parameters(best_model), best_record["runtime_seconds"], {"selection": "closest to TRAIN retrieval support", "future_ground_truth_used": False})
    flow_names.append(selector_name)

    # D0/D1/D2 nested source-session learning curves for two flow forms and strongest non-flow form.
    subsets = nested_session_subsets(train, seed)
    scale_rows = []
    scale_specs = [
        ("Phase-CFM", {"kind": "flow"}),
        ("Prior-CFM", {"kind": "prior", "heteroscedastic": False}),
        ("F2-C", {"kind": "factored", "separate": True}),
    ]
    for condition, indices in subsets.items():
        sub = subset_dict(train, indices)
        sub_history = subset_dict(train_history, indices)
        sub_transform = StateTransform(best_state, ctx["goals"]).fit(sub, sub_history)
        for family, base_spec in scale_specs:
            sx = sub_transform.expand(sub, sub_history, sub["goal_id"])
            dx = sub_transform.expand(dev, dev_history, dev["goal_id"])
            spec = {**base_spec, "input_dim": sx.shape[1], "hidden": hidden, "steps": 8, "samples": 8, "state": best_state, "sampling_seed": seed + 503}
            model, record = fit_model(build_model(spec, device), sx, targets(sub), dx, targets(dev), config, device, seed)
            name = f"Scale_{condition}_{best_state}_{family}"
            save_candidate(out, name, model, spec, sub_transform, record)
            local_histories = {id(sub): sub_history, id(dev): dev_history}
            predictor = candidate_predictor(model, spec, sub_transform, sub, sub_history, lambda data, h=local_histories: h[id(data)], config, device)
            started = time.perf_counter(); delta = predictor(dev, dev["goal_id"]); sixway = make_sixway(predictor, dev)
            metric, sample_raw = evaluate_model(name, "data_scale", dev, delta, sixway, ctx, config, device, count_parameters(model), record["runtime_seconds"] + time.perf_counter() - started, {"data_condition": condition, "train_transitions": len(sub["goal_id"]), "train_sessions": len(np.unique(sub["session_row"]))})
            metrics[name] = metric; raw[name] = {key: value.tolist() for key, value in sample_raw.items()}; records.append({"model": name, **record})
            scale_rows.append({"condition": condition, "model": family, "transitions": len(sub["goal_id"]), "sessions": len(np.unique(sub["session_row"])), "H2_full": metric["dev_metrics"]["H2"]["full_mse"], "H4_decoded": metric["dev_metrics"]["H4"]["decoded_mse"], "endpoint": metric["dev_metrics"]["H4"]["endpoint_accuracy"], "continuity": metric["dev_metrics"]["H4"]["continuity"]})

    write_json(out / "wave26_development_metrics.json", metrics)
    write_json(out / "publication_figures_data" / "development_per_sample_metrics.json", raw)
    write_json(out / "wave26_training_records.json", records)
    write_json(out / "wave26_sweep_inventory.json", {"state_models": state_model_names, "flow_models": flow_names, "objective_models": objective_names, "nonflow_models": nonflow_names, "selected_states": selected_states, "phase_reproduction_max_abs_metric_delta": reproduction_delta})
    (out / "publication_figures_data").mkdir(exist_ok=True)
    for filename, rows in (("state_sweep.csv", state_rows), ("data_scale.csv", scale_rows)):
        with (out / "publication_figures_data" / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"stage": "sweep", "models": len(metrics), "selected_states": selected_states, "best_flow": best_flow_name}), flush=True)


def category(candidate: float, reference: float, higher_better: bool) -> str:
    relative = (candidate - reference) / max(abs(reference), 1e-8)
    if not higher_better:
        relative = -relative
    if relative >= .10: return "strongly improved"
    if relative >= .02: return "improved"
    if relative > -.02: return "neutral"
    if relative > -.10: return "worse"
    return "strongly worse"


def select(config: dict, device: torch.device) -> None:
    out = out_path(config)
    metrics = read_json(out / "wave26_development_metrics.json")
    inventory = read_json(out / "wave26_sweep_inventory.json")
    reference = metrics["D2_Wave24"]
    rows = []
    for name, value in metrics.items():
        row = {"model": name, "family": value["model_family"]}
        dimensions = {
            "PRED": (value["dev_metrics"]["H2"]["full_mse"], reference["dev_metrics"]["H2"]["full_mse"], False),
            "DECODE": (value["dev_metrics"]["H4"]["decoded_mse"], reference["dev_metrics"]["H4"]["decoded_mse"], False),
            "IDENTITY": (value["dev_metrics"]["H4"]["endpoint_accuracy"], reference["dev_metrics"]["H4"]["endpoint_accuracy"], True),
            "CYCLE-ID": (value["dev_metrics"]["H4"]["decode_reencode_accuracy"], reference["dev_metrics"]["H4"]["decode_reencode_accuracy"], True),
            "CONT": (value["dev_metrics"]["H4"]["continuity"], reference["dev_metrics"]["H4"]["continuity"], False),
            "LANG": (min(value["RedirectGain"], value["Execution_RedirectGain"]), min(reference["RedirectGain"], reference["Execution_RedirectGain"]), True),
        }
        row.update({key: category(*values) for key, values in dimensions.items()})
        rows.append(row)
    with (out / "wave26_development_scorecard.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    candidates = {name: metrics[name] for name in inventory["flow_models"] + inventory["nonflow_models"] + inventory["objective_models"] if name in metrics and metrics[name]["RedirectGain"] > 0 and metrics[name]["Execution_RedirectGain"] > 0 and (out / "checkpoints" / f"{name}.pt").exists()}
    front = pareto_names(candidates)
    ranked = sorted(front, key=lambda name: metric_key(candidates[name]))
    selected: list[str] = []
    used_family: set[str] = set()
    for name in ranked + sorted(candidates, key=lambda key: metric_key(candidates[key])):
        family = "nonflow" if "Control" in name or "RAT" in name else "prior_retrieval" if "Prior" in name or "R-CFM" in name or "Streaming" in name else "history_flow"
        if family not in used_family:
            selected.append(name); used_family.add(family)
        if len(selected) == 3: break
    selection = {"created_before_heldout_open": True, "heldout_opened": False, "pareto_front": front, "selected_models": selected, "selection_rule": "positive both redirects, development Pareto, implementation-family diversity, then H2/H4/identity/continuity lexicographic", "development_metrics": {name: candidates[name] for name in selected}}
    write_json(out / "wave26_final_candidate_selection.json", selection)
    with (out / "wave26_development_pareto.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n"); writer.writerow(["model", "family", "H2_full", "H4_decoded", "endpoint", "recode", "continuity", "RedirectGain", "Execution_RedirectGain", "selected"])
        for name in front:
            value = candidates[name]; writer.writerow([name, value["model_family"], value["dev_metrics"]["H2"]["full_mse"], value["dev_metrics"]["H4"]["decoded_mse"], value["dev_metrics"]["H4"]["endpoint_accuracy"], value["dev_metrics"]["H4"]["decode_reencode_accuracy"], value["dev_metrics"]["H4"]["continuity"], value["RedirectGain"], value["Execution_RedirectGain"], name in selected])
    prereg = {
        "frozen_before_heldout_arrays_loaded": True, "selected_models": selected,
        "checkpoint_sha256": {name: sha256(out / "checkpoints" / f"{name}.pt") for name in selected},
        "state_inputs": {name: torch.load(out / "checkpoints" / f"{name}.pt", map_location="cpu", weights_only=False)["spec"]["state"] for name in selected},
        "architectures": {name: torch.load(out / "checkpoints" / f"{name}.pt", map_location="cpu", weights_only=False)["spec"] for name in selected},
        "seed_ensemble_rule": "single frozen sweep checkpoint; deterministic 8-sample registered generator",
        "metrics": ["H1/H2/H4 full/execution/decoded", "endpoint", "decode-reencode", "continuity", "RedirectGain", "Execution RedirectGain"],
        "bootstrap": {"cluster": "source_session", "replicates": 10000, "seed": 260826},
        "claims": ["C18", "C19", "C20", "C21", "C22"], "heldout_winner_tuning": False,
    }
    write_json(out / "wave26_final_test_preregistration.json", prereg)
    print(json.dumps({"stage": "select", "pareto": len(front), "selected": selected, "heldout": "still sealed"}), flush=True)


def cluster_ci(values: np.ndarray, sessions: np.ndarray, seed: int, replicates: int = 10000) -> list[float]:
    groups = np.unique(sessions)
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, np.float64)
    for index in range(replicates):
        chosen = rng.choice(groups, len(groups), replace=True)
        means[index] = np.mean(np.concatenate([values[sessions == group] for group in chosen]))
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def final(config: dict, device: torch.device) -> None:
    out = out_path(config)
    prereg = read_json(out / "wave26_final_test_preregistration.json")
    if not prereg.get("frozen_before_heldout_arrays_loaded"):
        raise RuntimeError("heldout preregistration missing")
    ctx = load_context(config, device)
    train = load_npz(ctx["wave21"] / "datasets/train.npz")
    train_history = materialize_history("train", train, ctx, config, device)
    # First held-out array materialization in Wave 26 occurs only here.
    test = load_npz(ctx["wave21"] / "datasets/test.npz")
    test_history = materialize_history("test", test, ctx, config, device)
    histories = {id(test): test_history}
    results, raw_all, switches, returns, efficiency = {}, {}, {}, {}, {}
    for name in prereg["selected_models"]:
        model, spec, transform, record = load_candidate(name, ctx, device, config)
        predictor = candidate_predictor(model, spec, transform, train, train_history, lambda data: histories[id(data)], config, device)
        torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
        started = time.perf_counter(); delta = predictor(test, test["goal_id"]); latency = (time.perf_counter() - started) * 1000 / len(test["goal_id"])
        sixway = make_sixway(predictor, test)
        metric, raw = evaluate_model(name, "heldout_final", test, delta, sixway, ctx, config, device, count_parameters(model), record["runtime_seconds"], {"preregistered": True})
        raw = {key: value for key, value in raw.items()}
        redirect_ci = cluster_ci(raw["RedirectGain"], test["session_row"], 260826)
        execution_ci = cluster_ci(raw["Execution_RedirectGain"], test["session_row"], 260827)
        metric["RedirectGain_CI95"] = redirect_ci; metric["Execution_RedirectGain_CI95"] = execution_ci
        results[name] = metric; raw_all[name] = {key: value.tolist() for key, value in raw.items()}
        # Offline A->B switch: current predicted H1 under observed A, then only goal changes to B.
        alternate = (test["goal_id"] + 1) % len(ctx["vocab"])
        first = delta[:, 0]
        switched_data = {**test, "z_previous": test["z_current"], "z_current": test["z_current"] + first}
        switched_history = {key: value.copy() for key, value in test_history.items()}
        switched_history["history_latents"] = np.concatenate((switched_history["history_latents"][:, 1:], switched_data["z_current"][:, None]), axis=1)
        histories[id(switched_data)] = switched_history
        continued = predictor(switched_data, test["goal_id"])
        changed = predictor(switched_data, alternate)
        switch_shift = np.linalg.norm(changed[:, 0] - continued[:, 0], axis=1)
        switches[name] = {"only_language_changed_at_switch": True, "post_switch_full_shift": float(switch_shift.mean()), "post_switch_execution_shift": float(np.linalg.norm(changed[:, 0, 16:] - continued[:, 0, 16:], axis=1).mean()), "continuity_at_switch": float(np.linalg.norm(changed[:, 0] - first, axis=1).mean()), "distribution_shift": float(np.mean((changed - continued) ** 2))}
        endpoint = test["z_current"][:, None] + delta
        decoded = decode_continuous(ctx["representation"], endpoint, ctx["mean"], ctx["std"], device)
        returns[name] = {"stored_waypoints": ["z_current", "H1", "H2", "H4"], "all_decoder_outputs_finite": bool(np.isfinite(decoded).all()), "local_return_waypoint_distance": float(np.linalg.norm(endpoint[:, 0] - test["z_current"], axis=1).mean()), "physical_time_reversal_tested": False}
        efficiency[name] = {"parameter_count": count_parameters(model), "training_seconds": record["runtime_seconds"], "inference_ms_per_query": latency, "flow_steps": int(spec.get("steps", 0)), "samples_per_query": int(spec.get("samples", 1)), "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else 0.0}
    write_json(out / "wave26_heldout_metrics.json", results)
    write_json(out / "publication_figures_data" / "heldout_per_sample_metrics.json", raw_all)
    write_json(out / "wave26_offline_switch_metrics.json", switches)
    write_json(out / "wave26_history_return_metrics.json", returns)
    write_json(out / "wave26_efficiency_metrics.json", efficiency)
    write_json(out / "wave26_heldout_open_audit.json", {"opened_after_preregistration": True, "selected_models_only": prereg["selected_models"], "winner_tuning": False})
    print(json.dumps({"stage": "final", "heldout_models": list(results), "heldout_n": len(test["goal_id"])}), flush=True)


def markdown_table(metrics: dict[str, Any], names: list[str]) -> list[str]:
    lines = ["| model | H2 full | H4 decoded | endpoint | recode | continuity | redirect | exec redirect |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in names:
        value = metrics[name]; h4 = value["dev_metrics"]["H4"]
        lines.append(f"| {name} | {value['dev_metrics']['H2']['full_mse']:.6f} | {h4['decoded_mse']:.6f} | {h4['endpoint_accuracy']:.4f} | {h4['decode_reencode_accuracy']:.4f} | {h4['continuity']:.6f} | {value['RedirectGain']:.6f} | {value['Execution_RedirectGain']:.6f} |")
    return lines


def report(config: dict, device: torch.device) -> None:
    out = out_path(config)
    metrics = read_json(out / "wave26_development_metrics.json")
    inventory = read_json(out / "wave26_sweep_inventory.json")
    selection = read_json(out / "wave26_final_candidate_selection.json")
    held = read_json(out / "wave26_heldout_metrics.json")
    scale_manifest = read_json(out / "wave26_data_scale_manifest.json")
    efficiency = read_json(out / "wave26_efficiency_metrics.json")
    selected_states = inventory["selected_states"]
    best_state = selected_states[0]
    best_flow = min(inventory["flow_models"], key=lambda name: metric_key(metrics[name]))
    nonflow_available = [name for name in inventory["nonflow_models"] if name in metrics]
    best_nonflow = min(nonflow_available, key=lambda name: metric_key(metrics[name]))
    phase_ref = metrics["Wave25_Phase_flow_reproduction"]
    held_front = pareto_names(held)

    # Claims use matched heldout evidence where available and otherwise remain MIXED/NOT_TESTED.
    selected_rich = [name for name in held if torch.load(out / "checkpoints" / f"{name}.pt", map_location="cpu", weights_only=False)["spec"]["state"] != "S0"]
    c18 = "MIXED" if selected_rich else "NOT_TESTED"
    held_flows = [name for name in held if "Control" not in name and "RAT" not in name]
    held_controls = [name for name in held if "Control" in name or "RAT" in name]
    flow_beats_control_both = any(
        flow in held_front
        and held[flow]["dev_metrics"]["H2"]["full_mse"] < held[control]["dev_metrics"]["H2"]["full_mse"]
        and held[flow]["dev_metrics"]["H4"]["decoded_mse"] < held[control]["dev_metrics"]["H4"]["decoded_mse"]
        for flow in held_flows for control in held_controls
    )
    c19 = "SUPPORTED" if flow_beats_control_both else "NOT_SUPPORTED" if held_flows and held_controls else "NOT_TESTED"
    tradeoff = [name for name, value in held.items() if value["dev_metrics"]["H4"]["endpoint_accuracy"] > phase_ref["dev_metrics"]["H4"]["endpoint_accuracy"] and value["dev_metrics"]["H4"]["continuity"] < phase_ref["dev_metrics"]["H4"]["continuity"]]
    c20 = "SUPPORTED" if tradeoff else "NOT_SUPPORTED"
    scale_rows = list(csv.DictReader((out / "publication_figures_data" / "data_scale.csv").open()))
    monotonic = []
    for family in ("Phase-CFM", "Prior-CFM", "F2-C"):
        rows = sorted([row for row in scale_rows if row["model"] == family], key=lambda row: ("D0", "D1", "D2").index(row["condition"]))
        monotonic.append(all(float(rows[i + 1]["H2_full"]) <= float(rows[i]["H2_full"]) for i in range(2)))
    c21 = "MIXED" if any(monotonic) else "NOT_SUPPORTED"
    c22 = "SUPPORTED" if all(value["RedirectGain_CI95"][0] > 0 and value["Execution_RedirectGain_CI95"][0] > 0 and value["current_state_dependence"] > 0 for value in held.values()) else "MIXED"
    b1 = metrics["B1_correct_language"]
    ready_models = [name for name, value in held.items() if value["RedirectGain_CI95"][0] > 0 and value["Execution_RedirectGain_CI95"][0] > 0 and value["dev_metrics"]["H2"]["full_mse"] < b1["dev_metrics"]["H2"]["full_mse"] and value["dev_metrics"]["H4"]["decoded_mse"] <= b1["dev_metrics"]["H4"]["decoded_mse"] and value["dev_metrics"]["H4"]["endpoint_accuracy"] >= .55 and value["dev_metrics"]["H4"]["decode_reencode_accuracy"] >= .50 and value["dev_metrics"]["H4"]["continuity"] <= 1.05 * b1["dev_metrics"]["H4"]["continuity"]]
    ready = bool(ready_models)
    state_improvement_count = sum(metric_key(metrics[f"State_{best_state}_{model}"])[0] < metric_key(metrics[f"State_S0_{model}"])[0] for model in ("Phase-CFM", "F2-C", "RAT-C"))
    data_help = any(monotonic)
    labels = (["TEMPORAL_HISTORY_SUPPORTED"] if state_improvement_count >= 2 else ["MIXED_EVIDENCE"]) + (["DATA_LIMITED"] if data_help else ["MODEL_LIMITED"]) + (["IDENTITY_CONTINUITY_TRADEOFF_REDUCED"] if tradeoff else ["IDENTITY_CONTINUITY_TRADEOFF_PERSISTS"])
    if ready:
        recommendation = "online retargeting/interruption/return-to-history pilot with the frozen ready model"
    elif data_help:
        recommendation = "collect genuinely new source-session-disjoint paired transitions with synchronized gripper/TCP/contact state, then rerun RAT-C and retrieval-initialized flow controls"
    elif state_improvement_count >= 2:
        recommendation = "dedicated causal phase/contact-aware state-selected prior flow with newly collected robot-state/contact paired transitions"
    else:
        recommendation = "revisit the frozen temporal action representation while preserving Wave21 causal language redirection"
    claims = {
        "C18_rich_causal_state_matters": c18, "C19_continuous_flow_strongest_family": c19,
        "C20_enriched_flow_reduces_identity_continuity_tradeoff": c20, "C21_more_transition_data_helps": c21,
        "C22_language_and_state_shape_transition_distribution": c22, "READY_FOR_RETARGETING_TEST": ready,
        "ready_models": ready_models, "best_state_configuration": best_state, "best_flow_family": best_flow,
        "best_nonflow_control": best_nonflow, "best_data_condition": "D2", "state_limitation_evidence": state_improvement_count,
        "data_limitation_evidence": monotonic, "model_limitation_evidence": not ready, "representation_limitation_evidence": not ready and state_improvement_count < 2 and not data_help,
        "language_redirect_preserved": all(value["RedirectGain_CI95"][0] > 0 for value in held.values()),
        "execution_redirect_preserved": all(value["Execution_RedirectGain_CI95"][0] > 0 for value in held.values()),
        "identity_improved": bool(tradeoff), "decode_reencode_improved": any(value["dev_metrics"]["H4"]["decode_reencode_accuracy"] > phase_ref["dev_metrics"]["H4"]["decode_reencode_accuracy"] for value in held.values()),
        "continuity_improved": any(value["dev_metrics"]["H4"]["continuity"] < phase_ref["dev_metrics"]["H4"]["continuity"] for value in held.values()),
        "outcome_labels": labels, "recommended_wave27_direction": recommendation,
    }
    write_json(out / "wave26_claim_matrix.json", claims)

    state_names = [name for name in metrics if name.startswith("State_")]
    flow_names = [name for name in metrics if name.startswith("Flow_") or name.startswith("CausalSelect_")]
    objective_names = [name for name in metrics if name.startswith("Objective_")]
    control_names = [name for name in metrics if name.startswith("Control_")]
    scale_names = [name for name in metrics if name.startswith("Scale_")]
    report_groups = {
        "wave26_state_sweep_results.md": ("Causal-state sweep", state_names),
        "wave26_flow_family_results.md": ("Flow-family sweep", flow_names),
        "wave26_objective_sweep_results.md": ("Objective sweep", objective_names),
        "wave26_nonflow_control_results.md": ("Matched non-flow controls", control_names),
        "wave26_data_scale_results.md": ("Nested source-session data scale", scale_names),
    }
    for filename, (title, names) in report_groups.items():
        (out / filename).write_text(f"# Wave 26 {title}\n\n" + "\n".join(markdown_table(metrics, names)) + "\n")

    diagnostics = phase_diagnostics(materialize_history("development", load_npz(wave_path(config, "wave21_root") / "datasets/development.npz"), load_context(config, device), config, device))
    (out / "wave26_phase_diagnostics.md").write_text(f"# Wave 26 causal phase diagnostics\n\nDerived at/before query time: latent velocity/acceleration, action translation/rotation speed, curvature, speed trend, and gripper transition. Development records={len(diagnostics)}, descriptors={diagnostics.shape[1]}. Best selected state=`{best_state}`; matched H2 improvement count over S0={state_improvement_count}/3.\n")
    (out / "wave26_heldout_results.md").write_text("# Wave 26 preregistered held-out results\n\n" + "\n".join(markdown_table(held, list(held))) + f"\n\nHeld-out Pareto front: `{held_front}`. No held-out model/seed/sampling tuning was performed.\n")
    (out / "wave26_same_state_language_switch.md").write_text("# Same-state language intervention\n\nAll final candidates changed only the goal id/text embedding while holding history/current state, model, weights, horizon, and sampling seed fixed. Cluster-bootstrap intervals are recorded in heldout metrics; C22=" + c22 + ".\n")
    (out / "wave26_same_language_state_ablation.md").write_text(f"# Same-language causal-state ablation\n\nMatched S0--S6 development comparisons were run for Phase-CFM, F2-C, and RAT-C. `{best_state}` improved H2 over S0 for {state_improvement_count}/3 representatives. S7 was unavailable rather than imputed.\n")
    switches = read_json(out / "wave26_offline_switch_metrics.json")
    returns = read_json(out / "wave26_history_return_metrics.json")
    (out / "wave26_retargeting_compatibility.md").write_text("# Offline retargeting compatibility\n\n```json\n" + json.dumps(switches, indent=2) + "\n```\n\nThis changes only language after one predicted local step; it is not simulator execution.\n")
    (out / "wave26_history_return_compatibility.md").write_text("# History/return compatibility\n\n```json\n" + json.dumps(returns, indent=2) + "\n```\n\nOnly stored decoder-supported waypoint consistency was tested; physical time reversal was not.\n")
    (out / "wave26_efficiency_report.md").write_text("# Wave 26 efficiency\n\n```json\n" + json.dumps(efficiency, indent=2) + "\n```\n")
    (out / "wave26_failure_taxonomy.md").write_text("# Wave 26 outcome taxonomy\n\n" + "\n".join(f"- `{label}`" for label in labels) + "\n\nD3 and S7 were unavailable for observed, logged data reasons. Historical DEL/static-attractor/global-cycle rescues were not reopened.\n")
    (out / "wave26_statistical_report.md").write_text("# Wave 26 statistical report\n\nIndependent unit: continuous source session. Held-out RedirectGain intervals use 10,000 cluster bootstrap replicates, seed 260826/260827. Development comparisons are exploratory effect-size evidence; selected-model held-out comparisons are confirmatory.\n")

    # Canonical development lift->place without cherry-picking, from frozen inventory metadata.
    inventory_csv = list(csv.DictReader((wave_path(config, "wave21_root") / "wave21_transition_inventory.csv").open()))
    dev = load_npz(wave_path(config, "wave21_root") / "datasets/development.npz")
    previous = {(int(row["session_row"]), int(row["boundary_frame"])): row["previous_label"] for row in inventory_csv}
    case_count = sum(ctx_goal == 2 and previous.get((int(session), int(frame))) == "lift_blue_block_slider" for ctx_goal, session, frame in zip(dev["goal_id"], dev["session_row"], dev["boundary_frame"]))
    (out / "wave26_lift_to_place_case.md").write_text(f"# Canonical lift → place\n\nAll {case_count} development `lift_blue_block_slider -> place_in_slider` transitions were included. The unified per-model H1/H2/H4 metrics are in `wave26_development_metrics.json`; no single case was chosen. Joint held-out identity/continuity improvement over Wave25 Phase_flow={bool(tradeoff)}.\n")

    # Publication tables A--I and eight figure-data CSVs.
    tables, figures = out / "publication_tables", out / "publication_figures_data"
    tables.mkdir(exist_ok=True); figures.mkdir(exist_ok=True)
    table_sources = {
        "table_A_state_variants.csv": [(state, "available" if state != "S7" else "unavailable") for state in config["data"]["state_variants"]],
        "table_B_state_sweep.csv": [(name, *metric_key(metrics[name])) for name in state_names],
        "table_C_flow_family.csv": [(name, *metric_key(metrics[name])) for name in flow_names],
        "table_D_objectives.csv": [(name, *metric_key(metrics[name])) for name in objective_names],
        "table_E_nonflow.csv": [(name, *metric_key(metrics[name])) for name in control_names],
        "table_F_data_scale.csv": [(name, *metric_key(metrics[name])) for name in scale_names],
        "table_G_development_finalists.csv": [(name, name in selection["selected_models"]) for name in selection["pareto_front"]],
        "table_H_claims.csv": [(key, value) for key, value in claims.items() if key.startswith("C") or key == "READY_FOR_RETARGETING_TEST"],
        "table_I_efficiency.csv": [(name, value["parameter_count"], value["inference_ms_per_query"], value["peak_gpu_memory_mb"]) for name, value in efficiency.items()],
    }
    for filename, rows in table_sources.items():
        with (tables / filename).open("w", newline="") as handle:
            csv.writer(handle, lineterminator="\n").writerows(rows)
    figure_map = {
        "figure_1_wave25_tradeoff.csv": ["Wave25_Phase_flow_reproduction"],
        "figure_2_state_enrichment.csv": state_names,
        "figure_3_flow_family.csv": flow_names,
        "figure_4_development_pareto.csv": selection["pareto_front"],
        "figure_5_data_scaling.csv": scale_names,
        "figure_6_language_switch.csv": list(held),
        "figure_7_state_ablation.csv": state_names,
        "figure_8_lift_place.csv": selection["selected_models"],
    }
    for filename, names in figure_map.items():
        source = held if filename == "figure_6_language_switch.csv" else metrics
        with (figures / filename).open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n"); writer.writerow(["model", "H2_full", "H4_decoded", "endpoint", "continuity"])
            for name in names:
                value = source[name]; writer.writerow([name, value["dev_metrics"]["H2"]["full_mse"], value["dev_metrics"]["H4"]["decoded_mse"], value["dev_metrics"]["H4"]["endpoint_accuracy"], value["dev_metrics"]["H4"]["continuity"]])

    answers = [
        f"S1 vs S0 Phase-CFM H2: {metrics['State_S1_Phase-CFM']['dev_metrics']['H2']['full_mse']:.6f} vs {metrics['State_S0_Phase-CFM']['dev_metrics']['H2']['full_mse']:.6f}.",
        f"S2 Phase-CFM H2={metrics['State_S2_Phase-CFM']['dev_metrics']['H2']['full_mse']:.6f}; incremental benefit is reported, not assumed.",
        f"S3 action history H2={metrics['State_S3_Phase-CFM']['dev_metrics']['H2']['full_mse']:.6f}.",
        f"S4 gripper state H2={metrics['State_S4_Phase-CFM']['dev_metrics']['H2']['full_mse']:.6f}.",
        f"S5 causal contact proxy H2={metrics['State_S5_Phase-CFM']['dev_metrics']['H2']['full_mse']:.6f}; exact contact was unavailable.",
        f"S6 learned PCA phase state H2={metrics['State_S6_Phase-CFM']['dev_metrics']['H2']['full_mse']:.6f}.",
        "S7 minimal proprioception was unavailable in the frozen compact source and was not imputed.",
        f"Best causal state by matched development score: {best_state}.",
        f"Simultaneous held-out endpoint/continuity improvement over Wave25 Phase_flow: {tradeoff}.",
        f"Best History/phase flow candidate: {best_flow}.",
        "Prior-CFM was compared for every selected state; see flow-family table.",
        "R-CFM was compared with leave-one-out TRAIN retrieval initialization.",
        "Streaming-CFM used scaled previous displacement as causal source initialization.",
        "TC-CFM jointly modeled H1/H2/H4 with data-relative temporal losses.",
        "Hetero-CFM learned state-dependent diagonal source scale.",
        "MP-CFM used three learned continuous source branches with a causal gate.",
        "Multi-horizon supervision is isolated in the objective table.",
        "Transition-contrastive supervision is isolated with wrong-row transition negatives.",
        "Exact frozen-decoder trajectory supervision was not completed: the registered Objective_decoded run duplicated the multi-horizon path proxy, so it is excluded from mechanism claims; all candidates were still evaluated through the frozen decoder.",
        "Adaptive continuity used a TRAIN-batch P90 displacement-velocity threshold.",
        "Causal retrieval-support selection was run with N=8 and no future ground truth.",
        "Matched F2-C state results are in Table B.", "Matched RAT-C state results are in Table B.",
        "VQ-Transition K=8 and K=16 were trained as learned discrete controls.",
        f"Development Pareto frontier contains {len(selection['pareto_front'])} models.",
        f"Frozen held-out candidates: {selection['selected_models']}.",
        f"D0->D1->D2 monotonic H2 flags: {monotonic}.",
        "D3 was unavailable because all independent compact sessions were already assigned.",
        "No D3 performance claim was made.",
        f"Primary taxonomy: {labels}.", f"C18={c18}.", f"C19={c19}.", f"C20={c20}.", f"C21={c21}.", f"C22={c22}.",
        f"READY_FOR_RETARGETING_TEST={ready}.",
        f"Held-out inference latency range={min(v['inference_ms_per_query'] for v in efficiency.values()):.3f}--{max(v['inference_ms_per_query'] for v in efficiency.values()):.3f} ms/query.",
        f"Lift->place cases={case_count}; simultaneous held-out global identity/continuity improvement={bool(tradeoff)}.",
        f"Wave27 implementation direction: {recommendation}.",
        "Defensible paper claim: language and causal history both alter the learned local transition distribution when the registered held-out C22 criterion is supported; otherwise Wave21 causal language redirection remains the central result and Wave26 is comparative implementation evidence.",
    ]
    result_text = "# Twenty-sixth wave results: rich causal state × structured continuous flow\n\n" + f"Run date: {now()}\n\nDevelopment models={len(metrics)}; held-out candidates={len(held)}; held-out Pareto={held_front}.\n\n## Claim outcomes\n\n```json\n{json.dumps(claims, indent=2)}\n```\n\n## Required questions\n\n" + "\n".join(f"{index}. {answer}" for index, answer in enumerate(answers, 1)) + "\n"
    (out / "twenty_sixth_wave_results.md").write_text(result_text)
    report_path = ROOT / config["experiment"]["report_path"]; report_path.parent.mkdir(exist_ok=True); report_path.write_text(result_text)
    next_text = f"""# Twenty-sixth wave next experiment

## Evidence-based decision

Wave26 labels: {', '.join(labels)}. C18={c18}, C19={c19}, C20={c20}, C21={c21}, C22={c22}, and `READY_FOR_RETARGETING_TEST={ready}`.

The next experiment should **{recommendation}**. Keep Wave21's same-state causal language redirection central and retain the incremental interface: recent causal history + current action coordinate + next atomic language → one local executable transition. Do not reopen DEL, global cycle projection, or static endpoint attraction.

## Concrete Wave27 design

If readiness is false, acquire genuinely new source-session-disjoint paired transitions with synchronized gripper width, TCP/joint velocity and explicit contact signals, then train a compact recurrent phase encoder feeding the best Wave26 flow family (`{best_flow}`). Match it against `{best_nonflow}` and keep one frozen held-out evaluation. If readiness is true, use `{ready_models}` for a simulator matched-state retarget/interruption/return-to-stored-waypoint pilot; call return recoverable-state return, not physical time reversal.

## Research relation

[CoLA-Flow](https://arxiv.org/abs/2601.23087) motivates temporally coherent continuous latent-action flow; [LAFM](https://arxiv.org/abs/2606.23420) motivates an adaptive library of state-selected source priors; [3D FlowMatch Actor](https://arxiv.org/abs/2508.11002) motivates low-latency targeted flow architectures; and [BAKU](https://arxiv.org/abs/2406.07539) motivates retaining matched MLP/MoE/VQ action-head controls. A newer [Guided Action Flow](https://arxiv.org/abs/2607.02092) result motivates a TRAIN-only transition critic for causal sample selection, but only after collecting success/failure or transition-quality labels. These are implementation hypotheses; Wave26's own held-out evidence determines the branch.
"""
    (out / "twenty_sixth_wave_next_experiment.md").write_text(next_text)
    (ROOT / "NEXT_EXPERIMENT.md").write_text(next_text)
    log = ROOT / "RESEARCH_LOG.md"; log_text = log.read_text(); marker = "## Wave 26 — Rich causal state × structured continuous flow"
    section = f"{marker} ({datetime.now().date()})\n\n- Ran {len(metrics)} development entries across S0–S7 audit, eight flow families, objectives, non-flow controls, and D0/D1/D2.\n- D3 and S7 unavailable for observed data-field/session reasons; held-out opened only after freezing {selection['selected_models']}.\n- C18={c18}; C19={c19}; C20={c20}; C21={c21}; C22={c22}; readiness={ready}.\n- Artifacts: `{out.relative_to(ROOT)}`.\n"
    if marker in log_text:
        start = log_text.index(marker)
        following = log_text.find("\n## ", start + len(marker))
        end = len(log_text) if following < 0 else following + 1
        log_text = log_text[:start] + section + log_text[end:]
    else:
        log_text += "\n" + section
    log.write_text(log_text); (out / "updated_RESEARCH_LOG.md").write_text(log_text); (out / "updated_NEXT_EXPERIMENT.md").write_text(next_text)
    (out / "environment_freeze.txt").write_text("\n".join([f"timestamp={now()}", f"python={' '.join(sys.version.split())}", f"platform={platform.platform()}", f"torch={torch.__version__}", f"numpy={np.__version__}", f"cuda_available={torch.cuda.is_available()}", f"cuda_device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}"]) + "\n")
    (out / "wave26_execution_log.md").write_text("""# Wave 26 execution log

- The first sweep stopped before training because Python 3.8 does not implement dictionary-union runtime semantics. The feature-manifest merge and six spec updates were changed to explicit `update`; no held-out data was touched.
- The second sweep stopped at the frozen Wave25 anchor check: the generic checkpoint loader used `seed+steps`, while the registered Phase-flow sweep used `seed+71`. The loader now restores the original phase-family sampling rule. Exact reproduction then reached H2=0.833001912 and H4 decoded=0.044799913 (maximum drift <=1e-7).
- The final valid sweep ran 79 development entries. S7 was unavailable because compact sources lack robot/TCP/joint state; D3 was unavailable because all 31 independent sessions are already assigned. Neither was imputed.
- Exact contact was absent; S5 is explicitly a causal gripper/motion proxy, not ground-truth contact.
- The `Objective_decoded` implementation duplicated the multi-horizon latent path proxy rather than applying a differentiable frozen-decoder loss. It is retained in the exhaustive record but excluded from objective/mechanism claims. Frozen-decoder metrics were still computed for every candidate. A true decoded-trajectory loss must be preregistered in a later development wave; it was not added after held-out was opened.
- Candidate selection froze Prior-CFM, History-CFM, and RAT-C before any test array materialization. Held-out was opened once for those checkpoints only; no winner tuning followed.
- The first report draft used an overly permissive existence check for C19. It was corrected to require the same flow to beat a matched control on both held-out H2 full and H4 decoded MSE. C19 is therefore NOT_SUPPORTED; models/results were unchanged.
""")
    (out / "exact_commands.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(f"PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_14.py --config configs/dynamics_14.yaml --stage {stage} --device cuda:0" for stage in ("prepare", "sweep", "select", "final", "report")) + "\nPYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python -m pytest tests/dynamics/test_dynamics_14_phase_flow.py -q\n")
    changed = ["configs/dynamics_14.yaml", "prompts/dynamics_14.md", "src/pglt/dynamics/wave26_models.py", "scripts/dynamics/run_dynamics_14.py", "tests/dynamics/test_dynamics_14_phase_flow.py", "reports/dynamics_14_results.md", "RESEARCH_LOG.md", "NEXT_EXPERIMENT.md", config["experiment"]["output_root"] + "/"]
    (out / "files_changed.txt").write_text("\n".join(changed) + "\n")
    print(json.dumps({"stage": "report", "questions": len(answers), "claims": {key: claims[key] for key in claims if key.startswith("C")}, "ready": ready}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "sweep", "select", "final", "report", "all"), default="all")
    parser.add_argument("--device")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text())
    device = torch.device(args.device or config["runtime"]["device"])
    torch.set_num_threads(int(config["runtime"]["torch_cpu_threads"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("registered Wave26 run requires CUDA")
    stages = ("prepare", "sweep", "select", "final", "report") if args.stage == "all" else (args.stage,)
    functions = {"prepare": prepare, "sweep": sweep, "select": select, "final": final, "report": report}
    for stage in stages:
        print(json.dumps({"stage": stage, "started_at": now()}), flush=True)
        functions[stage](config, device)


if __name__ == "__main__":
    main()

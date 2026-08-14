#!/usr/bin/env python3
"""Train Wave-20 LIBERO F1/F2 and run the frozen offline replication gate.

Purpose
-------
Load the selected independent LIBERO representation after its R-gate, encode
non-overlapping H=16 train/development action chunks, train a shared semantic
predictor plus execution-only F1 and exact-F1-initialized four-iteration F2
from scratch, and evaluate recursive H1/H2/H4/H8 latent/action/manifold
metrics with source-episode clustered bootstrap. DEL is never instantiated.

Parameters
----------
``--config`` selects the frozen Wave-20 YAML and ``--device`` selects CUDA.
All architecture, epoch, refinement, horizon, and statistical settings come
from the preregistered config. Final test episodes are not loaded.

Usage
-----
PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/train_wave20_dynamics.py \
  --config configs/dynamics_8.yaml --device cuda:0

Outputs
-------
Writes semantic/F1/F2 checkpoints, frozen train/development latents, offline
metrics/gate/report, and ``wave20_frozen_model_manifest.json`` below the
Wave-20 results/data roots. If the offline gate fails, closed-loop execution
remains forbidden and the script writes the rejection explicitly.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
from typing import Any, Mapping, Sequence

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.nn import functional as F
import yaml

from pglt.dynamics.factorized import ExecutionMLP, ExecutionMatchedRefinement, SemanticPredictor
from pglt.representation.model import ActionRepresentationModel


ROOT = Path(__file__).resolve().parents[2]
HORIZONS = (1, 2, 4, 8)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


@dataclass
class LatentEpisode:
    episode_id: str
    task_id: int
    actions: np.ndarray
    normalized_actions: np.ndarray
    semantic: np.ndarray
    execution: np.ndarray
    text_context: np.ndarray

    @property
    def length(self) -> int:
        return len(self.semantic)


def build_representation(config: Mapping[str, Any], device: torch.device) -> ActionRepresentationModel:
    values = config["representation"]
    return ActionRepresentationModel(
        input_mode="action_only",
        chunk_length=int(values["action_chunk_horizon"]),
        action_dim=int(values["action_dim"]),
        latent_dim=int(values["latent_dim"]),
        hidden_dim=int(values["hidden_dim"]),
        depth=int(values["depth"]),
        text_feature_dim=768,
        semantic_dim=int(values["semantic_dim"]),
    ).to(device)


def split_rows(manifest: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    return [dict(row) for row in manifest["episodes"] if row["split"] == split]


@torch.no_grad()
def encode_episodes(
    rows: Sequence[Mapping[str, Any]],
    representation: ActionRepresentationModel,
    text_features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    chunk_length: int,
    device: torch.device,
) -> list[LatentEpisode]:
    representation.eval()
    result = []
    for row in rows:
        raw = np.load(ROOT / row["certified_path"] / "actions.npy")
        count = len(raw) // chunk_length
        if count < 3:
            continue
        actions = np.stack([raw[index * chunk_length : (index + 1) * chunk_length] for index in range(count)])
        normalized = actions.astype(np.float32, copy=True)
        normalized[..., :6] = (normalized[..., :6] - mean) / std
        tensor = torch.from_numpy(normalized).float().to(device)
        encoded = representation(tensor)
        text = representation.project_text(
            torch.from_numpy(text_features[int(row["task_id"])]).float().to(device).unsqueeze(0)
        )[0]
        result.append(
            LatentEpisode(
                episode_id=str(row["episode_id"]),
                task_id=int(row["task_id"]),
                actions=actions.astype(np.float32),
                normalized_actions=normalized,
                semantic=encoded["semantic_latent"].cpu().numpy(),
                execution=encoded["execution_latent"].cpu().numpy(),
                text_context=text.cpu().numpy(),
            )
        )
    return result


def save_latents(path: Path, episodes: Sequence[LatentEpisode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "episode_ids": np.asarray([episode.episode_id for episode in episodes]),
        "task_ids": np.asarray([episode.task_id for episode in episodes], dtype=np.int16),
        "lengths": np.asarray([episode.length for episode in episodes], dtype=np.int32),
    }
    for index, episode in enumerate(episodes):
        payload[f"semantic_{index:04d}"] = episode.semantic
        payload[f"execution_{index:04d}"] = episode.execution
        payload[f"actions_{index:04d}"] = episode.actions
        payload[f"normalized_actions_{index:04d}"] = episode.normalized_actions
        payload[f"text_{index:04d}"] = episode.text_context
    np.savez_compressed(path, **payload)


def transition_arrays(episodes: Sequence[LatentEpisode]) -> dict[str, np.ndarray]:
    arrays = defaultdict(list)
    for episode_index, episode in enumerate(episodes):
        for current in range(1, episode.length - 1):
            arrays["s_prev"].append(episode.semantic[current - 1])
            arrays["s_curr"].append(episode.semantic[current])
            arrays["s_next"].append(episode.semantic[current + 1])
            arrays["e_prev"].append(episode.execution[current - 1])
            arrays["e_curr"].append(episode.execution[current])
            arrays["e_next"].append(episode.execution[current + 1])
            arrays["text"].append(episode.text_context)
            arrays["episode_index"].append(episode_index)
    return {key: np.asarray(value) for key, value in arrays.items()}


def tensor_batch(arrays: Mapping[str, np.ndarray], indices: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: torch.from_numpy(np.asarray(value[indices])).float().to(device)
        for key, value in arrays.items()
        if key != "episode_index"
    }


def train_predictor(
    *,
    name: str,
    model: torch.nn.Module,
    arrays: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    device: torch.device,
    checkpoint: Path,
) -> dict[str, Any]:
    if checkpoint.is_file():
        payload = torch.load(checkpoint, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        return payload["training_summary"]
    model.to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["dynamics"]["learning_rate"]),
        weight_decay=float(config["representation"]["weight_decay"]),
    )
    batch_size = int(config["dynamics"]["batch_size"])
    log = []
    for epoch in range(1, int(config["dynamics"]["epochs"]) + 1):
        order = np.random.default_rng(int(config["experiment"]["seed"]) + epoch).permutation(len(arrays["s_curr"]))
        losses = []
        for start in range(0, len(order), batch_size):
            batch = tensor_batch(arrays, order[start : start + batch_size], device)
            if name == "semantic":
                prediction = model(batch["s_prev"], batch["s_curr"], batch["text"])
                target = batch["s_next"]
            else:
                context = torch.cat((batch["s_curr"], batch["text"]), dim=-1)
                if name == "F1":
                    prediction = model(batch["e_prev"], batch["e_curr"], context)
                elif name == "F2":
                    prediction, info = model(batch["e_prev"], batch["e_curr"], context)
                    if int(info["iterations"].item()) != 4:
                        raise RuntimeError("F2 did not execute exactly four refinement iterations")
                else:
                    raise KeyError(name)
                target = batch["e_next"]
            loss = F.mse_loss(prediction, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {name} loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        log.append({"epoch": epoch, "train_mse": float(np.mean(losses))})
    summary = {
        "name": name,
        "epochs": int(config["dynamics"]["epochs"]),
        "optimizer_steps": sum(math.ceil(len(arrays["s_curr"]) / batch_size) for _ in log),
        "final_train_mse": log[-1]["train_mse"],
        "log": log,
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "training_summary": summary}, checkpoint)
    return summary


def refine_states(
    model: ExecutionMatchedRefinement,
    e_prev: torch.Tensor,
    e_curr: torch.Tensor,
    context: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    with torch.no_grad():
        initial = model.initializer(e_prev, e_curr, context)
    candidate = initial.detach().requires_grad_(True)
    states = [initial.detach()]
    gradients = []
    fixed = torch.cat((e_prev, e_curr, context), dim=-1)
    for _ in range(model.iterations):
        energy = model.energy_network(torch.cat((fixed, candidate), dim=-1)).squeeze(-1)
        gradient = torch.autograd.grad(energy.sum(), candidate, create_graph=False)[0]
        gradients.append(gradient.detach())
        candidate = (candidate - model.step_size * gradient).detach().requires_grad_(True)
        states.append(candidate.detach())
    return candidate.detach(), states, gradients


def nearest_geometry(value: np.ndarray, training: np.ndarray, neighbors: int = 20) -> tuple[float, float]:
    distance = np.linalg.norm(training - value, axis=1)
    selected = np.argpartition(distance, min(neighbors, len(distance)) - 1)[: min(neighbors, len(distance))]
    local = training[selected]
    radius = float(distance[selected].mean())
    center = local.mean(axis=0)
    _, singular, right = np.linalg.svd(local - center, full_matrices=False)
    variance = np.square(singular)
    if variance.sum() <= 1e-12:
        normal = value - center
    else:
        dimension = int(np.searchsorted(np.cumsum(variance) / variance.sum(), 0.9) + 1)
        basis = right[:dimension]
        delta = value - center
        normal = delta - basis.T @ (basis @ delta)
    return radius, float(np.linalg.norm(normal))


def decode_raw(
    representation: ActionRepresentationModel,
    semantic: torch.Tensor,
    execution: torch.Tensor,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    normalized = representation.decode(torch.cat((semantic, execution), dim=-1)).detach().cpu().numpy()[0]
    raw = normalized.copy()
    raw[:, :6] = raw[:, :6] * std + mean
    return raw


def evaluate_offline(
    *,
    episodes: Sequence[LatentEpisode],
    train_episodes: Sequence[LatentEpisode],
    representation: ActionRepresentationModel,
    semantic_model: SemanticPredictor,
    f1: ExecutionMLP,
    f2: ExecutionMatchedRefinement,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    training_execution = np.concatenate([episode.execution for episode in train_episodes])
    training_full = np.concatenate(
        [np.concatenate((episode.semantic, episode.execution), axis=1) for episode in train_episodes]
    )
    records = []
    correction_rows = []
    semantic_model.eval()
    f1.eval()
    f2.eval()
    representation.eval()
    for episode in episodes:
        text = torch.from_numpy(episode.text_context).float().to(device).unsqueeze(0)
        for horizon in HORIZONS:
            for offset in range(max(0, episode.length - horizon - 1)):
                s_prev = torch.from_numpy(episode.semantic[offset]).float().to(device).unsqueeze(0)
                s_curr = torch.from_numpy(episode.semantic[offset + 1]).float().to(device).unsqueeze(0)
                e_prev_f1 = torch.from_numpy(episode.execution[offset]).float().to(device).unsqueeze(0)
                e_curr_f1 = torch.from_numpy(episode.execution[offset + 1]).float().to(device).unsqueeze(0)
                e_prev_f2 = e_prev_f1.clone()
                e_curr_f2 = e_curr_f1.clone()
                final_states = []
                final_gradients = []
                final_initial = None
                for rollout_step in range(horizon):
                    with torch.no_grad():
                        s_next = semantic_model(s_prev, s_curr, text)
                        context = torch.cat((s_curr, text), dim=-1)
                        f1_next = f1(e_prev_f1, e_curr_f1, context)
                    f2_next, states, gradients = refine_states(f2, e_prev_f2, e_curr_f2, context)
                    final_initial = states[0]
                    final_states = states
                    final_gradients = gradients
                    s_prev, s_curr = s_curr, s_next
                    e_prev_f1, e_curr_f1 = e_curr_f1, f1_next
                    e_prev_f2, e_curr_f2 = e_curr_f2, f2_next
                target_index = offset + 1 + horizon
                target_s = episode.semantic[target_index]
                target_e = episode.execution[target_index]
                target_actions = episode.actions[target_index]
                predictions = {
                    "copy": torch.from_numpy(episode.execution[offset + 1]).float().to(device).unsqueeze(0),
                    "constant_velocity": torch.from_numpy(
                        episode.execution[offset + 1] + episode.execution[offset + 1] - episode.execution[offset]
                    ).float().to(device).unsqueeze(0),
                    "F1": e_curr_f1,
                    "F2": e_curr_f2,
                }
                for method, prediction in predictions.items():
                    execution = prediction.detach().cpu().numpy()[0]
                    semantic = s_curr.detach().cpu().numpy()[0]
                    decoded = decode_raw(representation, s_curr, prediction, mean, std)
                    execution_radius, execution_normal = nearest_geometry(execution, training_execution)
                    full = np.concatenate((semantic, execution))
                    full_radius, full_normal = nearest_geometry(full, training_full)
                    records.append(
                        {
                            "episode_id": episode.episode_id,
                            "task_id": episode.task_id,
                            "horizon": horizon,
                            "offset": offset,
                            "method": method,
                            "execution_mse": float(np.mean(np.square(execution - target_e))),
                            "full_latent_mse": float(
                                np.mean(np.square(full - np.concatenate((target_s, target_e))))
                            ),
                            "decoded_continuous_mse": float(np.mean(np.square(decoded[:, :6] - target_actions[:, :6]))),
                            "gripper_disagreement": float(
                                np.mean(np.sign(decoded[:, 6]) != np.sign(target_actions[:, 6]))
                            ),
                            "execution_knn_radius": execution_radius,
                            "full_knn_radius": full_radius,
                            "execution_normal_distance": execution_normal,
                            "full_normal_distance": full_normal,
                        }
                    )
                target_tensor = torch.from_numpy(target_e).float().to(device).unsqueeze(0)
                correction = e_curr_f2 - final_initial
                target_direction = target_tensor - final_initial
                cosine = F.cosine_similarity(correction, target_direction).item()
                iteration_latent = [float(F.mse_loss(state, target_tensor).item()) for state in final_states]
                iteration_decoded = [
                    float(
                        np.mean(
                            np.square(
                                decode_raw(representation, s_curr, state, mean, std)[:, :6]
                                - target_actions[:, :6]
                            )
                        )
                    )
                    for state in final_states
                ]
                iteration_radius = [
                    nearest_geometry(state.cpu().numpy()[0], training_execution)[0] for state in final_states
                ]
                iteration_normal = [
                    nearest_geometry(state.cpu().numpy()[0], training_execution)[1] for state in final_states
                ]
                correction_rows.append(
                    {
                        "episode_id": episode.episode_id,
                        "task_id": episode.task_id,
                        "horizon": horizon,
                        "offset": offset,
                        "correction_target_cosine": cosine,
                        "positive_cosine": cosine > 0,
                        "gradient_norms": [float(value.norm().item()) for value in final_gradients],
                        "iteration_latent_mse": iteration_latent,
                        "iteration_decoded_continuous_mse": iteration_decoded,
                        "iteration_execution_knn_radius": iteration_radius,
                        "iteration_execution_normal_distance": iteration_normal,
                    }
                )
    summary: dict[str, Any] = {method: {} for method in ("copy", "constant_velocity", "F1", "F2")}
    for method in summary:
        for horizon in HORIZONS:
            selected = [row for row in records if row["method"] == method and row["horizon"] == horizon]
            summary[method][str(horizon)] = {
                key: float(np.mean([row[key] for row in selected]))
                for key in (
                    "execution_mse",
                    "full_latent_mse",
                    "decoded_continuous_mse",
                    "gripper_disagreement",
                    "execution_knn_radius",
                    "full_knn_radius",
                    "execution_normal_distance",
                    "full_normal_distance",
                )
            }
            summary[method][str(horizon)]["sample_count"] = len(selected)
    return summary, records, correction_rows


def episode_auc(records: Sequence[Mapping[str, Any]], method: str, execution_variance: float) -> dict[str, float]:
    result = {}
    episodes = sorted({str(row["episode_id"]) for row in records})
    for episode in episodes:
        points = []
        for horizon in HORIZONS:
            selected = [
                row["execution_mse"]
                for row in records
                if row["episode_id"] == episode and row["method"] == method and row["horizon"] == horizon
            ]
            if selected:
                points.append((horizon, float(np.mean(selected)) / execution_variance))
        result[episode] = float(np.trapz([value for _, value in points], [h for h, _ in points]))
    return result


def paired_bootstrap(left: Mapping[str, float], right: Mapping[str, float], replicates: int, seed: int) -> dict[str, Any]:
    episodes = sorted(set(left) & set(right))
    delta = np.asarray([right[episode] - left[episode] for episode in episodes])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(replicates, len(delta)))
    sampled = delta[indices].mean(axis=1)
    return {
        "cluster": "source_episode",
        "episodes": len(episodes),
        "replicates": replicates,
        "seed": seed,
        "mean_delta_F2_minus_F1": float(delta.mean()),
        "lower_95": float(np.quantile(sampled, 0.025)),
        "upper_95": float(np.quantile(sampled, 0.975)),
        "episode_deltas": {episode: float(value) for episode, value in zip(episodes, delta)},
    }


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out = ROOT / config["experiment"]["output_root"]
    data = ROOT / config["experiment"]["data_root"]
    wave19_out = ROOT / config["sources"]["wave19_output_root"]
    wave19_data = ROOT / config["sources"]["wave19_data_root"]
    free_bytes = shutil.disk_usage(ROOT).free
    write_json(
        out / "wave20_dynamics_disk_record.json",
        {"recorded_at": now(), "phase": "dynamics", "free_bytes": free_bytes},
    )
    if free_bytes < int(config["runtime"]["minimum_free_disk_bytes"]):
        raise RuntimeError("Free disk is below the frozen Wave-20 minimum")
    gate = json.loads((out / "wave20_representation_gate.json").read_text(encoding="utf-8"))
    if gate.get("gate_pass") is not True:
        raise RuntimeError("Representation R-gate did not pass")
    selected_manifest = json.loads(
        (out / "wave20_frozen_libero_representation_manifest.json").read_text(encoding="utf-8")
    )
    split = json.loads((wave19_out / "wave19_dataset_split_manifest.json").read_text(encoding="utf-8"))
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Wave-20 dynamics training requires CUDA")
    representation = build_representation(config, device)
    representation_path = ROOT / selected_manifest["checkpoint"]
    representation.load_state_dict(torch.load(representation_path, map_location=device)["model_state_dict"])
    representation.eval()
    for parameter in representation.parameters():
        parameter.requires_grad_(False)
    normalization_values = selected_manifest["normalization"]
    mean = np.asarray(normalization_values["continuous_mean"], dtype=np.float32)
    std = np.asarray(normalization_values["continuous_std"], dtype=np.float32)
    text_features = np.asarray(
        np.load(wave19_data / "derived/representation/text_features.npz")["features"], dtype=np.float32
    )
    chunk_length = int(config["timebase"]["action_chunk_horizon"])
    train = encode_episodes(
        split_rows(split, "train"), representation, text_features, mean, std, chunk_length, device
    )
    development = encode_episodes(
        split_rows(split, "development"), representation, text_features, mean, std, chunk_length, device
    )
    if not train or not development:
        raise RuntimeError("Train/development latent episodes are empty")
    latent_root = data / "derived/dynamics"
    save_latents(latent_root / "train_latents.npz", train)
    save_latents(latent_root / "development_latents.npz", development)
    arrays = transition_arrays(train)
    set_seed(int(config["experiment"]["seed"]))
    semantic_model = SemanticPredictor(context_dim=16, hidden_dim=64, depth=3).to(device)
    f1 = ExecutionMLP(context_dim=32, hidden_dim=int(config["dynamics"]["hidden_dim"]), depth=int(config["dynamics"]["depth"])).to(device)
    checkpoint_root = out / "dynamics/checkpoints"
    summaries = {
        "semantic": train_predictor(
            name="semantic",
            model=semantic_model,
            arrays=arrays,
            config=config,
            device=device,
            checkpoint=checkpoint_root / "semantic.pt",
        ),
        "F1": train_predictor(
            name="F1", model=f1, arrays=arrays, config=config, device=device, checkpoint=checkpoint_root / "F1.pt"
        ),
    }
    f2 = ExecutionMatchedRefinement(
        f1,
        context_dim=32,
        hidden_dim=int(config["dynamics"]["hidden_dim"]),
        depth=int(config["dynamics"]["depth"]),
        iterations=int(config["dynamics"]["refinement_iterations"]),
        step_size=float(config["dynamics"]["refinement_step_size"]),
    ).to(device)
    if not all(torch.equal(f1.state_dict()[key], f2.initializer.state_dict()[key]) for key in f1.state_dict()):
        raise RuntimeError("F2 initializer differs from exact trained F1")
    summaries["F2"] = train_predictor(
        name="F2", model=f2, arrays=arrays, config=config, device=device, checkpoint=checkpoint_root / "F2.pt"
    )
    if f2.iterations != 4:
        raise RuntimeError("F2 refinement iteration count changed")
    if not all(torch.equal(f1.state_dict()[key], f2.initializer.state_dict()[key]) for key in f1.state_dict()):
        raise RuntimeError("Training changed frozen F2 initializer")
    execution_variance = float(np.var(np.concatenate([episode.execution for episode in train])))
    metrics, records, correction = evaluate_offline(
        episodes=development,
        train_episodes=train,
        representation=representation,
        semantic_model=semantic_model,
        f1=f1,
        f2=f2,
        mean=mean,
        std=std,
        device=device,
    )
    bootstrap = paired_bootstrap(
        episode_auc(records, "F1", execution_variance),
        episode_auc(records, "F2", execution_variance),
        int(config["statistics"]["bootstrap_replicates"]),
        int(config["statistics"]["bootstrap_seed"]),
    )
    cosine = np.asarray([row["correction_target_cosine"] for row in correction])
    f1_normal = np.mean(
        [row["execution_normal_distance"] for row in records if row["method"] == "F1" and row["horizon"] in (4, 8)]
    )
    f2_normal = np.mean(
        [row["execution_normal_distance"] for row in records if row["method"] == "F2" and row["horizon"] in (4, 8)]
    )
    conditions = {
        "O1_F2_AUC_upper_95_below_zero": bootstrap["upper_95"] < 0,
        "O2_F2_H4_execution_MSE_lower": metrics["F2"]["4"]["execution_mse"] < metrics["F1"]["4"]["execution_mse"],
        "O3_F2_H8_execution_MSE_lower": metrics["F2"]["8"]["execution_mse"] < metrics["F1"]["8"]["execution_mse"],
        "O4_F2_H8_decoded_MSE_lower": metrics["F2"]["8"]["decoded_continuous_mse"] < metrics["F1"]["8"]["decoded_continuous_mse"],
        "O5_F2_H8_execution_kNN_lower": metrics["F2"]["8"]["execution_knn_radius"] < metrics["F1"]["8"]["execution_knn_radius"],
        "O6_mean_correction_target_cosine_positive": float(cosine.mean()) > 0,
        "O7_positive_correction_fraction_above_half": float(np.mean(cosine > 0)) > 0.5,
        "O8_F2_normal_distance_lower": float(f2_normal) < float(f1_normal),
    }
    accepted = all(conditions.values())
    offline = {
        "created_at": now(),
        "cross_domain_offline_replication": "ACCEPTED" if accepted else "REJECTED",
        "conditions": conditions,
        "metrics": metrics,
        "bootstrap": bootstrap,
        "mechanism": {
            "mean_correction_target_cosine": float(cosine.mean()),
            "positive_correction_fraction": float(np.mean(cosine > 0)),
            "F1_H4_H8_normal_distance": float(f1_normal),
            "F2_H4_H8_normal_distance": float(f2_normal),
        },
        "training": summaries,
        "test_split_read": False,
        "DEL_instantiated": False,
    }
    write_json(out / "wave20_offline_replication_gate.json", offline)
    write_json(out / "dynamics/offline_metrics.json", metrics)
    write_json(out / "dynamics/offline_records.json", records)
    write_json(out / "dynamics/refinement_correction_alignment.json", correction)
    write_json(out / "publication_tables/offline_dynamics_gate.json", offline)
    write_json(out / "publication_figures_data/offline_horizons.json", metrics)
    write_json(
        out / "publication_figures_data/refinement_mechanism.json",
        {"bootstrap": bootstrap, "mechanism": offline["mechanism"], "correction_rows": correction},
    )
    report = f"""# Wave-20 LIBERO dynamics results

- offline replication gate: `{'PASS' if accepted else 'FAIL'}`
- clustered ΔAUC(F2-F1): `{bootstrap['mean_delta_F2_minus_F1']:.8f}`; 95% CI `[{bootstrap['lower_95']:.8f}, {bootstrap['upper_95']:.8f}]`
- H4 execution MSE F1/F2: `{metrics['F1']['4']['execution_mse']:.8f}` / `{metrics['F2']['4']['execution_mse']:.8f}`
- H8 execution MSE F1/F2: `{metrics['F1']['8']['execution_mse']:.8f}` / `{metrics['F2']['8']['execution_mse']:.8f}`
- H8 decoded continuous MSE F1/F2: `{metrics['F1']['8']['decoded_continuous_mse']:.8f}` / `{metrics['F2']['8']['decoded_continuous_mse']:.8f}`
- mean correction-target cosine: `{cosine.mean():.8f}`; positive fraction `{np.mean(cosine > 0):.6f}`
- DEL run: `no`

All learned models were initialized independently from CALVIN and trained only on the unchanged Wave-19 train split.
"""
    (out / "wave20_dynamics_results.md").write_text(report, encoding="utf-8")
    model_manifest = {
        "frozen_at": now(),
        "independent_from_calvin": True,
        "calvin_weights_loaded": False,
        "representation": {"path": str(representation_path.relative_to(ROOT)), "sha256": sha256_file(representation_path)},
        "semantic": {"path": str((checkpoint_root / "semantic.pt").relative_to(ROOT)), "sha256": sha256_file(checkpoint_root / "semantic.pt")},
        "F1": {"path": str((checkpoint_root / "F1.pt").relative_to(ROOT)), "sha256": sha256_file(checkpoint_root / "F1.pt")},
        "F2": {"path": str((checkpoint_root / "F2.pt").relative_to(ROOT)), "sha256": sha256_file(checkpoint_root / "F2.pt")},
        "F2_exact_F1_initializer": True,
        "F2_refinement_iterations": 4,
        "DEL_forbidden_and_absent": True,
        "test_split_read": False,
        "closed_loop_authorized": accepted,
    }
    write_json(out / "wave20_frozen_model_manifest.json", model_manifest)
    with (out / "exact_commands.sh").open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n# {now()} phase=dynamics-training-and-offline-gate\n"
            "PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 "
            "/home/jinjaguo/anaconda3/envs/libero/bin/python "
            "scripts/dynamics/train_wave20_dynamics.py --config configs/dynamics_8.yaml --device cuda:0\n"
        )
    if not accepted:
        (out / "wave20_offline_replication_rejection.md").write_text(
            "# Wave-20 offline replication rejected\n\nAt least one frozen O1–O8 condition failed. "
            "Per protocol, exact-state closed-loop B0–B5 evaluation did not run.\n",
            encoding="utf-8",
        )
        raise RuntimeError("Wave-20 offline replication gate failed")


if __name__ == "__main__":
    main()

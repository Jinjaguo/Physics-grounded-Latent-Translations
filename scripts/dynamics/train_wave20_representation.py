#!/usr/bin/env python3
"""Train and gate the independent Wave-20 LIBERO action representation.

Purpose
-------
Train six preregistered seeds of the unchanged action-only representation on
the frozen Wave-19 140-episode train split. Pair reconstruction-only R0 with
motor-weighted R1 (2*reconstruction + semantic), train one shuffled-language
control for the frozen semantic delta, and gate only on the fresh 50-episode
Wave-20 confirmation set. The old final test remains unopened.

Parameters
----------
``--config`` selects the frozen Wave-20 YAML and ``--device`` selects the CUDA
device. Dataset scale, conditions, seeds, epochs, and gate rules come from the
frozen configuration and the pre-training manifest written by this script.

Usage
-----
PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/train_wave20_representation.py \
  --config configs/dynamics_8.yaml --device cuda:0

Outputs
-------
Checkpoints and per-seed metrics are written below
``results/dynamics/twentieth_wave/2026-08-14_dynamics_8/representation``.
The top level receives the training/statistical reports, motor diagnostics,
``wave20_representation_gate.json``, and—only after a pass—the frozen
representation manifest. A failed R-gate writes
``wave20_representation_gate_failure.md`` and forbids dynamics training.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
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

from pglt.representation.ema import ParameterEMA
from pglt.representation.losses import symmetric_contrastive_loss
from pglt.representation.model import ActionRepresentationModel


ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ("reconstruction_only", "correct_language", "shuffled_language_control")


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


@dataclass(frozen=True)
class Chunk:
    episode_id: str
    task_id: int
    start: int
    actions: np.ndarray


def load_episode_rows(manifest: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    rows = [dict(row) for row in manifest["episodes"] if row["split"] == split]
    if len(rows) != len({row["episode_id"] for row in rows}):
        raise RuntimeError(f"Duplicate {split} episode")
    return rows


def load_chunks(rows: Sequence[Mapping[str, Any]], chunk_length: int) -> tuple[list[Chunk], dict[str, np.ndarray]]:
    chunks = []
    episodes = {}
    for row in rows:
        episode_id = str(row["episode_id"])
        actions = np.load(ROOT / row["certified_path"] / "actions.npy")
        if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
            raise RuntimeError(f"Invalid certified actions for {episode_id}: {actions.shape}")
        episodes[episode_id] = actions
        for start in range(0, len(actions) - chunk_length + 1, chunk_length):
            chunks.append(Chunk(episode_id, int(row["task_id"]), start, actions[start : start + chunk_length]))
    return chunks, episodes


def normalization(train_episodes: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    values = np.concatenate(list(train_episodes.values()), axis=0)[:, :6]
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 1e-6):
        raise RuntimeError("Invalid train-only continuous action normalization")
    return mean.astype(np.float32), std.astype(np.float32)


def normalized_actions(chunk: Chunk, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    value = chunk.actions.astype(np.float32, copy=True)
    value[:, :6] = (value[:, :6] - mean) / std
    return torch.from_numpy(value)


def unique_task_batches(chunks: Sequence[Chunk], seed: int) -> list[list[int]]:
    by_task: dict[int, list[int]] = defaultdict(list)
    for index, chunk in enumerate(chunks):
        by_task[chunk.task_id].append(index)
    rng = np.random.default_rng(seed)
    for indices in by_task.values():
        rng.shuffle(indices)
    batches = []
    while any(by_task.values()):
        tasks = [task for task, values in by_task.items() if values]
        rng.shuffle(tasks)
        batches.append([by_task[task].pop() for task in tasks])
    return batches


def build_model(config: Mapping[str, Any], device: torch.device) -> ActionRepresentationModel:
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


def derangement(seed: int) -> dict[int, int]:
    shift = 1 + int(seed) % 9
    return {task: (task + shift) % 10 for task in range(10)}


def batch_tensors(
    chunks: Sequence[Chunk],
    indices: Sequence[int],
    mean: np.ndarray,
    std: np.ndarray,
    text_features: np.ndarray,
    condition: str,
    shuffle: Mapping[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    actions = torch.stack([normalized_actions(chunks[index], mean, std) for index in indices]).to(device)
    text_ids = [
        shuffle[chunks[index].task_id]
        if condition == "shuffled_language_control"
        else chunks[index].task_id
        for index in indices
    ]
    text = torch.from_numpy(text_features[text_ids]).float().to(device)
    return actions, text


def representation_objective(
    reconstruction: torch.Tensor,
    semantic: torch.Tensor,
    condition: str,
    reconstruction_weight: float,
) -> torch.Tensor:
    """Apply the frozen R0 or Wave-20 R1/control objective."""

    weight = 1.0 if condition == "reconstruction_only" else reconstruction_weight
    return weight * reconstruction + semantic


def train_one(
    *,
    config: Mapping[str, Any],
    chunks: Sequence[Chunk],
    mean: np.ndarray,
    std: np.ndarray,
    text_features: np.ndarray,
    seed_base: int,
    condition: str,
    device: torch.device,
    output: Path,
) -> ActionRepresentationModel:
    checkpoint = output / "checkpoint_ema.pt"
    if checkpoint.is_file():
        payload = torch.load(checkpoint, map_location=device)
        model = build_model(config, device)
        model.load_state_dict(payload["model_state_dict"])
        return model
    output.mkdir(parents=True, exist_ok=True)
    effective_seed = seed_base
    set_seed(effective_seed)
    model = build_model(config, device)
    ema = ParameterEMA(model, float(config["representation"]["ema_decay"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["representation"]["learning_rate"]),
        weight_decay=float(config["representation"]["weight_decay"]),
    )
    shuffle = derangement(effective_seed + 20)
    log_path = output / "train_log.jsonl"
    log_path.write_text("", encoding="utf-8")
    for epoch in range(1, int(config["representation"]["epochs"]) + 1):
        model.train()
        totals = defaultdict(float)
        count = 0
        for indices in unique_task_batches(chunks, effective_seed + epoch):
            actions, text = batch_tensors(
                chunks, indices, mean, std, text_features, condition, shuffle, device
            )
            result = model(actions, isolate_clip_shared=condition != "reconstruction_only")
            reconstruction = F.mse_loss(result["reconstruction"], actions)
            if condition == "reconstruction_only":
                semantic = reconstruction.new_zeros(())
            else:
                projected = model.project_text(text)
                semantic, _ = symmetric_contrastive_loss(
                    result["clip_semantic_latent"], projected, float(config["representation"]["temperature"])
                )
            loss = representation_objective(
                reconstruction,
                semantic,
                condition,
                float(config["representation"]["reconstruction_weight"]),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            ema.update(model)
            count += len(indices)
            totals["loss"] += float(loss.detach()) * len(indices)
            totals["reconstruction"] += float(reconstruction.detach()) * len(indices)
            totals["semantic"] += float(semantic.detach()) * len(indices)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"epoch": epoch, **{key: value / count for key, value in totals.items()}}) + "\n")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "seed_base": seed_base,
            "effective_seed": effective_seed,
            "condition": condition,
            "epoch": int(config["representation"]["epochs"]),
        },
        output / "checkpoint_raw.pt",
    )
    ema.store(model)
    ema.copy_to(model)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "seed_base": seed_base,
            "effective_seed": effective_seed,
            "condition": condition,
            "epoch": int(config["representation"]["epochs"]),
            "ema_decay": float(config["representation"]["ema_decay"]),
            "ema_updates": ema.updates,
        },
        checkpoint,
    )
    return model


def load_checkpoint_model(
    config: Mapping[str, Any], checkpoint: Path, device: torch.device
) -> ActionRepresentationModel:
    payload = torch.load(checkpoint, map_location=device)
    model = build_model(config, device)
    model.load_state_dict(payload["model_state_dict"])
    return model


@torch.no_grad()
def evaluate(
    model: ActionRepresentationModel,
    chunks: Sequence[Chunk],
    mean: np.ndarray,
    std: np.ndarray,
    text_features: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    model.eval()
    by_episode: dict[str, list[np.ndarray]] = defaultdict(list)
    episode_task = {}
    squared = []
    decoded = []
    gripper_correct = []
    saturation = []
    for start in range(0, len(chunks), 256):
        selected = chunks[start : start + 256]
        actions = torch.stack([normalized_actions(chunk, mean, std) for chunk in selected]).to(device)
        result = model(actions)
        prediction = result["reconstruction"].cpu().numpy()
        latent = F.normalize(result["semantic_latent"], dim=-1).cpu().numpy()
        target = actions.cpu().numpy()
        raw_prediction = prediction.copy()
        raw_prediction[..., :6] = raw_prediction[..., :6] * std + mean
        raw_target = np.stack([chunk.actions for chunk in selected])
        squared.append(np.square(raw_prediction - raw_target))
        decoded.append(raw_prediction)
        gripper_correct.append(np.sign(raw_prediction[..., 6]) == np.sign(raw_target[..., 6]))
        saturation.append(np.abs(raw_prediction[..., :6]) > 1.0)
        for chunk, value in zip(selected, latent):
            by_episode[chunk.episode_id].append(value)
            episode_task[chunk.episode_id] = chunk.task_id
    episode_ids = sorted(by_episode)
    episode_embeddings = np.stack(
        [np.mean(np.stack(by_episode[episode]), axis=0) for episode in episode_ids]
    )
    episode_embeddings /= np.linalg.norm(episode_embeddings, axis=1, keepdims=True).clip(1e-12)
    task_ids = np.asarray([episode_task[episode] for episode in episode_ids], dtype=np.int16)
    text_latents = model.project_text(torch.from_numpy(text_features).float().to(device))
    text_latents = F.normalize(text_latents, dim=-1).cpu().numpy()
    action_scores = episode_embeddings @ text_latents.T
    action_correct = np.argmax(action_scores, axis=1) == task_ids
    prototypes = np.stack([episode_embeddings[task_ids == task].mean(axis=0) for task in range(10)])
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True).clip(1e-12)
    text_correct = np.argmax(text_latents @ prototypes.T, axis=1) == np.arange(10)
    errors = np.concatenate(squared, axis=0)
    grip = np.concatenate(gripper_correct, axis=0)
    sat = np.concatenate(saturation, axis=0)
    metrics = {
        "action_to_text_R1": float(action_correct.mean()),
        "text_to_action_R1": float(text_correct.mean()),
        "macro_action_to_text_R1": float(
            np.mean([action_correct[task_ids == task].mean() for task in range(10)])
        ),
        "continuous_action_mse": float(errors[..., :6].mean()),
        "translation_mse": float(errors[..., :3].mean()),
        "rotation_mse": float(errors[..., 3:6].mean()),
        "gripper_mse": float(errors[..., 6].mean()),
        "per_dimension_mse": errors.mean(axis=(0, 1)).tolist(),
        "gripper_sign_accuracy": float(grip.mean()),
        "decoder_continuous_saturation_fraction": float(sat.mean()),
        "action_clipping_saturation_rate": float(sat.mean()),
        "decoded_action_norm": float(
            np.linalg.norm(np.concatenate(decoded, axis=0)[..., :6], axis=-1).mean()
        ),
        "development_episode_count": len(episode_ids),
        "development_chunk_count": len(chunks),
    }
    arrays = {
        "episode_ids": np.asarray(episode_ids),
        "task_ids": task_ids,
        "episode_embeddings": episode_embeddings.astype(np.float32),
        "text_embeddings": text_latents.astype(np.float32),
        "action_correct": action_correct.astype(np.bool_),
        "text_correct": text_correct.astype(np.bool_),
    }
    return metrics, arrays


def clustered_gate_bootstrap(
    arrays: Mapping[tuple[int, str], Mapping[str, np.ndarray]],
    seeds: Sequence[int],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    reference = arrays[(seeds[0], "correct_language")]
    task_ids = reference["task_ids"]
    rng = np.random.default_rng(seed)
    action_samples = np.empty(replicates, dtype=np.float64)
    text_samples = np.empty(replicates, dtype=np.float64)
    by_task = {task: np.flatnonzero(task_ids == task) for task in range(10)}
    for replicate in range(replicates):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in by_task.values()]
        )
        action_seed_delta = []
        text_seed_delta = []
        for seed_base in seeds:
            condition_action = {}
            condition_text = {}
            for condition in CONDITIONS:
                item = arrays[(seed_base, condition)]
                condition_action[condition] = float(item["action_correct"][sampled].mean())
                prototypes = np.stack(
                    [item["episode_embeddings"][sampled[task_ids[sampled] == task]].mean(axis=0) for task in range(10)]
                )
                prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True).clip(1e-12)
                prediction = np.argmax(item["text_embeddings"] @ prototypes.T, axis=1)
                condition_text[condition] = float(np.mean(prediction == np.arange(10)))
            action_seed_delta.append(
                condition_action["correct_language"]
                - max(condition_action["shuffled_language_control"], condition_action["reconstruction_only"])
            )
            text_seed_delta.append(
                condition_text["correct_language"]
                - max(condition_text["shuffled_language_control"], condition_text["reconstruction_only"])
            )
        action_samples[replicate] = np.mean(action_seed_delta)
        text_samples[replicate] = np.mean(text_seed_delta)
    return {
        "cluster": "source_episode, stratified within official task",
        "replicates": replicates,
        "seed": seed,
        "action_to_text": {
            "mean": float(action_samples.mean()),
            "lower_95": float(np.quantile(action_samples, 0.025)),
            "upper_95": float(np.quantile(action_samples, 0.975)),
        },
        "text_to_action": {
            "mean": float(text_samples.mean()),
            "lower_95": float(np.quantile(text_samples, 0.025)),
            "upper_95": float(np.quantile(text_samples, 0.975)),
        },
    }


def finite_tree(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    if isinstance(value, (float, int, np.number)):
        return bool(np.isfinite(value))
    return True


def final_train_row(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 40 or rows[-1]["epoch"] != 40:
        raise RuntimeError(f"Incomplete 40-epoch log: {path}")
    return rows[-1]


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out = ROOT / config["experiment"]["output_root"]
    fresh_data = ROOT / config["experiment"]["data_root"]
    wave19_out = ROOT / config["sources"]["wave19_output_root"]
    wave19_data = ROOT / config["sources"]["wave19_data_root"]
    free_bytes = shutil.disk_usage(ROOT).free
    write_json(out / "wave20_representation_disk_record.json", {"recorded_at": now(), "free_bytes": free_bytes})
    if free_bytes < int(config["runtime"]["minimum_free_disk_bytes"]):
        raise RuntimeError("Free disk is below the frozen Wave-20 minimum")
    required = [
        out / "wave20_existing_split_freeze.json",
        out / "wave20_seed_preregistration.json",
        out / "wave20_checkpoint_selection_rule.json",
        out / "wave20_fresh_confirmation_manifest.json",
    ]
    if not all(path.is_file() for path in required):
        raise RuntimeError("Wave-20 preregistration or fresh confirmation freeze is missing")
    split = json.loads((wave19_out / "wave19_dataset_split_manifest.json").read_text(encoding="utf-8"))
    confirmation = json.loads((out / "wave20_fresh_confirmation_manifest.json").read_text(encoding="utf-8"))
    if split["counts"] != {"train": 140, "development": 50, "test": 50}:
        raise RuntimeError("Wave-19 split counts changed")
    if confirmation.get("total_certified_episodes") != 50 or confirmation.get("wave19_final_test_read") is not False:
        raise RuntimeError("Fresh confirmation completeness/leakage gate failed")
    train_rows = load_episode_rows(split, "train")
    old_dev_rows = load_episode_rows(split, "development")
    confirmation_rows = [dict(row) for row in confirmation["episodes"]]
    train_ids = {row["episode_id"] for row in train_rows}
    test_ids = set(split["assignments"]["test"])
    confirmation_ids = {row["episode_id"] for row in confirmation_rows}
    if train_ids & confirmation_ids or test_ids & confirmation_ids:
        raise RuntimeError("Fresh confirmation entered an existing Wave-19 split")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Wave-20 representation training requires CUDA")
    text_archive = np.load(wave19_data / "derived/representation/text_features.npz")
    text_features = np.asarray(text_archive["features"], dtype=np.float32)
    chunk_length = int(config["representation"]["action_chunk_horizon"])
    train_chunks, train_episodes = load_chunks(train_rows, chunk_length)
    old_dev_chunks, _ = load_chunks(old_dev_rows, chunk_length)
    confirmation_chunks, _ = load_chunks(confirmation_rows, chunk_length)
    mean, std = normalization(train_episodes)
    wave19_training = json.loads(
        (wave19_out / "wave19_representation_training_manifest.json").read_text(encoding="utf-8")
    )
    frozen_mean = np.asarray(wave19_training["train_only_normalization"]["continuous_mean"])
    frozen_std = np.asarray(wave19_training["train_only_normalization"]["continuous_std"])
    if not np.array_equal(mean.astype(np.float64), frozen_mean) or not np.array_equal(std.astype(np.float64), frozen_std):
        raise RuntimeError("Wave-19 frozen normalization changed")
    seeds = [int(value) for value in config["representation"]["seeds"]]
    training_manifest_path = out / "wave20_representation_training_manifest.json"
    write_json(
        training_manifest_path,
        {
            "frozen_at": now(),
            "written_before_representation_outputs": True,
            "primary_pair": {"R0": "L_rec", "R1": "2.0*L_rec + L_sem"},
            "shuffled_control": "2.0*L_rec + L_sem with task-label derangement; semantic control only",
            "paired_initialization_and_batch_order": True,
            "train_only_normalization": {"continuous_mean": mean.tolist(), "continuous_std": std.tolist()},
            "seeds": seeds,
            "conditions": list(CONDITIONS),
            "epochs": 40,
            "ema_decay": 0.999,
            "checkpoint_rule": "EMA epoch 40",
            "train_episode_count": len(train_rows),
            "train_chunk_count": len(train_chunks),
            "fresh_confirmation_episode_count": len(confirmation_rows),
            "fresh_confirmation_chunk_count": len(confirmation_chunks),
            "old_development_chunk_count": len(old_dev_chunks),
            "confirmation_used_for_training": False,
            "old_final_test_read": False,
            "hyperparameter_sweep": False,
        },
    )
    with (out / "exact_commands.sh").open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n# {now()} phase=R0-R1-shuffled-control-training-confirmation-evaluation-bootstrap-gate\n"
            "PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/jinjaguo/anaconda3/envs/libero/bin/python "
            "scripts/dynamics/train_wave20_representation.py --config configs/dynamics_8.yaml --device cuda:0\n"
        )
    metrics: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    arrays_by_key: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    root = out / "representation"
    for seed_base in seeds:
        for condition in CONDITIONS:
            unit = root / f"seed_{seed_base}" / condition
            train_one(
                config=config,
                chunks=train_chunks,
                mean=mean,
                std=std,
                text_features=text_features,
                seed_base=seed_base,
                condition=condition,
                device=device,
                output=unit,
            )
            print(json.dumps({"seed": seed_base, "condition": condition, "status": "trained"}), flush=True)
    for seed_base in seeds:
        for condition in CONDITIONS:
            unit = root / f"seed_{seed_base}" / condition
            for checkpoint_kind, filename in (("raw", "checkpoint_raw.pt"), ("ema", "checkpoint_ema.pt")):
                model = load_checkpoint_model(config, unit / filename, device)
                fresh_metrics, fresh_arrays = evaluate(
                    model, confirmation_chunks, mean, std, text_features, device
                )
                old_dev_metrics, _ = evaluate(model, old_dev_chunks, mean, std, text_features, device)
                metrics[(seed_base, condition, checkpoint_kind, "fresh")] = fresh_metrics
                metrics[(seed_base, condition, checkpoint_kind, "old_dev")] = old_dev_metrics
                write_json(unit / f"{checkpoint_kind}_fresh_confirmation_metrics.json", fresh_metrics)
                write_json(unit / f"{checkpoint_kind}_old_development_metrics.json", old_dev_metrics)
                if checkpoint_kind == "ema":
                    arrays_by_key[(seed_base, condition)] = fresh_arrays
                    np.savez_compressed(unit / "ema_fresh_confirmation_arrays.npz", **fresh_arrays)
            print(json.dumps({"seed": seed_base, "condition": condition, "status": "evaluated"}), flush=True)
    bootstrap = clustered_gate_bootstrap(
        arrays_by_key,
        seeds,
        int(config["statistics"]["bootstrap_replicates"]),
        int(config["statistics"]["bootstrap_seed"]),
    )
    seed_rows = []
    for seed_base in seeds:
        correct = metrics[(seed_base, "correct_language", "ema", "fresh")]
        shuffled = metrics[(seed_base, "shuffled_language_control", "ema", "fresh")]
        reconstruction = metrics[(seed_base, "reconstruction_only", "ema", "fresh")]
        raw_correct = metrics[(seed_base, "correct_language", "raw", "fresh")]
        raw_reconstruction = metrics[(seed_base, "reconstruction_only", "raw", "fresh")]
        seed_rows.append(
            {
                "seed": seed_base,
                "correct": correct,
                "shuffled_control": shuffled,
                "reconstruction_only": reconstruction,
                "action_to_text_semantic_delta": correct["action_to_text_R1"]
                - max(shuffled["action_to_text_R1"], reconstruction["action_to_text_R1"]),
                "text_to_action_semantic_delta": correct["text_to_action_R1"]
                - max(shuffled["text_to_action_R1"], reconstruction["text_to_action_R1"]),
                "ema_motor_ratio": correct["continuous_action_mse"] / reconstruction["continuous_action_mse"],
                "raw_motor_ratio": raw_correct["continuous_action_mse"] / raw_reconstruction["continuous_action_mse"],
                "ema_gripper_drop": reconstruction["gripper_sign_accuracy"] - correct["gripper_sign_accuracy"],
                "train_epoch40": {
                    condition: final_train_row(root / f"seed_{seed_base}" / condition / "train_log.jsonl")
                    for condition in CONDITIONS
                },
                "old_development": {
                    condition: metrics[(seed_base, condition, "ema", "old_dev")]
                    for condition in CONDITIONS
                },
            }
        )
    mean_action_delta = float(np.mean([row["action_to_text_semantic_delta"] for row in seed_rows]))
    mean_text_delta = float(np.mean([row["text_to_action_semantic_delta"] for row in seed_rows]))
    correct_mse = float(np.mean([row["correct"]["continuous_action_mse"] for row in seed_rows]))
    recon_mse = float(np.mean([row["reconstruction_only"]["continuous_action_mse"] for row in seed_rows]))
    motor_ratio = correct_mse / recon_mse
    correct_grip = float(np.mean([row["correct"]["gripper_sign_accuracy"] for row in seed_rows]))
    recon_grip = float(np.mean([row["reconstruction_only"]["gripper_sign_accuracy"] for row in seed_rows]))
    gripper_drop = recon_grip - correct_grip
    all_outputs_finite = finite_tree(seed_rows) and finite_tree(bootstrap)
    conditions = {
        "semantic_A2T_mean_positive": mean_action_delta > 0.0,
        "semantic_T2A_mean_positive": mean_text_delta > 0.0,
        "semantic_A2T_lower95_positive": bootstrap["action_to_text"]["lower_95"] > 0.0,
        "semantic_T2A_lower95_positive": bootstrap["text_to_action"]["lower_95"] > 0.0,
        "motor_ratio_at_most_1_15": motor_ratio <= float(config["gates"]["motor_ratio_maximum"]),
        "gripper_drop_at_most_0_02": gripper_drop <= float(config["gates"]["gripper_accuracy_drop_maximum"]),
        "all_six_seeds_complete": len(seed_rows) == 6,
        "all_outputs_finite": all_outputs_finite,
        "all_tasks_and_confirmation_episodes_present": len(confirmation_rows) == 50
        and {row["task_id"] for row in confirmation_rows} == set(range(10)),
    }
    ready = all(conditions.values())
    per_task = []
    for task in range(10):
        a2t = []
        t2a = []
        for seed_base in seeds:
            items = {condition: arrays_by_key[(seed_base, condition)] for condition in CONDITIONS}
            mask = items["correct_language"]["task_ids"] == task
            a2t.append(
                float(items["correct_language"]["action_correct"][mask].mean())
                - max(
                    float(items["shuffled_language_control"]["action_correct"][mask].mean()),
                    float(items["reconstruction_only"]["action_correct"][mask].mean()),
                )
            )
            t2a.append(
                float(items["correct_language"]["text_correct"][task])
                - max(
                    float(items["shuffled_language_control"]["text_correct"][task]),
                    float(items["reconstruction_only"]["text_correct"][task]),
                )
            )
        per_task.append({"task_id": task, "A2T_delta": float(np.mean(a2t)), "T2A_delta": float(np.mean(t2a))})
    eligible = [
        row for row in seed_rows
        if row["action_to_text_semantic_delta"] > 0 and row["text_to_action_semantic_delta"] > 0
    ]
    ranked = sorted(eligible, key=lambda row: (row["ema_motor_ratio"], row["seed"]))
    selected_seed = int(ranked[(len(ranked) - 1) // 2]["seed"]) if ready and ranked else None
    gate = {
        "created_at": now(),
        "semantic_A2T_mean_delta": mean_action_delta,
        "semantic_A2T_lower95": bootstrap["action_to_text"]["lower_95"],
        "semantic_T2A_mean_delta": mean_text_delta,
        "semantic_T2A_lower95": bootstrap["text_to_action"]["lower_95"],
        "correct_language_continuous_MSE": correct_mse,
        "reconstruction_only_continuous_MSE": recon_mse,
        "motor_ratio": motor_ratio,
        "correct_gripper_accuracy": correct_grip,
        "reconstruction_gripper_accuracy": recon_grip,
        "gripper_drop": gripper_drop,
        "all_six_seeds_complete": len(seed_rows) == 6,
        "all_outputs_finite": all_outputs_finite,
        "conditions": conditions,
        "gate_pass": ready,
        "selected_seed": selected_seed,
        "seed_rows": seed_rows,
        "per_task_semantic_delta": per_task,
        "bootstrap": bootstrap,
        "fresh_confirmation_episode_count": 50,
        "old_final_test_read": False,
    }
    write_json(out / "wave20_representation_gate.json", gate)
    write_json(out / "publication_tables/representation_gate.json", gate)
    write_json(
        out / "publication_figures_data/representation_motor_margin.json",
        {"seed_rows": seed_rows, "per_task": per_task, "bootstrap": bootstrap},
    )
    correct_dims = np.mean([row["correct"]["per_dimension_mse"] for row in seed_rows], axis=0)
    recon_dims = np.mean([row["reconstruction_only"]["per_dimension_mse"] for row in seed_rows], axis=0)
    diagnostics = {
        "per_dimension_correct_mse": correct_dims.tolist(),
        "per_dimension_reconstruction_mse": recon_dims.tolist(),
        "per_dimension_ratio": (correct_dims / recon_dims).tolist(),
        "translation_ratio": float(
            np.mean([row["correct"]["translation_mse"] for row in seed_rows])
            / np.mean([row["reconstruction_only"]["translation_mse"] for row in seed_rows])
        ),
        "rotation_ratio": float(
            np.mean([row["correct"]["rotation_mse"] for row in seed_rows])
            / np.mean([row["reconstruction_only"]["rotation_mse"] for row in seed_rows])
        ),
        "ema_improves_margin_by_seed": [row["ema_motor_ratio"] < row["raw_motor_ratio"] for row in seed_rows],
        "below_wave19_ratio_by_seed": [row["ema_motor_ratio"] < 1.200444393 for row in seed_rows],
        "seed_rows": seed_rows,
    }
    write_json(out / "publication_figures_data/motor_margin_diagnostics.json", diagnostics)
    training_lines = [
        "# Wave-20 representation training report", "",
        "Six preregistered seeds completed exactly 40 epochs for paired R0/R1 and the shuffled semantic control.",
        "R0 and R1 shared initialization seed, train episodes, and epoch batch order. R1 used exactly `2*L_rec + L_sem`.",
        "Fresh confirmation episodes never entered training; EMA epoch 40 was evaluated without epoch search.", "",
        "| seed | R0 epoch-40 rec | R1 epoch-40 rec | shuffled epoch-40 rec |", "|---:|---:|---:|---:|",
    ]
    for row in seed_rows:
        logs = row["train_epoch40"]
        training_lines.append(
            f"| {row['seed']} | {logs['reconstruction_only']['reconstruction']:.8f} | "
            f"{logs['correct_language']['reconstruction']:.8f} | "
            f"{logs['shuffled_language_control']['reconstruction']:.8f} |"
        )
    (out / "wave20_representation_training_report.md").write_text("\n".join(training_lines) + "\n", encoding="utf-8")
    statistical = f"""# Wave-20 representation statistical report

- bootstrap: 10,000 source-episode clusters, task-stratified, seed 200820
- action-to-text delta: mean `{mean_action_delta:.9f}`, 95% CI `[{bootstrap['action_to_text']['lower_95']:.9f}, {bootstrap['action_to_text']['upper_95']:.9f}]`
- text-to-action delta: mean `{mean_text_delta:.9f}`, 95% CI `[{bootstrap['text_to_action']['lower_95']:.9f}, {bootstrap['text_to_action']['upper_95']:.9f}]`
- motor ratio: `{motor_ratio:.12f}` (gate `<=1.15`)
- gripper accuracy drop: `{gripper_drop:.12f}` (gate `<=0.02`)
- final Wave-19 test read: `false`
"""
    (out / "wave20_representation_statistical_report.md").write_text(statistical, encoding="utf-8")
    motor_lines = [
        "# Wave-20 motor-margin diagnostics", "",
        f"Translation MSE ratio: `{diagnostics['translation_ratio']:.9f}`.",
        f"Rotation MSE ratio: `{diagnostics['rotation_ratio']:.9f}`.",
        f"Per-dimension ratios: `{diagnostics['per_dimension_ratio']}`.", "",
        "| seed | raw ratio | EMA ratio | EMA improves | below Wave-19 aggregate |", "|---:|---:|---:|:---:|:---:|",
    ]
    for row, improves, below in zip(seed_rows, diagnostics["ema_improves_margin_by_seed"], diagnostics["below_wave19_ratio_by_seed"]):
        motor_lines.append(
            f"| {row['seed']} | {row['raw_motor_ratio']:.9f} | {row['ema_motor_ratio']:.9f} | {improves} | {below} |"
        )
    motor_lines.extend(
        ["", "Train epoch-40 reconstruction, old-development MSE, and fresh-confirmation MSE are preserved in the gate seed rows.",
         "No diagnostic was used to change the frozen gate."]
    )
    (out / "wave20_motor_margin_diagnostics.md").write_text("\n".join(motor_lines) + "\n", encoding="utf-8")
    if ready:
        selected = root / f"seed_{selected_seed}" / "correct_language" / "checkpoint_ema.pt"
        write_json(
            out / "wave20_frozen_libero_representation_manifest.json",
            {
                "frozen_at": now(),
                "selected_seed": selected_seed,
                "selection_rule": json.loads((out / "wave20_checkpoint_selection_rule.json").read_text())["rule"],
                "checkpoint": str(selected.relative_to(ROOT)),
                "checkpoint_sha256": sha256_file(selected),
                "decoder_frozen_inside_checkpoint": True,
                "normalization": {"continuous_mean": mean.tolist(), "continuous_std": std.tolist()},
                "text_encoder": config["representation"]["text_encoder"],
                "optimizer_steps_after_freeze": 0,
                "ema_updates_after_freeze": 0,
                "old_final_test_read": False,
            },
        )
    else:
        (out / "wave20_representation_gate_failure.md").write_text(
            "# Wave-20 representation gate failure\n\nThe preregistered fresh-confirmation gate failed. "
            "F1/F2, offline dynamics, and the old final test are forbidden.\n",
            encoding="utf-8",
        )
        raise RuntimeError("Wave-20 representation R-gate failed")


if __name__ == "__main__":
    main()

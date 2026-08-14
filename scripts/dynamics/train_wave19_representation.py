#!/usr/bin/env python3
"""Train and gate the independent Wave-19 LIBERO action representation.

Purpose
-------
Train from scratch six seeds of the 32-D (16 semantic + 16 execution)
action-only representation for correct-language, shuffled-language, and
reconstruction-only conditions. Use only the frozen train split for fitting,
only the development split for the preregistered R-gate, and keep final test
episodes unopened. Select one correct-language EMA checkpoint for dynamics by
the frozen development rule after the multi-seed gate passes.

Parameters
----------
``--config`` selects the frozen Wave-19 YAML and ``--device`` selects the CUDA
device. Dataset scale, conditions, seeds, epochs, and gate rules come from the
frozen configuration and the pre-training manifest written by this script.

Usage
-----
PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/train_wave19_representation.py \
  --config configs/dynamics_7.yaml --device cuda:0

Outputs
-------
Checkpoints and per-seed metrics are written below
``results/dynamics/nineteenth_wave/2026-08-14_dynamics_7/representation``.
The top level receives ``wave19_representation_gate.json``, the representation
report, and the selected frozen checkpoint manifest. A failed R-gate writes
``wave19_representation_gate_failure.md`` and forbids dynamics training.
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
CONDITIONS = ("correct_language", "shuffled_language", "reconstruction_only")
CONDITION_OFFSETS = {"correct_language": 2, "shuffled_language": 1, "reconstruction_only": 0}


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
    text_ids = [shuffle[chunks[index].task_id] if condition == "shuffled_language" else chunks[index].task_id for index in indices]
    text = torch.from_numpy(text_features[text_ids]).float().to(device)
    return actions, text


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
    effective_seed = seed_base + CONDITION_OFFSETS[condition]
    set_seed(effective_seed)
    model = build_model(config, device)
    ema = ParameterEMA(model, float(config["representation"]["ema_decay"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["representation"]["learning_rate"]),
        weight_decay=float(config["representation"]["weight_decay"]),
    )
    shuffle = derangement(effective_seed)
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
            loss = reconstruction + semantic
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
        "gripper_mse": float(errors[..., 6].mean()),
        "per_dimension_mse": errors.mean(axis=(0, 1)).tolist(),
        "gripper_sign_accuracy": float(grip.mean()),
        "decoder_continuous_saturation_fraction": float(sat.mean()),
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
                - max(condition_action["shuffled_language"], condition_action["reconstruction_only"])
            )
            text_seed_delta.append(
                condition_text["correct_language"]
                - max(condition_text["shuffled_language"], condition_text["reconstruction_only"])
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


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out = ROOT / config["experiment"]["output_root"]
    data = ROOT / config["experiment"]["data_root"]
    free_bytes = shutil.disk_usage(ROOT).free
    write_json(
        out / "wave19_representation_disk_record.json",
        {"recorded_at": now(), "phase": "representation", "free_bytes": free_bytes},
    )
    if free_bytes < int(config["runtime"]["minimum_free_disk_bytes"]):
        raise RuntimeError("Free disk is below the frozen Wave-19 minimum")
    split_path = out / "wave19_dataset_split_manifest.json"
    if not split_path.is_file():
        raise RuntimeError("Dataset split must be frozen before representation training")
    manifest = json.loads(split_path.read_text(encoding="utf-8"))
    if not manifest.get("episode_disjoint") or not manifest.get("final_test_unread_for_model_selection"):
        raise RuntimeError("Dataset leakage gate failed")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Wave-19 representation training requires CUDA")
    text_archive = np.load(data / "derived/representation/text_features.npz")
    text_features = np.asarray(text_archive["features"], dtype=np.float32)
    chunk_length = int(config["representation"]["action_chunk_horizon"])
    train_rows = load_episode_rows(manifest, "train")
    development_rows = load_episode_rows(manifest, "development")
    train_chunks, train_episodes = load_chunks(train_rows, chunk_length)
    development_chunks, _ = load_chunks(development_rows, chunk_length)
    mean, std = normalization(train_episodes)
    training_manifest_path = out / "wave19_representation_training_manifest.json"
    if not training_manifest_path.exists():
        write_json(
            training_manifest_path,
            {
                "frozen_at": now(),
                "written_before_representation_outputs": True,
                "certified_action_origin": "policy step 0; collection stabilization wait already removed at certification",
                "train_only_normalization": {"continuous_mean": mean.tolist(), "continuous_std": std.tolist()},
                "gripper_training_target": "actual continuous postprocessed command; unstandardized MSE",
                "gripper_evaluation": "sign accuracy plus continuous MSE",
                "loss": "all-seven-dimension normalized/raw-gripper MSE + isolated symmetric contrastive loss",
                "motor_gate": {
                    "maximum_correct_language_continuous_mse_ratio_vs_reconstruction_only": 1.2,
                    "maximum_gripper_accuracy_drop": 0.05,
                },
                "selected_seed_rule": (
                    "highest minimum of development action-to-text and text-to-action R@1; "
                    "tie by lower continuous MSE, then lower registered seed"
                ),
                "test_split_read": False,
                "train_episode_count": len(train_rows),
                "development_episode_count": len(development_rows),
                "train_chunk_count": len(train_chunks),
                "development_chunk_count": len(development_chunks),
            },
        )
    seeds = [int(value) for value in config["representation"]["seeds"]]
    metrics_by_key = {}
    arrays_by_key = {}
    root = out / "representation"
    for seed_base in seeds:
        for condition in CONDITIONS:
            unit = root / f"seed_{seed_base}" / condition
            model = train_one(
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
            metrics, arrays = evaluate(model, development_chunks, mean, std, text_features, device)
            write_json(unit / "development_metrics.json", metrics)
            np.savez_compressed(unit / "development_arrays.npz", **arrays)
            metrics_by_key[(seed_base, condition)] = metrics
            arrays_by_key[(seed_base, condition)] = arrays
            print(json.dumps({"seed": seed_base, "condition": condition, **metrics}), flush=True)
    bootstrap = clustered_gate_bootstrap(
        arrays_by_key,
        seeds,
        int(config["statistics"]["bootstrap_replicates"]),
        int(config["statistics"]["bootstrap_seed"]),
    )
    seed_rows = []
    for seed_base in seeds:
        correct = metrics_by_key[(seed_base, "correct_language")]
        shuffled = metrics_by_key[(seed_base, "shuffled_language")]
        reconstruction = metrics_by_key[(seed_base, "reconstruction_only")]
        seed_rows.append(
            {
                "seed": seed_base,
                "correct": correct,
                "shuffled": shuffled,
                "reconstruction_only": reconstruction,
                "action_to_text_semantic_delta": correct["action_to_text_R1"]
                - max(shuffled["action_to_text_R1"], reconstruction["action_to_text_R1"]),
                "text_to_action_semantic_delta": correct["text_to_action_R1"]
                - max(shuffled["text_to_action_R1"], reconstruction["text_to_action_R1"]),
            }
        )
    mean_action_delta = float(np.mean([row["action_to_text_semantic_delta"] for row in seed_rows]))
    mean_text_delta = float(np.mean([row["text_to_action_semantic_delta"] for row in seed_rows]))
    correct_mse = float(np.mean([row["correct"]["continuous_action_mse"] for row in seed_rows]))
    recon_mse = float(np.mean([row["reconstruction_only"]["continuous_action_mse"] for row in seed_rows]))
    correct_grip = float(np.mean([row["correct"]["gripper_sign_accuracy"] for row in seed_rows]))
    recon_grip = float(np.mean([row["reconstruction_only"]["gripper_sign_accuracy"] for row in seed_rows]))
    conditions = {
        "six_registered_seeds_completed": len(seed_rows) == 6,
        "positive_mean_action_to_text_delta": mean_action_delta > 0,
        "positive_mean_text_to_action_delta": mean_text_delta > 0,
        "action_to_text_clustered_lower_95_positive": bootstrap["action_to_text"]["lower_95"] > 0,
        "text_to_action_clustered_lower_95_positive": bootstrap["text_to_action"]["lower_95"] > 0,
        "continuous_motor_fidelity_preserved": correct_mse <= 1.2 * recon_mse,
        "gripper_motor_fidelity_preserved": correct_grip >= recon_grip - 0.05,
    }
    ready = all(conditions.values())
    ranking = sorted(
        seeds,
        key=lambda value: (
            -min(
                metrics_by_key[(value, "correct_language")]["action_to_text_R1"],
                metrics_by_key[(value, "correct_language")]["text_to_action_R1"],
            ),
            metrics_by_key[(value, "correct_language")]["continuous_action_mse"],
            value,
        ),
    )
    selected_seed = ranking[0] if ready else None
    gate = {
        "created_at": now(),
        "representation_ready_LIBERO": ready,
        "conditions": conditions,
        "mean_action_to_text_semantic_delta": mean_action_delta,
        "mean_text_to_action_semantic_delta": mean_text_delta,
        "bootstrap": bootstrap,
        "motor": {
            "correct_continuous_mse": correct_mse,
            "reconstruction_only_continuous_mse": recon_mse,
            "correct_gripper_accuracy": correct_grip,
            "reconstruction_only_gripper_accuracy": recon_grip,
        },
        "selected_seed": selected_seed,
        "seed_rows": seed_rows,
        "test_split_read": False,
    }
    write_json(out / "wave19_representation_gate.json", gate)
    write_json(
        out / "publication_figures_data/representation_retrieval_and_motor.json",
        {"seed_rows": seed_rows, "bootstrap": bootstrap, "motor": gate["motor"], "conditions": conditions},
    )
    write_json(out / "publication_tables/representation_gate.json", gate)
    report = f"""# Wave-19 LIBERO representation results

- six-seed R-gate: `{'PASS' if ready else 'FAIL'}`
- mean action-to-text semantic delta: `{mean_action_delta:.6f}`
- mean text-to-action semantic delta: `{mean_text_delta:.6f}`
- clustered lower 95%: action `{bootstrap['action_to_text']['lower_95']:.6f}`, text `{bootstrap['text_to_action']['lower_95']:.6f}`
- continuous MSE, correct / reconstruction-only: `{correct_mse:.8f}` / `{recon_mse:.8f}`
- gripper sign accuracy, correct / reconstruction-only: `{correct_grip:.6f}` / `{recon_grip:.6f}`
- selected correct-language seed: `{selected_seed}`

All selection and R-gate calculations used only episode-disjoint development data. Final test episodes remained unopened.
"""
    (out / "wave19_representation_results.md").write_text(report, encoding="utf-8")
    with (out / "exact_commands.sh").open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n# {now()} phase=representation-training-and-gate\n"
            "PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 "
            "/home/jinjaguo/anaconda3/envs/libero/bin/python "
            "scripts/dynamics/train_wave19_representation.py --config configs/dynamics_7.yaml --device cuda:0\n"
        )
    if not ready:
        (out / "wave19_representation_gate_failure.md").write_text(
            "# Wave-19 representation gate failure\n\nThe independent six-seed LIBERO R-gate failed. "
            "Per protocol, F1/F2 training and closed-loop evaluation did not run.\n",
            encoding="utf-8",
        )
        raise RuntimeError("Wave-19 representation R-gate failed")
    selected = root / f"seed_{selected_seed}" / "correct_language" / "checkpoint_ema.pt"
    write_json(
        out / "wave19_selected_representation_manifest.json",
        {
            "frozen_at": now(),
            "selected_seed": selected_seed,
            "checkpoint": str(selected.relative_to(ROOT)),
            "checkpoint_sha256": sha256_file(selected),
            "selection_rule": json.loads(training_manifest_path.read_text())["selected_seed_rule"],
            "independent_from_calvin": True,
            "test_split_read": False,
            "normalization": {"continuous_mean": mean.tolist(), "continuous_std": std.tolist()},
        },
    )


if __name__ == "__main__":
    main()

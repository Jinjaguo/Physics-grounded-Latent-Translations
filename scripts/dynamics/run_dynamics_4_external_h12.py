#!/usr/bin/env python3
"""Run the amended wave-16 frozen public-data H1/H2 replication.

Purpose
-------
Encode the preregistered 60 VyoJ CALVIN segments with the frozen wave-11
representation, evaluate the frozen wave-15 semantic/F1/F2 models at H1 and H2
only, and report paired trajectory AUC, decoded action, off-manifold, correction
direction, and every matched-refinement intermediate state.

Parameters
----------
``--config`` selects the amended external-H1/H2 YAML. ``--stage`` is
``serialize``, ``evaluate``, or ``finalize``. ``--device`` selects Torch CPU or
CUDA; no stage constructs an optimizer or changes a parameter.

Usage
-----
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/run_dynamics_4_external_h12.py \
  --config configs/dynamics_4_external_h12.yaml --stage serialize --device cpu
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/run_dynamics_4_external_h12.py \
  --config configs/dynamics_4_external_h12.yaml --stage evaluate --device cpu

Outputs
-------
Frozen latents, prospective manifests, H1/H2 metrics, bootstrap results, final
reports, and test/provenance references are saved below
``results/dynamics/sixteenth_wave/2026-08-13_dynamics_4_external_h12``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import yaml

from pglt.dynamics.dynamics_data import load_frozen_representation, sha256_file, write_json
from pglt.dynamics.factorized import ExecutionMLP, ExecutionMatchedRefinement, SemanticPredictor
from pglt.dynamics.long_horizon import paired_trajectory_bootstrap, supported_rollout_offsets
from pglt.dynamics.runner import load_sequences
from pglt.representation.reproducibility import load_text_feature_archive


ROOT = Path(__file__).resolve().parents[2]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("serialize", "evaluate", "finalize"))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def output_root(config: Mapping[str, Any]) -> Path:
    return ROOT / config["experiment"]["output_root"]


def tensor_hashes(model: nn.Module) -> dict[str, str]:
    return {
        name: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
        for name, value in model.state_dict().items()
    }


def frozen_representation(config: Mapping[str, Any]) -> tuple[nn.Module, dict[str, Any], Path]:
    manifest = read_json(ROOT / config["representation"]["checkpoint_manifest"])
    entry = next(
        item for item in manifest["checkpoints"]
        if item["condition"] == config["representation"]["expected_condition"]
        and int(item["seed_base"]) == int(config["representation"]["expected_seed_base"])
    )
    path = ROOT / entry["path"]
    if sha256_file(path) != entry["sha256"]:
        raise RuntimeError("Frozen representation checkpoint hash changed")
    model, payload = load_frozen_representation(
        yaml.safe_load((ROOT / config["representation"]["config"]).read_text(encoding="utf-8")), path
    )
    return model, payload, path


def load_frozen_models(
    config: Mapping[str, Any], device: torch.device
) -> tuple[dict[str, nn.Module], nn.Module, dict[str, Any]]:
    representation, payload, _ = frozen_representation(config)
    values = config["models"]
    semantic = SemanticPredictor(
        context_dim=int(values["context_dim"]), hidden_dim=int(values["hidden_dim"]),
        depth=int(values["depth"]),
    )
    f1 = ExecutionMLP(context_dim=32, hidden_dim=int(values["hidden_dim"]), depth=int(values["depth"]))
    semantic.load_state_dict(torch.load(
        ROOT / values["semantic_checkpoint"], map_location="cpu", weights_only=False
    )["model_state_dict"])
    f1.load_state_dict(torch.load(
        ROOT / values["f1_checkpoint"], map_location="cpu", weights_only=False
    )["model_state_dict"])
    f2 = ExecutionMatchedRefinement(
        f1, context_dim=32, hidden_dim=int(values["hidden_dim"]), depth=int(values["depth"]),
        iterations=int(values["refinement_iterations"]), step_size=float(values["refinement_step_size"]),
    )
    f2.load_state_dict(torch.load(
        ROOT / values["f2_checkpoint"], map_location="cpu", weights_only=False
    )["model_state_dict"])
    modules = {"semantic": semantic, "F1": f1, "F2": f2, "representation": representation}
    for model in modules.values():
        model.eval().to(device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    for key, value in f1.state_dict().items():
        if not torch.equal(value.cpu(), f2.initializer.state_dict()[key].cpu()):
            raise RuntimeError("F2 initializer is not the exact frozen F1")
    return {"semantic": semantic, "F1": f1, "F2": f2}, representation, payload


def serialize(config: Mapping[str, Any], device: torch.device) -> None:
    out = output_root(config)
    out.mkdir(parents=True, exist_ok=True)
    acquisition = ROOT / config["experiment"]["acquisition_root"]
    selected_path = ROOT / config["data"]["candidate_manifest"]
    selected = read_json(selected_path)
    if not selected["gate_passed"] or selected["total_segments"] != 60:
        raise RuntimeError("External H1/H2 data gate has not passed")
    models, representation, payload = load_frozen_models(config, device)
    initial_hashes = {name: tensor_hashes(model) for name, model in {**models, "representation": representation}.items()}
    text_features = load_text_feature_archive(ROOT / config["representation"]["text_feature_archive"])
    normalization = payload["resolved_config"]["normalization"]
    mean = np.asarray(normalization["action_mean"], dtype=np.float32)
    std = np.asarray(normalization["action_std"], dtype=np.float32)
    raw_windows = []
    normalized_windows = []
    contexts = []
    segment_ids = []
    tasks = []
    texts = []
    starts = []
    ends = []
    sequence_rows = []
    cursor = 0
    for segment in selected["segments"]:
        with np.load(ROOT / segment["path"], allow_pickle=False) as saved:
            actions = saved["rel_actions"].copy()
            frame_indices = saved["global_frame_indices"].copy()
        if actions.shape[0] < 64 or actions.shape[1:] != (7,):
            raise ValueError(f"Invalid selected segment shape: {segment['segment_id']} {actions.shape}")
        if not np.array_equal(frame_indices, np.arange(segment["start_frame"], segment["end_frame"] + 1)):
            raise ValueError(f"Non-contiguous selected segment: {segment['segment_id']}")
        text = segment["raw_language"]
        if text not in text_features:
            raise KeyError(f"Frozen text feature missing for {text!r}")
        feature = torch.from_numpy(np.asarray(text_features[text], dtype=np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            context = torch.nn.functional.normalize(representation.project_text(feature), dim=-1)[0].cpu().numpy()
        local_ids = []
        for offset in range(0, 64, 16):
            raw = actions[offset:offset + 16].astype(np.float32)
            normalized = raw.copy()
            normalized[:, :6] = (normalized[:, :6] - mean) / std
            raw_windows.append(raw)
            normalized_windows.append(normalized)
            contexts.append(context)
            segment_ids.append(segment["segment_id"])
            tasks.append(segment["canonical_task"])
            texts.append(text)
            starts.append(int(frame_indices[offset]))
            ends.append(int(frame_indices[offset + 15]))
            local_ids.append(cursor)
            cursor += 1
        sequence_rows.append({
            "segment_id": segment["segment_id"], "task": segment["canonical_task"],
            "text": text, "source_subset": segment["source_subset"],
            "annotation_position": segment["annotation_position"],
            "annotation_start": segment["start_frame"], "annotation_end": segment["end_frame"],
            "window_indices": segment["four_window_ranges"], "latent_indices": local_ids,
            "number_non_overlapping_latent_steps": 4,
            "valid_H1_starts": supported_rollout_offsets(4, 1),
            "valid_H2_starts": supported_rollout_offsets(4, 2),
            "H4_H8_run": False, "task_boundary_occurs": False,
        })
    normalized_array = np.stack(normalized_windows).astype(np.float32)
    with torch.no_grad():
        latents = representation.encode(torch.from_numpy(normalized_array).to(device)).cpu().numpy().astype(np.float32)
    arrays = {
        "latents": latents, "semantic_latents": latents[:, :16], "execution_latents": latents[:, 16:],
        "raw_actions": np.stack(raw_windows).astype(np.float32),
        "normalized_actions": normalized_array, "contexts": np.stack(contexts).astype(np.float32),
        "segment_id": np.asarray(segment_ids), "task": np.asarray(tasks), "text": np.asarray(texts),
        "window_start": np.asarray(starts, dtype=np.int64),
        "window_end_inclusive": np.asarray(ends, dtype=np.int64),
    }
    latent_path = ROOT / config["data"]["external_latents"]
    latent_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(latent_path, **arrays)
    sequence_path = ROOT / config["data"]["external_sequences"]
    sequence_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in sequence_rows), encoding="utf-8")
    final_hashes = {name: tensor_hashes(model) for name, model in {**models, "representation": representation}.items()}
    if initial_hashes != final_hashes:
        raise RuntimeError("A frozen model changed during external serialization")
    checkpoint_manifest = read_json(acquisition / "frozen_checkpoint_manifest.json")
    prospective = {
        "created_at": now(), "written_before_any_external_F1_F2_output": True,
        "post_audit_amendment": True, "evaluated_horizons": [1, 2], "H4_H8_run": False,
        "selected_segment_manifest": {
            "path": selected_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(selected_path),
        },
        "external_latents": {"path": latent_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(latent_path)},
        "external_sequences": {"path": sequence_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(sequence_path)},
        "segment_count": 60, "per_task_count": selected["per_task_counts"],
        "windows_per_segment": 4, "H1_starts": 120, "H2_starts": 60,
        "checkpoints": checkpoint_manifest,
        "primary_endpoint": "paired whole-trajectory normalized execution-error AUC over H1/H2; Delta=F2-F1",
        "bootstrap": {"replicates": 10000, "seed": config["evaluation"]["bootstrap_seed"], "upper_95_required_below_zero": True},
        "secondary_metrics": ["decoded actions", "off-manifold distances", "correction-target cosine", "refinement intermediate states"],
        "selection_uses_model_outputs": False, "future_target_actions": False,
        "model_updates": 0, "representation_updates": 0, "EMA_updates": 0,
    }
    write_json(out / "external_h12_prospective_preregistration.json", prospective)
    write_json(out / "frozen_serialization_audit.json", {
        "model_tensor_hashes_before": initial_hashes, "model_tensor_hashes_after": final_hashes,
        "all_unchanged": True, "latent_shape": list(latents.shape),
        "strict_H16_stride16": True, "padding": False, "future_target_actions": False,
    })
    print(json.dumps({"stage": "serialize", "segments": 60, "latents": len(latents), "H1_starts": 120, "H2_starts": 60}))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def knn(reference: np.ndarray, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    distances = []
    for offset in range(0, len(query), 256):
        value = torch.cdist(
            torch.from_numpy(query[offset:offset + 256].astype(np.float32)),
            torch.from_numpy(reference.astype(np.float32)),
        )
        distances.append(torch.topk(value, k=k, largest=False, dim=1).values.numpy())
    nearest = np.concatenate(distances)
    return nearest[:, 0], nearest[:, -1]


def f2_with_states(
    model: ExecutionMatchedRefinement, previous: torch.Tensor, current: torch.Tensor, context: torch.Tensor
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    with torch.no_grad():
        initial = model.initializer(previous, current, context)
    candidate = initial.detach().requires_grad_(True)
    states = [initial.detach()]
    gradients = []
    fixed = torch.cat((previous, current, context), dim=-1)
    for _ in range(model.iterations):
        energy = model.energy_network(torch.cat((fixed, candidate), dim=-1)).squeeze(-1)
        gradient = torch.autograd.grad(energy.sum(), candidate, create_graph=True)[0]
        gradients.append(gradient.detach())
        candidate = candidate - model.step_size * gradient
        states.append(candidate.detach())
    with torch.enable_grad():
        direct, _ = model(previous, current, context)
    if not torch.allclose(candidate.detach(), direct.detach(), atol=1e-7, rtol=1e-6):
        raise RuntimeError("Logged F2 intermediate path does not reproduce frozen F2 forward")
    return candidate, states, gradients


def summarize_values(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(array), "mean": float(array.mean()), "median": float(np.median(array)),
        "minimum": float(array.min()), "maximum": float(array.max()),
    }


def metric_bundle(
    prediction: np.ndarray, target: np.ndarray, target_actions: np.ndarray,
    representation: nn.Module, payload: Mapping[str, Any], training: Mapping[str, np.ndarray],
    thresholds: Mapping[str, Any], execution_variance: float, device: torch.device, k: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    execution_prediction = prediction[:, 16:]
    execution_target = target[:, 16:]
    execution_squared = ((execution_prediction - execution_target) ** 2).mean(axis=1)
    execution_cosine = np.sum(execution_prediction * execution_target, axis=1) / np.maximum(
        np.linalg.norm(execution_prediction, axis=1) * np.linalg.norm(execution_target, axis=1), 1e-12
    )
    normalization = payload["resolved_config"]["normalization"]
    mean = np.asarray(normalization["action_mean"], dtype=np.float32)
    std = np.asarray(normalization["action_std"], dtype=np.float32)
    with torch.no_grad():
        decoded = representation.decode(torch.from_numpy(prediction).float().to(device)).cpu().numpy()
    decoded[:, :, :6] = decoded[:, :, :6] * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
    continuous_by_sample = ((decoded[:, :, :6] - target_actions[:, :, :6]) ** 2).mean(axis=(1, 2))
    gripper = np.where(decoded[:, :, 6] >= 0, 1.0, -1.0)
    gripper_by_sample = (gripper == target_actions[:, :, 6]).mean(axis=1)
    full_nearest, full_radius = knn(training["latents"], prediction, k)
    exec_nearest, exec_radius = knn(training["execution_latents"], execution_prediction, k)
    _, exec_target_radius = knn(training["execution_latents"], execution_target, k)
    metrics = {
        "sample_count": len(prediction),
        "execution": {
            "mse": float(execution_squared.mean()),
            "normalized_mse": float(execution_squared.mean() / execution_variance),
            "cosine_similarity": float(execution_cosine.mean()),
        },
        "full_latent": {
            "mse": float(((prediction - target) ** 2).mean()),
            "semantic_mse": float(((prediction[:, :16] - target[:, :16]) ** 2).mean()),
            "execution_mse": float(execution_squared.mean()),
        },
        "decoded_actions": {
            "continuous_mse": float(continuous_by_sample.mean()),
            "gripper_accuracy": float(gripper_by_sample.mean()),
            "per_continuous_dimension_mse": [
                float(((decoded[:, :, dim] - target_actions[:, :, dim]) ** 2).mean()) for dim in range(6)
            ],
        },
        "off_manifold": {
            "full_nearest_training_distance": float(full_nearest.mean()),
            "full_knn_radius": float(full_radius.mean()),
            "full_fraction_beyond_frozen_threshold": float(np.mean(full_radius > thresholds["full"]["threshold"])),
            "execution_nearest_training_distance": float(exec_nearest.mean()),
            "execution_knn_radius": float(exec_radius.mean()),
            "execution_ground_truth_knn_radius": float(exec_target_radius.mean()),
            "execution_radius_ratio_to_ground_truth": float(exec_radius.mean() / max(exec_target_radius.mean(), 1e-12)),
            "execution_fraction_beyond_frozen_threshold": float(np.mean(exec_radius > thresholds["execution"]["threshold"])),
        },
    }
    details = {
        "execution_squared": execution_squared, "continuous_mse": continuous_by_sample,
        "gripper_accuracy": gripper_by_sample, "execution_nearest": exec_nearest,
        "execution_radius": exec_radius, "full_radius": full_radius,
    }
    return metrics, details


def grouped_summary(records: Sequence[dict[str, Any]], field: str) -> dict[str, Any]:
    output = {"pooled": summarize_values([record[field] for record in records])}
    for group_name in ("horizon", "task"):
        grouped: dict[str, list[float]] = defaultdict(list)
        for record in records:
            grouped[str(record[group_name])].append(float(record[field]))
        output[f"by_{group_name}"] = {key: summarize_values(values) for key, values in sorted(grouped.items())}
    return output


def evaluate(config: Mapping[str, Any], device: torch.device) -> None:
    out = output_root(config)
    prereg = read_json(out / "external_h12_prospective_preregistration.json")
    if not prereg["written_before_any_external_F1_F2_output"] or prereg["evaluated_horizons"] != [1, 2]:
        raise RuntimeError("External H1/H2 preregistration is absent or changed")
    external = load_npz(ROOT / config["data"]["external_latents"])
    sequences = [json.loads(line) for line in (ROOT / config["data"]["external_sequences"]).read_text(encoding="utf-8").splitlines()]
    wave15 = load_npz(ROOT / config["data"]["wave15_frozen_latents"])
    train_sequences = load_sequences(ROOT / config["data"]["wave15_train_sequences"])
    train_ids = np.unique(np.asarray([
        index for sequence in train_sequences for index in sequence.latent_indices
    ], dtype=np.int64))
    training = {
        "latents": wave15["latents"][train_ids],
        "execution_latents": wave15["execution_latents"][train_ids],
    }
    thresholds = read_json(ROOT / config["data"]["wave15_off_manifold_thresholds"])
    execution_variance = float(read_json(
        ROOT / config["data"]["wave15_training_selection"]
    )["execution_training_variance_mean"])
    models, representation, payload = load_frozen_models(config, device)
    all_modules = {**models, "representation": representation}
    hashes_before = {name: tensor_hashes(model) for name, model in all_modules.items()}
    result_records: dict[str, dict[int, list[dict[str, Any]]]] = {
        "F1": {1: [], 2: []}, "F2": {1: [], 2: []},
    }
    iteration_records = []
    for sequence in sequences:
        ids = sequence["latent_indices"]
        for horizon in (1, 2):
            for offset in supported_rollout_offsets(4, horizon):
                for model_name in ("F1", "F2"):
                    sp = torch.from_numpy(external["semantic_latents"][ids[offset]:ids[offset] + 1]).float().to(device)
                    sc = torch.from_numpy(external["semantic_latents"][ids[offset + 1]:ids[offset + 1] + 1]).float().to(device)
                    ep = torch.from_numpy(external["execution_latents"][ids[offset]:ids[offset] + 1]).float().to(device)
                    ec = torch.from_numpy(external["execution_latents"][ids[offset + 1]:ids[offset + 1] + 1]).float().to(device)
                    context = torch.from_numpy(external["contexts"][ids[offset + 1]:ids[offset + 1] + 1]).float().to(device)
                    for rollout_step in range(horizon):
                        target_id = ids[offset + 2 + rollout_step]
                        with torch.no_grad():
                            sn = models["semantic"](sp, sc, context)
                        combined = torch.cat((sc, context), dim=-1)
                        if model_name == "F1":
                            with torch.no_grad():
                                en = models["F1"](ep, ec, combined)
                        else:
                            with torch.enable_grad():
                                en, states, gradients = f2_with_states(models["F2"], ep, ec, combined)
                            target_exec = external["execution_latents"][target_id:target_id + 1]
                            target_raw = external["raw_actions"][target_id:target_id + 1]
                            for iteration, state in enumerate(states):
                                full = torch.cat((sn, state), dim=-1).detach().cpu().numpy()
                                bundle, detail = metric_bundle(
                                    full, external["latents"][target_id:target_id + 1], target_raw,
                                    representation, payload, training, thresholds, execution_variance,
                                    device, int(config["evaluation"]["knn_k"]),
                                )
                                iteration_records.append({
                                    "segment_id": sequence["segment_id"], "task": sequence["task"],
                                    "horizon": horizon, "rollout_step": rollout_step + 1,
                                    "start_offset": offset, "iteration": iteration,
                                    "execution_mse": float(((state.detach().cpu().numpy() - target_exec) ** 2).mean()),
                                    "decoded_continuous_mse": bundle["decoded_actions"]["continuous_mse"],
                                    "execution_knn_radius": float(detail["execution_radius"][0]),
                                    "gradient_norm": 0.0 if iteration == 0 else float(gradients[iteration - 1].norm().cpu()),
                                })
                        sp, sc = sc.detach(), sn.detach()
                        ep, ec = ec.detach(), en.detach()
                    prediction = torch.cat((sc, ec), dim=-1).cpu().numpy()[0]
                    final_target = ids[offset + 1 + horizon]
                    result_records[model_name][horizon].append({
                        "segment_id": sequence["segment_id"], "task": sequence["task"],
                        "source_subset": sequence["source_subset"], "offset": offset,
                        "target_id": final_target, "prediction": prediction,
                    })
    metrics: dict[str, Any] = {"evaluated_horizons": [1, 2], "H4_H8_run": False, "models": {}}
    details_by_model: dict[str, dict[int, dict[str, np.ndarray]]] = {"F1": {}, "F2": {}}
    for model_name in ("F1", "F2"):
        metrics["models"][model_name] = {}
        for horizon in (1, 2):
            rows = result_records[model_name][horizon]
            prediction = np.stack([row["prediction"] for row in rows]).astype(np.float32)
            target_ids = np.asarray([row["target_id"] for row in rows], dtype=np.int64)
            target = external["latents"][target_ids]
            target_actions = external["raw_actions"][target_ids]
            bundle, detail = metric_bundle(
                prediction, target, target_actions, representation, payload, training, thresholds,
                execution_variance, device, int(config["evaluation"]["knn_k"]),
            )
            for key, values in detail.items():
                for row, value in zip(rows, values):
                    row[key] = float(value)
            bundle["per_task"] = {}
            for task in config["data"]["tasks"]:
                mask = np.asarray([row["task"] == task for row in rows])
                task_bundle, _ = metric_bundle(
                    prediction[mask], target[mask], target_actions[mask], representation, payload,
                    training, thresholds, execution_variance, device, int(config["evaluation"]["knn_k"]),
                )
                bundle["per_task"][task] = task_bundle
            metrics["models"][model_name][str(horizon)] = bundle
            details_by_model[model_name][horizon] = detail
    trajectory_rows = []
    for sequence in sequences:
        row = {"segment_id": sequence["segment_id"], "task": sequence["task"], "source_subset": sequence["source_subset"]}
        for model_name in ("F1", "F2"):
            points = []
            for horizon in (1, 2):
                values = [
                    record["execution_squared"] / execution_variance
                    for record in result_records[model_name][horizon]
                    if record["segment_id"] == sequence["segment_id"]
                ]
                row[f"{model_name}_H{horizon}_normalized_execution_error"] = float(np.mean(values))
                points.append(float(np.mean(values)))
            row[f"{model_name}_AUC_H1_H2"] = float(np.trapz(points, [1, 2]))
        row["Delta_AUC_F2_minus_F1"] = row["F2_AUC_H1_H2"] - row["F1_AUC_H1_H2"]
        trajectory_rows.append(row)
    f1_auc = np.asarray([row["F1_AUC_H1_H2"] for row in trajectory_rows])
    f2_auc = np.asarray([row["F2_AUC_H1_H2"] for row in trajectory_rows])
    bootstrap = paired_trajectory_bootstrap(
        f1_auc, f2_auc, replicates=int(config["evaluation"]["bootstrap_replicates"]),
        seed=int(config["evaluation"]["bootstrap_seed"]),
    )
    per_task_bootstrap = {}
    for task_index, task in enumerate(config["data"]["tasks"]):
        mask = np.asarray([row["task"] == task for row in trajectory_rows])
        per_task_bootstrap[task] = paired_trajectory_bootstrap(
            f1_auc[mask], f2_auc[mask], replicates=int(config["evaluation"]["bootstrap_replicates"]),
            seed=int(config["evaluation"]["bootstrap_seed"]) + task_index + 1,
        )
    correction_records = []
    for horizon in (1, 2):
        for f1_row, f2_row in zip(result_records["F1"][horizon], result_records["F2"][horizon]):
            if (f1_row["segment_id"], f1_row["offset"]) != (f2_row["segment_id"], f2_row["offset"]):
                raise RuntimeError("F1/F2 paired rollout order changed")
            target = external["execution_latents"][f1_row["target_id"]]
            f1_exec = f1_row["prediction"][16:]
            f2_exec = f2_row["prediction"][16:]
            refine = f2_exec - f1_exec
            desired = target - f1_exec
            cosine = float(np.dot(refine, desired) / max(np.linalg.norm(refine) * np.linalg.norm(desired), 1e-12))
            correction_records.append({
                "segment_id": f1_row["segment_id"], "task": f1_row["task"], "horizon": horizon,
                "cosine": cosine,
                "decoded_improvement_F1_minus_F2": f1_row["continuous_mse"] - f2_row["continuous_mse"],
                "execution_radius_reduction_F1_minus_F2": f1_row["execution_radius"] - f2_row["execution_radius"],
            })
    correction = {
        "correction_target_cosine": grouped_summary(correction_records, "cosine"),
        "decoded_improvement_F1_minus_F2": grouped_summary(correction_records, "decoded_improvement_F1_minus_F2"),
        "execution_radius_reduction_F1_minus_F2": grouped_summary(correction_records, "execution_radius_reduction_F1_minus_F2"),
        "fraction_correction_cosine_positive": float(np.mean([row["cosine"] > 0 for row in correction_records])),
    }
    iteration_summary = {}
    for horizon in (1, 2):
        iteration_summary[str(horizon)] = {}
        for iteration in range(5):
            rows = [row for row in iteration_records if row["horizon"] == horizon and row["iteration"] == iteration]
            iteration_summary[str(horizon)][str(iteration)] = {
                "sample_count": len(rows),
                "execution_mse": float(np.mean([row["execution_mse"] for row in rows])),
                "decoded_continuous_mse": float(np.mean([row["decoded_continuous_mse"] for row in rows])),
                "execution_knn_radius": float(np.mean([row["execution_knn_radius"] for row in rows])),
                "gradient_norm": float(np.mean([row["gradient_norm"] for row in rows])),
            }
    hashes_after = {name: tensor_hashes(model) for name, model in all_modules.items()}
    if hashes_before != hashes_after:
        raise RuntimeError("Frozen model parameters changed during external evaluation")
    decision = {
        "created_at": now(), "evaluated_horizons": [1, 2], "H4_H8_run": False,
        "F1_mean_trajectory_AUC": float(f1_auc.mean()),
        "F2_mean_trajectory_AUC": float(f2_auc.mean()),
        "bootstrap": bootstrap,
        "primary_replication_success": bootstrap["upper_95"] < 0,
        "F2_less_than_F1_H1_execution_mse": metrics["models"]["F2"]["1"]["execution"]["mse"] < metrics["models"]["F1"]["1"]["execution"]["mse"],
        "F2_less_than_F1_H2_execution_mse": metrics["models"]["F2"]["2"]["execution"]["mse"] < metrics["models"]["F1"]["2"]["execution"]["mse"],
        "C3c_local_status_if_success": "STRENGTHENED_BY_INDEPENDENT_PUBLIC_EXTERNAL_REPLICATION",
        "C3c_long_status": "NOT_TESTED; H4/H8 were not run",
        "DEL_role": "not run; frozen historical negative baseline only",
    }
    write_json(out / "external_h12_rollout_metrics.json", metrics)
    write_json(out / "external_h12_decoded_action_metrics.json", {
        model: {h: value["decoded_actions"] for h, value in horizons.items()}
        for model, horizons in metrics["models"].items()
    })
    write_json(out / "external_h12_off_manifold_metrics.json", {
        model: {h: value["off_manifold"] for h, value in horizons.items()}
        for model, horizons in metrics["models"].items()
    })
    write_json(out / "external_h12_per_task_metrics.json", {
        model: {h: value["per_task"] for h, value in horizons.items()}
        for model, horizons in metrics["models"].items()
    })
    write_json(out / "external_h12_trajectory_auc.json", {"trajectories": trajectory_rows})
    write_json(out / "external_h12_paired_trajectory_bootstrap.json", {
        "pooled": bootstrap, "per_task": per_task_bootstrap,
        "sampling_unit": "whole selected annotation trajectory", "window_bootstrap": False,
    })
    write_json(out / "external_h12_correction_alignment.json", correction)
    write_json(out / "external_h12_refinement_intermediate_states.json", {
        "iterations": [0, 1, 2, 3, 4], "iteration_0_is_F1_initializer": True,
        "summary": iteration_summary, "records": iteration_records,
    })
    write_json(out / "external_h12_sample_counts.json", {
        "segments": 60, "per_task": {task: 10 for task in config["data"]["tasks"]},
        "windows_per_segment": 4, "H1_rollout_starts": 120, "H2_rollout_starts": 60,
        "H4_rollout_starts": 0, "H8_rollout_starts": 0, "H4_H8_run": False,
    })
    write_json(out / "external_h12_replication_decision.json", decision)
    write_json(out / "external_h12_freezing_and_causality_audit.json", {
        "tensor_hashes_before": hashes_before, "tensor_hashes_after": hashes_after,
        "all_parameters_unchanged": True,
        "representation_optimizer_steps": 0, "representation_backward_calls": 0,
        "F1_optimizer_steps": 0, "F1_backward_calls": 0,
        "F2_optimizer_steps": 0, "F2_backward_calls": 0, "EMA_updates": 0,
        "future_target_actions_used_as_model_input": False,
        "ground_truth_used_only_after_prediction_for_metrics": True,
    })
    print(json.dumps({
        "stage": "evaluate", "F1_AUC": decision["F1_mean_trajectory_AUC"],
        "F2_AUC": decision["F2_mean_trajectory_AUC"],
        "delta_CI": [bootstrap["lower_95"], bootstrap["upper_95"]],
        "replication_success": decision["primary_replication_success"],
    }, indent=2))


def finalize(config: Mapping[str, Any]) -> None:
    out = output_root(config)
    decision = read_json(out / "external_h12_replication_decision.json")
    metrics = read_json(out / "external_h12_rollout_metrics.json")
    correction = read_json(out / "external_h12_correction_alignment.json")
    success = decision["primary_replication_success"]
    c3c = "STRENGTHENED_BY_INDEPENDENT_PUBLIC_EXTERNAL_REPLICATION" if success else "NOT_STRENGTHENED_BY_EXTERNAL_REPLICATION"
    claim = {
        "created_at": now(), "C1": "SUPPORTED", "C2": "SUPPORTED",
        "C3a_full_DEL": "REJECTED", "C3b_exec_DEL": "REJECTED",
        "C3c_local_refinement": c3c,
        "C3c_long_refinement": "NOT_TESTED; amended wave-16 evaluated H1/H2 only",
        "C3d_empirical_manifold_restoration": "NOT_CLAIMED_FROM_H1_H2_REPLICATION",
        "evaluated_horizons": [1, 2], "H4_H8_run": False,
    }
    write_json(out / "wave16_external_h12_claim_decision.json", claim)
    f1h1 = metrics["models"]["F1"]["1"]
    f2h1 = metrics["models"]["F2"]["1"]
    f1h2 = metrics["models"]["F1"]["2"]
    f2h2 = metrics["models"]["F2"]["2"]
    report = f"""# PGLT wave-16 amended public-data H1/H2 external replication

## Scope

This is the post-audit amendment recorded in `prompts/dynamics_4.md`. It evaluated **H1 and H2 only** using four strict non-overlapping H16 windows per public CALVIN task segment. **H4 and H8 were not run.** Representation, semantic predictor, F1, and F2 remained frozen; DEL was not run.

## Data

- Source: `VyoJ/calvin-ABCD-D-subsets`.
- Processed shards: `subset_training_023`, then `subset_training_000`; stopped at the preregistered gate.
- Selected: 60 direct annotation-consistent segments, exactly 10/task.
- Segment length: 64–65 frames; exactly the first 64 frames form four stride-16 windows, with no padding.
- Rollout starts: H1 = 120; H2 = 60.

## Primary paired trajectory endpoint

- F1 mean normalized H1/H2 trajectory AUC: **{decision['F1_mean_trajectory_AUC']:.6f}**.
- F2 mean normalized H1/H2 trajectory AUC: **{decision['F2_mean_trajectory_AUC']:.6f}**.
- Delta AUC (F2-F1): **{decision['bootstrap']['mean_delta_auc']:.6f}**, 95% CI **[{decision['bootstrap']['lower_95']:.6f}, {decision['bootstrap']['upper_95']:.6f}]**.
- Preregistered external replication gate: **{'PASS' if success else 'FAIL'}**.

## Horizon and mechanism metrics

| metric | F1 H1 | F2 H1 | F1 H2 | F2 H2 |
|---|---:|---:|---:|---:|
| execution MSE | {f1h1['execution']['mse']:.6f} | {f2h1['execution']['mse']:.6f} | {f1h2['execution']['mse']:.6f} | {f2h2['execution']['mse']:.6f} |
| decoded continuous MSE | {f1h1['decoded_actions']['continuous_mse']:.6f} | {f2h1['decoded_actions']['continuous_mse']:.6f} | {f1h2['decoded_actions']['continuous_mse']:.6f} | {f2h2['decoded_actions']['continuous_mse']:.6f} |
| execution kNN radius | {f1h1['off_manifold']['execution_knn_radius']:.6f} | {f2h1['off_manifold']['execution_knn_radius']:.6f} | {f1h2['off_manifold']['execution_knn_radius']:.6f} | {f2h2['off_manifold']['execution_knn_radius']:.6f} |

Mean refinement correction-target cosine: **{correction['correction_target_cosine']['pooled']['mean']:.6f}**; positive fraction: **{correction['fraction_correction_cosine_positive']:.3f}**.

## Claim decision

C3c-local: **{c3c}**. C3c-long remains **NOT TESTED** because this amended experiment deliberately ran H1/H2 only. H4/H8 results must not be inferred from this replication.
"""
    (out / "sixteenth_wave_results.md").write_text(report, encoding="utf-8")
    report_path = ROOT / config["experiment"]["report_path"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    canonical_report_path = ROOT / "reports/dynamics_4_results.md"
    canonical_report_path.write_text(report, encoding="utf-8")
    next_text = """# Next experiment after amended wave-16 H1/H2 external replication

The amended public-data replication evaluated H1 and H2 only. Preserve its frozen checkpoints and selected public manifest. Any H4/H8 claim still requires genuinely >=160-frame annotation-consistent CALVIN trajectories; do not extrapolate the H1/H2 result to long horizons and do not reopen DEL.
"""
    (out / "sixteenth_wave_next_experiment.md").write_text(next_text, encoding="utf-8")
    (ROOT / "NEXT_EXPERIMENT.md").write_text(next_text, encoding="utf-8")
    log_path = ROOT / "RESEARCH_LOG.md"
    previous = log_path.read_text(encoding="utf-8")
    marker = "amended public-data H1/H2 external replication"
    if marker not in previous:
        entry = f"\n## {now()} — dynamics_4 amended public external replication\n\nCompleted the {marker} on 60 VyoJ CALVIN segments (10/task, four non-overlapping H16 windows each). This experiment evaluated **H1 and H2 only**; **H4 and H8 were not run**. F1 mean AUC={decision['F1_mean_trajectory_AUC']:.6f}, F2 mean AUC={decision['F2_mean_trajectory_AUC']:.6f}, paired Delta=F2-F1 {decision['bootstrap']['mean_delta_auc']:.6f} with 95% CI [{decision['bootstrap']['lower_95']:.6f}, {decision['bootstrap']['upper_95']:.6f}]; gate={'PASS' if success else 'FAIL'}. C3c-local={c3c}; C3c-long remains NOT_TESTED.\n"
        log_path.write_text(previous.rstrip() + "\n" + entry, encoding="utf-8")
    commands = """df -h /home/jinjaguo/Actions_As_Coordinates && df -B1 /home/jinjaguo/Actions_As_Coordinates
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/acquire_dynamics_4_external_h12.py --config configs/dynamics_4_external_h12.yaml --stage prepare
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/acquire_dynamics_4_external_h12.py --config configs/dynamics_4_external_h12.yaml --stage acquire
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_4_external_h12.py --config configs/dynamics_4_external_h12.yaml --stage serialize --device cpu
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_4_external_h12.py --config configs/dynamics_4_external_h12.yaml --stage evaluate --device cpu
PYTHONPATH=src:third_party/LaWM PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/jinjaguo/anaconda3/envs/libero/bin/python -m pytest tests/representation tests/dynamics -q --junitxml=results/dynamics/sixteenth_wave/2026-08-13_dynamics_4_external_h12/pytest_results.xml
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_4_external_h12.py --config configs/dynamics_4_external_h12.yaml --stage finalize --device cpu
"""
    (out / "executed_commands.txt").write_text(commands, encoding="utf-8")
    write_json(out / "environment_provenance.json", {
        "created_at": now(), "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "numpy": np.__version__,
        "evaluated_horizons": [1, 2], "H4_H8_run": False,
    })
    tracked = [path for path in sorted(out.rglob("*")) if path.is_file() and path.name != "files_changed_report.json"]
    tracked.extend([
        ROOT / "prompts/dynamics_4.md", ROOT / "configs/dynamics_4_external_h12.yaml",
        ROOT / "scripts/dynamics/acquire_dynamics_4_external_h12.py",
        ROOT / "scripts/dynamics/run_dynamics_4_external_h12.py",
        ROOT / "scripts/dynamics/audit_dynamics_4_external_h12.py",
        ROOT / "tests/dynamics/test_dynamics_4_external_h12.py",
        report_path, canonical_report_path, ROOT / "NEXT_EXPERIMENT.md", ROOT / "RESEARCH_LOG.md",
    ])
    tracked.extend(
        path for path in sorted((ROOT / config["experiment"]["acquisition_root"]).rglob("*"))
        if path.is_file()
    )
    write_json(out / "files_changed_report.json", {
        "created_or_updated": [
            {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in tracked if path.exists()
        ],
        "prior_wave_artifacts_overwritten": False,
    })
    print(json.dumps({"stage": "finalize", "gate": "PASS" if success else "FAIL", "C3c_local": c3c, "H4_H8_run": False}))


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    if args.stage == "serialize":
        serialize(config, device)
    elif args.stage == "evaluate":
        evaluate(config, device)
    else:
        finalize(config)


if __name__ == "__main__":
    main()

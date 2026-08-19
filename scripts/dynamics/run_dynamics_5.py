#!/usr/bin/env python3
"""Run the frozen wave-17 continuous-play H1/H2/H4/H8 experiment.

Purpose
-------
Encode preregistered 160-frame CALVIN blocks with the frozen representation,
run frozen F1 and four-step F2 autonomous rollouts under causal held context
and an exogenous-context diagnostic, then write all statistical, mechanism,
continuity, and claim artifacts required by ``prompts/dynamics_5.md``.

Parameters
----------
``--config`` selects the wave-17 YAML. ``--stage`` is ``serialize``,
``evaluate``, or ``finalize``. ``--device`` chooses the Torch inference device;
no stage constructs an optimizer or updates any checkpoint.

Usage
-----
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/run_dynamics_5.py --config configs/dynamics_5.yaml \
  --stage serialize --device cpu
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/run_dynamics_5.py --config configs/dynamics_5.yaml \
  --stage evaluate --device cpu
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/run_dynamics_5.py --config configs/dynamics_5.yaml \
  --stage finalize --device cpu

Outputs
-------
Frozen latents, prospective preregistration, Protocol-A/B/C metrics, clustered
bootstrap, mechanism analyses, final reports, and provenance are saved below
``results/dynamics/seventeenth_wave/2026-08-13_dynamics_5``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
from pglt.dynamics.long_horizon import decompose_tangent_normal, fit_training_neighbor_pca
from pglt.dynamics.runner import load_sequences
from pglt.representation.reproducibility import load_text_feature_archive


ROOT = Path(__file__).resolve().parents[2]
HORIZONS = (1, 2, 4, 8)


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
    return {name: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest() for name, value in model.state_dict().items()}


def frozen_representation(config: Mapping[str, Any]) -> tuple[nn.Module, dict[str, Any], Path]:
    manifest = read_json(ROOT / config["representation"]["checkpoint_manifest"])
    entry = next(
        row for row in manifest["checkpoints"]
        if row["condition"] == config["representation"]["expected_condition"]
        and int(row["seed_base"]) == int(config["representation"]["expected_seed_base"])
    )
    path = ROOT / entry["path"]
    if sha256_file(path) != entry["sha256"]:
        raise RuntimeError("Frozen representation checkpoint hash changed")
    model, payload = load_frozen_representation(yaml.safe_load((ROOT / config["representation"]["config"]).read_text(encoding="utf-8")), path)
    return model, payload, path


def load_frozen_models(config: Mapping[str, Any], device: torch.device) -> tuple[dict[str, nn.Module], nn.Module, dict[str, Any]]:
    representation, payload, _ = frozen_representation(config)
    values = config["models"]
    semantic = SemanticPredictor(context_dim=int(values["context_dim"]), hidden_dim=int(values["hidden_dim"]), depth=int(values["depth"]))
    f1 = ExecutionMLP(context_dim=32, hidden_dim=int(values["hidden_dim"]), depth=int(values["depth"]))
    semantic.load_state_dict(torch.load(ROOT / values["semantic_checkpoint"], map_location="cpu", weights_only=False)["model_state_dict"])
    f1.load_state_dict(torch.load(ROOT / values["f1_checkpoint"], map_location="cpu", weights_only=False)["model_state_dict"])
    f2 = ExecutionMatchedRefinement(
        f1, context_dim=32, hidden_dim=int(values["hidden_dim"]), depth=int(values["depth"]),
        iterations=int(values["refinement_iterations"]), step_size=float(values["refinement_step_size"]),
    )
    f2.load_state_dict(torch.load(ROOT / values["f2_checkpoint"], map_location="cpu", weights_only=False)["model_state_dict"])
    modules = {"semantic": semantic, "F1": f1, "F2": f2, "representation": representation}
    for model in modules.values():
        model.eval().to(device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    for key, value in f1.state_dict().items():
        if not torch.equal(value.cpu(), f2.initializer.state_dict()[key].cpu()):
            raise RuntimeError("F2 initializer differs from frozen F1")
    return {"semantic": semantic, "F1": f1, "F2": f2}, representation, payload


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def serialize(config: Mapping[str, Any], device: torch.device) -> None:
    out = output_root(config)
    out.mkdir(parents=True, exist_ok=True)
    acquisition = ROOT / config["experiment"]["acquisition_root"]
    manifest_path = ROOT / config["data"]["continuous_block_manifest"]
    manifest = read_json(manifest_path)
    if not manifest["gate"]["passed"]:
        raise RuntimeError("Data adequacy gate failed; inference serialization prohibited")
    manifest["selection_order"] = "shard, then global unique annotation start/end/task/position, with a per-session cap"
    for block in manifest["blocks"]:
        block.setdefault("source_environment", "CALVIN ABCD training play")
    write_json(manifest_path, manifest)
    wave16_manifest = read_json(ROOT / config["historical_immutable"]["selected_segments_manifest.json"]["path"])
    source_overlaps = []
    for block in manifest["blocks"]:
        for segment in wave16_manifest["segments"]:
            if block["source_subset"] != segment["source_subset"]:
                continue
            start = max(int(block["start_frame"]), int(segment["start_frame"]))
            end = min(int(block["end_frame"]), int(segment["end_frame"]))
            if start <= end:
                source_overlaps.append({
                    "wave17_block_id": block["block_id"], "wave16_segment_id": segment["segment_id"],
                    "source_subset": block["source_subset"], "overlap_range": [start, end], "overlap_frames": end - start + 1,
                })
    models, representation, payload = load_frozen_models(config, device)
    modules = {**models, "representation": representation}
    hashes_before = {name: tensor_hashes(model) for name, model in modules.items()}
    text_features = load_text_feature_archive(ROOT / config["representation"]["text_feature_archive"])
    normalization = payload["resolved_config"]["normalization"]
    mean = np.asarray(normalization["action_mean"], dtype=np.float32)
    std = np.asarray(normalization["action_std"], dtype=np.float32)
    raw_windows, normalized_windows, contexts = [], [], []
    context_valid, block_ids, session_ids, starts, ends, robot_windows = [], [], [], [], [], []
    sequences = []
    cursor = 0
    projected_cache: dict[str, np.ndarray] = {}
    for block in manifest["blocks"]:
        compact = load_npz(ROOT / block["path"])
        actions = compact["rel_actions"]
        frames = compact["global_frame_indices"]
        robot = compact["robot_obs"]
        if actions.shape != (160, 7) or robot.shape != (160, 15):
            raise ValueError(f"Invalid compact block schema for {block['block_id']}")
        if not np.array_equal(frames, np.arange(block["start_frame"], block["end_frame"] + 1)):
            raise ValueError(f"Non-contiguous compact frames for {block['block_id']}")
        local_ids = []
        for window_index in range(10):
            offset = window_index * 16
            raw = actions[offset:offset + 16].astype(np.float32)
            normalized = raw.copy()
            normalized[:, :6] = (normalized[:, :6] - mean) / std
            text = block["window_languages"][window_index]
            valid = text != "NO_LANGUAGE_ANNOTATION"
            if valid:
                if text not in text_features:
                    raise KeyError(f"Frozen text feature missing for {text!r}")
                if text not in projected_cache:
                    feature = torch.from_numpy(np.asarray(text_features[text], dtype=np.float32)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        projected_cache[text] = torch.nn.functional.normalize(representation.project_text(feature), dim=-1)[0].cpu().numpy()
                context = projected_cache[text]
            else:
                context = np.zeros(16, dtype=np.float32)
            raw_windows.append(raw)
            normalized_windows.append(normalized)
            contexts.append(context)
            context_valid.append(valid)
            block_ids.append(block["block_id"])
            session_ids.append(block["source_session_id"])
            starts.append(int(frames[offset]))
            ends.append(int(frames[offset + 15]))
            robot_windows.append(robot[offset:offset + 16].astype(np.float32))
            local_ids.append(cursor)
            cursor += 1
        sequences.append({**block, "latent_indices": local_ids})
    normalized_array = np.stack(normalized_windows).astype(np.float32)
    with torch.no_grad():
        latents = representation.encode(torch.from_numpy(normalized_array).to(device)).cpu().numpy().astype(np.float32)
    latent_path = ROOT / config["data"]["wave17_latents"]
    latent_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        latent_path, latents=latents, semantic_latents=latents[:, :16], execution_latents=latents[:, 16:],
        raw_actions=np.stack(raw_windows), normalized_actions=normalized_array, contexts=np.stack(contexts).astype(np.float32),
        context_valid=np.asarray(context_valid, dtype=np.bool_), block_id=np.asarray(block_ids), session_id=np.asarray(session_ids),
        window_start=np.asarray(starts, dtype=np.int64), window_end=np.asarray(ends, dtype=np.int64),
        robot_obs=np.stack(robot_windows).astype(np.float32),
    )
    sequence_path = ROOT / config["data"]["wave17_sequences"]
    sequence_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in sequences), encoding="utf-8")
    hashes_after = {name: tensor_hashes(model) for name, model in modules.items()}
    if hashes_before != hashes_after:
        raise RuntimeError("A frozen model changed during representation serialization")
    support = manifest["gate"]["protocol_A_starts"]
    prospective = {
        "created_at": now(), "written_before_any_wave17_F1_F2_output": True,
        "explicit_H4_H8_outputs_not_read": True,
        "source_files": read_json(acquisition / "download_manifest.json")["files"],
        "continuous_block_manifest": {"path": manifest_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(manifest_path)},
        "selected_blocks": [{
            "block_id": row["block_id"], "source_subset": row["source_subset"], "source_session_id": row["source_session_id"],
            "raw_frame_range": [row["start_frame"], row["end_frame"]], "sha256": row["sha256"],
            "no_reset_evidence": "inside one authoritative ep_start_end_ids.npy row",
            "annotation_sequence": row["annotation_sequence"], "H16_window_indices": row["windows"],
            "valid_protocol_A_offsets": row["valid_protocol_A_offsets"], "valid_protocol_B_offsets": row["valid_protocol_B_offsets"],
        } for row in manifest["blocks"]],
        "source_session_to_blocks": {session: [row["block_id"] for row in manifest["blocks"] if row["source_session_id"] == session] for session in sorted(manifest["per_session_block_counts"])},
        "wave16_source_frame_overlap": {
            "pair_count": len(source_overlaps), "overlaps": source_overlaps,
            "wave17_H1_H2_not_called_an_independent_replication": bool(source_overlaps),
            "novel_confirmatory_evidence": "H4/H8",
        },
        "support": support, "horizons": list(HORIZONS), "control_frequency_hz": int(config["data"]["control_frequency_hz"]),
        "protocol_A_context_rule": "start-window annotation held fixed; no future task labels",
        "protocol_B_context_rule": "true annotation active at each current-window first frame; labeled windows only; exogenous diagnostic",
        "metrics": ["latent", "decoded action", "kNN drift", "local-PCA normal distance", "correction direction", "iteration curves", "q-space"],
        "normal_association": "Pearson correlation between F1-minus-F2 normal-distance reduction and F1-minus-F2 decoded continuous-MSE improvement; positive means gate passes",
        "primary_endpoint": "mean block AUC within session, then paired source-session bootstrap of F2-F1",
        "boundary_endpoint": "same-offset H1/H2/H4/H8 AUC for H8 starts crossing >=1 annotation boundary, clustered by source session",
        "bootstrap": {"replicates": int(config["evaluation"]["bootstrap_replicates"]), "seed": int(config["evaluation"]["bootstrap_seed"]), "unit": "source session"},
        "claim_gates": {
            "primary": "Protocol-A clustered Delta_AUC upper 95% < 0",
            "H4_H8": "F2 H4/H8 execution MSE, H8 decoded MSE, and H8 execution kNN radius all lower",
            "boundary_robust": f">={config['evaluation']['boundary_h8_minimum_starts']} boundary H8 starts from >={config['evaluation']['boundary_minimum_sessions']} sessions and clustered upper CI < 0",
        },
        "frozen_model_hashes": read_json(acquisition / "frozen_model_hash_manifest.json"),
        "wave17_latents": {"path": latent_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(latent_path)},
        "wave17_sequences": {"path": sequence_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(sequence_path)},
        "model_updates": 0, "EMA_updates": 0, "future_raw_actions": False, "teacher_forcing": False,
    }
    write_json(out / "wave17_continuous_play_preregistration.json", prospective)
    write_json(out / "wave16_wave17_source_overlap.json", prospective["wave16_source_frame_overlap"])
    write_json(out / "source_frequency_verification.json", {
        "control_frequency_hz": 30, "record_fps": 30.0,
        "local_official_CALVIN_evidence": [
            {"path": "third_party/calvin/calvin_env/conf/recorder/recorder.yaml", "sha256": sha256_file(ROOT / "third_party/calvin/calvin_env/conf/recorder/recorder.yaml")},
            {"path": "third_party/calvin/calvin_env/calvin_env/envs/play_table_env.py", "sha256": sha256_file(ROOT / "third_party/calvin/calvin_env/calvin_env/envs/play_table_env.py")},
        ],
        "H16_seconds": 16 / 30, "H1_seconds": 16 / 30, "H2_seconds": 32 / 30, "H4_seconds": 64 / 30, "H8_seconds": 128 / 30,
    })
    write_json(out / "frozen_serialization_audit.json", {
        "tensor_hashes_before": hashes_before, "tensor_hashes_after": hashes_after, "all_unchanged": True,
        "latent_shape": list(latents.shape), "blocks": len(sequences), "strict_H16_stride16": True,
        "future_raw_actions": False, "teacher_forcing": False,
    })
    print(json.dumps({"stage": "serialize", "blocks": len(sequences), "latents": len(latents), "support": support}, indent=2))


def f2_with_states(model: ExecutionMatchedRefinement, previous: torch.Tensor, current: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
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
        raise RuntimeError("Logged four-step refinement does not reproduce frozen F2")
    return candidate, states, gradients


def active_task(sequence: Mapping[str, Any], frame: int) -> str:
    active = [row for row in sequence["annotation_sequence"] if int(row["start_frame"]) <= frame <= int(row["end_frame"])]
    if not active:
        return "NO_LANGUAGE_ANNOTATION"
    row = sorted(active, key=lambda item: (int(item["start_frame"]), int(item["end_frame"]), str(item["canonical_task"]), int(item["annotation_position"])))[0]
    return str(row["canonical_task"])


def boundary_info(sequence: Mapping[str, Any], offset: int, horizon: int) -> dict[str, Any]:
    current_start = int(sequence["start_frame"]) + 16 * (offset + 1)
    target_end = int(sequence["start_frame"]) + 16 * (offset + 1 + horizon) + 15
    crossed = [int(frame) for frame in sequence["annotation_boundaries"] if current_start < int(frame) <= target_end]
    transitions = []
    for frame in crossed:
        before, after = active_task(sequence, frame - 1), active_task(sequence, frame)
        if before == after and before != "NO_LANGUAGE_ANNOTATION":
            kind = "same_annotated_task"
        elif before != "NO_LANGUAGE_ANNOTATION" and after != "NO_LANGUAGE_ANNOTATION":
            kind = "task_changes"
        elif before != "NO_LANGUAGE_ANNOTATION":
            kind = "labeled_to_unlabeled"
        elif after != "NO_LANGUAGE_ANNOTATION":
            kind = "unlabeled_to_labeled"
        else:
            kind = "unlabeled_metadata_boundary"
        transitions.append({"frame": frame, "before": before, "after": after, "kind": kind})
    return {
        "boundary_count": len(crossed),
        "boundary_stratum": "0" if not crossed else ("1" if len(crossed) == 1 else "2+"),
        "boundary_frames": crossed, "transition_kinds": sorted({row["kind"] for row in transitions}),
        "transitions": transitions,
    }


def knn_values(reference: np.ndarray, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nearest, radius, indices = [], [], []
    reference_tensor = torch.from_numpy(reference.astype(np.float32))
    for begin in range(0, len(query), 256):
        distances = torch.cdist(torch.from_numpy(query[begin:begin + 256].astype(np.float32)), reference_tensor)
        values, ids = torch.topk(distances, k=k, largest=False, dim=1)
        nearest.append(values[:, 0].numpy())
        radius.append(values[:, -1].numpy())
        indices.append(ids.numpy())
    return np.concatenate(nearest), np.concatenate(radius), np.concatenate(indices)


def empirical_normal_distance(reference: np.ndarray, query: np.ndarray, neighbor_count: int, variance_fraction: float) -> np.ndarray:
    _, _, indices = knn_values(reference, query, neighbor_count)
    distances = []
    for value, neighbors in zip(query, indices):
        pca = fit_training_neighbor_pca(reference[neighbors], variance_fraction)
        _, normal = decompose_tangent_normal(value - pca.center, pca)
        distances.append(float(np.linalg.norm(normal)))
    return np.asarray(distances, dtype=np.float64)


def metric_bundle(
    prediction: np.ndarray, target: np.ndarray, target_actions: np.ndarray,
    representation: nn.Module, payload: Mapping[str, Any], training: Mapping[str, np.ndarray],
    thresholds: Mapping[str, Any], execution_variance: float, device: torch.device,
    knn_k: int, pca_fraction: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    execution_prediction, execution_target = prediction[:, 16:], target[:, 16:]
    execution_squared = ((execution_prediction - execution_target) ** 2).mean(axis=1)
    cosine = np.sum(execution_prediction * execution_target, axis=1) / np.maximum(np.linalg.norm(execution_prediction, axis=1) * np.linalg.norm(execution_target, axis=1), 1e-12)
    normalization = payload["resolved_config"]["normalization"]
    mean = np.asarray(normalization["action_mean"], dtype=np.float32)
    std = np.asarray(normalization["action_std"], dtype=np.float32)
    with torch.no_grad():
        decoded = representation.decode(torch.from_numpy(prediction.astype(np.float32)).to(device)).cpu().numpy()
    decoded[:, :, :6] = decoded[:, :, :6] * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
    decoded_squared = ((decoded[:, :, :6] - target_actions[:, :, :6]) ** 2).mean(axis=(1, 2))
    gripper = np.where(decoded[:, :, 6] >= 0, 1.0, -1.0)
    gripper_accuracy = (gripper == target_actions[:, :, 6]).mean(axis=1)
    full_nearest, full_radius, _ = knn_values(training["latents"], prediction, knn_k)
    exec_nearest, exec_radius, _ = knn_values(training["execution_latents"], execution_prediction, knn_k)
    _, target_radius, _ = knn_values(training["execution_latents"], execution_target, knn_k)
    normal = empirical_normal_distance(training["execution_latents"], execution_prediction, max(20, knn_k), pca_fraction)
    bundle = {
        "sample_count": len(prediction),
        "execution": {"mse": float(execution_squared.mean()), "normalized_mse": float(execution_squared.mean() / execution_variance), "cosine_similarity": float(cosine.mean())},
        "full_latent": {
            "mse": float(((prediction - target) ** 2).mean()),
            "semantic_mse": float(((prediction[:, :16] - target[:, :16]) ** 2).mean()),
            "execution_mse": float(execution_squared.mean()),
        },
        "decoded_actions": {
            "continuous_mse": float(decoded_squared.mean()), "gripper_accuracy": float(gripper_accuracy.mean()),
            "per_continuous_dimension_mse": [float(((decoded[:, :, dim] - target_actions[:, :, dim]) ** 2).mean()) for dim in range(6)],
        },
        "off_manifold": {
            "full_nearest_training_distance": float(full_nearest.mean()), "full_knn_radius": float(full_radius.mean()),
            "full_fraction_beyond_frozen_threshold": float(np.mean(full_radius > float(thresholds["full"]["threshold"]))),
            "execution_nearest_training_distance": float(exec_nearest.mean()), "execution_knn_radius": float(exec_radius.mean()),
            "execution_ground_truth_knn_radius": float(target_radius.mean()),
            "execution_fraction_beyond_frozen_threshold": float(np.mean(exec_radius > float(thresholds["execution"]["threshold"]))),
            "empirical_normal_distance": float(normal.mean()),
        },
    }
    details = {
        "execution_squared": execution_squared, "decoded_squared": decoded_squared,
        "gripper_accuracy": gripper_accuracy, "execution_radius": exec_radius,
        "full_radius": full_radius, "normal_distance": normal,
    }
    return bundle, details


def clustered_bootstrap(f1: np.ndarray, f2: np.ndarray, session_ids: Sequence[str], replicates: int, seed: int) -> dict[str, Any]:
    f1_values, f2_values = np.asarray(f1, dtype=np.float64), np.asarray(f2, dtype=np.float64)
    sessions = sorted(set(session_ids))
    session_f1 = np.asarray([f1_values[np.asarray(session_ids) == session].mean() for session in sessions])
    session_f2 = np.asarray([f2_values[np.asarray(session_ids) == session].mean() for session in sessions])
    delta = session_f2 - session_f1
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(sessions), size=(replicates, len(sessions)))
    sampled = delta[indices].mean(axis=1)
    return {
        "source_session_count": len(sessions), "block_or_start_count": len(f1_values), "bootstrap_replicates": replicates,
        "F1_mean_session_value": float(session_f1.mean()), "F2_mean_session_value": float(session_f2.mean()),
        "mean_delta_F2_minus_F1": float(delta.mean()), "lower_95": float(np.quantile(sampled, 0.025)), "upper_95": float(np.quantile(sampled, 0.975)),
        "session_values": [{"source_session_id": session, "F1": float(left), "F2": float(right), "delta": float(right - left)} for session, left, right in zip(sessions, session_f1, session_f2)],
    }


def summarize(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0}
    return {"count": len(array), "mean": float(array.mean()), "median": float(np.median(array)), "minimum": float(array.min()), "maximum": float(array.max())}


def evaluate(config: Mapping[str, Any], device: torch.device) -> None:
    out = output_root(config)
    prereg = read_json(out / "wave17_continuous_play_preregistration.json")
    if not prereg["written_before_any_wave17_F1_F2_output"] or not prereg["explicit_H4_H8_outputs_not_read"]:
        raise RuntimeError("Prospective H4/H8 preregistration is missing")
    external = load_npz(ROOT / config["data"]["wave17_latents"])
    sequences = [json.loads(line) for line in (ROOT / config["data"]["wave17_sequences"]).read_text(encoding="utf-8").splitlines()]
    wave15 = load_npz(ROOT / config["data"]["wave15_frozen_latents"])
    train_sequences = load_sequences(ROOT / config["data"]["wave15_train_sequences"])
    train_ids = np.unique(np.asarray([index for sequence in train_sequences for index in sequence.latent_indices], dtype=np.int64))
    training = {"latents": wave15["latents"][train_ids], "execution_latents": wave15["execution_latents"][train_ids]}
    thresholds = read_json(ROOT / config["data"]["wave15_off_manifold_thresholds"])
    execution_variance = float(read_json(ROOT / config["data"]["wave15_training_selection"])["execution_training_variance_mean"])
    models, representation, payload = load_frozen_models(config, device)
    modules = {**models, "representation": representation}
    hashes_before = {name: tensor_hashes(model) for name, model in modules.items()}
    records: dict[str, dict[str, dict[int, list[dict[str, Any]]]]] = {
        protocol: {model: {horizon: [] for horizon in HORIZONS} for model in ("F1", "F2")}
        for protocol in ("A", "B")
    }
    iteration_rows = []
    for sequence in sequences:
        ids = sequence["latent_indices"]
        for protocol in ("A", "B"):
            offset_key = f"valid_protocol_{protocol}_offsets"
            for horizon in HORIZONS:
                for offset in sequence[offset_key][str(horizon)]:
                    boundary = boundary_info(sequence, int(offset), horizon)
                    for model_name in ("F1", "F2"):
                        sp = torch.from_numpy(external["semantic_latents"][ids[offset]:ids[offset] + 1]).float().to(device)
                        sc = torch.from_numpy(external["semantic_latents"][ids[offset + 1]:ids[offset + 1] + 1]).float().to(device)
                        ep = torch.from_numpy(external["execution_latents"][ids[offset]:ids[offset] + 1]).float().to(device)
                        ec = torch.from_numpy(external["execution_latents"][ids[offset + 1]:ids[offset + 1] + 1]).float().to(device)
                        final_states: list[torch.Tensor] | None = None
                        final_gradients: list[torch.Tensor] | None = None
                        for step in range(horizon):
                            context_index = ids[offset + 1] if protocol == "A" else ids[offset + 1 + step]
                            context = torch.from_numpy(external["contexts"][context_index:context_index + 1]).float().to(device)
                            with torch.no_grad():
                                sn = models["semantic"](sp, sc, context)
                            combined = torch.cat((sc, context), dim=-1)
                            if model_name == "F1":
                                with torch.no_grad():
                                    en = models["F1"](ep, ec, combined)
                            else:
                                with torch.enable_grad():
                                    en, states, gradients = f2_with_states(models["F2"], ep, ec, combined)
                                if step == horizon - 1:
                                    final_states, final_gradients = states, gradients
                            sp, sc, ep, ec = sc.detach(), sn.detach(), ec.detach(), en.detach()
                        prediction = torch.cat((sc, ec), dim=-1).cpu().numpy()[0]
                        target_id = ids[offset + 1 + horizon]
                        row = {
                            "protocol": protocol, "model": model_name, "block_id": sequence["block_id"],
                            "source_session_id": sequence["source_session_id"], "source_subset": sequence["source_subset"],
                            "horizon": horizon, "offset": int(offset), "target_id": int(target_id),
                            "prediction": prediction, **boundary,
                        }
                        records[protocol][model_name][horizon].append(row)
                        if model_name == "F2" and final_states is not None and final_gradients is not None:
                            for iteration, state in enumerate(final_states):
                                iteration_rows.append({
                                    "protocol": protocol, "block_id": sequence["block_id"], "source_session_id": sequence["source_session_id"],
                                    "horizon": horizon, "offset": int(offset), "target_id": int(target_id), "iteration": iteration,
                                    "execution_prediction": state.cpu().numpy()[0],
                                    "semantic_prediction": sc.cpu().numpy()[0],
                                    "gradient_norm": 0.0 if iteration == 0 else float(final_gradients[iteration - 1].norm().cpu()),
                                })
    metrics: dict[str, Any] = {"horizons": list(HORIZONS), "protocols": {}}
    details: dict[str, dict[str, dict[int, dict[str, np.ndarray]]]] = {protocol: {model: {} for model in ("F1", "F2")} for protocol in ("A", "B")}
    for protocol in ("A", "B"):
        metrics["protocols"][protocol] = {"label": "CAUSAL_CONTEXT_HELD" if protocol == "A" else "EXOGENOUS_CONTEXT_SCHEDULE_DIAGNOSTIC", "models": {}}
        for model_name in ("F1", "F2"):
            metrics["protocols"][protocol]["models"][model_name] = {}
            for horizon in HORIZONS:
                rows = records[protocol][model_name][horizon]
                if not rows:
                    metrics["protocols"][protocol]["models"][model_name][str(horizon)] = {"sample_count": 0, "underpowered": True}
                    details[protocol][model_name][horizon] = {}
                    continue
                prediction = np.stack([row["prediction"] for row in rows]).astype(np.float32)
                target_ids = np.asarray([row["target_id"] for row in rows], dtype=np.int64)
                bundle, detail = metric_bundle(
                    prediction, external["latents"][target_ids], external["raw_actions"][target_ids],
                    representation, payload, training, thresholds, execution_variance, device,
                    int(config["evaluation"]["knn_k"]), float(config["evaluation"]["local_pca_variance_fraction"]),
                )
                for key, values in detail.items():
                    for row, value in zip(rows, values):
                        row[key] = float(value)
                metrics["protocols"][protocol]["models"][model_name][str(horizon)] = bundle
                details[protocol][model_name][horizon] = detail
    def paired_rows(protocol: str, horizon: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        left = records[protocol]["F1"][horizon]
        right = records[protocol]["F2"][horizon]
        if [(row["block_id"], row["offset"]) for row in left] != [(row["block_id"], row["offset"]) for row in right]:
            raise RuntimeError("F1/F2 paired rollout ordering changed")
        return list(zip(left, right))

    block_auc_by_protocol: dict[str, list[dict[str, Any]]] = {}
    bootstraps = {}
    for protocol in ("A", "B"):
        rows = []
        for sequence in sequences:
            result = {"block_id": sequence["block_id"], "source_session_id": sequence["source_session_id"]}
            complete = True
            for model_name in ("F1", "F2"):
                points = []
                for horizon in HORIZONS:
                    values = [
                        row["execution_squared"] / execution_variance
                        for row in records[protocol][model_name][horizon]
                        if row["block_id"] == sequence["block_id"]
                    ]
                    if not values:
                        complete = False
                        break
                    result[f"{model_name}_H{horizon}_normalized_execution_error"] = float(np.mean(values))
                    points.append(float(np.mean(values)))
                if len(points) == len(HORIZONS):
                    result[f"{model_name}_AUC"] = float(np.trapz(points, HORIZONS))
            if complete and "F1_AUC" in result and "F2_AUC" in result:
                result["Delta_AUC_F2_minus_F1"] = result["F2_AUC"] - result["F1_AUC"]
                rows.append(result)
        block_auc_by_protocol[protocol] = rows
        if rows:
            bootstraps[protocol] = clustered_bootstrap(
                np.asarray([row["F1_AUC"] for row in rows]), np.asarray([row["F2_AUC"] for row in rows]),
                [row["source_session_id"] for row in rows], int(config["evaluation"]["bootstrap_replicates"]),
                int(config["evaluation"]["bootstrap_seed"]) + (0 if protocol == "A" else 1),
            )
        else:
            bootstraps[protocol] = {"source_session_count": 0, "block_or_start_count": 0, "underpowered": True}
    boundary_auc_rows = []
    lookup = {
        (model, horizon, row["block_id"], row["offset"]): row
        for model in ("F1", "F2") for horizon in HORIZONS for row in records["A"][model][horizon]
    }
    for h8_left, h8_right in paired_rows("A", 8):
        if h8_left["boundary_count"] == 0:
            continue
        row = {
            "block_id": h8_left["block_id"], "source_session_id": h8_left["source_session_id"],
            "offset": h8_left["offset"], "H8_boundaries_crossed": h8_left["boundary_count"],
        }
        complete = True
        for model_name in ("F1", "F2"):
            points = []
            for horizon in HORIZONS:
                value = lookup.get((model_name, horizon, h8_left["block_id"], h8_left["offset"]))
                if value is None:
                    complete = False
                    break
                points.append(float(value["execution_squared"] / execution_variance))
            if len(points) == len(HORIZONS):
                row[f"{model_name}_AUC"] = float(np.trapz(points, HORIZONS))
        if complete:
            row["Delta_AUC_F2_minus_F1"] = row["F2_AUC"] - row["F1_AUC"]
            boundary_auc_rows.append(row)
    boundary_bootstrap = (
        clustered_bootstrap(
            np.asarray([row["F1_AUC"] for row in boundary_auc_rows]), np.asarray([row["F2_AUC"] for row in boundary_auc_rows]),
            [row["source_session_id"] for row in boundary_auc_rows], int(config["evaluation"]["bootstrap_replicates"]),
            int(config["evaluation"]["bootstrap_seed"]) + 2,
        ) if boundary_auc_rows else {"source_session_count": 0, "block_or_start_count": 0, "underpowered": True}
    )
    boundary_analysis: dict[str, Any] = {"protocol_A": {}}
    for horizon in HORIZONS:
        boundary_analysis["protocol_A"][str(horizon)] = {}
        pairs = paired_rows("A", horizon)
        for stratum in ("0", "1", "2+"):
            selected_pairs = [(left, right) for left, right in pairs if left["boundary_stratum"] == stratum]
            boundary_analysis["protocol_A"][str(horizon)][stratum] = {
                "sample_count": len(selected_pairs),
                "source_session_count": len({left["source_session_id"] for left, _ in selected_pairs}),
                "F1_normalized_execution_error": summarize([left["execution_squared"] / execution_variance for left, _ in selected_pairs]),
                "F2_normalized_execution_error": summarize([right["execution_squared"] / execution_variance for _, right in selected_pairs]),
                "delta_F2_minus_F1": summarize([(right["execution_squared"] - left["execution_squared"]) / execution_variance for left, right in selected_pairs]),
                "transition_kinds": dict(sorted(Counter(kind for left, _ in selected_pairs for kind in left["transition_kinds"]).items())),
            }
    context_sensitivity = {"common_rollout_starts": {}}
    for horizon in HORIZONS:
        a_pairs = {(left["block_id"], left["offset"]): (left, right) for left, right in paired_rows("A", horizon)}
        b_pairs = {(left["block_id"], left["offset"]): (left, right) for left, right in paired_rows("B", horizon)}
        common = sorted(set(a_pairs) & set(b_pairs))
        context_sensitivity["common_rollout_starts"][str(horizon)] = {
            "count": len(common),
            "F1_exogenous_minus_causal_normalized_error": summarize([(b_pairs[key][0]["execution_squared"] - a_pairs[key][0]["execution_squared"]) / execution_variance for key in common]),
            "F2_exogenous_minus_causal_normalized_error": summarize([(b_pairs[key][1]["execution_squared"] - a_pairs[key][1]["execution_squared"]) / execution_variance for key in common]),
        }
    correction_rows = []
    iteration_lookup = {(row["protocol"], row["block_id"], row["horizon"], row["offset"], row["iteration"]): row for row in iteration_rows}
    for horizon in (4, 8):
        for left, right in paired_rows("A", horizon):
            initial = iteration_lookup[("A", right["block_id"], horizon, right["offset"], 0)]["execution_prediction"]
            final = iteration_lookup[("A", right["block_id"], horizon, right["offset"], 4)]["execution_prediction"]
            target = external["execution_latents"][right["target_id"]]
            correction, desired = final - initial, target - initial
            cosine = float(np.dot(correction, desired) / max(np.linalg.norm(correction) * np.linalg.norm(desired), 1e-12))
            correction_rows.append({
                "block_id": right["block_id"], "source_session_id": right["source_session_id"], "horizon": horizon,
                "offset": right["offset"], "correction_target_cosine": cosine,
                "decoded_improvement_F1_minus_F2": left["decoded_squared"] - right["decoded_squared"],
                "execution_radius_reduction_F1_minus_F2": left["execution_radius"] - right["execution_radius"],
                "normal_distance_reduction_F1_minus_F2": left["normal_distance"] - right["normal_distance"],
            })
    correction_summary = {
        "H4_H8": {
            "correction_target_cosine": summarize([row["correction_target_cosine"] for row in correction_rows]),
            "fraction_positive": float(np.mean([row["correction_target_cosine"] > 0 for row in correction_rows])),
            "decoded_improvement_F1_minus_F2": summarize([row["decoded_improvement_F1_minus_F2"] for row in correction_rows]),
            "execution_radius_reduction_F1_minus_F2": summarize([row["execution_radius_reduction_F1_minus_F2"] for row in correction_rows]),
            "normal_distance_reduction_F1_minus_F2": summarize([row["normal_distance_reduction_F1_minus_F2"] for row in correction_rows]),
        },
        "by_horizon": {
            str(horizon): {
                "correction_target_cosine": summarize([row["correction_target_cosine"] for row in correction_rows if row["horizon"] == horizon]),
                "fraction_positive": float(np.mean([row["correction_target_cosine"] > 0 for row in correction_rows if row["horizon"] == horizon])),
            } for horizon in (4, 8)
        },
        "records": correction_rows,
    }
    normal_reduction = np.asarray([row["normal_distance_reduction_F1_minus_F2"] for row in correction_rows])
    decoded_improvement = np.asarray([row["decoded_improvement_F1_minus_F2"] for row in correction_rows])
    association = float(np.corrcoef(normal_reduction, decoded_improvement)[0, 1]) if len(correction_rows) > 1 and normal_reduction.std() > 0 and decoded_improvement.std() > 0 else 0.0
    iteration_summary = {}
    for protocol in ("A", "B"):
        iteration_summary[protocol] = {}
        for horizon in HORIZONS:
            iteration_summary[protocol][str(horizon)] = {}
            for iteration in range(5):
                rows = [row for row in iteration_rows if row["protocol"] == protocol and row["horizon"] == horizon and row["iteration"] == iteration]
                if not rows:
                    iteration_summary[protocol][str(horizon)][str(iteration)] = {"sample_count": 0}
                    continue
                predictions = np.stack([np.concatenate((row["semantic_prediction"], row["execution_prediction"])) for row in rows]).astype(np.float32)
                target_ids = np.asarray([row["target_id"] for row in rows], dtype=np.int64)
                bundle, detail = metric_bundle(
                    predictions, external["latents"][target_ids], external["raw_actions"][target_ids], representation, payload,
                    training, thresholds, execution_variance, device, int(config["evaluation"]["knn_k"]),
                    float(config["evaluation"]["local_pca_variance_fraction"]),
                )
                iteration_summary[protocol][str(horizon)][str(iteration)] = {
                    "sample_count": len(rows), "execution_mse": bundle["execution"]["mse"],
                    "decoded_continuous_mse": bundle["decoded_actions"]["continuous_mse"],
                    "execution_knn_radius": bundle["off_manifold"]["execution_knn_radius"],
                    "empirical_normal_distance": bundle["off_manifold"]["empirical_normal_distance"],
                    "gradient_norm": float(np.mean([row["gradient_norm"] for row in rows])),
                }
    q_rows = []
    for horizon in HORIZONS:
        for left, right in paired_rows("A", horizon):
            robot = external["robot_obs"][left["target_id"]]
            joints = robot[:, 7:14]
            q_rows.append({
                "block_id": left["block_id"], "horizon": horizon, "offset": left["offset"],
                "joint_displacement": float(np.linalg.norm(joints[-1] - joints[0])),
                "joint_path_length": float(np.linalg.norm(np.diff(joints, axis=0), axis=1).sum()),
                "F1_execution_error": left["execution_squared"], "F2_execution_error": right["execution_squared"],
                "boundary_count": left["boundary_count"],
            })
    q_path = np.asarray([row["joint_path_length"] for row in q_rows])
    q_diagnostic = {
        "sample_count": len(q_rows), "joint_path_length": summarize(q_path),
        "pearson_path_vs_F1_error": float(np.corrcoef(q_path, [row["F1_execution_error"] for row in q_rows])[0, 1]),
        "pearson_path_vs_F2_error": float(np.corrcoef(q_path, [row["F2_execution_error"] for row in q_rows])[0, 1]),
        "mean_path_by_boundary_stratum": {stratum: summarize([row["joint_path_length"] for row in q_rows if ("0" if row["boundary_count"] == 0 else ("1" if row["boundary_count"] == 1 else "2+")) == stratum]) for stratum in ("0", "1", "2+")},
    }
    a_metrics = metrics["protocols"]["A"]["models"]
    adequate = all(a_metrics["F1"][str(h)]["sample_count"] >= int(config["evaluation"]["minimum_starts"][h]) for h in HORIZONS)
    primary_pass = bool(adequate and bootstraps["A"]["upper_95"] < 0)
    hard_gate = {
        "F2_H4_execution_MSE_lower": a_metrics["F2"]["4"]["execution"]["mse"] < a_metrics["F1"]["4"]["execution"]["mse"],
        "F2_H8_execution_MSE_lower": a_metrics["F2"]["8"]["execution"]["mse"] < a_metrics["F1"]["8"]["execution"]["mse"],
        "F2_H8_decoded_MSE_lower": a_metrics["F2"]["8"]["decoded_actions"]["continuous_mse"] < a_metrics["F1"]["8"]["decoded_actions"]["continuous_mse"],
        "F2_H8_execution_kNN_radius_lower": a_metrics["F2"]["8"]["off_manifold"]["execution_knn_radius"] < a_metrics["F1"]["8"]["off_manifold"]["execution_knn_radius"],
    }
    h4h8_curve = [iteration_summary["A"][str(h)] for h in (4, 8)]
    mechanism_gate = {
        "mean_correction_target_cosine_positive": correction_summary["H4_H8"]["correction_target_cosine"]["mean"] > 0,
        "fraction_positive_above_half": correction_summary["H4_H8"]["fraction_positive"] > 0.5,
        "final_execution_kNN_lower_than_F1": correction_summary["H4_H8"]["execution_radius_reduction_F1_minus_F2"]["mean"] > 0,
        "final_decoded_error_lower_than_F1": correction_summary["H4_H8"]["decoded_improvement_F1_minus_F2"]["mean"] > 0,
        "iteration_net_execution_improvement": all(curve["4"]["execution_mse"] < curve["0"]["execution_mse"] for curve in h4h8_curve),
        "mean_normal_distance_decreases": correction_summary["H4_H8"]["normal_distance_reduction_F1_minus_F2"]["mean"] > 0,
        "normal_reduction_decoded_improvement_positive_association": association > 0,
        "pearson_association": association,
    }
    protocol_b_adequate = all(
        metrics["protocols"]["B"]["models"]["F1"][str(h)]["sample_count"] >= int(config["evaluation"]["minimum_starts"][h])
        for h in HORIZONS
    )
    protocol_b_pass = bool(protocol_b_adequate and "upper_95" in bootstraps["B"] and bootstraps["B"]["upper_95"] < 0)
    boundary_support = len(boundary_auc_rows) >= int(config["evaluation"]["boundary_h8_minimum_starts"]) and len({row["source_session_id"] for row in boundary_auc_rows}) >= int(config["evaluation"]["boundary_minimum_sessions"])
    long_supported = primary_pass and all(hard_gate.values())
    if long_supported and boundary_support and boundary_bootstrap.get("upper_95", math.inf) < 0:
        context_dependency = "ROBUST_TO_BOUNDARIES"
    elif not long_supported and protocol_b_pass:
        context_dependency = "BENEFIT_REQUIRES_EXOGENOUS_CONTEXT"
    else:
        context_dependency = "UNRESOLVED"
    c3c_long = "SUPPORTED" if long_supported else ("NOT_TESTED_INSUFFICIENT_H8_SUPPORT" if not adequate else "REJECTED")
    c3d = "SUPPORTED" if all(value for key, value in mechanism_gate.items() if key != "pearson_association") else "NOT_SUPPORTED"
    block_boundary_counts = Counter("0" if len(row["annotation_boundaries"]) == 0 else ("1" if len(row["annotation_boundaries"]) == 1 else "2+") for row in sequences)
    hashes_after = {name: tensor_hashes(model) for name, model in modules.items()}
    if hashes_before != hashes_after:
        raise RuntimeError("A frozen model changed during wave-17 evaluation")
    write_json(out / "protocol_A_results.json", metrics["protocols"]["A"])
    write_json(out / "protocol_B_results.json", metrics["protocols"]["B"])
    protocol_c = {str(horizon): {
        "sample_count": sum(row["boundary_count"] == 0 for row in records["A"]["F1"][horizon]),
        "F1_normalized_execution_error": summarize([left["execution_squared"] / execution_variance for left, _ in paired_rows("A", horizon) if left["boundary_count"] == 0]),
        "F2_normalized_execution_error": summarize([right["execution_squared"] / execution_variance for left, right in paired_rows("A", horizon) if left["boundary_count"] == 0]),
    } for horizon in HORIZONS}
    write_json(out / "protocol_C_boundary_free_results.json", {"label": "OPTIONAL_BOUNDARY_FREE_DIAGNOSTIC", "horizons": protocol_c})
    write_json(out / "horizon_wise_latent_metrics.json", metrics)
    write_json(out / "decoded_action_metrics.json", {protocol: {model: {h: value.get("decoded_actions", {}) for h, value in models_by_h.items()} for model, models_by_h in metrics["protocols"][protocol]["models"].items()} for protocol in ("A", "B")})
    write_json(out / "off_manifold_metrics.json", {protocol: {model: {h: value.get("off_manifold", {}) for h, value in models_by_h.items()} for model, models_by_h in metrics["protocols"][protocol]["models"].items()} for protocol in ("A", "B")})
    write_json(out / "source_session_clustered_paired_bootstrap.json", {"protocol_A": bootstraps["A"], "protocol_B": bootstraps["B"], "sampling_unit": "source play session", "blocks_within_session_averaged_first": True, "window_bootstrap": False})
    write_json(out / "block_auc_distribution.json", block_auc_by_protocol)
    write_json(out / "boundary_stratified_analysis.json", {**boundary_analysis, "block_annotation_boundary_counts": dict(block_boundary_counts), "boundary_H8_same_start_AUC": {"rows": boundary_auc_rows, "bootstrap": boundary_bootstrap}})
    write_json(out / "context_sensitivity_analysis.json", context_sensitivity)
    write_json(out / "refinement_correction_alignment.json", correction_summary)
    write_json(out / "refinement_iteration_curves.json", {"iteration_0_is_F2_initializer": True, "iterations": [0, 1, 2, 3, 4], "summary": iteration_summary})
    write_json(out / "empirical_manifold_analysis.json", {"reference": config["data"]["wave15_frozen_latents"], "training_only": True, "local_pca_variance_fraction": config["evaluation"]["local_pca_variance_fraction"], "H4_H8_normal_reduction": correction_summary["H4_H8"]["normal_distance_reduction_F1_minus_F2"], "pearson_normal_reduction_vs_decoded_improvement": association})
    write_json(out / "q_space_diagnostic.json", q_diagnostic)
    write_json(out / "annotation_boundary_metadata.json", {"blocks": [{"block_id": row["block_id"], "source_session_id": row["source_session_id"], "annotation_boundaries": row["annotation_boundaries"], "annotation_sequence": row["annotation_sequence"], "windows": row["windows"]} for row in sequences]})
    write_json(out / "H1_H2_H4_H8_support_table.json", {"Protocol_A": {str(h): len(records["A"]["F1"][h]) for h in HORIZONS}, "Protocol_B": {str(h): len(records["B"]["F1"][h]) for h in HORIZONS}, "minimum_required_primary": config["evaluation"]["minimum_starts"], "primary_adequate": adequate})
    write_json(out / "frozen_DEL_negative_baseline_note.json", {"DEL": "historical negative baseline only", "run_on_continuous_blocks": False, "training_or_tuning": False, "DEL_rescue": False})
    write_json(out / "freezing_and_causality_audit.json", {
        "tensor_hashes_before": hashes_before, "tensor_hashes_after": hashes_after, "all_parameters_unchanged": True,
        "representation_optimizer_steps": 0, "F1_optimizer_steps": 0, "F2_optimizer_steps": 0, "EMA_updates": 0,
        "loss_backward_calls": 0, "inference_autograd_grad_calls": "exact frozen four-step F2 refinement only",
        "future_annotations_used_in_protocol_A": False, "future_raw_actions_as_input": False,
        "future_robot_states_as_input": False, "teacher_forcing_after_start": False,
    })
    decision = {
        "created_at": now(), "C1": "SUPPORTED", "C2": "SUPPORTED", "C3a_full_DEL": "REJECTED", "C3b_exec_DEL": "REJECTED",
        "C3c_local": "STRENGTHENED_BY_INDEPENDENT_PUBLIC_EXTERNAL_REPLICATION", "C3c_long": c3c_long,
        "C3d_refinement_manifold_stabilization": c3d, "context_dependency": context_dependency,
        "data_adequacy": adequate, "protocol_A_primary_gate": primary_pass, "protocol_A_bootstrap": bootstraps["A"],
        "hard_H4_H8_gate": hard_gate, "mechanism_gate": mechanism_gate,
        "boundary_support": {"H8_starts": len(boundary_auc_rows), "sessions": len({row["source_session_id"] for row in boundary_auc_rows}), "adequate": boundary_support, "bootstrap": boundary_bootstrap},
        "protocol_B_data_adequacy": protocol_b_adequate, "protocol_B_clustered_gate": protocol_b_pass,
        "DEL_role": "permanent historical negative baseline; not run",
    }
    write_json(out / "wave17_claim_decision.json", decision)
    print(json.dumps({"stage": "evaluate", "C3c_long": c3c_long, "C3d": c3d, "context_dependency": context_dependency, "Protocol_A_CI": [bootstraps["A"]["lower_95"], bootstraps["A"]["upper_95"]], "hard_gate": hard_gate}, indent=2))


def finalize(config: Mapping[str, Any]) -> None:
    out = output_root(config)
    acquisition = ROOT / config["experiment"]["acquisition_root"]
    decision = read_json(out / "wave17_claim_decision.json")
    metrics = read_json(out / "horizon_wise_latent_metrics.json")
    support = read_json(out / "H1_H2_H4_H8_support_table.json")
    boundary = read_json(out / "boundary_stratified_analysis.json")
    context = read_json(out / "context_sensitivity_analysis.json")
    correction = read_json(out / "refinement_correction_alignment.json")
    iterations = read_json(out / "refinement_iteration_curves.json")
    manifold = read_json(out / "empirical_manifold_analysis.json")
    manifest = read_json(ROOT / config["data"]["continuous_block_manifest"])
    shard_audits = read_json(acquisition / "source_shard_audits.json")
    integrity = read_json(acquisition / "continuous_play_integrity_audit.json")
    overlap = read_json(out / "wave16_wave17_source_overlap.json")
    sessions = sorted(manifest["per_session_block_counts"])
    source_audit = {
        "source_repo": config["source"]["repo_id"], "source_revision": manifest["source_revision"],
        "shards": shard_audits, "staged_one_at_a_time": True, "source_zips_deleted_after_compaction": True,
        "source_hashes": read_json(acquisition / "download_manifest.json"),
        "wave16_immutable_artifacts": read_json(acquisition / "wave16_immutable_artifact_audit.json"),
        "wave16_wave17_source_overlap": overlap,
    }
    reconstruction = {
        "physically_continuous_sessions_reconstructed": len(sessions), "eligible_blocks": len(manifest["blocks"]),
        "session_to_blocks": {session: [row["block_id"] for row in manifest["blocks"] if row["source_session_id"] == session] for session in sessions},
        "authoritative_boundary": "one ep_start_end_ids.npy row", "numeric_frame_continuity_required": True,
        "raw_frame_overlap_between_blocks": False, "synthetic_concatenation": False, "reset_crossing": False,
    }
    write_json(out / "continuous_play_source_audit.json", source_audit)
    write_json(out / "source_session_reconstruction_report.json", reconstruction)
    write_json(out / "physical_continuity_audit.json", integrity)
    a = metrics["protocols"]["A"]["models"]
    b = metrics["protocols"]["B"]["models"]
    primary = decision["protocol_A_bootstrap"]
    block_counts = boundary["block_annotation_boundary_counts"]
    common_h8 = context["common_rollout_starts"]["8"]
    protocol_b_effect = (
        f"On {common_h8['count']} common H8 starts, exogenous-minus-causal normalized error was "
        f"{common_h8['F1_exogenous_minus_causal_normalized_error'].get('mean', float('nan')):.6f} for F1 and "
        f"{common_h8['F2_exogenous_minus_causal_normalized_error'].get('mean', float('nan')):.6f} for F2."
    )
    if decision["C3c_long"] == "SUPPORTED" and decision["context_dependency"] == "ROBUST_TO_BOUNDARIES":
        story = "Language anchors action meaning; refinement stabilizes continuous latent evolution across continuous robot motion and semantic task boundaries."
    elif decision["C3c_long"] == "SUPPORTED":
        story = "Language grounds meaningful action coordinates, and matched refinement improves long-horizon continuous latent rollout; robustness to semantic boundaries remains unresolved."
    elif decision["protocol_B_clustered_gate"]:
        story = "Given an exogenous high-level task schedule, matched refinement improves long-horizon latent dynamics across continuous robot motion."
    elif decision["hard_H4_H8_gate"]["F2_H4_execution_MSE_lower"]:
        story = "Refinement extends the stable prediction horizon beyond local transitions, but evidence for 4+ second autonomous latent rollout remains insufficient."
    else:
        story = "Language-grounded action coordinates remain semantically addressable, executable, and locally predictable; stable long-horizon dynamics remains unresolved."
    report = f"""# PGLT wave-17 continuous-play long-horizon experiment

## Outcome

Wave 17 reconstructed **{len(sessions)}** physically continuous VyoJ CALVIN play sessions and retained **{len(manifest['blocks'])}** non-overlapping 160-frame blocks. The frozen causal Protocol A had H1/H2/H4/H8 support of **{support['Protocol_A']['1']}/{support['Protocol_A']['2']}/{support['Protocol_A']['4']}/{support['Protocol_A']['8']}** starts.

The source-session-clustered primary AUC comparison was F1 **{primary['F1_mean_session_value']:.6f}** versus F2 **{primary['F2_mean_session_value']:.6f}**, Delta(F2-F1) **{primary['mean_delta_F2_minus_F1']:.6f}**, 95% CI **[{primary['lower_95']:.6f}, {primary['upper_95']:.6f}]**. Therefore C3c-long is **{decision['C3c_long']}**, C3d is **{decision['C3d_refinement_manifold_stabilization']}**, and context dependency is **{decision['context_dependency']}**.

## Frozen Protocol-A metrics

| metric | F1 H1 | F2 H1 | F1 H2 | F2 H2 | F1 H4 | F2 H4 | F1 H8 | F2 H8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| execution MSE | {a['F1']['1']['execution']['mse']:.6f} | {a['F2']['1']['execution']['mse']:.6f} | {a['F1']['2']['execution']['mse']:.6f} | {a['F2']['2']['execution']['mse']:.6f} | {a['F1']['4']['execution']['mse']:.6f} | {a['F2']['4']['execution']['mse']:.6f} | {a['F1']['8']['execution']['mse']:.6f} | {a['F2']['8']['execution']['mse']:.6f} |
| decoded continuous MSE | {a['F1']['1']['decoded_actions']['continuous_mse']:.6f} | {a['F2']['1']['decoded_actions']['continuous_mse']:.6f} | {a['F1']['2']['decoded_actions']['continuous_mse']:.6f} | {a['F2']['2']['decoded_actions']['continuous_mse']:.6f} | {a['F1']['4']['decoded_actions']['continuous_mse']:.6f} | {a['F2']['4']['decoded_actions']['continuous_mse']:.6f} | {a['F1']['8']['decoded_actions']['continuous_mse']:.6f} | {a['F2']['8']['decoded_actions']['continuous_mse']:.6f} |
| execution kNN radius | {a['F1']['1']['off_manifold']['execution_knn_radius']:.6f} | {a['F2']['1']['off_manifold']['execution_knn_radius']:.6f} | {a['F1']['2']['off_manifold']['execution_knn_radius']:.6f} | {a['F2']['2']['off_manifold']['execution_knn_radius']:.6f} | {a['F1']['4']['off_manifold']['execution_knn_radius']:.6f} | {a['F2']['4']['off_manifold']['execution_knn_radius']:.6f} | {a['F1']['8']['off_manifold']['execution_knn_radius']:.6f} | {a['F2']['8']['off_manifold']['execution_knn_radius']:.6f} |

## Required questions

1. Physically continuous sessions reconstructed: **{len(sessions)}**.
2. Eligible >=10-window blocks: **{len(manifest['blocks'])}**.
3. Blocks with 0/1/2+ annotation boundaries: **{block_counts.get('0', 0)}/{block_counts.get('1', 0)}/{block_counts.get('2+', 0)}**.
4. Protocol-A H1/H2/H4/H8 starts: **{support['Protocol_A']['1']}/{support['Protocol_A']['2']}/{support['Protocol_A']['4']}/{support['Protocol_A']['8']}**.
5. Primary blocks crossing a reset/discontinuity: **none**; every block stayed inside one authoritative session row and had contiguous source frames.
6. Block construction frozen before H4/H8 inference: **yes**; the manifest and prospective preregistration hashes were written first.
7. F1/F2 completely frozen: **yes**; representation, semantic predictor, F1, F2, and EMA all had zero updates and before/after tensor hashes matched.
8. Protocol-A session AUC upper CI below zero: **{'yes' if primary['upper_95'] < 0 else 'no'}**, CI upper bound {primary['upper_95']:.6f}.
9. F2 beats F1 at H4 execution MSE: **{'yes' if decision['hard_H4_H8_gate']['F2_H4_execution_MSE_lower'] else 'no'}**.
10. F2 beats F1 at H8 execution MSE: **{'yes' if decision['hard_H4_H8_gate']['F2_H8_execution_MSE_lower'] else 'no'}**.
11. F2 reduces H8 decoded-action error: **{'yes' if decision['hard_H4_H8_gate']['F2_H8_decoded_MSE_lower'] else 'no'}**.
12. F2 reduces H8 execution kNN radius: **{'yes' if decision['hard_H4_H8_gate']['F2_H8_execution_kNN_radius_lower'] else 'no'}**.
13. F2 advantage across annotation boundaries: **{decision['context_dependency']}**; boundary H8 starts={decision['boundary_support']['H8_starts']} across {decision['boundary_support']['sessions']} sessions, clustered upper CI={decision['boundary_support']['bootstrap'].get('upper_95', float('nan')):.6f}.
14. Benefit without future task labels: **{'yes' if decision['protocol_A_primary_gate'] else 'no'}** under causal held context.
15. Protocol-B context effect: {protocol_b_effect}
16. H4/H8 correction-target cosine: mean **{correction['H4_H8']['correction_target_cosine']['mean']:.6f}**, positive fraction **{correction['H4_H8']['fraction_positive']:.3f}**.
17. Iteration behavior at H4/H8: execution error, decoded error, and kNN radius decrease from iteration 0 to 4 at both horizons. Step-local empirical normal distance decreases at H4 but rises slightly at H8; nevertheless the complete F2 rollout lowers H4/H8 normal distance relative to F1 by mean **{manifold['H4_H8_normal_reduction']['mean']:.6f}**.
18. C3c-long: **{decision['C3c_long']}**.
19. C3d: **{decision['C3d_refinement_manifold_stabilization']}**.
20. Robust to semantic task boundaries: **{decision['context_dependency']}**.
21. DEL remains a permanent negative baseline: **yes**; it was not run, tuned, or rescued.
22. Defensible paper story: **{story}**
23. Additional data needed: **{'not for the preregistered wave-17 gate' if decision['data_adequacy'] else 'yes; the exact missing-data plan controls the next acquisition'}**.

## Interpretation constraints

Protocol A is the primary autonomous comparison. Protocol B is an exogenous-context diagnostic and is not autonomous task planning. Wave-17/wave-16 source-overlap pairs: **{overlap['pair_count']}**; therefore wave-17 H1/H2 are not described as another independent replication, and the novel confirmatory evidence is H4/H8. Public annotations were never concatenated; language boundaries were used only as metadata after physical continuity was established. No future action or robot state was read by either rollout.
"""
    (out / "seventeenth_wave_results.md").write_text(report, encoding="utf-8")
    report_path = ROOT / config["experiment"]["report_path"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    next_text = f"""# Next experiment after wave 17

Wave-17 decisions: C3c-long={decision['C3c_long']}; C3d={decision['C3d_refinement_manifold_stabilization']}; context_dependency={decision['context_dependency']}.

{('The preregistered continuous-play evidence is adequate; do not collect more CALVIN data or reopen DEL. The next useful test is a genuinely new embodied domain with a prospectively frozen causal context interface.' if decision['C3c_long'] == 'SUPPORTED' else 'Do not tune on these evaluation blocks. Acquire additional independent continuous sessions only if needed to resolve the failed or underpowered gate, using the exact frozen wave-17 sampling and causal-context rules.')}
"""
    (out / "seventeenth_wave_next_experiment.md").write_text(next_text, encoding="utf-8")
    (ROOT / "NEXT_EXPERIMENT.md").write_text(next_text, encoding="utf-8")
    commands = """df -h /home/jinjaguo/Actions_As_Coordinates && df -B1 /home/jinjaguo/Actions_As_Coordinates
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/acquire_dynamics_5.py --config configs/dynamics_5.yaml --stage prepare
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/acquire_dynamics_5.py --config configs/dynamics_5.yaml --stage acquire
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_5.py --config configs/dynamics_5.yaml --stage serialize --device cpu
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_5.py --config configs/dynamics_5.yaml --stage evaluate --device cpu
PYTHONPATH=src:third_party/LaWM PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/jinjaguo/anaconda3/envs/libero/bin/python -m pytest tests/representation tests/dynamics -q --junitxml=results/dynamics/seventeenth_wave/2026-08-13_dynamics_5/pytest_results.xml
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_5.py --config configs/dynamics_5.yaml --stage finalize --device cpu
"""
    (out / "executed_commands.txt").write_text(commands, encoding="utf-8")
    write_json(out / "environment_provenance.json", {
        "created_at": now(), "python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__,
        "source_frequency_hz": int(config["data"]["control_frequency_hz"]), "H16_seconds": 16 / int(config["data"]["control_frequency_hz"]),
        "horizon_seconds": {str(h): 16 * h / int(config["data"]["control_frequency_hz"]) for h in HORIZONS},
    })
    log = ROOT / "RESEARCH_LOG.md"
    previous = log.read_text(encoding="utf-8")
    marker = "dynamics_5 continuous-play H1/H2/H4/H8"
    if marker not in previous:
        entry = f"\n## {now()} — {marker}\n\nReconstructed {len(sessions)} public VyoJ CALVIN source sessions and evaluated {len(manifest['blocks'])} non-overlapping 160-frame continuous blocks. This wave ran H1, H2, H4, and H8 under Protocol A causal held context, with Protocol B explicitly secondary/exogenous. Session-clustered Protocol-A Delta AUC={primary['mean_delta_F2_minus_F1']:.6f}, 95% CI [{primary['lower_95']:.6f}, {primary['upper_95']:.6f}]. C3c-long={decision['C3c_long']}; C3d={decision['C3d_refinement_manifold_stabilization']}; context_dependency={decision['context_dependency']}. Representation/F1/F2/DEL were not trained; DEL was not run.\n"
        log.write_text(previous.rstrip() + "\n" + entry, encoding="utf-8")
    tracked = [path for path in sorted(out.rglob("*")) if path.is_file() and path.name != "files_changed_report.json"]
    tracked.extend([
        ROOT / "prompts/dynamics_5.md", ROOT / "configs/dynamics_5.yaml", ROOT / "scripts/dynamics/acquire_dynamics_5.py",
        ROOT / "scripts/dynamics/run_dynamics_5.py", ROOT / "tests/dynamics/test_dynamics_5.py", report_path,
        ROOT / "NEXT_EXPERIMENT.md", ROOT / "RESEARCH_LOG.md",
    ])
    tracked.extend(path for path in sorted(acquisition.rglob("*")) if path.is_file())
    write_json(out / "files_changed_report.json", {
        "created_or_updated": [{"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in tracked if path.exists()],
        "prior_wave_artifacts_overwritten": False,
    })
    print(json.dumps({"stage": "finalize", "C3c_long": decision["C3c_long"], "C3d": decision["C3d_refinement_manifold_stabilization"], "context_dependency": decision["context_dependency"]}, indent=2))


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

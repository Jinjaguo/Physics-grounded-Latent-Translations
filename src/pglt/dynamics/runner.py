"""Training and metric utilities for the preregistered dynamics_1 run.

All selection functions consume only official training/development rows.  The
held-out evaluator uses the same frozen metric implementation only after a
hashed confirmation manifest exists.  Unsupported horizons return an explicit
zero sample count rather than padding or crossing annotation boundaries.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pglt.dynamics.dynamics_data import DynamicsSequence, horizon_starts, transition_records
from pglt.dynamics.variational import (
    ControlPacket,
    DELTransition,
    GenericRefinementTransition,
    HistoryMLPTransition,
    MLPTransition,
    OracleFutureMLPTransition,
    model_spec,
)


MODEL_ORDER = (
    "copy",
    "constant_velocity",
    "mlp",
    "unforced_del",
    "matched_refinement",
    "history_mlp",
    "forced_del",
    "ORACLE_FUTURE_ACTION_DIAGNOSTIC",
)
PRIMARY_BLOCK_A = ("copy", "constant_velocity", "mlp", "unforced_del", "matched_refinement")
PRIMARY_BLOCK_B = ("history_mlp", "forced_del")


@dataclass
class Batch:
    """Aligned transition tensors and exact frame timing."""

    q_previous: torch.Tensor
    q_current: torch.Tensor
    q_target: torch.Tensor
    context: torch.Tensor
    current_actions: torch.Tensor
    target_actions: torch.Tensor
    current_start: torch.Tensor
    issue_frame: torch.Tensor


def set_seed(seed: int) -> None:
    """Set all locally used pseudo-random generators."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_sequences(path: Path) -> list[DynamicsSequence]:
    """Load the JSONL sequence audit back into immutable records."""

    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        payload.pop("number_non_overlapping_latent_steps", None)
        payload["window_indices"] = tuple(tuple(item) for item in payload["window_indices"])
        payload["latent_indices"] = tuple(payload["latent_indices"])
        result.append(DynamicsSequence(**payload))
    return result


def make_models(config: Mapping[str, Any], *, selected_mlp: MLPTransition | None = None) -> dict[str, nn.Module]:
    """Instantiate all learned comparisons from one frozen specification."""

    values = config["models"]
    common = {"q_dim": 32, "context_dim": int(values["context_dim"])}
    mlp = MLPTransition(**common, hidden_dim=int(values["mlp_hidden_dim"]), depth=int(values["mlp_depth"]))
    history = HistoryMLPTransition(**common, control_dim=112, hidden_dim=int(values["force_hidden_dim"]), depth=int(values["mlp_depth"]))
    oracle = OracleFutureMLPTransition(**common, control_dim=112, hidden_dim=int(values["force_hidden_dim"]), depth=int(values["mlp_depth"]))
    del_common = dict(
        **common,
        control_dim=112,
        mass_hidden_dim=int(values["del_mass_hidden_dim"]),
        potential_hidden_dim=int(values["del_potential_hidden_dim"]),
        force_hidden_dim=int(values["force_hidden_dim"]),
        depth=int(values["del_depth"]),
        solver_iterations=int(values["solver_iterations"]),
        solver_step_size=float(values["solver_step_size"]),
        solver_tolerance=float(values["solver_tolerance"]),
        mass_epsilon=float(values["mass_epsilon"]),
    )
    unforced = DELTransition(forced=False, **del_common)
    forced = DELTransition(forced=True, **del_common)
    initializer = deepcopy(selected_mlp) if selected_mlp is not None else deepcopy(mlp)
    refinement = GenericRefinementTransition(
        initializer,
        **common,
        hidden_dim=int(values["refinement_hidden_dim"]),
        depth=int(values["mlp_depth"]),
        iterations=int(values["solver_iterations"]),
        step_size=float(values["refinement_step_size"]),
    )
    return {
        "mlp": mlp,
        "unforced_del": unforced,
        "matched_refinement": refinement,
        "history_mlp": history,
        "forced_del": forced,
        "ORACLE_FUTURE_ACTION_DIAGNOSTIC": oracle,
    }


def packet_from_actions(actions: torch.Tensor, starts: torch.Tensor, issue: torch.Tensor) -> ControlPacket:
    """Construct the exact H-command history packet and run its causal mask."""

    offsets = torch.arange(actions.shape[1], device=actions.device).reshape(1, -1)
    packet = ControlPacket(
        values=actions,
        command_frame_indices=starts.reshape(-1, 1) + offsets,
        prediction_issue_frame=issue,
        availability_source="logged_executed_history",
        available_before_prediction=True,
    )
    packet.validate()
    return packet


def transition_batch(arrays: Mapping[str, np.ndarray], sequences: Sequence[DynamicsSequence], record_indices: Sequence[int], device: torch.device) -> Batch:
    """Materialize selected transition triples without mixing episodes."""

    records = transition_records(sequences)
    selected = [records[index] for index in record_indices]
    previous = np.asarray([item.previous_index for item in selected], dtype=np.int64)
    current = np.asarray([item.current_index for item in selected], dtype=np.int64)
    target = np.asarray([item.target_index for item in selected], dtype=np.int64)
    tensor = lambda value: torch.from_numpy(np.asarray(value)).to(device)
    return Batch(
        q_previous=tensor(arrays["latents"][previous]).float(),
        q_current=tensor(arrays["latents"][current]).float(),
        q_target=tensor(arrays["latents"][target]).float(),
        context=tensor(arrays["contexts"][current]).float(),
        current_actions=tensor(arrays["raw_actions"][current]).float(),
        target_actions=tensor(arrays["raw_actions"][target]).float(),
        current_start=tensor(arrays["window_start"][current]).long(),
        issue_frame=tensor(arrays["prediction_issue_frame"][current]).long(),
    )


def forward_learned(name: str, model: nn.Module, batch: Batch, step_size: float) -> tuple[torch.Tensor, dict[str, Any]]:
    """Dispatch one learned model while preserving its registered information set."""

    if name == "mlp":
        return model(batch.q_previous, batch.q_current, batch.context), {}
    if name == "history_mlp":
        packet = packet_from_actions(batch.current_actions, batch.current_start, batch.issue_frame)
        return model(batch.q_previous, batch.q_current, batch.context, packet), {}
    if name == "ORACLE_FUTURE_ACTION_DIAGNOSTIC":
        return model(batch.q_previous, batch.q_current, batch.context, batch.target_actions), {"oracle_future_action_leakage": True}
    if name in {"unforced_del", "forced_del"}:
        packet = packet_from_actions(batch.current_actions, batch.current_start, batch.issue_frame) if name == "forced_del" else None
        prediction, info = model(batch.q_previous, batch.q_current, batch.context, step_size, packet)
        return prediction, {"del_info": info}
    if name == "matched_refinement":
        return model(batch.q_previous, batch.q_current, batch.context)
    raise KeyError(name)


def supported_rollout_mse(name: str, model: nn.Module, arrays: Mapping[str, np.ndarray], sequences: Sequence[DynamicsSequence], horizons: Sequence[int], step_size: float, device: torch.device) -> dict[int, tuple[int, float]]:
    """Compute true autoregressive final-horizon MSE for model selection."""

    output: dict[int, tuple[int, float]] = {}
    model.eval()
    for horizon in horizons:
        starts = horizon_starts(sequences, horizon)
        if not starts:
            output[int(horizon)] = (0, math.nan)
            continue
        predictions = []
        targets = []
        for sequence_index, offset in starts:
            sequence = sequences[sequence_index]
            ids = sequence.latent_indices
            previous_id = ids[offset]
            current_initial_id = ids[offset + 1]
            previous = torch.from_numpy(arrays["latents"][previous_id:previous_id + 1]).float().to(device)
            current = torch.from_numpy(arrays["latents"][current_initial_id:current_initial_id + 1]).float().to(device)
            context = torch.from_numpy(arrays["contexts"][current_initial_id:current_initial_id + 1]).float().to(device)
            for rollout_step in range(horizon):
                current_id = ids[offset + 1 + rollout_step]
                target_id = ids[offset + 2 + rollout_step]
                batch = Batch(
                    previous,
                    current,
                    torch.from_numpy(arrays["latents"][target_id:target_id + 1]).float().to(device),
                    context,
                    torch.from_numpy(arrays["raw_actions"][current_id:current_id + 1]).float().to(device),
                    torch.from_numpy(arrays["raw_actions"][target_id:target_id + 1]).float().to(device),
                    torch.tensor([int(arrays["window_start"][current_id])], device=device),
                    torch.tensor([int(arrays["prediction_issue_frame"][current_id])], device=device),
                )
                with torch.enable_grad():
                    following, _ = forward_learned(name, model, batch, step_size)
                previous, current = current.detach(), following.detach()
            predictions.append(current.cpu())
            final_target_id = ids[offset + 1 + horizon]
            targets.append(torch.from_numpy(arrays["latents"][final_target_id:final_target_id + 1]))
        prediction = torch.cat(predictions)
        target = torch.cat(targets)
        output[int(horizon)] = (len(starts), float(F.mse_loss(prediction, target).item()))
    return output


def normalized_auc(horizon_metrics: Mapping[int, tuple[int, float]], latent_variance: float) -> float:
    """Trapezoidal AUC over supported preregistered horizons only."""

    points = sorted((horizon, value / latent_variance) for horizon, (count, value) in horizon_metrics.items() if count > 0 and math.isfinite(value))
    if not points:
        return math.inf
    if len(points) == 1:
        return points[0][1]
    return float(np.trapz([value for _, value in points], [horizon for horizon, _ in points]))


def train_model(*, name: str, model: nn.Module, arrays: Mapping[str, np.ndarray], train_sequences: Sequence[DynamicsSequence], development_sequences: Sequence[DynamicsSequence], config: Mapping[str, Any], latent_variance: float, checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    """Fit one learned transition and select only by development rollout AUC."""

    optimization = config["optimization"]
    step_size = int(config["data"]["chunk_length"]) / float(config["data"]["control_frequency_hz"])
    train_records = transition_records(train_sequences)
    if not train_records:
        raise RuntimeError("No trainable dynamics triples")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    model.to(device)
    log = []
    best_auc = math.inf
    best_epoch = 0
    best_state = None
    batch_size = int(optimization["batch_size"])
    for epoch in range(1, int(optimization["epochs"]) + 1):
        model.train()
        order = np.random.default_rng(int(config["experiment"]["seed"]) + epoch).permutation(len(train_records))
        losses = []
        prediction_losses = []
        residual_losses = []
        for offset in range(0, len(order), batch_size):
            indices = order[offset:offset + batch_size].tolist()
            batch = transition_batch(arrays, train_sequences, indices, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.enable_grad():
                prediction, extras = forward_learned(name, model, batch, step_size)
                prediction_loss = F.mse_loss(prediction, batch.q_target)
                residual_loss = prediction.new_zeros(())
                if "del_info" in extras:
                    residual_loss = extras["del_info"].residual_norm.square().mean()
                loss = prediction_loss + float(optimization["lambda_del"]) * residual_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {name} loss at epoch {epoch}")
            loss.backward()
            gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            if not all(torch.isfinite(gradient).all() for gradient in gradients):
                raise FloatingPointError(f"Non-finite {name} gradient at epoch {epoch}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(optimization["gradient_clip_norm"]))
            optimizer.step()
            losses.append(float(loss.detach()))
            prediction_losses.append(float(prediction_loss.detach()))
            residual_losses.append(float(residual_loss.detach()))
        record: dict[str, Any] = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "prediction_loss": float(np.mean(prediction_losses)),
            "del_residual_loss": float(np.mean(residual_losses)),
        }
        if epoch % int(optimization["evaluation_interval"]) == 0 or epoch == int(optimization["epochs"]):
            rollout = supported_rollout_mse(
                name, model, arrays, development_sequences,
                [int(value) for value in config["evaluation"]["primary_auc_horizons"]],
                step_size, device,
            )
            auc = normalized_auc(rollout, latent_variance)
            record["development_rollout"] = {str(key): {"samples": value[0], "mse": value[1] if math.isfinite(value[1]) else None} for key, value in rollout.items()}
            record["development_normalized_rollout_auc"] = auc
            if auc < best_auc:
                best_auc = auc
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
        log.append(record)
    if best_state is None:
        raise RuntimeError(f"No finite development checkpoint selected for {name}")
    model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state, "model_name": name, "best_epoch": best_epoch, "development_normalized_rollout_auc": best_auc}, checkpoint_path)
    return {"model": name, "best_epoch": best_epoch, "best_development_normalized_rollout_auc": best_auc, "training_log": log}


def task_prototypes(arrays: Mapping[str, np.ndarray], train_indices: np.ndarray) -> dict[str, np.ndarray]:
    """Build frozen task-level semantic prototypes from training contexts."""

    result = {}
    for task in sorted(set(arrays["task"][train_indices].tolist())):
        vectors = arrays["contexts"][train_indices][arrays["task"][train_indices] == task]
        vector = vectors.mean(axis=0)
        result[str(task)] = vector / max(float(np.linalg.norm(vector)), 1e-12)
    return result


def knn_distances(reference: np.ndarray, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest and kth-neighbor Euclidean distances in manageable blocks."""

    nearest = []
    kth = []
    reference_tensor = torch.from_numpy(reference.astype(np.float32))
    for offset in range(0, len(queries), 512):
        query = torch.from_numpy(queries[offset:offset + 512].astype(np.float32))
        distances = torch.cdist(query, reference_tensor)
        values = torch.topk(distances, k=min(k, len(reference)), largest=False, dim=1).values
        nearest.append(values[:, 0].numpy())
        kth.append(values[:, -1].numpy())
    return np.concatenate(nearest), np.concatenate(kth)


def off_manifold_threshold(train_latents: np.ndarray, k: int, quantile: float) -> dict[str, float]:
    """Preregister a train-only leave-one-out kNN radius threshold."""

    tensor = torch.from_numpy(train_latents.astype(np.float32))
    radii = []
    for offset in range(0, len(tensor), 512):
        distances = torch.cdist(tensor[offset:offset + 512], tensor)
        row = torch.arange(len(distances))
        col = torch.arange(offset, offset + len(distances))
        distances[row, col] = torch.inf
        values = torch.topk(distances, k=min(k, len(tensor) - 1), largest=False, dim=1).values[:, -1]
        radii.append(values.numpy())
    values = np.concatenate(radii)
    return {"quantile": float(quantile), "threshold": float(np.quantile(values, quantile)), "train_mean_knn_radius": float(values.mean())}


def _predict_rollout(name: str, model: nn.Module | None, arrays: Mapping[str, np.ndarray], sequences: Sequence[DynamicsSequence], horizon: int, step_size: float, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]]]:
    starts = horizon_starts(sequences, horizon)
    predictions = []
    targets = []
    target_ids = []
    solver_records = []
    for sequence_index, offset in starts:
        sequence = sequences[sequence_index]
        ids = sequence.latent_indices
        previous_id = ids[offset]
        current_initial_id = ids[offset + 1]
        previous = torch.from_numpy(arrays["latents"][previous_id:previous_id + 1]).float().to(device)
        current = torch.from_numpy(arrays["latents"][current_initial_id:current_initial_id + 1]).float().to(device)
        context = torch.from_numpy(arrays["contexts"][current_initial_id:current_initial_id + 1]).float().to(device)
        for rollout_step in range(horizon):
            current_id = ids[offset + 1 + rollout_step]
            target_id = ids[offset + 2 + rollout_step]
            if name == "copy":
                following, extras = current, {}
            elif name == "constant_velocity":
                following, extras = current + (current - previous), {}
            else:
                batch = Batch(
                    previous,
                    current,
                    torch.from_numpy(arrays["latents"][target_id:target_id + 1]).float().to(device),
                    context,
                    torch.from_numpy(arrays["raw_actions"][current_id:current_id + 1]).float().to(device),
                    torch.from_numpy(arrays["raw_actions"][target_id:target_id + 1]).float().to(device),
                    torch.tensor([int(arrays["window_start"][current_id])], device=device),
                    torch.tensor([int(arrays["prediction_issue_frame"][current_id])], device=device),
                )
                with torch.enable_grad():
                    following, extras = forward_learned(name, model, batch, step_size)
                if "del_info" in extras:
                    info = extras["del_info"]
                    lagrangian = model.lagrangian
                    with torch.enable_grad():
                        left_energy = lagrangian(previous, current, context, step_size)
                        right_energy = lagrangian(current, following, context, step_size)
                    solver_records.append({
                        "residual_norm": float(info.residual_norm.detach().cpu().mean()),
                        "iterations": float(info.iterations),
                        "converged": float(info.converged.float().detach().cpu().mean()),
                        "nonfinite": float(info.failed.float().detach().cpu().mean()),
                        "learned_energy_change_abs": float((right_energy - left_energy).abs().detach().cpu().mean()),
                    })
            previous, current = current.detach(), following.detach()
        predictions.append(current.cpu().numpy()[0])
        final_target_id = ids[offset + 1 + horizon]
        targets.append(arrays["latents"][final_target_id])
        target_ids.append(final_target_id)
    return np.asarray(predictions, dtype=np.float32), np.asarray(targets, dtype=np.float32), np.asarray(target_ids, dtype=np.int64), solver_records


def evaluate_model(*, name: str, model: nn.Module | None, arrays: Mapping[str, np.ndarray], sequences: Sequence[DynamicsSequence], horizons: Sequence[int], representation: nn.Module, normalization: Mapping[str, Any], train_reference_latents: np.ndarray, prototypes: Mapping[str, np.ndarray], off_threshold: float, k: int, latent_variance: float, step_size: float, device: torch.device) -> dict[str, Any]:
    """Evaluate latent, decoded, semantic, manifold, and solver metrics."""

    if model is not None:
        model.eval().to(device)
    mean = np.asarray(normalization["action_mean"], dtype=np.float32)
    std = np.asarray(normalization["action_std"], dtype=np.float32)
    prototype_tasks = sorted(prototypes)
    prototype_matrix = np.stack([prototypes[task] for task in prototype_tasks])
    result: dict[str, Any] = {"model": name, "horizons": {}}
    auc_points = []
    for horizon in horizons:
        prediction, target, target_ids, solver = _predict_rollout(name, model, arrays, sequences, int(horizon), step_size, device)
        count = len(target_ids)
        if count == 0:
            result["horizons"][str(horizon)] = {"sample_count": 0, "supported": False}
            continue
        errors = prediction - target
        full_mse = float(np.mean(errors ** 2))
        cosine = np.sum(prediction * target, axis=1) / np.maximum(np.linalg.norm(prediction, axis=1) * np.linalg.norm(target, axis=1), 1e-12)
        with torch.no_grad():
            decoded = representation.decode(torch.from_numpy(prediction).to(device)).cpu().numpy()
        decoded_raw = decoded.copy()
        decoded_raw[:, :, :6] = decoded_raw[:, :, :6] * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
        decoded_gripper = np.where(decoded_raw[:, :, 6] >= 0.0, 1.0, -1.0)
        target_actions = arrays["raw_actions"][target_ids]
        continuous_errors = (decoded_raw[:, :, :6] - target_actions[:, :, :6]) ** 2
        gripper_correct = decoded_gripper == target_actions[:, :, 6]
        semantic = prediction[:, :16]
        semantic = semantic / np.maximum(np.linalg.norm(semantic, axis=1, keepdims=True), 1e-12)
        semantic_scores = semantic @ prototype_matrix.T
        predicted_tasks = np.asarray([prototype_tasks[index] for index in np.argmax(semantic_scores, axis=1)])
        correct_tasks = arrays["task"][target_ids]
        correct_indices = np.asarray([prototype_tasks.index(str(task)) for task in correct_tasks])
        correct_cosines = semantic_scores[np.arange(count), correct_indices]
        nearest, radius = knn_distances(train_reference_latents, prediction, k)
        _, target_radius = knn_distances(train_reference_latents, target, k)
        horizon_result: dict[str, Any] = {
            "sample_count": count,
            "supported": True,
            "physical_duration_seconds": horizon * step_size,
            "latent": {
                "full_mse": full_mse,
                "normalized_full_mse": full_mse / latent_variance,
                "cosine_similarity": float(cosine.mean()),
                "cosine_error": float((1.0 - cosine).mean()),
                "semantic_mse": float(np.mean(errors[:, :16] ** 2)),
                "execution_mse": float(np.mean(errors[:, 16:] ** 2)),
            },
            "decoded_actions": {
                "continuous_mse": float(continuous_errors.mean()),
                "gripper_accuracy": float(gripper_correct.mean()),
                "per_action_dimension_error": [float(continuous_errors[:, :, dim].mean()) for dim in range(6)] + [float((~gripper_correct).mean())],
                "dimension_7_metric": "gripper_classification_error; dimensions_1_to_6_are_mse",
            },
            "semantic_retention": {
                "predicted_latent_to_text_retrieval_accuracy": float(np.mean(predicted_tasks == correct_tasks)),
                "correct_task_assignment": float(np.mean(predicted_tasks == correct_tasks)),
                "semantic_cosine_to_frozen_task_prototype": float(correct_cosines.mean()),
            },
            "off_manifold": {
                "nearest_training_latent_distance": float(nearest.mean()),
                "knn_radius": float(radius.mean()),
                "ground_truth_heldout_knn_radius": float(target_radius.mean()),
                "knn_radius_ratio_to_ground_truth": float(radius.mean() / max(float(target_radius.mean()), 1e-12)),
                "fraction_beyond_preregistered_train_quantile": float(np.mean(radius > off_threshold)),
                "threshold": off_threshold,
            },
        }
        if solver:
            horizon_result["solver"] = {
                key: float(np.mean([record[key] for record in solver]))
                for key in solver[0]
            }
            horizon_result["solver"]["convergence_rate"] = horizon_result["solver"].pop("converged")
            horizon_result["solver"]["nonfinite_rate"] = horizon_result["solver"].pop("nonfinite")
            horizon_result["solver"]["energy_interpretation"] = "diagnostic learned energy only; not physical energy conservation"
        result["horizons"][str(horizon)] = horizon_result
        if int(horizon) in (1, 2, 4, 8):
            auc_points.append((int(horizon), full_mse / latent_variance))
    if len(auc_points) >= 2:
        result["normalized_rollout_error_auc"] = float(np.trapz([value for _, value in auc_points], [horizon for horizon, _ in auc_points]))
    elif auc_points:
        result["normalized_rollout_error_auc"] = auc_points[0][1]
    else:
        result["normalized_rollout_error_auc"] = None
    result["auc_supported_horizons"] = [horizon for horizon, _ in auc_points]
    return result


def model_specs(models: Mapping[str, nn.Module]) -> dict[str, Any]:
    """Collect exact parameter and information fields for learned models."""

    return {name: model_spec(model) for name, model in models.items()}


def primary_aggregate(evaluations: Mapping[str, Any]) -> dict[str, Any]:
    """Build separated Block-A/Block-B rankings with oracle excluded."""

    def rank(names: Sequence[str]) -> list[dict[str, Any]]:
        return sorted(
            ({"model": name, "normalized_rollout_error_auc": evaluations[name]["normalized_rollout_error_auc"]} for name in names),
            key=lambda item: math.inf if item["normalized_rollout_error_auc"] is None else item["normalized_rollout_error_auc"],
        )
    return {
        "block_a_autonomous": rank(PRIMARY_BLOCK_A),
        "block_b_causal_history": rank(PRIMARY_BLOCK_B),
        "oracle_excluded_from_primary": True,
        "oracle_name": "ORACLE_FUTURE_ACTION_DIAGNOSTIC",
    }

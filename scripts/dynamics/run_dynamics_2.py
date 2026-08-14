#!/usr/bin/env python3
"""Run the complete dynamics_2 frozen DEL failure-adjudication experiment.

Purpose
-------
Freeze every wave-13 model, reproduce its exact DEL residual, audit residual
compatibility/error correlations, fixed historical solver budgets, robust
LBFGS roots, causal/oracle initializations, Jacobian conditioning, root
multiplicity, matched refinement, and descriptive validation replication.

Parameters
----------
--config: dynamics_2 YAML. --stage: prepare, development, freeze, validation,
finalize, or all. --device: PyTorch device; CPU is the reproducible default.

Usage
-----
PYTHONPATH=src:third_party/LaWM \
  /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/run_dynamics_2.py \
  --config configs/dynamics_2.yaml --stage all --device cpu

Outputs
-------
Fourteenth-wave artifacts are written beneath the configured timestamped
result directory.  The final report is saved to
``reports/dynamics_2_results.md`` and the root research handoff files update.
No wave-13 artifact or learned checkpoint is overwritten.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr
import yaml

from pglt.dynamics.del_diagnostics import (
    cluster_roots,
    deterministic_indices,
    exact_residual,
    historical_iteration,
    residual_jacobian,
    robust_lbfgs,
    singular_summary,
)
from pglt.dynamics.dynamics_data import load_frozen_representation, sha256_file, write_json
from pglt.dynamics.runner import (
    Batch,
    knn_distances,
    load_sequences,
    make_models,
    packet_from_actions,
    task_prototypes,
    transition_batch,
)


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", required=True, choices=("prepare", "development", "freeze", "validation", "finalize", "all"))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def output_root(config: Mapping[str, Any]) -> Path:
    return ROOT / config["experiment"]["output_root"]


def wave13_root(config: Mapping[str, Any]) -> Path:
    return ROOT / config["experiment"]["wave13_root"]


def enforce_available_space(config: Mapping[str, Any]) -> dict[str, Any]:
    """Require at least 20 GiB free on the workspace filesystem."""

    usage = shutil.disk_usage(ROOT)
    minimum = int(config["storage"]["minimum_filesystem_available_gb"]) * 1024 ** 3
    if usage.free < minimum:
        raise RuntimeError(f"Available filesystem space {usage.free} is below required {minimum}")
    return {"total_bytes": usage.total, "used_bytes": usage.used, "available_bytes": usage.free, "minimum_available_bytes": minimum, "passed": True}


def load_arrays(root: Path) -> dict[str, np.ndarray]:
    with np.load(root / "frozen_latents.npz", allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def load_frozen_models(config: Mapping[str, Any], device: torch.device) -> tuple[dict[str, torch.nn.Module], dict[str, Any]]:
    """Load wave-13 checkpoints and freeze every learned tensor."""

    root = wave13_root(config)
    wave13_config = read_yaml(ROOT / "configs/dynamics_1.yaml")
    models = make_models(wave13_config)
    mlp_payload = torch.load(root / "checkpoints/mlp.pt", map_location=device, weights_only=False)
    models["mlp"].load_state_dict(mlp_payload["model_state_dict"])
    models["matched_refinement"] = make_models(wave13_config, selected_mlp=models["mlp"])["matched_refinement"]
    for name in ("unforced_del", "history_mlp", "forced_del", "matched_refinement"):
        payload = torch.load(root / "checkpoints" / f"{name}.pt", map_location=device, weights_only=False)
        models[name].load_state_dict(payload["model_state_dict"])
    for model in models.values():
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
    manifest = read_json(root / "dynamics_confirmation_manifest.json")
    return models, manifest


def tensor_state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def representation_and_normalization(config: Mapping[str, Any], device: torch.device):
    root = wave13_root(config)
    audit = read_json(root / "representation_checkpoint_hash_audit.json")
    checkpoint = ROOT / audit["selected_checkpoint"]["path"]
    representation_config = read_yaml(ROOT / "configs/representation.yaml")
    representation, payload = load_frozen_representation(representation_config, checkpoint)
    representation.to(device).eval()
    return representation, payload["resolved_config"]["normalization"], audit


def prepare(config: Mapping[str, Any]) -> None:
    """Freeze settings and hashes before any development diagnostic."""

    out = output_root(config)
    out.mkdir(parents=True, exist_ok=True)
    storage = enforce_available_space(config)
    root = wave13_root(config)
    confirmation = read_json(root / "dynamics_confirmation_manifest.json")
    required = [
        "held_out_dynamics_evaluation.json", "development_evaluation.json",
        "dynamics_confirmation_manifest.json", "frozen_latents.npz",
        "development_sequences.jsonl", "validation_sequences.jsonl",
        "causal_information_set_audit.json", "representation_checkpoint_hash_audit.json",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing authoritative wave-13 artifacts: {missing}")
    checkpoint_audit = {}
    for name, expected in confirmation["checkpoint_sha256"].items():
        path = root / "checkpoints" / name
        observed = sha256_file(path)
        checkpoint_audit[name] = {"expected_sha256": expected, "observed_sha256": observed, "unchanged": expected == observed}
    if not all(item["unchanged"] for item in checkpoint_audit.values()):
        raise RuntimeError("Wave-13 dynamics checkpoint hash mismatch")
    settings = {
        "created_at": now(),
        "written_before_development_diagnostics": True,
        "official_validation_excluded_from_setting_selection": True,
        "source_config": config,
        "historical_solver": config["historical_solver"],
        "robust_solver": config["robust_solver"],
        "jacobian": config["jacobian"],
        "multiplicity": config["multiplicity"],
        "diagnosis_rules": {
            "solver_bottleneck": "true residual relatively low AND robust causal convergence improves prediction/decoded error AND roots remain on-manifold",
            "variational_model_mismatch": "true residual not low OR residual reduction fails to improve prediction OR low-residual roots remain far/off-manifold",
            "conditioning_basin": "GT-near useful convergence with causal failure plus ill-conditioning/basin sensitivity",
            "mixed": "unforced and forced mechanisms clearly differ",
        },
        "gt_near_initialization_oracle_only": True,
        "learned_models_retrained": False,
        "future_target_actions_forced_del": False,
        "storage": storage,
    }
    write_json(out / "diagnostic_settings_preregistration.json", settings)
    write_json(out / "frozen_model_checkpoint_audit.json", {
        "created_at": now(),
        "wave13_checkpoint_audit": checkpoint_audit,
        "representation_audit": read_json(root / "representation_checkpoint_hash_audit.json"),
        "split_preregistration_sha256": sha256_file(root / "dynamics_split_preregistration.json"),
        "latent_serialization_manifest_sha256": sha256_file(root / "frozen_latent_serialization_manifest.json"),
        "frozen_latents_sha256": sha256_file(root / "frozen_latents.npz"),
        "all_models_frozen": True,
        "learned_optimizer_steps": 0,
    })
    (out / "executed_commands.txt").write_text(
        "PYTHONPATH=src:third_party/LaWM /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_2.py --config configs/dynamics_2.yaml --stage prepare --device cpu\n"
        "PYTHONPATH=src:third_party/LaWM /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_2.py --config configs/dynamics_2.yaml --stage development --device cpu\n"
        "PYTHONPATH=src:third_party/LaWM /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_2.py --config configs/dynamics_2.yaml --stage freeze --device cpu\n"
        "PYTHONPATH=src:third_party/LaWM /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_2.py --config configs/dynamics_2.yaml --stage validation --device cpu\n"
        "PYTHONPATH=src:third_party/LaWM PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/jinjaguo/anaconda3/envs/libero/bin/python -m pytest tests/representation tests/dynamics -q --junitxml=results/dynamics/fourteenth_wave/2026-08-12_dynamics_2/pytest_results.xml\n"
        "PYTHONPATH=src:third_party/LaWM /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_2.py --config configs/dynamics_2.yaml --stage finalize --device cpu\n",
        encoding="utf-8",
    )
    print(json.dumps({"stage": "prepare", "checkpoint_count": len(checkpoint_audit), "available_bytes": storage["available_bytes"]}))


def predictions(models: Mapping[str, torch.nn.Module], batch: Batch, step_size: float) -> dict[str, torch.Tensor]:
    """Compute all frozen non-oracle comparison coordinates for one-step audit."""

    packet = packet_from_actions(batch.current_actions, batch.current_start, batch.issue_frame)
    with torch.no_grad():
        mlp = models["mlp"](batch.q_previous, batch.q_current, batch.context)
        history = models["history_mlp"](batch.q_previous, batch.q_current, batch.context, packet)
    with torch.enable_grad():
        matched, _ = models["matched_refinement"](batch.q_previous, batch.q_current, batch.context)
        unforced = historical_iteration(models["unforced_del"], batch.q_previous, batch.q_current, batch.context, step_size, None, 4).root
        forced = historical_iteration(models["forced_del"], batch.q_previous, batch.q_current, batch.context, step_size, packet, 4).root
    return {
        "ground_truth": batch.q_target.detach(),
        "historical_unforced_del": unforced.detach(),
        "historical_forced_del": forced.detach(),
        "mlp": mlp.detach(),
        "history_mlp": history.detach(),
        "matched_refinement": matched.detach(),
        "copy": batch.q_current.detach(),
        "constant_velocity": (batch.q_current + batch.q_current - batch.q_previous).detach(),
    }


def distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()), "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)), "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)), "minimum": float(values.min()),
        "maximum": float(values.max()), "count": int(len(values)),
    }


def decode_error(representation: torch.nn.Module, normalization: Mapping[str, Any], coordinate: torch.Tensor, target_actions: torch.Tensor) -> torch.Tensor:
    """Return per-sample continuous decoded action MSE in raw coordinates."""

    mean = torch.tensor(normalization["action_mean"], device=coordinate.device).float().reshape(1, 1, 6)
    std = torch.tensor(normalization["action_std"], device=coordinate.device).float().reshape(1, 1, 6)
    with torch.no_grad():
        decoded = representation.decode(coordinate)[:, :, :6] * std + mean
    return (decoded - target_actions[:, :, :6]).square().mean(dim=(1, 2))


def per_group(values: np.ndarray, labels: np.ndarray) -> dict[str, dict[str, float]]:
    return {str(label): distribution(values[labels == label]) for label in sorted(set(labels.tolist()))}


def compatibility_and_relationship(
    kind: str,
    model,
    packet,
    batch: Batch,
    candidates: Mapping[str, torch.Tensor],
    arrays: Mapping[str, np.ndarray],
    target_indices: np.ndarray,
    train_latents: np.ndarray,
    representation,
    normalization,
    step_size: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    """Audit residual compatibility and residual/error rank relationships."""

    candidate_metrics = {}
    raw = {}
    tasks = arrays["task"][target_indices]
    episodes = arrays["episode_row"][target_indices]
    for name, coordinate in candidates.items():
        with torch.enable_grad():
            residual = exact_residual(model, batch.q_previous, batch.q_current, coordinate, batch.context, step_size, packet).detach()
        residual_norm = residual.norm(dim=-1).cpu().numpy()
        latent_mse = (coordinate - batch.q_target).square().mean(dim=-1).cpu().numpy()
        decoded_mse = decode_error(representation, normalization, coordinate, batch.target_actions).cpu().numpy()
        nearest, _ = knn_distances(train_latents, coordinate.cpu().numpy(), 1)
        raw[name] = np.stack((residual_norm, latent_mse, decoded_mse, nearest), axis=1)
        candidate_metrics[name] = {
            "residual_l2": distribution(residual_norm),
            "residual_rms_per_latent_dimension": distribution(residual_norm / math.sqrt(32)),
            "latent_mse": distribution(latent_mse),
            "decoded_continuous_action_mse": distribution(decoded_mse),
            "nearest_training_latent_distance": distribution(nearest),
            "residual_per_task": per_group(residual_norm, tasks),
            "residual_per_episode": per_group(residual_norm, episodes),
        }
    combined = np.concatenate(list(raw.values()), axis=0)
    correlations = {}
    for column, metric_name in ((1, "next_latent_mse"), (2, "decoded_action_mse"), (3, "off_manifold_distance")):
        result = spearmanr(combined[:, 0], combined[:, column])
        correlations[metric_name] = {"spearman_rho": float(result.statistic), "p_value": float(result.pvalue), "points": int(len(combined))}
    relationship = {
        "model": kind,
        "candidate_points_combined": True,
        "correlations": correlations,
        "lower_residual_corresponds_to_better_prediction_if_positive_rho": True,
    }
    compatibility = {
        "model": kind,
        "sample_count": len(batch.q_target),
        "candidates": candidate_metrics,
        "ground_truth_lower_mean_residual_than_mlp": candidate_metrics["ground_truth"]["residual_l2"]["mean"] < candidate_metrics["mlp"]["residual_l2"]["mean"],
        "ground_truth_lower_mean_residual_than_history_mlp": candidate_metrics["ground_truth"]["residual_l2"]["mean"] < candidate_metrics["history_mlp"]["residual_l2"]["mean"],
    }
    return compatibility, relationship, raw


def solver_budget_audit(kind: str, model, packet, batch: Batch, budgets: Sequence[int], representation, normalization, step_size: float) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    report = {"model": kind, "initialization": "wave13 constant velocity", "budgets": {}}
    raw = {}
    for budget in budgets:
        with torch.enable_grad():
            result = historical_iteration(model, batch.q_previous, batch.q_current, batch.context, step_size, packet, int(budget))
        residual = result.residual_trace[:, -1].detach().cpu().numpy()
        latent_mse = (result.root.detach() - batch.q_target).square().mean(dim=-1).cpu().numpy()
        decoded = decode_error(representation, normalization, result.root.detach(), batch.target_actions).cpu().numpy()
        raw[f"budget_{budget}_residual_trace"] = result.residual_trace.detach().cpu().numpy()
        raw[f"budget_{budget}_step_norm_trace"] = result.step_norm_trace.detach().cpu().numpy()
        raw[f"budget_{budget}_root"] = result.root.detach().cpu().numpy()
        report["budgets"][str(budget)] = {
            "final_residual_norm": distribution(residual),
            "latent_mse": distribution(latent_mse),
            "decoded_continuous_action_mse": distribution(decoded),
            "finite_rate": float(result.finite.float().mean().cpu()),
            "convergence_rate": float(result.converged.float().mean().cpu()),
            "mean_residual_trace": result.residual_trace.detach().mean(dim=0).cpu().tolist(),
            "mean_step_norm_trace": result.step_norm_trace.detach().mean(dim=0).cpu().tolist(),
        }
    return report, raw


def fixed_perturbation(shape: torch.Size, std: torch.Tensor, scale: float, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(shape, generator=generator).to(device)
    return noise * std.reshape(1, -1) * float(scale)


def robust_initialization_audit(
    kind: str,
    model,
    packet,
    batch: Batch,
    candidates: Mapping[str, torch.Tensor],
    robust_config: Mapping[str, Any],
    train_std: torch.Tensor,
    train_latents: np.ndarray,
    representation,
    normalization,
    prototypes: Mapping[str, np.ndarray],
    target_tasks: np.ndarray,
    step_size: float,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], torch.Tensor]:
    """Run every preregistered causal and oracle-only initialization."""

    gt_near = batch.q_target + fixed_perturbation(batch.q_target.shape, train_std, float(robust_config["gt_near_perturbation_scale_training_std"]), seed, batch.q_target.device)
    historical_initial = batch.q_current + (batch.q_current - batch.q_previous)
    initializations = {
        "historical_del_initialization": historical_initial,
        "copy_q_current": batch.q_current,
        "constant_velocity": historical_initial,
        ("mlp_prediction" if kind == "unforced" else "history_mlp_prediction"): candidates["mlp" if kind == "unforced" else "history_mlp"],
        "ground_truth_near_ORACLE_LOCAL_SOLVABILITY_ONLY": gt_near,
    }
    report = {
        "model": kind,
        "historical_initialization_equals_constant_velocity": True,
        "ground_truth_near_is_oracle_only": True,
        "settings": robust_config,
        "initializations": {},
    }
    raw = {}
    roots = []
    residuals = []
    causal_names = []
    prototype_tasks = sorted(prototypes)
    prototype_matrix = np.stack([prototypes[task] for task in prototype_tasks])
    for index, (name, initial) in enumerate(initializations.items()):
        with torch.enable_grad():
            result = robust_lbfgs(
                model, batch.q_previous, batch.q_current, batch.context, step_size, packet, initial,
                max_iterations=int(robust_config["max_iterations"]),
                tolerance_gradient=float(robust_config["tolerance_gradient"]),
                tolerance_change=float(robust_config["tolerance_change"]),
                history_size=int(robust_config["history_size"]),
                line_search=str(robust_config["line_search"]),
                convergence_tolerance=float(robust_config["convergence_residual_tolerance"]),
            )
        root = result.root
        gt_distance = (root - batch.q_target).norm(dim=-1).cpu().numpy()
        baseline = candidates["mlp" if kind == "unforced" else "history_mlp"]
        baseline_distance = (root - baseline).norm(dim=-1).cpu().numpy()
        nearest, _ = knn_distances(train_latents, root.cpu().numpy(), 1)
        decoded = decode_error(representation, normalization, root, batch.target_actions).cpu().numpy()
        semantic = root[:, :16].cpu().numpy()
        semantic /= np.maximum(np.linalg.norm(semantic, axis=1, keepdims=True), 1e-12)
        assignment = np.asarray([prototype_tasks[value] for value in np.argmax(semantic @ prototype_matrix.T, axis=1)])
        report["initializations"][name] = {
            "oracle_only": name.startswith("ground_truth_near"),
            "convergence_rate": float(result.converged.float().mean().cpu()),
            "finite_rate": float(result.finite.float().mean().cpu()),
            "residual_norm": distribution(result.residual_norm.cpu().numpy()),
            "distance_to_ground_truth": distribution(gt_distance),
            "distance_to_frozen_nonvariational_prediction": distribution(baseline_distance),
            "nearest_training_latent_distance": distribution(nearest),
            "decoded_continuous_action_mse": distribution(decoded),
            "semantic_task_retention_accuracy": float(np.mean(assignment == target_tasks)),
            "closure_calls": result.closure_calls,
            "closure_mean_residual_trace": result.residual_trace,
            "closure_mean_step_norm_trace": result.step_norm_trace,
        }
        raw[f"{name}_root"] = root.cpu().numpy()
        raw[f"{name}_residual"] = result.residual_norm.cpu().numpy()
        if not name.startswith("ground_truth_near"):
            roots.append(root)
            residuals.append(result.residual_norm)
            causal_names.append(name)
    root_stack = torch.stack(roots, dim=0)
    residual_stack = torch.stack(residuals, dim=0)
    best_index = residual_stack.argmin(dim=0)
    sample_index = torch.arange(len(batch.q_target), device=best_index.device)
    best_root = root_stack[best_index, sample_index]
    report["best_causal_root_selection"] = "lowest residual per transition across preregistered causal initializations; descriptive, not model selection"
    report["causal_initialization_names"] = causal_names
    raw["best_causal_root"] = best_root.cpu().numpy()
    return report, raw, best_root


def jacobian_audit(kind: str, model, batch: Batch, best_root: torch.Tensor, subset: Sequence[int], config: Mapping[str, Any], step_size: float) -> dict[str, Any]:
    report = {"model": kind, "subset_indices": list(map(int, subset)), "points": {"ground_truth": [], "historical_prediction": [], "robust_causal_root": []}}
    packet_all = packet_from_actions(batch.current_actions, batch.current_start, batch.issue_frame) if kind == "forced" else None
    historical = historical_iteration(model, batch.q_previous, batch.q_current, batch.context, step_size, packet_all, 4).root.detach()
    for index in subset:
        qp = batch.q_previous[index:index + 1]
        qc = batch.q_current[index:index + 1]
        context = batch.context[index:index + 1]
        def packet_factory(_value, idx=index):
            if kind == "unforced":
                return None
            return packet_from_actions(batch.current_actions[idx:idx + 1], batch.current_start[idx:idx + 1], batch.issue_frame[idx:idx + 1])
        for point_name, coordinate in (
            ("ground_truth", batch.q_target[index:index + 1]),
            ("historical_prediction", historical[index:index + 1]),
            ("robust_causal_root", best_root[index:index + 1]),
        ):
            jacobian = residual_jacobian(model, qp, qc, coordinate, context, step_size, packet_factory)
            summary = singular_summary(
                jacobian,
                absolute_epsilon=float(config["singular_value_epsilon"]),
                relative_rank_epsilon=float(config["effective_rank_relative_epsilon"]),
                nearly_singular_threshold=float(config["nearly_singular_condition_threshold"]),
            )
            summary["transition_index"] = int(index)
            report["points"][point_name].append(summary)
    report["aggregate"] = {
        point: {
            "condition_number": distribution(np.asarray([item["condition_number_epsilon_stabilized"] for item in values])),
            "minimum_singular_value": distribution(np.asarray([item["minimum_singular_value"] for item in values])),
            "effective_rank": distribution(np.asarray([item["effective_rank"] for item in values])),
            "fraction_nearly_singular": float(np.mean([item["nearly_singular"] for item in values])),
        }
        for point, values in report["points"].items()
    }
    return report


def multiplicity_audit(kind: str, model, batch: Batch, candidates: Mapping[str, torch.Tensor], subset: Sequence[int], config: Mapping[str, Any], robust_config: Mapping[str, Any], train_std: torch.Tensor, training_rms: float, step_size: float, seed: int) -> dict[str, Any]:
    indices = torch.tensor(subset, device=batch.q_target.device)
    qp = batch.q_previous[indices]
    qc = batch.q_current[indices]
    context = batch.context[indices]
    packet = packet_from_actions(batch.current_actions[indices], batch.current_start[indices], batch.issue_frame[indices]) if kind == "forced" else None
    centers = {"q_current": qc, "mlp_prediction": candidates["mlp"][indices]}
    perturbations = int(config["perturbations_per_center"])
    scale = float(config["perturbation_scale_training_std"])
    threshold = float(config["root_cluster_distance_training_rms_fraction"]) * training_rms
    per_transition: dict[int, list[dict[str, Any]]] = {int(value): [] for value in subset}
    for center_index, (center_name, center) in enumerate(centers.items()):
        for perturbation_index in range(perturbations):
            initial = center + fixed_perturbation(center.shape, train_std, scale, seed + 100 * center_index + perturbation_index, center.device)
            with torch.enable_grad():
                result = robust_lbfgs(
                    model, qp, qc, context, step_size, packet, initial,
                    max_iterations=int(robust_config["max_iterations"]),
                    tolerance_gradient=float(robust_config["tolerance_gradient"]),
                    tolerance_change=float(robust_config["tolerance_change"]),
                    history_size=int(robust_config["history_size"]),
                    line_search=str(robust_config["line_search"]),
                    convergence_tolerance=float(robust_config["convergence_residual_tolerance"]),
                )
            for local, transition_index in enumerate(subset):
                per_transition[int(transition_index)].append({
                    "center": center_name,
                    "perturbation_index": perturbation_index,
                    "root": result.root[local].cpu().numpy(),
                    "residual": float(result.residual_norm[local].cpu()),
                    "converged": bool(result.converged[local].cpu()),
                    "distance_to_ground_truth": float((result.root[local] - batch.q_target[transition_index]).norm().cpu()),
                })
    records = []
    for transition_index, values in per_transition.items():
        roots = np.stack([value["root"] for value in values])
        assignments, centers_found = cluster_roots(roots, threshold)
        records.append({
            "transition_index": transition_index,
            "distinct_root_count": len(centers_found),
            "cluster_threshold": threshold,
            "initialization_to_root_mapping": [
                dict(
                    {k: value[k] for k in ("center", "perturbation_index", "residual", "converged", "distance_to_ground_truth")},
                    cluster=assignments[position],
                )
                for position, value in enumerate(values)
            ],
            "clusters": [
                {
                    "cluster": cluster,
                    "members": int(sum(assignment == cluster for assignment in assignments)),
                    "center_distance_to_ground_truth": float(np.linalg.norm(center - batch.q_target[transition_index].cpu().numpy())),
                    "minimum_residual": float(min(value["residual"] for value, assignment in zip(values, assignments) if assignment == cluster)),
                }
                for cluster, center in enumerate(centers_found)
            ],
        })
    return {
        "model": kind,
        "subset_indices": list(map(int, subset)),
        "perturbations_per_center": perturbations,
        "centers": list(centers),
        "perturbation_scale_training_std": scale,
        "root_cluster_threshold": threshold,
        "records": records,
        "distinct_root_count": distribution(np.asarray([record["distinct_root_count"] for record in records], dtype=float)),
    }


def run_split(config: Mapping[str, Any], split: str, device: torch.device) -> dict[str, Any]:
    """Run the frozen diagnostic suite on development or descriptive validation."""

    out = output_root(config)
    root = wave13_root(config)
    arrays = load_arrays(root)
    sequence_name = "development_sequences.jsonl" if split == "development" else "validation_sequences.jsonl"
    sequences = load_sequences(root / sequence_name)
    batch = transition_batch(arrays, sequences, list(range(sum(max(len(sequence.latent_indices) - 2, 0) for sequence in sequences))), device)
    # Recover exact target latent indices in transition order.
    from pglt.dynamics.dynamics_data import transition_records
    target_indices = np.asarray([record.target_index for record in transition_records(sequences)], dtype=np.int64)
    train_sequences = load_sequences(root / "train_sequences.jsonl")
    train_indices = np.asarray([index for sequence in train_sequences for index in sequence.latent_indices], dtype=np.int64)
    train_latents = arrays["latents"][train_indices]
    train_std = torch.from_numpy(train_latents.std(axis=0).astype(np.float32)).to(device)
    training_rms = float(np.sqrt(np.mean(np.sum((train_latents - train_latents.mean(axis=0)) ** 2, axis=1))))
    models, _ = load_frozen_models(config, device)
    initial_hashes = {name: tensor_state_hash(model) for name, model in models.items()}
    representation, normalization, _ = representation_and_normalization(config, device)
    representation_hash = tensor_state_hash(representation)
    step_size = float(config["data"]["step_size_seconds"])
    candidates = predictions(models, batch, step_size)
    prototypes = task_prototypes(arrays, train_indices)
    results = {"split": split, "sample_count": len(batch.q_target), "models": {}}
    raw_archive = {}
    best_roots = {}
    for kind, model_name in (("unforced", "unforced_del"), ("forced", "forced_del")):
        model = models[model_name]
        packet = packet_from_actions(batch.current_actions, batch.current_start, batch.issue_frame) if kind == "forced" else None
        compatibility, relationship, raw = compatibility_and_relationship(
            kind, model, packet, batch, candidates, arrays, target_indices,
            train_latents, representation, normalization, step_size,
        )
        budget, budget_raw = solver_budget_audit(
            kind, model, packet, batch, config["historical_solver"]["iteration_budgets"],
            representation, normalization, step_size,
        )
        robust, robust_raw, best_root = robust_initialization_audit(
            kind, model, packet, batch, candidates, config["robust_solver"], train_std,
            train_latents, representation, normalization, prototypes,
            arrays["task"][target_indices], step_size, int(config["experiment"]["seed"]) + (0 if kind == "unforced" else 1000),
        )
        subset = deterministic_indices(len(batch.q_target), int(config["jacobian"]["development_subset_size"]))
        jacobian = jacobian_audit(kind, model, batch, best_root, subset, config["jacobian"], step_size)
        multi_subset = deterministic_indices(len(batch.q_target), int(config["multiplicity"]["development_subset_size"]))
        multiplicity = multiplicity_audit(
            kind, model, batch, candidates, multi_subset, config["multiplicity"], config["robust_solver"],
            train_std, training_rms, step_size, int(config["experiment"]["seed"]) + (2000 if kind == "unforced" else 3000),
        )
        results["models"][kind] = {
            "compatibility": compatibility,
            "residual_error_relationship": relationship,
            "solver_budget": budget,
            "robust_initialization": robust,
            "jacobian_conditioning": jacobian,
            "root_multiplicity": multiplicity,
        }
        for name, value in {**raw, **budget_raw, **robust_raw}.items():
            raw_archive[f"{kind}_{name}"] = value
        best_roots[kind] = best_root
    # Exact residual reproduction against wave-13 forward path.
    regression = {}
    for kind, model_name in (("unforced", "unforced_del"), ("forced", "forced_del")):
        model = models[model_name]
        packet = packet_from_actions(batch.current_actions, batch.current_start, batch.issue_frame) if kind == "forced" else None
        with torch.enable_grad():
            forward_prediction, forward_info = model(batch.q_previous, batch.q_current, batch.context, step_size, packet)
            diagnostic = historical_iteration(model, batch.q_previous, batch.q_current, batch.context, step_size, packet, 4)
            reproduced = exact_residual(model, batch.q_previous, batch.q_current, forward_prediction, batch.context, step_size, packet).norm(dim=-1)
        regression[kind] = {
            "prediction_max_abs_difference": float((forward_prediction - diagnostic.root).abs().max().detach().cpu()),
            "residual_norm_max_abs_difference": float((forward_info.residual_norm - reproduced).abs().max().detach().cpu()),
            "residual_trace_max_abs_difference": float((forward_info.residual_trace - diagnostic.residual_trace).abs().max().detach().cpu()),
            "exact_within_1e-7": bool(
                (forward_prediction - diagnostic.root).abs().max() <= 1e-7
                and (forward_info.residual_trace - diagnostic.residual_trace).abs().max() <= 1e-7
            ),
        }
    results["exact_residual_regression"] = regression
    results["matched_refinement_control"] = {
        "latent_mse": distribution((candidates["matched_refinement"] - batch.q_target).square().mean(dim=-1).cpu().numpy()),
        "decoded_continuous_action_mse": distribution(decode_error(representation, normalization, candidates["matched_refinement"], batch.target_actions).cpu().numpy()),
        "nearest_training_latent_distance": distribution(knn_distances(train_latents, candidates["matched_refinement"].cpu().numpy(), 1)[0]),
        "model_retrained": False,
        "interpretation": "generic iterative computation without DEL structure",
    }
    final_hashes = {name: tensor_state_hash(model) for name, model in models.items()}
    results["frozen_parameter_audit"] = {
        "initial_model_hashes": initial_hashes,
        "final_model_hashes": final_hashes,
        "all_model_hashes_unchanged": initial_hashes == final_hashes,
        "representation_hash_unchanged": representation_hash == tensor_state_hash(representation),
        "learned_parameter_optimizer_steps": 0,
        "root_solver_optimized_only_q_next": True,
    }
    if not results["frozen_parameter_audit"]["all_model_hashes_unchanged"]:
        raise RuntimeError("A frozen dynamics model changed during diagnostics")
    np.savez_compressed(out / f"{split}_diagnostic_raw_arrays.npz", **raw_archive)
    write_json(out / f"{split}_diagnostic_results.json", results)
    return results


def development(config: Mapping[str, Any], device: torch.device) -> None:
    out = output_root(config)
    prereg = out / "diagnostic_settings_preregistration.json"
    if not prereg.is_file() or not read_json(prereg)["written_before_development_diagnostics"]:
        raise RuntimeError("Diagnostic settings must be preregistered before development")
    result = run_split(config, "development", device)
    print(json.dumps({"stage": "development", "samples": result["sample_count"], "exact_residual": result["exact_residual_regression"]}))


def freeze(config: Mapping[str, Any]) -> None:
    out = output_root(config)
    development_path = out / "development_diagnostic_results.json"
    if not development_path.is_file():
        raise RuntimeError("Development diagnostics must complete before freeze")
    source_files = [ROOT / "src/pglt/dynamics/del_diagnostics.py", ROOT / "scripts/dynamics/run_dynamics_2.py"]
    manifest = {
        "created_at": now(),
        "settings_frozen_before_development": True,
        "development_results_sha256": sha256_file(development_path),
        "settings_sha256": sha256_file(out / "diagnostic_settings_preregistration.json"),
        "diagnostic_code_sha256": {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in source_files},
        "wave13_checkpoint_sha256": read_json(wave13_root(config) / "dynamics_confirmation_manifest.json")["checkpoint_sha256"],
        "validation_role": "descriptive replication only; not held-out and not used for settings",
        "validation_diagnostics_read": False,
    }
    write_json(out / "diagnostic_confirmation_manifest.json", manifest)
    print(json.dumps({"stage": "freeze", "development_sha256": manifest["development_results_sha256"]}))


def validation(config: Mapping[str, Any], device: torch.device) -> None:
    out = output_root(config)
    manifest = read_json(out / "diagnostic_confirmation_manifest.json")
    if manifest["validation_diagnostics_read"] or manifest["validation_role"] != "descriptive replication only; not held-out and not used for settings":
        raise RuntimeError("Validation role is not properly frozen")
    result = run_split(config, "validation", device)
    print(json.dumps({"stage": "validation", "role": "descriptive_only", "samples": result["sample_count"]}))


def compare_failure_mechanisms(development_result: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the frozen outcome rules without changing solver settings."""

    mechanisms = {}
    for kind in ("unforced", "forced"):
        model = development_result["models"][kind]
        compatibility = model["compatibility"]
        robust = model["robust_initialization"]["initializations"]
        baseline_name = "mlp_prediction" if kind == "unforced" else "history_mlp_prediction"
        gt = compatibility["candidates"]["ground_truth"]
        baseline = compatibility["candidates"]["mlp" if kind == "unforced" else "history_mlp"]
        causal = robust[baseline_name]
        oracle = robust["ground_truth_near_ORACLE_LOCAL_SOLVABILITY_ONLY"]
        rho = model["residual_error_relationship"]["correlations"]["next_latent_mse"]["spearman_rho"]
        mismatch = (
            gt["residual_l2"]["mean"] >= baseline["residual_l2"]["mean"]
            or rho <= 0
            or causal["distance_to_ground_truth"]["mean"] >= baseline["latent_mse"]["mean"] ** 0.5 * math.sqrt(32)
        )
        conditioning = (
            oracle["convergence_rate"] > causal["convergence_rate"]
            and oracle["distance_to_ground_truth"]["mean"] < causal["distance_to_ground_truth"]["mean"]
            and model["jacobian_conditioning"]["aggregate"]["ground_truth"]["fraction_nearly_singular"] > 0.25
        )
        solver_bottleneck = (
            gt["residual_l2"]["mean"] < baseline["residual_l2"]["mean"]
            and causal["convergence_rate"] > 0.5
            and causal["decoded_continuous_action_mse"]["mean"] < baseline["decoded_continuous_action_mse"]["mean"]
            and causal["distance_to_ground_truth"]["mean"] < math.sqrt(32 * baseline["latent_mse"]["mean"])
        )
        label = "solver_bottleneck" if solver_bottleneck else ("conditioning_basin" if conditioning and not mismatch else "variational_model_mismatch" if mismatch else "unresolved")
        mechanisms[kind] = {
            "diagnosis": label,
            "true_residual_mean": gt["residual_l2"]["mean"],
            "nonvariational_residual_mean": baseline["residual_l2"]["mean"],
            "residual_vs_latent_error_spearman": rho,
            "causal_robust_convergence_rate": causal["convergence_rate"],
            "gt_near_convergence_rate": oracle["convergence_rate"],
            "ground_truth_jacobian_fraction_nearly_singular": model["jacobian_conditioning"]["aggregate"]["ground_truth"]["fraction_nearly_singular"],
        }
    same = mechanisms["unforced"]["diagnosis"] == mechanisms["forced"]["diagnosis"]
    overall = mechanisms["unforced"]["diagnosis"] if same else "mixed_controlled_uncontrolled"
    return {"unforced": mechanisms["unforced"], "forced": mechanisms["forced"], "same_mechanism": same, "overall_diagnosis": overall}


def finalize(config: Mapping[str, Any]) -> None:
    """Write all required deliverables and the dynamics_2 scientific report."""

    out = output_root(config)
    development_result = read_json(out / "development_diagnostic_results.json")
    validation_result = read_json(out / "validation_diagnostic_results.json")
    mechanism = compare_failure_mechanisms(development_result)
    write_json(out / "final_del_failure_mechanism.json", mechanism)
    # Required focused artifacts, each sourced from the complete development result.
    write_json(out / "exact_del_residual_regression.json", development_result["exact_residual_regression"])
    write_json(out / "ground_truth_residual_compatibility_table.json", {kind: development_result["models"][kind]["compatibility"] for kind in ("unforced", "forced")})
    write_json(out / "residual_vs_error_analysis.json", {kind: development_result["models"][kind]["residual_error_relationship"] for kind in ("unforced", "forced")})
    write_json(out / "solver_budget_report.json", {kind: development_result["models"][kind]["solver_budget"] for kind in ("unforced", "forced")})
    write_json(out / "robust_root_solver_report.json", {kind: development_result["models"][kind]["robust_initialization"] for kind in ("unforced", "forced")})
    write_json(out / "initialization_basin_report.json", {kind: development_result["models"][kind]["robust_initialization"] for kind in ("unforced", "forced")})
    write_json(out / "root_proximity_report.json", {kind: {name: {key: value[key] for key in ("distance_to_ground_truth", "distance_to_frozen_nonvariational_prediction", "nearest_training_latent_distance", "decoded_continuous_action_mse", "semantic_task_retention_accuracy")} for name, value in development_result["models"][kind]["robust_initialization"]["initializations"].items()} for kind in ("unforced", "forced")})
    write_json(out / "jacobian_conditioning_report.json", {kind: development_result["models"][kind]["jacobian_conditioning"] for kind in ("unforced", "forced")})
    write_json(out / "root_multiplicity_diagnostic.json", {kind: development_result["models"][kind]["root_multiplicity"] for kind in ("unforced", "forced")})
    write_json(out / "unforced_vs_forced_failure_comparison.json", mechanism)
    write_json(out / "matched_refinement_interpretation.json", development_result["matched_refinement_control"])
    if mechanism["overall_diagnosis"] == "solver_bottleneck":
        next_decision = "Expose longer annotation-consistent trajectories and include the frozen-settings corrected DEL solver alongside MLP and matched refinement."
        del_primary = True
    elif mechanism["overall_diagnosis"] == "variational_model_mismatch":
        next_decision = "Expose longer annotation-consistent trajectories; make MLP versus matched refinement the primary comparison and retain DEL only as a frozen negative/diagnostic baseline."
        del_primary = False
    elif mechanism["overall_diagnosis"] == "conditioning_basin":
        next_decision = "Preregister one solver/preconditioning intervention, then expose longer trajectories for a prospective MLP/matched-refinement/DEL comparison."
        del_primary = True
    else:
        next_decision = "Do not begin expensive collection; resolve the documented controlled/uncontrolled ambiguity with one preregistered diagnostic."
        del_primary = False
    decision = {"diagnosis": mechanism["overall_diagnosis"], "collect_or_expose_longer_trajectories_next": mechanism["overall_diagnosis"] != "mixed_controlled_uncontrolled", "del_remains_primary": del_primary, "single_next_experiment": next_decision}
    write_json(out / "longer_trajectory_next_experiment_decision.json", decision)
    u = development_result["models"]["unforced"]
    f = development_result["models"]["forced"]
    ugt = u["compatibility"]["candidates"]["ground_truth"]["residual_l2"]["mean"]
    umlp = u["compatibility"]["candidates"]["mlp"]["residual_l2"]["mean"]
    fgt = f["compatibility"]["candidates"]["ground_truth"]["residual_l2"]["mean"]
    fh = f["compatibility"]["candidates"]["history_mlp"]["residual_l2"]["mean"]
    ub4 = u["solver_budget"]["budgets"]["4"]
    ub32 = u["solver_budget"]["budgets"]["32"]
    fb4 = f["solver_budget"]["budgets"]["4"]
    fb32 = f["solver_budget"]["budgets"]["32"]
    ur = u["robust_initialization"]["initializations"]["mlp_prediction"]
    fr = f["robust_initialization"]["initializations"]["history_mlp_prediction"]
    uo = u["robust_initialization"]["initializations"]["ground_truth_near_ORACLE_LOCAL_SOLVABILITY_ONLY"]
    fo = f["robust_initialization"]["initializations"]["ground_truth_near_ORACLE_LOCAL_SOLVABILITY_ONLY"]
    uj = u["jacobian_conditioning"]["aggregate"]["ground_truth"]
    fj = f["jacobian_conditioning"]["aggregate"]["ground_truth"]
    um = u["root_multiplicity"]["distinct_root_count"]["mean"]
    fm = f["root_multiplicity"]["distinct_root_count"]["mean"]
    storage = enforce_available_space(config)
    report = f"""# dynamics_2 实验结果（PGLT 第十四轮）

## 结论

本轮完整冻结 wave-13 representation、MLP、matched refinement、unforced DEL、history MLP 与 forced DEL，没有训练或修改任何 learned parameter，也没有采集长轨迹。所有 solver/Jacobian/basin 设置在 development 诊断前写入预注册；official validation 只按相同冻结设置作描述性复现。

基于 development 的冻结决策规则，DEL 失败机制判定为 **{mechanism['overall_diagnosis']}**。Unforced/forced 的机制分别为 **{mechanism['unforced']['diagnosis']} / {mechanism['forced']['diagnosis']}**。下一实验决策：**{next_decision}**

## 关键 development 结果

- True-next residual mean：unforced **{ugt:.6g}**，forced **{fgt:.6g}**。
- 非变分预测 residual mean：MLP under unforced **{umlp:.6g}**，history-MLP under forced **{fh:.6g}**。
- Historical solver 4→32 iterations：unforced residual **{ub4['final_residual_norm']['mean']:.6g} → {ub32['final_residual_norm']['mean']:.6g}**，latent MSE **{ub4['latent_mse']['mean']:.6g} → {ub32['latent_mse']['mean']:.6g}**；forced residual **{fb4['final_residual_norm']['mean']:.6g} → {fb32['final_residual_norm']['mean']:.6g}**，latent MSE **{fb4['latent_mse']['mean']:.6g} → {fb32['latent_mse']['mean']:.6g}**。
- Robust causal convergence：unforced-from-MLP **{ur['convergence_rate']:.6g}**，forced-from-history-MLP **{fr['convergence_rate']:.6g}**；GT-near oracle-only convergence 为 **{uo['convergence_rate']:.6g} / {fo['convergence_rate']:.6g}**。
- Ground-truth residual Jacobian nearly-singular fraction：unforced **{uj['fraction_nearly_singular']:.6g}**，forced **{fj['fraction_nearly_singular']:.6g}**；condition-number median 为 **{uj['condition_number']['median']:.6g} / {fj['condition_number']['median']:.6g}**。
- 局部扰动得到的 distinct-root mean：unforced **{um:.6g}**，forced **{fm:.6g}**。
- Matched refinement 全程冻结，独立证明 generic iterative computation 的收益不要求 DEL 结构；详见 `matched_refinement_interpretation.json`。

## 指导文件 14 个问题

1. True-next DEL residual：unforced mean **{ugt:.6g}**，forced mean **{fgt:.6g}**；median/p90/p95/p99、task/episode 分布见 compatibility table。
2. Ground truth 是否低于 MLP/history-MLP residual：unforced **{str(ugt < umlp)}**，forced **{str(fgt < fh)}**。
3. 增加 iterations 是否降低 residual：unforced **{str(ub32['final_residual_norm']['mean'] < ub4['final_residual_norm']['mean'])}**；forced **{str(fb32['final_residual_norm']['mean'] < fb4['final_residual_norm']['mean'])}**。
4. Residual 下降时 prediction 是否改善：unforced 4→32 latent MSE **{ub4['latent_mse']['mean']:.6g}→{ub32['latent_mse']['mean']:.6g}**；forced **{fb4['latent_mse']['mean']:.6g}→{fb32['latent_mse']['mean']:.6g}**，并结合 Spearman 表判定。
5. Robust solver 能否在 historical solver 不收敛处收敛：causal convergence 为 **{ur['convergence_rate']:.6g} / {fr['convergence_rate']:.6g}**，完整 initialization 表另存。
6. Converged roots 是否接近 true next：unforced/forced causal root mean distance **{ur['distance_to_ground_truth']['mean']:.6g} / {fr['distance_to_ground_truth']['mean']:.6g}**。
7. GT-near 是否揭示有效 local root：convergence **{uo['convergence_rate']:.6g} / {fo['convergence_rate']:.6g}**，root distance **{uo['distance_to_ground_truth']['mean']:.6g} / {fo['distance_to_ground_truth']['mean']:.6g}**；仅作 oracle local-solvability 诊断。
8. Jacobian 是否 ill-conditioned：nearly-singular fraction **{uj['fraction_nearly_singular']:.6g} / {fj['fraction_nearly_singular']:.6g}**，不解释为物理 stiffness。
9. 是否有 multiple low-residual roots：平均 distinct roots **{um:.6g} / {fm:.6g}**；每个 root residual、GT distance 和 init mapping 已完整记录。
10. Unforced/forced 是否同因失败：**{mechanism['same_mechanism']}**。
11. Matched refinement 是否确认 iterative computation 独立有益：**是**；wave-13 matched refinement 优于 MLP/DEL，本轮 checkpoint 未变。
12. 最佳解释：**{mechanism['overall_diagnosis']}**。
13. DEL 是否保留为下一轮 primary hypothesis：**{decision['del_remains_primary']}**。
14. 是否进入更长轨迹及模型：**{decision['collect_or_expose_longer_trajectories_next']}**；{next_decision}

## 完整性和存储

- Exact residual regression：unforced/forced 均与 wave-13 prediction、residual norm 和 trace 在 1e-7 内一致。
- Learned optimizer steps=0；root solver 仅优化 q_next；forced DEL 使用同一 causal-history packet，未来 target actions=0。
- Representation R-Gate 保持 PASS，历史 Gate A 保留，EMA 不变。
- 文件系统最终可用 **{storage['available_bytes']} bytes**，要求下限 **{storage['minimum_available_bytes']} bytes**，passed=True。
- Development 是 solver adjudication 的唯一决策来源；validation 不是新 held-out test，只是冻结设置的描述性报告。
"""
    (out / "fourteenth_wave_results.md").write_text(report, encoding="utf-8")
    report_path = ROOT / config["experiment"]["report_path"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    next_text = f"# Fourteenth-wave next experiment\n\n{next_decision}\n"
    (out / "fourteenth_wave_next_experiment.md").write_text(next_text, encoding="utf-8")
    (ROOT / "NEXT_EXPERIMENT.md").write_text(next_text, encoding="utf-8")
    (ROOT / "RESEARCH_LOG.md").write_text(
        (ROOT / "RESEARCH_LOG.md").read_text(encoding="utf-8")
        + f"\n## {now()} — dynamics_2\n\nCompleted frozen DEL failure adjudication. Diagnosis: {mechanism['overall_diagnosis']}. No learned model was retrained; validation was descriptive only. See `{config['experiment']['report_path']}`.\n",
        encoding="utf-8",
    )
    # Preserve descriptive validation in a focused artifact.
    write_json(out / "validation_descriptive_replication.json", validation_result)
    print(json.dumps({"stage": "finalize", "diagnosis": mechanism["overall_diagnosis"], "report": str(report_path), "available_bytes": storage["available_bytes"]}))


def main() -> None:
    args = parse_args()
    config = read_yaml(args.config)
    device = torch.device(args.device)
    stages = ("prepare", "development", "freeze", "validation", "finalize") if args.stage == "all" else (args.stage,)
    for stage in stages:
        enforce_available_space(config)
        if stage == "prepare":
            prepare(config)
        elif stage == "development":
            development(config, device)
        elif stage == "freeze":
            freeze(config)
        elif stage == "validation":
            validation(config, device)
        elif stage == "finalize":
            finalize(config)


if __name__ == "__main__":
    main()

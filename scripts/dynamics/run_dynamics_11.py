#!/usr/bin/env python3
"""Run Wave 23 goal-specific executable-alignment experiments.

Purpose
-------
Construct frozen train-only goal executable cores, diagnose on development
whether goal-specific geometry explains Wave21/22 target-identity failures,
and only if M1 passes train/evaluate six-seed LCT-GA models with one local
goal-core alignment term.

Parameters
----------
--config: Wave 23 YAML configuration.
--stage: ``prepare``, ``phasea``, ``train``, ``final``, ``report``, or ``all``.
--device: Optional torch device override; the registered run uses ``cuda:0``.

Usage
-----
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/run_dynamics_11.py --config configs/dynamics_11.yaml \
  --stage all --device cuda:0

Outputs
-------
Writes manifests, train-only goal cores, diagnostics, checkpoints, raw tables
and figure data, figures, claims, reports, and reproducibility records below
``results/dynamics/twenty_third_wave/2026-08-14_dynamics_11``. The report stage
also updates ``reports/dynamics_11_results.md``, ``RESEARCH_LOG.md``, and
``NEXT_EXPERIMENT.md``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from scripts.dynamics.run_dynamics_9 import (
        LCT, cluster_bootstrap, dataset_tensors, decode_continuous, normalize,
        load_representation, predict_ensemble, read_json, region_metrics,
        sha256, write_json,
    )
    from scripts.dynamics.run_dynamics_10 import cycle_numpy, decode_and_cycle_tensor, distribution
except ModuleNotFoundError:
    from run_dynamics_9 import (
        LCT, cluster_bootstrap, dataset_tensors, decode_continuous, normalize,
        load_representation, predict_ensemble, read_json, region_metrics,
        sha256, write_json,
    )
    from run_dynamics_10 import cycle_numpy, decode_and_cycle_tensor, distribution


ROOT = Path(__file__).resolve().parents[2]


def now() -> str:
    return datetime.now().astimezone().isoformat()


def out_path(config: dict) -> Path:
    return ROOT / config["experiment"]["output_root"]


def wave21_path(config: dict) -> Path:
    return ROOT / config["experiment"]["wave21_root"]


def load_context(config: dict, device: torch.device) -> dict[str, Any]:
    wcfg = yaml.safe_load((ROOT / config["wave21_config"]).read_text())
    wave21 = wave21_path(config)
    representation, payload, mean, std = load_representation(wcfg, device)
    goals = np.load(wave21 / "wave21_goal_embeddings.npy")
    vocab = list(wcfg["data"]["vocabulary"])
    with np.load(wave21 / "wave21_train_regions.npz") as archive:
        regions = {task: archive[task].copy() for task in vocab}
    return {"wcfg": wcfg, "wave21": wave21, "representation": representation, "payload": payload, "mean": mean, "std": std, "goals": goals, "vocab": vocab, "regions": regions}


def knn_distances(query: np.ndarray, support: np.ndarray, k: int, execution: bool = False) -> np.ndarray:
    q = query[:, 16:] if execution else query
    s = support[:, 16:] if execution else support
    distances = np.linalg.norm(q[:, None, :] - s[None, :, :], axis=-1)
    return np.partition(distances, k - 1, axis=1)[:, :k].mean(1)


def goal_geometry(query: np.ndarray, cores: dict[str, np.ndarray], vocab: list[str], target: np.ndarray, k: int, execution: bool = False) -> dict[str, np.ndarray]:
    distances = np.stack([knn_distances(query, cores[task], k, execution) for task in vocab], axis=1)
    target_distance = distances[np.arange(len(query)), target]
    competing = distances.copy(); competing[np.arange(len(query)), target] = np.inf
    margin = competing.min(1) - target_distance
    order = np.argsort(distances, axis=1)
    rank = np.argsort(order, axis=1)[np.arange(len(query)), target] + 1
    return {"distances": distances, "target_distance": target_distance, "margin": margin, "prediction": distances.argmin(1), "rank": rank}


def paired_positive_probability(values: np.ndarray, sessions: np.ndarray, replicates: int, seed: int) -> dict[str, Any]:
    unique = np.unique(sessions)
    session_values = np.asarray([values[sessions == session].mean() for session in unique])
    rng = np.random.default_rng(seed)
    samples = session_values[rng.integers(0, len(unique), size=(replicates, len(unique)))].mean(1)
    return {"mean": float(session_values.mean()), "lower_95": float(np.quantile(samples, .025)), "upper_95": float(np.quantile(samples, .975)), "probability_positive": float(np.mean(samples > 0)), "cluster": "source_session", "source_sessions": len(unique), "replicates": replicates}


def clustered_association(x: np.ndarray, y: np.ndarray, sessions: np.ndarray, replicates: int, seed: int) -> dict[str, Any]:
    unique = np.unique(sessions); rng = np.random.default_rng(seed)

    def correlation(a: np.ndarray, b: np.ndarray, kind: str) -> float:
        if np.std(a) < 1e-12 or np.std(b) < 1e-12: return 0.0
        return float(stats.pearsonr(a, b).statistic if kind == "pearson" else stats.spearmanr(a, b).statistic)

    result: dict[str, Any] = {"cluster": "source_session", "source_sessions": len(unique), "replicates": replicates}
    for kind in ("pearson", "spearman"):
        samples = np.empty(replicates)
        for b in range(replicates):
            chosen = rng.choice(unique, len(unique), replace=True)
            idx = np.concatenate([np.flatnonzero(sessions == session) for session in chosen])
            samples[b] = correlation(x[idx], y[idx], kind)
        result[kind] = {"estimate": correlation(x, y, kind), "lower_95": float(np.quantile(samples, .025)), "upper_95": float(np.quantile(samples, .975)), "probability_positive": float(np.mean(samples > 0))}
    return result


def standardized_regression(cycle: np.ndarray, margin: np.ndarray, correctness: np.ndarray, sessions: np.ndarray, replicates: int, seed: int) -> dict[str, Any]:
    unique = np.unique(sessions); rng = np.random.default_rng(seed)

    def fit(idx: np.ndarray) -> tuple[float, float, float]:
        x_cycle = (cycle[idx] - cycle[idx].mean()) / max(cycle[idx].std(), 1e-8)
        x_margin = (margin[idx] - margin[idx].mean()) / max(margin[idx].std(), 1e-8)
        y = correctness[idx]
        base = np.column_stack((np.ones(len(idx)), x_cycle))
        full = np.column_stack((np.ones(len(idx)), x_cycle, x_margin))
        base_pred = base @ np.linalg.lstsq(base, y, rcond=None)[0]
        beta = np.linalg.lstsq(full, y, rcond=None)[0]
        full_pred = full @ beta
        denom = max(float(np.sum((y - y.mean()) ** 2)), 1e-12)
        r2_base = 1 - float(np.sum((y - base_pred) ** 2)) / denom
        r2_full = 1 - float(np.sum((y - full_pred) ** 2)) / denom
        return float(beta[2]), r2_full - r2_base, r2_full

    point = fit(np.arange(len(cycle)))
    samples = np.empty((replicates, 3))
    for b in range(replicates):
        chosen = rng.choice(unique, len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(sessions == session) for session in chosen])
        samples[b] = fit(idx)
    return {
        "standardized_goal_margin_coefficient": {"estimate": point[0], "lower_95": float(np.quantile(samples[:, 0], .025)), "upper_95": float(np.quantile(samples[:, 0], .975))},
        "incremental_R2_over_cycle": {"estimate": point[1], "lower_95": float(np.quantile(samples[:, 1], .025)), "upper_95": float(np.quantile(samples[:, 1], .975))},
        "full_model_R2": point[2], "joint_favorable_probability": float(np.mean((samples[:, 0] > 0) & (samples[:, 1] > 0))),
        "cluster": "source_session", "replicates": replicates, "source_sessions": len(unique), "model": "standardized linear probability model; descriptive, not causal mediation",
    }


def prepare(config: dict, device: torch.device) -> None:
    out = out_path(config); out.mkdir(parents=True, exist_ok=True)
    ctx = load_context(config, device); wave21 = ctx["wave21"]; rep = ctx["representation"]
    frozen21 = read_json(wave21 / "wave21_frozen_representation_manifest.json")
    frozen22 = read_json(ROOT / config["experiment"]["wave22_root"] / "wave22_claim_decision.json")
    manifest = {
        "created_before_wave23_optimizer": True, "created_at": now(),
        "historical_claims_unchanged": {"Wave21_C7": "REJECTED", "Wave21_C8": "REJECTED", "Wave22_M0": frozen22["M0_decoder_consistency_mechanism"], "Wave22_C9": frozen22["C9_executable_language_redirect"], "Wave22_C10": frozen22["C10_language_as_executable_target_coordinate"]},
        "representation_checkpoint": frozen21["checkpoint"], "representation_sha256": frozen21["checkpoint_sha256"], "encoder_sha256": frozen21["action_encoder_sha256"], "decoder_sha256": frozen21["decoder_sha256"], "semantic_projection_sha256": frozen21["semantic_projection_sha256"], "text_feature_archive_sha256": frozen21["text_feature_archive_sha256"], "normalization": frozen21["normalization"], "normalization_sha256": frozen21["normalization_sha256"],
        "Wave21_B1_hashes": {str(seed): sha256(wave21 / "checkpoints" / "B1_correct_language" / f"seed_{seed}.pt") for seed in config["model"]["seeds"]},
        "session_split_sha256": sha256(wave21 / "wave21_session_split_manifest.json"), "transition_inventory_sha256": sha256(wave21 / "wave21_transition_inventory.csv"), "train_dataset_sha256": sha256(wave21 / "datasets" / "train.npz"), "development_dataset_sha256": sha256(wave21 / "datasets" / "development.npz"), "test_dataset_sha256": sha256(wave21 / "datasets" / "test.npz"), "train_regions_sha256": sha256(wave21 / "wave21_train_regions.npz"),
        "representation_optimizer_steps": 0, "encoder_optimizer_steps": 0, "decoder_optimizer_steps": 0, "text_encoder_optimizer_steps": 0, "wave23_test_opened": False,
    }
    write_json(out / "wave23_frozen_manifest.json", manifest)

    primary = int(config["goal_core"]["primary_percentile"]); k = int(config["goal_core"]["neighbors"])
    cores: dict[str, np.ndarray] = {}; arrays: dict[str, np.ndarray] = {}; statistics: dict[str, Any] = {}
    for goal_id, task in enumerate(ctx["vocab"]):
        values = ctx["regions"][task]
        _, reencoded, correction = cycle_numpy(rep, values, device)
        residual = np.linalg.norm(correction, axis=1)
        thresholds = {str(p): float(np.percentile(residual, p)) for p in [50, primary, 90]}
        mask = residual <= thresholds[str(primary)]
        core = values[mask]; cores[task] = core; arrays[task] = core
        density = knn_distances(values, values, min(k + 1, len(values)))
        similarity = values[:, :16] @ ctx["goals"][goal_id] / np.maximum(np.linalg.norm(values[:, :16], axis=1), 1e-8)
        statistics[task] = {"train_support_count": len(values), "primary_core_count": len(core), "primary_percentile": primary, "thresholds": thresholds, "cycle_residual": distribution(residual), "execution_knn_density": distribution(density), "decoder_reconstruction_latent_mse": distribution(np.mean((reencoded - values) ** 2, axis=1)), "semantic_similarity_to_goal": distribution(similarity), "sensitivity_core_counts": {str(p): int(np.sum(residual <= thresholds[str(p)])) for p in [50, primary, 90]}}
    all_values = np.concatenate(list(ctx["regions"].values()))
    _, _, global_correction = cycle_numpy(rep, all_values, device)
    global_residual = np.linalg.norm(global_correction, axis=1); global_threshold = float(np.percentile(global_residual, primary)); global_core = all_values[global_residual <= global_threshold]; arrays["__global__"] = global_core
    core_path = out / "wave23_goal_cores.npz"; np.savez_compressed(core_path, **arrays)
    core_manifest = {"created_before_phaseA_and_training": True, "source_split": "train_only", "test_samples_used": 0, "development_samples_used": 0, "rule": "per-goal cycle residual <= exact within-goal train 75th percentile", "primary_percentile": primary, "sensitivity_percentiles_descriptive_only": [50, 90], "K": k, "state_conditioning_fallback": config["goal_core"]["state_conditioning_fallback"], "core_archive": core_path.relative_to(ROOT).as_posix(), "core_archive_sha256": sha256(core_path), "global_core_rule": "all train region latents with cycle residual <= global train 75th percentile", "global_core_count": len(global_core), "global_core_threshold": global_threshold, "goals": statistics}
    write_json(out / "wave23_goal_core_manifest.json", core_manifest)
    write_json(out / "wave23_seed_preregistration.json", {"created_before_training": True, "seeds": config["model"]["seeds"], "paired_with_Wave21_B1": True, "no_seed_replacement": True, "no_post_evaluation_seeds": True})
    write_json(out / "wave23_model_preregistration.json", {"created_before_training": True, "architecture": "exact Wave21 B1 LCT", "inputs": ["z_previous", "z_current", "next_language_embedding"], "new_factor": "state-conditioned softmin distance to requested train-only 75th-percentile goal core", "K": k, "lambda_candidates": config["model"]["lambda_align_candidates"], "selection_split": "development_only", "objective": {"latent_prediction": 1.0, "decoded_action": 1.0, "goal_executable_alignment": "lambda_align"}, "forbidden": {"target_classification_loss": 0.0, "prototype_loss": 0.0, "cycle_loss": 0.0, "F2_refinement": False, "DEL": False, "generic_cycle_projection": False}, "frozen": ["encoder", "decoder", "semantic projection", "text encoder", "goal cores"]})
    write_json(out / "wave23_phaseA_preregistration.json", {"created_before_phaseA_metrics": True, "splits": ["train", "development"], "test_opened": False, "bootstrap": {"cluster": "source_session", "replicates": 10000, "seed": 230823}, "M1_definitions": {"A1": "goal-core margin/correctness Pearson and Spearman point estimates positive and max bootstrap P(positive)>=0.95", "A2": "goal-core distance/decoded-error Pearson and Spearman point estimates positive and max bootstrap P(positive)>=0.95", "A3": "standardized margin coefficient and incremental R2 over cycle positive with bootstrap joint-favorable probability>=0.90", "A4": "cycle0-cycle4 clustered lower95>0 and full margin decreases on average or in >=50% samples", "A5": "execution margin decreases on average or in >=50% samples"}, "substantial_fraction": config["evaluation"]["mechanism_gate"]["substantial_projection_fraction"]})
    print(json.dumps({"stage": "prepare", "goal_core_counts": {task: len(value) for task, value in cores.items()}, "test_opened": False}), flush=True)


def phase_a(config: dict, device: torch.device) -> None:
    out = out_path(config)
    if not (out / "wave23_phaseA_preregistration.json").exists(): raise RuntimeError("prepare must freeze M1 definitions before Phase A")
    ctx = load_context(config, device); rep = ctx["representation"]; wave21 = ctx["wave21"]
    with np.load(out / "wave23_goal_cores.npz") as archive:
        cores = {task: archive[task].copy() for task in ctx["vocab"]}; global_core = archive["__global__"].copy()
    dev = dataset_tensors(wave21 / "datasets" / "development.npz", ctx["goals"], device)
    ids = dev["goal_id"].cpu().numpy(); sessions = dev["session_row_np"]; k = int(config["goal_core"]["neighbors"])
    pred, _ = predict_ensemble(ctx["wcfg"], "B1_correct_language", dev, dev["goal"], device, wave21)
    endpoint = pred[:, 3]; decoded = decode_continuous(rep, endpoint, ctx["mean"], ctx["std"], device)
    true_action = dev["future_actions"].cpu().numpy()[:, 3, :, :6]
    decoded_error = np.mean((decoded - true_action) ** 2, axis=(1, 2))
    _, reencoded, correction = cycle_numpy(rep, endpoint, device); cycle_residual = np.linalg.norm(correction, axis=1)
    full = goal_geometry(endpoint, cores, ctx["vocab"], ids, k); execution = goal_geometry(endpoint, cores, ctx["vocab"], ids, k, True)
    recoded = goal_geometry(reencoded, cores, ctx["vocab"], ids, k)
    global_distance = knn_distances(endpoint, global_core, k)
    current_action = dev["current_action"].cpu().numpy()[..., :6]
    continuity = np.linalg.norm(decoded[:, 0] - current_action[:, -1], axis=1)
    correctness = (full["prediction"] == ids).astype(float)

    projected = endpoint.copy()
    for _ in range(int(config["evaluation"]["projection_iterations"])):
        _, projected, _ = cycle_numpy(rep, projected, device)
    _, _, projected_correction = cycle_numpy(rep, projected, device)
    projected_full = goal_geometry(projected, cores, ctx["vocab"], ids, k)
    projected_execution = goal_geometry(projected, cores, ctx["vocab"], ids, k, True)
    projected_global_distance = knn_distances(projected, global_core, k)

    reps = int(config["evaluation"]["bootstrap_replicates"]); seed = int(config["evaluation"]["bootstrap_seed"])
    margin_correct = clustered_association(full["margin"], correctness, sessions, reps, seed)
    distance_decoded = clustered_association(full["target_distance"], decoded_error, sessions, reps, seed + 1)
    regression = standardized_regression(cycle_residual, full["margin"], correctness, sessions, reps, seed + 2)
    cycle_reduction = paired_positive_probability(cycle_residual - np.linalg.norm(projected_correction, axis=1), sessions, reps, seed + 3)
    margin_decrease = paired_positive_probability(full["margin"] - projected_full["margin"], sessions, reps, seed + 4)
    exec_margin_decrease = paired_positive_probability(execution["margin"] - projected_execution["margin"], sessions, reps, seed + 5)
    fraction_margin_decreased = float(np.mean(projected_full["margin"] < full["margin"]))
    fraction_exec_margin_decreased = float(np.mean(projected_execution["margin"] < execution["margin"]))
    threshold_assoc = float(config["evaluation"]["mechanism_gate"]["association_bootstrap_positive_fraction"])
    threshold_reg = float(config["evaluation"]["mechanism_gate"]["regression_bootstrap_joint_favorable_fraction"])
    threshold_fraction = float(config["evaluation"]["mechanism_gate"]["substantial_projection_fraction"])
    a1 = margin_correct["pearson"]["estimate"] > 0 and margin_correct["spearman"]["estimate"] > 0 and max(margin_correct[kind]["probability_positive"] for kind in ("pearson", "spearman")) >= threshold_assoc
    a2 = distance_decoded["pearson"]["estimate"] > 0 and distance_decoded["spearman"]["estimate"] > 0 and max(distance_decoded[kind]["probability_positive"] for kind in ("pearson", "spearman")) >= threshold_assoc
    a3 = regression["standardized_goal_margin_coefficient"]["estimate"] > 0 and regression["incremental_R2_over_cycle"]["estimate"] > 0 and regression["joint_favorable_probability"] >= threshold_reg
    a4 = cycle_reduction["lower_95"] > 0 and (margin_decrease["mean"] > 0 or fraction_margin_decreased >= threshold_fraction)
    a5 = exec_margin_decrease["mean"] > 0 or fraction_exec_margin_decreased >= threshold_fraction
    authorized = all((a1, a2, a3, a4, a5)); gates = {"A1": a1, "A2": a2, "A3": a3, "A4": a4, "A5": a5}

    per_sample = []
    for i in range(len(ids)):
        per_sample.append({"sample": i, "session": int(sessions[i]), "goal": ctx["vocab"][ids[i]], "cycle_residual": float(cycle_residual[i]), "global_distance": float(global_distance[i]), "goal_core_distance": float(full["target_distance"][i]), "goal_core_margin": float(full["margin"][i]), "execution_goal_core_distance": float(execution["target_distance"][i]), "execution_goal_core_margin": float(execution["margin"][i]), "endpoint_correct": int(correctness[i]), "decoded_reencoded_correct": int(recoded["prediction"][i] == ids[i]), "decoded_action_mse": float(decoded_error[i]), "continuity": float(continuity[i]), "cycle4_residual": float(np.linalg.norm(projected_correction[i])), "cycle4_global_distance": float(projected_global_distance[i]), "cycle4_goal_core_margin": float(projected_full["margin"][i]), "cycle4_execution_goal_core_margin": float(projected_execution["margin"][i])})
    figures = out / "publication_figures_data"; figures.mkdir(exist_ok=True)
    with (figures / "phaseA_goal_geometry_per_sample.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_sample[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(per_sample)
    results = {"M1_goal_specific_executable_alignment": "SUPPORTED_FOR_INTERVENTION" if authorized else "REJECTED", "gates": gates, "optimizer_steps_before_decision": 0, "test_opened": False, "development_samples": len(ids), "source_sessions": len(np.unique(sessions)), "D1_margin_vs_correctness": margin_correct, "D2_distance_vs_decoded_error": distance_decoded, "D3_regression_beyond_cycle": regression, "D4_cycle_projection": {"cycle_reduction": cycle_reduction, "full_margin_decrease": margin_decrease, "fraction_full_margin_decreased": fraction_margin_decreased, "global_distance_before": float(global_distance.mean()), "global_distance_after": float(projected_global_distance.mean())}, "D5_execution_projection": {"execution_margin_decrease": exec_margin_decrease, "fraction_execution_margin_decreased": fraction_exec_margin_decreased}, "development_identity": {"B1_goal_core_top1": float(correctness.mean()), "B1_decode_reencode_top1": float(np.mean(recoded["prediction"] == ids)), "B1_mean_margin": float(full["margin"].mean()), "B1_execution_margin": float(execution["margin"].mean()), "cycle4_goal_core_top1": float(np.mean(projected_full["prediction"] == ids)), "cycle4_mean_margin": float(projected_full["margin"].mean()), "cycle4_execution_margin": float(projected_execution["margin"].mean())}}
    write_json(out / "wave23_phaseA_results.json", results)
    write_json(out / "wave23_mechanism_gate.json", {"M1_goal_specific_executable_alignment": results["M1_goal_specific_executable_alignment"], "gates": gates, "definitions_frozen_before_metrics": True, "optimizer_steps": 0})
    (out / "wave23_phaseA_goal_geometry.md").write_text("# Wave 23 Phase-A goal geometry\n\n" + f"M1: **{results['M1_goal_specific_executable_alignment']}**; gates `{json.dumps(gates, sort_keys=True)}`. Analysis used train cores plus {len(ids)} development transitions from {len(np.unique(sessions))} source sessions; test remained unopened and optimizer steps were zero.\n\n" + f"B1 goal-core Top-1={correctness.mean():.6f}, margin={full['margin'].mean():.6f}; cycle4 Top-1={np.mean(projected_full['prediction']==ids):.6f}, margin={projected_full['margin'].mean():.6f}. Cycle4 reduced residual by {cycle_reduction['mean']:.6f} while full margin change (before-after) was {margin_decrease['mean']:.6f}.\n")
    (out / "wave23_goal_core_association_report.md").write_text("# Wave 23 goal-core association report\n\nSource-session clustered bootstrap, 10,000 replicates, seed family rooted at 230823.\n\n" + f"- Margin vs correctness: Pearson {margin_correct['pearson']['estimate']:.4f} [{margin_correct['pearson']['lower_95']:.4f}, {margin_correct['pearson']['upper_95']:.4f}], Spearman {margin_correct['spearman']['estimate']:.4f} [{margin_correct['spearman']['lower_95']:.4f}, {margin_correct['spearman']['upper_95']:.4f}].\n- Goal-core distance vs decoded MSE: Pearson {distance_decoded['pearson']['estimate']:.4f} [{distance_decoded['pearson']['lower_95']:.4f}, {distance_decoded['pearson']['upper_95']:.4f}].\n- Standardized margin coefficient beyond cycle={regression['standardized_goal_margin_coefficient']['estimate']:.4f}; incremental R2={regression['incremental_R2_over_cycle']['estimate']:.4f}; joint favorable probability={regression['joint_favorable_probability']:.4f}.\n")
    if not authorized:
        (out / "wave23_goal_alignment_mechanism_rejected.md").write_text("# Wave 23 goal-alignment mechanism rejected\n\nM1 failed at least one prospectively frozen development-only gate. No LCT-GA optimizer, lambda sweep, held-out inference, or rescue mechanism is permitted.\n")
    print(json.dumps({"stage": "phasea", "M1": results["M1_goal_specific_executable_alignment"], "gates": gates}), flush=True)


def train(config: dict, device: torch.device) -> None:
    gate = read_json(out_path(config) / "wave23_mechanism_gate.json")
    if gate["M1_goal_specific_executable_alignment"] != "SUPPORTED_FOR_INTERVENTION": raise RuntimeError("STOP: M1 rejected; LCT-GA optimizer is forbidden")
    out = out_path(config); ctx = load_context(config, device); wave21 = ctx["wave21"]
    with np.load(out / "wave23_goal_cores.npz") as archive: cores = {task: archive[task].copy() for task in ctx["vocab"]}
    train_data = dataset_tensors(wave21 / "datasets" / "train.npz", ctx["goals"], device)
    development = dataset_tensors(wave21 / "datasets" / "development.npz", ctx["goals"], device)
    k = int(config["goal_core"]["neighbors"]); temperature = float(config["model"]["softmin_temperature"])

    def neighborhoods(data: dict) -> torch.Tensor:
        current = data["z_current"].detach().cpu().numpy(); ids = data["goal_id"].detach().cpu().numpy(); selected = []
        for z, goal_id in zip(current, ids):
            core = cores[ctx["vocab"][goal_id]]
            distance = np.linalg.norm(core[:, 16:] - z[None, 16:], axis=1)
            indices = np.argpartition(distance, k - 1)[:k]
            selected.append(core[indices])
        return torch.from_numpy(np.stack(selected)).float().to(device)

    train_neighbors = neighborhoods(train_data)

    def lambda_label(value: float) -> str:
        return f"lambda_{value:g}".replace(".", "p")

    records = []
    for lambda_align in config["model"]["lambda_align_candidates"]:
        for seed_value in config["model"]["seeds"]:
            seed = int(seed_value); torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            model = LCT(True, ctx["wcfg"]).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["model"]["learning_rate"]), weight_decay=float(config["model"]["weight_decay"]))
            tensor_set = TensorDataset(train_data["z_previous"], train_data["z_current"], train_data["future_latents"], train_data["future_actions"], train_data["goal"], train_neighbors)
            loader = DataLoader(tensor_set, batch_size=int(config["model"]["batch_size"]), shuffle=True, generator=torch.Generator().manual_seed(seed))
            mean_t = torch.from_numpy(ctx["mean"]).to(device); std_t = torch.from_numpy(ctx["std"]).to(device)
            losses = []; gradient_audit = None
            for epoch in range(int(config["model"]["epochs"])):
                total = 0.0
                for zp, zc, zf, af, goal, neighbor in loader:
                    optimizer.zero_grad(set_to_none=True)
                    predicted = model.rollout(zp, zc, goal)
                    latent_loss = (predicted - zf).square().mean()
                    decoded = ctx["representation"].decode(predicted.flatten(0, 1)).view(*predicted.shape[:2], 16, 7)
                    target = torch.from_numpy(normalize(af.detach().cpu().numpy(), ctx["mean"], ctx["std"])).to(device)
                    decode_loss = (decoded[..., :6] - target[..., :6]).square().mean()
                    distance = (predicted[:, :, None, :] - neighbor[:, None, :, :]).square().mean(-1)
                    alignment_loss = -temperature * (torch.logsumexp(-distance / temperature, dim=-1) - math.log(k))
                    alignment_loss = alignment_loss.mean()
                    loss = latent_loss + float(config["model"]["lambda_decode"]) * decode_loss + float(lambda_align) * alignment_loss
                    loss.backward()
                    if gradient_audit is None:
                        gradient_audit = {"transition_gradients_nonzero": bool(any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters())), "representation_gradients_none": bool(all(parameter.grad is None for parameter in ctx["representation"].parameters())), "neighbor_requires_grad": neighbor.requires_grad, "classification_loss": 0.0, "prototype_loss": 0.0, "cycle_loss": 0.0}
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["model"]["gradient_clip_norm"])); optimizer.step()
                    total += float(loss.detach()) * len(zp)
                losses.append(total / len(tensor_set))
            path = out / "checkpoints" / lambda_label(float(lambda_align)) / f"seed_{seed}.pt"; path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "seed": seed, "lambda_align": float(lambda_align), "architecture": "Wave21_B1_LCT"}, path)
            records.append({"lambda_align": float(lambda_align), "seed": seed, "epochs": len(losses), "final_loss": losses[-1], "minimum_loss": min(losses), "checkpoint": path.relative_to(ROOT).as_posix(), "checkpoint_sha256": sha256(path), "gradient_audit": gradient_audit, "K": k, "goal_core_split": "train_only", "future_actions_as_input": False})
            print(json.dumps({"trained": str(path.relative_to(out)), "final_loss": losses[-1]}), flush=True)
    write_json(out / "wave23_training_records.json", records)

    def predict_ga(data: dict, goals: torch.Tensor, lambda_align: float) -> tuple[np.ndarray, list[np.ndarray]]:
        values = []
        for seed in config["model"]["seeds"]:
            payload = torch.load(out / "checkpoints" / lambda_label(lambda_align) / f"seed_{seed}.pt", map_location=device, weights_only=False)
            model = LCT(True, ctx["wcfg"]).to(device); model.load_state_dict(payload["model_state_dict"]); model.eval()
            with torch.no_grad(): values.append(model.rollout(data["z_previous"], data["z_current"], goals).cpu().numpy())
        return np.mean(np.stack(values), axis=0), values

    def condition_metrics(prediction_fn) -> dict[str, Any]:
        n = len(development["goal_id"]); ids = development["goal_id"].cpu().numpy(); sessions = development["session_row_np"]
        sixway = []
        goals_t = torch.from_numpy(ctx["goals"]).float().to(device)
        for goal_id in range(len(ctx["vocab"])): sixway.append(prediction_fn(goals_t[goal_id].expand(n, -1)))
        sixway = np.stack(sixway, axis=1); endpoint = sixway[:, :, 3]; requested = np.tile(np.arange(len(ctx["vocab"])), n); flat = endpoint.reshape(-1, 32)
        original_regions = ctx["regions"]
        target_ep = endpoint[np.arange(n), ids]; wrong_ep = np.stack([np.mean(np.delete(endpoint[i], ids[i], axis=0), axis=0) for i in range(n)])
        target_d = region_metrics(target_ep, original_regions, ctx["vocab"], ids, k)["target_distance"]; wrong_d = region_metrics(wrong_ep, original_regions, ctx["vocab"], ids, k)["target_distance"]
        target_e = region_metrics(target_ep, original_regions, ctx["vocab"], ids, k, slice(16, None))["target_distance"]; wrong_e = region_metrics(wrong_ep, original_regions, ctx["vocab"], ids, k, slice(16, None))["target_distance"]
        full_core = goal_geometry(flat, cores, ctx["vocab"], requested, k)
        _, reencoded, _ = cycle_numpy(ctx["representation"], flat, device); recoded = goal_geometry(reencoded, cores, ctx["vocab"], requested, k)
        observed = target_ep; true = development["future_latents"].cpu().numpy(); decoded = decode_continuous(ctx["representation"], observed, ctx["mean"], ctx["std"], device); true_action = development["future_actions"].cpu().numpy()[:, 3, :, :6]
        decoded_mse = np.mean((decoded - true_action) ** 2, axis=(1, 2)); h2_mse = np.mean((sixway[np.arange(n), ids, 1] - true[:, 1]) ** 2, axis=1)
        current_last = development["current_action"].cpu().numpy()[:, -1, :6]; gt_jump = np.linalg.norm(development["future_actions"].cpu().numpy()[:, 0, 0, :6] - current_last, axis=1); pred_jump = np.linalg.norm(decoded[:, 0] - current_last, axis=1); continuity_error = np.abs(pred_jump - gt_jump)
        accuracy_by_goal = [(full_core["prediction"][requested == goal_id] == goal_id).mean() for goal_id in range(len(ctx["vocab"]))]
        recoded_by_goal = [(recoded["prediction"][requested == goal_id] == goal_id).mean() for goal_id in range(len(ctx["vocab"]))]
        return {"RedirectGain": float(np.mean(wrong_d - target_d)), "Execution_RedirectGain": float(np.mean(wrong_e - target_e)), "endpoint_macro_accuracy": float(np.mean(accuracy_by_goal)), "decode_reencode_macro_accuracy": float(np.mean(recoded_by_goal)), "H4_decoded_action_MSE": float(decoded_mse.mean()), "H2_full_MSE": float(h2_mse.mean()), "continuity_error": float(continuity_error.mean()), "goal_core_margin": float(full_core["margin"].mean()), "sample_count": n, "source_sessions": len(np.unique(sessions))}

    def b1_prediction(goal_tensor: torch.Tensor) -> np.ndarray:
        return predict_ensemble(ctx["wcfg"], "B1_correct_language", development, goal_tensor, device, wave21)[0]

    b1_metrics = condition_metrics(b1_prediction); candidates = {}
    for value in config["model"]["lambda_align_candidates"]:
        lam = float(value)
        candidates[str(lam)] = condition_metrics(lambda goal_tensor, lam=lam: predict_ga(development, goal_tensor, lam)[0])
    chosen = None; decisions = {}
    for value in config["model"]["lambda_align_candidates"]:
        key = str(float(value)); row = candidates[key]
        conditions = {
            "full_redirect_retained": row["RedirectGain"] >= float(config["evaluation"]["redirect_retention"]) * b1_metrics["RedirectGain"],
            "execution_redirect_retained": row["Execution_RedirectGain"] >= float(config["evaluation"]["redirect_retention"]) * b1_metrics["Execution_RedirectGain"],
            "endpoint_improved_0p05": row["endpoint_macro_accuracy"] >= b1_metrics["endpoint_macro_accuracy"] + float(config["evaluation"]["development_endpoint_improvement"]),
            "decode_reencode_improved_0p05": row["decode_reencode_macro_accuracy"] >= b1_metrics["decode_reencode_macro_accuracy"] + float(config["evaluation"]["development_endpoint_improvement"]),
            "decoded_MSE_within_5pct": row["H4_decoded_action_MSE"] <= float(config["evaluation"]["development_decode_mse_max_ratio"]) * b1_metrics["H4_decoded_action_MSE"],
            "continuity_no_worse": row["continuity_error"] <= b1_metrics["continuity_error"],
        }
        decisions[key] = conditions
        if chosen is None and all(conditions.values()): chosen = float(value)
    selection = {"status": "SELECTED" if chosen is not None else "NO_CANDIDATE_PASSED", "selected_lambda_align": chosen, "candidate_set": config["model"]["lambda_align_candidates"], "selection_split": "development_only", "Wave21_B1_development": b1_metrics, "candidates": candidates, "conditions": decisions, "smallest_passing_rule": True, "held_out_test_opened": False, "no_new_lambda_allowed": True}
    write_json(out / "wave23_alignment_weight_selection.json", selection)
    lines = ["# Wave 23 training report", "", "Only exact Wave21 B1 LCT parameters were trainable. Encoder, decoder, text projection, goal cores, and neighborhood candidates were frozen. The only new term was state-conditioned K=20 goal-core softmin alignment.", "", "| lambda | seed | final loss | gradient audit |", "|---:|---:|---:|---|"]
    lines += [f"| {row['lambda_align']} | {row['seed']} | {row['final_loss']:.8f} | {row['gradient_audit']} |" for row in records]
    lines += ["", f"Development selection status: **{selection['status']}**; selected lambda={chosen}."]
    (out / "wave23_training_report.md").write_text("\n".join(lines) + "\n")
    if chosen is None:
        (out / "wave23_no_alignment_weight_passed.md").write_text("# No Wave 23 alignment weight passed\n\nNone of {0.03, 0.1, 0.3} satisfied every frozen development preservation condition. Per preregistration, no new lambda or held-out evaluation is permitted.\n")
    else:
        selected_records = [row for row in records if row["lambda_align"] == chosen]
        frozen = read_json(out / "wave23_frozen_manifest.json")
        write_json(out / "wave23_final_test_preregistration.json", {"created_before_heldout_inference": True, "held_out_test_opened": False, "M1_decision": "SUPPORTED_FOR_INTERVENTION", "selected_lambda_align": chosen, "selection_rule": "smallest registered development candidate satisfying all conditions", "seeds": config["model"]["seeds"], "checkpoint_rule": "last epoch per each of six registered seeds; arithmetic mean predictions", "checkpoint_hashes": {str(row["seed"]): row["checkpoint_sha256"] for row in selected_records}, "goal_core_manifest_sha256": sha256(out / "wave23_goal_core_manifest.json"), "goal_core_archive_sha256": sha256(out / "wave23_goal_cores.npz"), "K": k, "representation_sha256": frozen["representation_sha256"], "session_split_sha256": frozen["session_split_sha256"], "transition_inventory_sha256": frozen["transition_inventory_sha256"], "thresholds": {"endpoint": 0.60, "stronger_same_state": 0.65, "breadth": "5/6", "dominance": 0.40}, "claim_gates": "prompts/dynamics_11.md C11 G1-G8 and C12", "metrics": ["RedirectGain", "Execution RedirectGain", "goal-core endpoint identity", "decode/reencode identity", "H2 full MSE", "H4 decoded MSE", "continuity", "cycle residual", "breadth"], "bootstrap": {"cluster": "source_session", "replicates": 10000, "seed": 230823}, "post_test_retraining": False, "post_test_rescue": False})
    print(json.dumps({"stage": "train", "selection": selection["status"], "selected_lambda": chosen}), flush=True)


def evaluate_final(config: dict, device: torch.device) -> None:
    raise NotImplementedError("final evaluation is activated only after development lambda selection")


def report(config: dict, device: torch.device) -> None:
    out = out_path(config); selection = read_json(out / "wave23_alignment_weight_selection.json"); phase = read_json(out / "wave23_phaseA_results.json")
    if selection["status"] != "NO_CANDIDATE_PASSED": raise RuntimeError("This report branch expects the registered development stop")
    ctx = load_context(config, device); wave21 = ctx["wave21"]; vocab = ctx["vocab"]; k = int(config["goal_core"]["neighbors"])
    with np.load(out / "wave23_goal_cores.npz") as archive: cores = {task: archive[task].copy() for task in vocab}
    dev = dataset_tensors(wave21 / "datasets" / "development.npz", ctx["goals"], device); n = len(dev["goal_id"]); ids = dev["goal_id"].cpu().numpy(); sessions = dev["session_row_np"]
    goals_t = torch.from_numpy(ctx["goals"]).float().to(device)

    def label(value: float) -> str: return f"lambda_{value:g}".replace(".", "p")

    def predict_ga(goal_tensor: torch.Tensor, value: float) -> np.ndarray:
        predictions = []
        for seed in config["model"]["seeds"]:
            payload = torch.load(out / "checkpoints" / label(value) / f"seed_{seed}.pt", map_location=device, weights_only=False)
            model = LCT(True, ctx["wcfg"]).to(device); model.load_state_dict(payload["model_state_dict"]); model.eval()
            with torch.no_grad(): predictions.append(model.rollout(dev["z_previous"], dev["z_current"], goal_tensor).cpu().numpy())
        return np.mean(np.stack(predictions), axis=0)

    models: dict[str, np.ndarray] = {}
    b1_goals = []
    for goal_id in range(len(vocab)):
        goal = goals_t[goal_id].expand(n, -1)
        b1_goals.append(predict_ensemble(ctx["wcfg"], "B1_correct_language", dev, goal, device, wave21)[0])
    models["Wave21_B1"] = np.stack(b1_goals, axis=1)
    for value in config["model"]["lambda_align_candidates"]:
        models[f"GA_lambda_{float(value):g}"] = np.stack([predict_ga(goals_t[goal_id].expand(n, -1), float(value)) for goal_id in range(len(vocab))], axis=1)
    cycle4 = models["Wave21_B1"].copy().reshape(-1, 32)
    for _ in range(4): _, cycle4, _ = cycle_numpy(ctx["representation"], cycle4, device)
    models["Wave22_cycle4"] = cycle4.reshape(n, len(vocab), 4, 32)
    prototype_endpoints = np.stack([ctx["regions"][task].mean(0) for task in vocab])
    models["language_prototype"] = np.repeat(np.repeat(prototype_endpoints[None, :, None, :], n, axis=0), 4, axis=2)
    retrieval = np.empty((n, len(vocab), 4, 32), np.float32)
    current = dev["z_current"].cpu().numpy()
    for i in range(n):
        for goal_id, task in enumerate(vocab):
            core = cores[task]; nearest = np.argmin(np.linalg.norm(core[:, 16:] - current[i, None, 16:], axis=1)); retrieval[i, goal_id] = np.repeat(core[nearest][None], 4, axis=0)
    models["goal_core_retrieval"] = retrieval
    np.savez_compressed(out / "publication_figures_data" / "development_same_state_trajectories.npz", **models, z_current=current, goal_id=ids, session_row=sessions, boundary_frame=dev["boundary_frame_np"])

    requested = np.tile(np.arange(len(vocab)), n); current_last = dev["current_action"].cpu().numpy()[:, -1, :6]; gt_first = dev["future_actions"].cpu().numpy()[:, 0, 0, :6]; gt_jump = np.linalg.norm(gt_first - current_last, axis=1); true = dev["future_latents"].cpu().numpy(); true_action = dev["future_actions"].cpu().numpy()[:, 3, :, :6]
    comparison: dict[str, Any] = {}; per_goal: dict[str, Any] = {}; confusions: dict[str, Any] = {}
    for name, trajectory in models.items():
        endpoints = trajectory[:, :, 3]; flat = endpoints.reshape(-1, 32); full = goal_geometry(flat, cores, vocab, requested, k); execution = goal_geometry(flat, cores, vocab, requested, k, True)
        _, recoded_latent, correction = cycle_numpy(ctx["representation"], flat, device); recoded = goal_geometry(recoded_latent, cores, vocab, requested, k)
        observed = endpoints[np.arange(n), ids]; decoded = decode_continuous(ctx["representation"], observed, ctx["mean"], ctx["std"], device); decoded_mse = np.mean((decoded - true_action) ** 2, axis=(1, 2)); pred_jump = np.linalg.norm(decoded[:, 0] - current_last, axis=1); continuity_error = np.abs(pred_jump - gt_jump)
        h2 = trajectory[np.arange(n), ids, 1]; h2_mse = np.mean((h2 - true[:, 1]) ** 2, axis=1)
        endpoint_accuracy = []; execution_accuracy = []; recoded_accuracy = []; margins = []
        confusion = np.zeros((len(vocab), len(vocab)), int)
        for target, prediction in zip(requested, full["prediction"]): confusion[target, prediction] += 1
        for goal_id, task in enumerate(vocab):
            mask = requested == goal_id; endpoint_accuracy.append(float(np.mean(full["prediction"][mask] == goal_id))); execution_accuracy.append(float(np.mean(execution["prediction"][mask] == goal_id))); recoded_accuracy.append(float(np.mean(recoded["prediction"][mask] == goal_id))); margins.append(float(full["margin"][mask].mean()))
            per_goal.setdefault(task, {})[name] = {"endpoint_accuracy": endpoint_accuracy[-1], "execution_accuracy": execution_accuracy[-1], "decode_reencode_accuracy": recoded_accuracy[-1], "goal_core_margin": margins[-1], "cycle_residual": float(np.linalg.norm(correction[mask], axis=1).mean())}
        target_endpoint = endpoints[np.arange(n), ids]; wrong_endpoint = np.stack([np.mean(np.delete(endpoints[i], ids[i], axis=0), axis=0) for i in range(n)])
        td = region_metrics(target_endpoint, ctx["regions"], vocab, ids, k)["target_distance"]; wd = region_metrics(wrong_endpoint, ctx["regions"], vocab, ids, k)["target_distance"]
        te = region_metrics(target_endpoint, ctx["regions"], vocab, ids, k, slice(16, None))["target_distance"]; we = region_metrics(wrong_endpoint, ctx["regions"], vocab, ids, k, slice(16, None))["target_distance"]
        residual_center = np.stack([observed[ids == goal].mean(0) for goal in ids]); state_dependence = float(np.corrcoef(np.linalg.norm(current - current.mean(0), axis=1), np.linalg.norm(observed - residual_center, axis=1))[0, 1])
        comparison[name] = {"RedirectGain": float(np.mean(wd - td)), "Execution_RedirectGain": float(np.mean(we - te)), "endpoint_macro_accuracy": float(np.mean(endpoint_accuracy)), "execution_macro_accuracy": float(np.mean(execution_accuracy)), "decode_reencode_macro_accuracy": float(np.mean(recoded_accuracy)), "H2_full_MSE": float(h2_mse.mean()), "H4_decoded_action_MSE": float(decoded_mse.mean()), "goal_core_margin": float(full["margin"].mean()), "execution_goal_core_margin": float(execution["margin"].mean()), "cycle_residual": float(np.linalg.norm(correction, axis=1).mean()), "continuity_error": float(continuity_error.mean()), "within_goal_endpoint_variance": float(np.mean([np.var(observed[ids == goal], axis=0).mean() for goal in range(len(vocab))])), "current_state_endpoint_residual_correlation": state_dependence, "decoded_action_diversity": float(np.var(decoded, axis=0).mean()), "split": "development_only"}
        confusions[name] = confusion.tolist()
    write_json(out / "publication_figures_data" / "development_main_comparison.json", comparison)
    write_json(out / "publication_figures_data" / "development_per_goal.json", per_goal)
    write_json(out / "publication_figures_data" / "development_confusion_matrices.json", confusions)

    # Pairwise train-only core geometry.
    pairwise = []
    for i, first in enumerate(vocab):
        for j, second in enumerate(vocab):
            if j <= i: continue
            a, b = cores[first], cores[second]
            cross_ab = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1); cross_exec = np.linalg.norm(a[:, None, 16:] - b[None, :, 16:], axis=-1)
            within_a = np.linalg.norm(a[:, None, :] - a[None, :, :], axis=-1); np.fill_diagonal(within_a, np.inf)
            within_b = np.linalg.norm(b[:, None, :] - b[None, :, :], axis=-1); np.fill_diagonal(within_b, np.inf)
            overlap = .5 * (np.mean(cross_ab.min(1) <= within_a.min(1)) + np.mean(cross_ab.min(0) <= within_b.min(1)))
            confusion = .5 * (np.mean(cross_ab.min(1) < within_a.min(1)) + np.mean(cross_ab.min(0) < within_b.min(1)))
            semantic_cosine = float(ctx["goals"][i] @ ctx["goals"][j] / max(np.linalg.norm(ctx["goals"][i]) * np.linalg.norm(ctx["goals"][j]), 1e-8))
            pairwise.append({"goal_a": first, "goal_b": second, "symmetric_nearest_distance": float(.5 * (cross_ab.min(1).mean() + cross_ab.min(0).mean())), "overlap_rate": float(overlap), "nearest_neighbor_confusion": float(confusion), "semantic_cosine": semantic_cosine, "execution_space_separation": float(.5 * (cross_exec.min(1).mean() + cross_exec.min(0).mean()))})
    write_json(out / "publication_figures_data" / "pairwise_goal_geometry.json", pairwise)

    # All eligible development lift->place cases; held-out stays unopened.
    inventory = list(csv.DictReader((wave21 / "wave21_transition_inventory.csv").open())); lookup = {(int(sessions[i]), int(dev["boundary_frame_np"][i])): i for i in range(n)}
    candidates = [row for row in inventory if row["split"] == "development" and row["previous_label"] == "lift_blue_block_slider" and row["next_label"] == "place_in_slider" and (int(row["session_row"]), int(row["boundary_frame"])) in lookup]
    lift_rows = []
    place_id = vocab.index("place_in_slider")
    for row in candidates:
        sample = lookup[(int(row["session_row"]), int(row["boundary_frame"]))]
        for name, trajectory in models.items():
            endpoint = trajectory[sample, place_id, 3][None]; geom = goal_geometry(endpoint, cores, vocab, np.asarray([place_id]), k); ex = goal_geometry(endpoint, cores, vocab, np.asarray([place_id]), k, True); decoded = decode_continuous(ctx["representation"], endpoint, ctx["mean"], ctx["std"], device)[0]; _, _, corr = cycle_numpy(ctx["representation"], endpoint, device)
            lift_rows.append({"sample": sample, "session": int(sessions[sample]), "boundary_frame": int(dev["boundary_frame_np"][sample]), "model": name, "goal_core_distance": float(geom["target_distance"][0]), "goal_core_margin": float(geom["margin"][0]), "execution_goal_core_distance": float(ex["target_distance"][0]), "decoded_action_MSE": float(np.mean((decoded - true_action[sample]) ** 2)), "cycle_residual": float(np.linalg.norm(corr[0]))})
    with (out / "publication_figures_data" / "development_lift_to_place.csv").open("w", newline="") as handle:
        fields = list(lift_rows[0]) if lift_rows else ["sample", "session", "boundary_frame", "model", "goal_core_distance", "goal_core_margin", "execution_goal_core_distance", "decoded_action_MSE", "cycle_residual"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(lift_rows)

    write_json(out / "wave23_final_test_preregistration.json", {"status": "NOT_ACTIVATED_NO_DEVELOPMENT_CANDIDATE", "held_out_test_opened": False, "selected_lambda_align": None, "candidate_set_exhausted": config["model"]["lambda_align_candidates"], "six_seed_checkpoints_trained_for_each_candidate": True, "goal_core_manifest_sha256": sha256(out / "wave23_goal_core_manifest.json"), "K": 20, "bootstrap": {"cluster": "source_session", "replicates": 10000, "seed": 230823}, "post_stop_rescue": False})
    claim = {"M1_goal_specific_executable_alignment": "SUPPORTED_FOR_INTERVENTION", "C11_goal_specific_executable_alignment": "NOT_TESTED", "C12_language_as_goal_specific_executable_coordinate": "NOT_TESTED", "full_redirect_preserved": "inconclusive", "execution_redirect_preserved": "inconclusive", "endpoint_identity_repaired": "inconclusive", "decode_reencode_identity_repaired": "inconclusive", "current_state_matters": "inconclusive", "continuity_preserved": "inconclusive", "cycle_error_reduced_without_cycle_loss": "inconclusive", "stop_reason": "No registered lambda_align satisfied all development rules; endpoint and decode/reencode identity did not improve by 0.05 for any candidate", "held_out_test_opened": False}
    write_json(out / "wave23_claim_decision.json", claim)

    tables = out / "publication_tables"; figures = out / "publication_figures"; tables.mkdir(exist_ok=True); figures.mkdir(exist_ok=True)
    core_manifest = read_json(out / "wave23_goal_core_manifest.json")
    with (tables / "table_A_goal_core_statistics.csv").open("w", newline="") as handle:
        writer=csv.writer(handle,lineterminator="\n");writer.writerow(["goal","support_count","core_count","cycle_threshold_75","cycle_mean","semantic_similarity"])
        for task in vocab:
            row=core_manifest["goals"][task];writer.writerow([task,row["train_support_count"],row["primary_core_count"],row["thresholds"]["75"],row["cycle_residual"]["mean"],row["semantic_similarity_to_goal"]["mean"]])
    with (tables / "table_B_phaseA_geometry.csv").open("w", newline="") as handle:
        writer=csv.writer(handle,lineterminator="\n");writer.writerow(["diagnostic","estimate","lower95","upper95"]);writer.writerow(["margin_correctness_Pearson",phase["D1_margin_vs_correctness"]["pearson"]["estimate"],phase["D1_margin_vs_correctness"]["pearson"]["lower_95"],phase["D1_margin_vs_correctness"]["pearson"]["upper_95"]]);writer.writerow(["distance_decoded_Pearson",phase["D2_distance_vs_decoded_error"]["pearson"]["estimate"],phase["D2_distance_vs_decoded_error"]["pearson"]["lower_95"],phase["D2_distance_vs_decoded_error"]["pearson"]["upper_95"]]);writer.writerow(["incremental_R2",phase["D3_regression_beyond_cycle"]["incremental_R2_over_cycle"]["estimate"],phase["D3_regression_beyond_cycle"]["incremental_R2_over_cycle"]["lower_95"],phase["D3_regression_beyond_cycle"]["incremental_R2_over_cycle"]["upper_95"]]);writer.writerow(["cycle4_margin_decrease",phase["D4_cycle_projection"]["full_margin_decrease"]["mean"],phase["D4_cycle_projection"]["full_margin_decrease"]["lower_95"],phase["D4_cycle_projection"]["full_margin_decrease"]["upper_95"]])
    with (tables / "table_C_development_comparison.csv").open("w", newline="") as handle:
        writer=csv.writer(handle,lineterminator="\n");writer.writerow(["metric",*comparison]);
        for metric in ("RedirectGain","Execution_RedirectGain","endpoint_macro_accuracy","decode_reencode_macro_accuracy","H2_full_MSE","H4_decoded_action_MSE","goal_core_margin","cycle_residual","continuity_error"):writer.writerow([metric,*[comparison[name][metric] for name in comparison]])
    with (tables / "table_D_per_goal_development.csv").open("w", newline="") as handle:
        writer=csv.writer(handle,lineterminator="\n");writer.writerow(["goal","model","endpoint_accuracy","decode_reencode_accuracy","goal_core_margin"])
        for task in vocab:
            for name,row in per_goal[task].items():writer.writerow([task,name,row["endpoint_accuracy"],row["decode_reencode_accuracy"],row["goal_core_margin"]])
    with (tables / "table_E_claim_gates.csv").open("w", newline="") as handle:
        writer=csv.writer(handle,lineterminator="\n");writer.writerow(["claim_or_gate","decision"]);[writer.writerow([key,value]) for key,value in phase["gates"].items()];writer.writerow(["M1","SUPPORTED_FOR_INTERVENTION"]);writer.writerow(["development_lambda_selection","NO_CANDIDATE_PASSED"]);writer.writerow(["C11","NOT_TESTED"]);writer.writerow(["C12","NOT_TESTED"])

    try:
        import matplotlib.pyplot as plt
        # 1: schematic
        fig,ax=plt.subplots(figsize=(6,4));rng=np.random.default_rng(23);colors=plt.cm.tab10(np.arange(6));
        for g in range(6): center=np.asarray([math.cos(g*math.pi/3),math.sin(g*math.pi/3)])*2;pts=center+rng.normal(0,.35,(60,2));core=center+rng.normal(0,.16,(25,2));ax.scatter(*pts.T,s=7,color=colors[g],alpha=.25);ax.scatter(*core.T,s=9,color=colors[g],label=vocab[g])
        ax.set_title("Global support contains goal-specific cores");ax.legend(fontsize=5,ncol=2);fig.tight_layout();fig.savefig(figures/"figure_1_goal_specific_support_schematic.png",dpi=160);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(7,3));axes[0].bar(["B1","cycle4"],[phase["D4_cycle_projection"]["cycle_reduction"]["mean"],0]);axes[0].set_title("residual reduction (B1→cycle4)");axes[1].bar(["B1","cycle4"],[phase["development_identity"]["B1_mean_margin"],phase["development_identity"]["cycle4_mean_margin"]]);axes[1].set_title("goal-core margin worsens");fig.tight_layout();fig.savefig(figures/"figure_2_global_vs_goal_support.png",dpi=160);plt.close(fig)
        phase_rows=list(csv.DictReader((out/"publication_figures_data"/"phaseA_goal_geometry_per_sample.csv").open()));x=np.asarray([float(row["goal_core_margin"]) for row in phase_rows]);y=np.asarray([int(row["endpoint_correct"]) for row in phase_rows]);bins=np.quantile(x,[0,.2,.4,.6,.8,1]);fig,ax=plt.subplots();ax.scatter(x,y+rng.normal(0,.025,len(y)),s=8,alpha=.25);ax.plot([x[(x>=lo)&(x<=hi)].mean() for lo,hi in zip(bins[:-1],bins[1:])],[y[(x>=lo)&(x<=hi)].mean() for lo,hi in zip(bins[:-1],bins[1:])],marker="o",color="black");ax.set(xlabel="goal-core margin",ylabel="endpoint correctness");fig.tight_layout();fig.savefig(figures/"figure_3_margin_vs_identity.png",dpi=160);plt.close(fig)
        display=["Wave21_B1","Wave22_cycle4","language_prototype","GA_lambda_0.03","GA_lambda_0.1","GA_lambda_0.3"];fig,axes=plt.subplots(1,3,figsize=(11,3));
        for ax,metric in zip(axes,("RedirectGain","endpoint_macro_accuracy","goal_core_margin")):ax.bar(range(len(display)),[comparison[name][metric] for name in display]);ax.set_title(metric);ax.set_xticks(range(len(display)),display,rotation=55,ha="right",fontsize=6)
        fig.suptitle("Development only — no GA selected / test unopened");fig.tight_layout();fig.savefig(figures/"figure_4_development_model_comparison.png",dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,3));xx=np.arange(6);width=.18
        for offset,name in enumerate(("Wave21_B1","GA_lambda_0.03","GA_lambda_0.1","GA_lambda_0.3")):ax.bar(xx+(offset-1.5)*width,[per_goal[task][name]["endpoint_accuracy"] for task in vocab],width,label=name)
        ax.set_xticks(xx,vocab,rotation=35,ha="right",fontsize=6);ax.legend(fontsize=6);ax.set_title("Development same-state six-way identity");fig.tight_layout();fig.savefig(figures/"figure_5_same_state_goal_cores.png",dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,3));names=["Wave21_B1","Wave22_cycle4","GA_lambda_0.03","GA_lambda_0.1","GA_lambda_0.3"];ax.bar(names,[comparison[name]["decode_reencode_macro_accuracy"] for name in names]);ax.axhline(.6,color="red",ls="--");ax.tick_params(axis="x",rotation=35);ax.set_title("Development decode/reencode identity");fig.tight_layout();fig.savefig(figures/"figure_6_decode_reencode_identity.png",dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,3));
        if lift_rows:
            names=sorted({row["model"] for row in lift_rows});ax.bar(names,[np.mean([row["goal_core_margin"] for row in lift_rows if row["model"]==name]) for name in names]);ax.tick_params(axis="x",rotation=45);ax.set_title("Development lift→place: all eligible cases")
        else:ax.text(.5,.5,"No eligible development lift→place case\nheld-out test not opened",ha="center",va="center");ax.set_axis_off()
        fig.tight_layout();fig.savefig(figures/"figure_7_lift_to_place_development.png",dpi=160);plt.close(fig)
    except ImportError: write_json(figures/"matplotlib_unavailable.json",{"raw_figure_data_complete":True})

    overlap_mean=float(np.mean([row["overlap_rate"] for row in pairwise])); overlap_max=max(pairwise,key=lambda row:row["overlap_rate"])
    (out/"wave23_main_comparison.md").write_text("# Wave 23 main comparison\n\nNo held-out comparison was run. Development-only B1, cycle4, prototype, retrieval, and all three GA candidates are in Table C. All GA candidates failed both required +0.05 identity improvements; λ=0.3 also failed full RedirectGain retention.\n")
    (out/"wave23_same_state_language_swap.md").write_text("# Wave 23 same-state language swap\n\nDevelopment-only six-way trajectories for B1, cycle4, prototype, retrieval, and all GA candidates are saved. Within every boundary only language changes. No held-out GA intervention exists because no λ passed selection.\n")
    (out/"wave23_decode_reencode_results.md").write_text(f"# Wave 23 decode/re-encode results\n\nDevelopment B1={selection['Wave21_B1_development']['decode_reencode_macro_accuracy']:.6f}; GA λ=0.03/0.1/0.3={selection['candidates']['0.03']['decode_reencode_macro_accuracy']:.6f}/{selection['candidates']['0.1']['decode_reencode_macro_accuracy']:.6f}/{selection['candidates']['0.3']['decode_reencode_macro_accuracy']:.6f}. None improved by the required 0.05. Held-out was not opened.\n")
    (out/"wave23_continuity_results.md").write_text(f"# Wave 23 continuity results\n\nDevelopment B1 continuity error={selection['Wave21_B1_development']['continuity_error']:.6f}; GA λ=0.03/0.1/0.3={selection['candidates']['0.03']['continuity_error']:.6f}/{selection['candidates']['0.1']['continuity_error']:.6f}/{selection['candidates']['0.3']['continuity_error']:.6f}. All candidates were no worse, but identity selection still failed.\n")
    (out/"wave23_goal_geometry_analysis.md").write_text(f"# Wave 23 goal geometry analysis\n\nSix train-only 75% cores contain 118–243 latents. Across 15 goal pairs, mean overlap rate={overlap_mean:.6f}; maximum={overlap_max['overlap_rate']:.6f} for `{overlap_max['goal_a']} / {overlap_max['goal_b']}`. Pairwise distance, overlap, nearest-neighbor confusion, semantic cosine, and execution separation are raw in `publication_figures_data/pairwise_goal_geometry.json`. No contact/physical-phase stratification was possible because the frozen region archive does not carry phase labels; primary samples were unchanged.\n")
    (out/"wave23_lift_to_place_case.md").write_text(f"# Wave 23 lift-to-place case\n\nHeld-out cases were not opened because development selection stopped. All {len(candidates)} eligible development `lift_blue_block_slider -> place_in_slider` cases were analyzed without cherry-picking; raw rows compare B1, cycle4, prototypes/retrieval, and every GA candidate.\n")
    (out/"wave23_statistical_report.md").write_text(f"# Wave 23 statistical report\n\nIndependent unit: continuous source session. Phase A used six development sessions, 10,000 clustered bootstrap replicates, seed family 230823. Margin/correctness Pearson={phase['D1_margin_vs_correctness']['pearson']['estimate']:.6f} [{phase['D1_margin_vs_correctness']['pearson']['lower_95']:.6f}, {phase['D1_margin_vs_correctness']['pearson']['upper_95']:.6f}]; distance/decoded-error Pearson={phase['D2_distance_vs_decoded_error']['pearson']['estimate']:.6f} [{phase['D2_distance_vs_decoded_error']['pearson']['lower_95']:.6f}, {phase['D2_distance_vs_decoded_error']['pearson']['upper_95']:.6f}]; incremental R²={phase['D3_regression_beyond_cycle']['incremental_R2_over_cycle']['estimate']:.6f}. No held-out inference or C11/C12 statistics were performed.\n")
    (out/"wave23_failure_taxonomy.md").write_text("# Wave 23 failure taxonomy\n\nFrozen categories: goal-core misalignment; global-support / target-support conflict; semantic-execution mismatch; prototype collapse; current-state ignored; decoder inconsistency; goal overlap ambiguity; continuity failure; goal-specific failure; long-horizon accumulation; other.\n\nActivated: goal-core misalignment, global-support / target-support conflict, goal overlap ambiguity, and goal-specific failure. The alignment loss preserved continuity but moved development goal-core margin/identity in the wrong direction. Held-out failure categories were not assigned.\n")

    best="0.03"; q=[
        f"Yes, descriptively: all six train-only cores are nonempty (118–243 points), but they are not disjoint; mean pairwise overlap={overlap_mean:.4f}.",
        f"Mean pairwise overlap is {overlap_mean:.4f}; maximum {overlap_max['overlap_rate']:.4f} for {overlap_max['goal_a']} / {overlap_max['goal_b']}.",
        f"Yes. Development Pearson r={phase['D2_distance_vs_decoded_error']['pearson']['estimate']:.4f} for distance vs decoded error; distance also tracks poorer target geometry.",
        f"Yes. Margin/correctness Pearson r={phase['D1_margin_vs_correctness']['pearson']['estimate']:.4f} and Spearman ρ={phase['D1_margin_vs_correctness']['spearman']['estimate']:.4f}.",
        f"Yes descriptively. Standardized margin coefficient={phase['D3_regression_beyond_cycle']['standardized_goal_margin_coefficient']['estimate']:.4f}; incremental R²={phase['D3_regression_beyond_cycle']['incremental_R2_over_cycle']['estimate']:.4f}.",
        f"Yes. Global distance improved 3.285295→2.594105 while mean goal-core margin worsened by {phase['D4_cycle_projection']['full_margin_decrease']['mean']:.6f}; 72.7% moved to lower margin.",
        f"Only weakly. Mean execution-margin decrease was {phase['D5_execution_projection']['execution_margin_decrease']['mean']:.6f}; the frozen A5 directional definition passed, but CI crossed zero and 49.6% decreased.",
        "Yes. M1 passed A1–A5 before training.",
        "None. All registered λ values failed development identity-improvement rules.",
        "No held-out answer. On development λ=0.03/0.1 retained >=90%; λ=0.3 did not.",
        "No held-out answer. All three retained >=90% execution RedirectGain on development.",
        f"Not tested held-out. Development endpoint macro fell from {selection['Wave21_B1_development']['endpoint_macro_accuracy']:.6f} to {selection['candidates'][best]['endpoint_macro_accuracy']:.6f} for the smallest λ.",
        f"Not tested held-out. Development decode/reencode fell from {selection['Wave21_B1_development']['decode_reencode_macro_accuracy']:.6f} to {selection['candidates'][best]['decode_reencode_macro_accuracy']:.6f} for λ=0.03.",
        "Not tested held-out; no GA was selected.", "Not tested held-out; no GA was selected.", "Not adjudicated for a selected GA. Development diversity/state-dependence diagnostics are saved, but selection stopped first.",
        "Development continuity was no worse for all candidates, but no selected/held-out GA result exists.", "Not tested held-out; no selected GA exists.", "Not tested held-out.",
        f"Held-out was not opened. {len(candidates)} development cases were analyzed descriptively without cherry-picking.", "C11 is NOT_TESTED.", "C12 is NOT_TESTED.",
        "Goal-specific geometry explains identity beyond global cycle residual, but the implemented fallback neighborhood lacks source-preceding transition pairing; simply pulling predictions toward execution-nearest core points worsens target margin. Wave21/22 therefore reflect both global/target support conflict and missing transition-conditioned correspondence within each goal core.",
        "Defensible claim: train-only goal-core geometry strongly predicts development target identity beyond global cycle consistency, while a preregistered local softmin alignment preserves redirection/continuity but does not repair identity and is not authorized for held-out evaluation.",
        "If C11 had passed, the next experiment would be a separately preregistered closed-loop CALVIN comparison of frozen B1 and selected GA from matched simulator states.",
        "The language-target-coordinate hypothesis remains supported only as a diagnostic association: goals have structured executable cores and their margins explain identity. What fails is the current alignment intervention, not the earlier causal language-redirection component.",
    ]
    lines=["# Twenty-third wave results: goal-specific executable alignment","",f"Run date: {now()}","","## Outcome","","- M1 goal-specific executable alignment: **SUPPORTED_FOR_INTERVENTION**","- Development lambda selection: **NO_CANDIDATE_PASSED**","- C11/C12: **NOT_TESTED**","- Held-out Wave23 test opened: **false**","","Phase A supported goal-specific geometry, so all 18 preregistered models were trained. However, no λ achieved either required +0.05 identity improvement. The smallest candidates preserved language redirection, decoded MSE, and continuity, but endpoint and decode/reencode identity worsened. The experiment therefore stopped before held-out inference.","","## Required questions",""]+[f"{i}. {answer}" for i,answer in enumerate(q,1)]+["","## Scientific conclusion","","Goal-specific core geometry is explanatory but the registered alignment operator is not corrective. The likely missing information is transition-conditioned correspondence: train core latents were available, but their source-preceding states were not stored, so the preregistered fallback matched current states in execution space rather than learning a valid path into the goal core.","","## Discipline disclosure","","Goal cores used train only. M1 and λ selection used development only. No Wave23 held-out inference, post-selection tuning, replacement seed, new λ, cycle loss, classification loss, prototype loss, F2, or DEL was used."]
    result_text="\n".join(lines)+"\n";(out/"twenty_third_wave_results.md").write_text(result_text);report_path=ROOT/config["experiment"]["report_path"];report_path.parent.mkdir(parents=True,exist_ok=True);report_path.write_text(result_text)
    next_text="# Twenty-third wave next experiment\n\nWave 24 should diagnose transition-conditioned correspondence within each goal core before adding another loss. Reconstruct train-only pairs linking each core endpoint to its true source-preceding latent and test whether source-conditioned neighbors improve goal margin over the Wave23 execution-nearest fallback. Freeze the correspondence rule on train, authorize intervention on development only, and retain the same identity, redirection, current-state, continuity, six-seed, and source-session gates. Do not rescue Wave23 with another λ, cycle loss, classification loss, F2, or DEL; held-out Wave23 remains unopened.\n"
    (out/"twenty_third_wave_next_experiment.md").write_text(next_text);(ROOT/"NEXT_EXPERIMENT.md").write_text(next_text)
    log_path=ROOT/"RESEARCH_LOG.md";log_text=log_path.read_text();marker="## Wave 23 — Goal-specific executable alignment"
    if marker not in log_text:
        log_text+=f"\n{marker} ({datetime.now().date()})\n\n- Built exact train-only 75% goal cores (118–243 points/class), K=20.\n- M1 **SUPPORTED_FOR_INTERVENTION**: all five development-only gates passed.\n- Trained 18 LCT-GA models (3 lambdas × 6 paired seeds); no candidate passed both +0.05 identity improvements.\n- Development λ=0.03 preserved redirects and continuity but endpoint/decode-reencode accuracy worsened.\n- C11/C12 **NOT_TESTED**; Wave23 held-out test remained unopened.\n- Full artifacts: `{out.relative_to(ROOT)}`.\n";log_path.write_text(log_text)
    (out/"updated_RESEARCH_LOG.md").write_text(log_path.read_text());(out/"updated_NEXT_EXPERIMENT.md").write_text((ROOT/"NEXT_EXPERIMENT.md").read_text())
    (out/"environment_freeze.txt").write_text("\n".join([f"timestamp={now()}",f"python={' '.join(sys.version.split())}",f"platform={platform.platform()}",f"torch={torch.__version__}",f"numpy={np.__version__}",f"cuda_available={torch.cuda.is_available()}",f"cuda_device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}"])+"\n")
    (out/"exact_commands.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\nPYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_11.py --config configs/dynamics_11.yaml --stage prepare --device cuda:0\nPYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_11.py --config configs/dynamics_11.yaml --stage phasea --device cuda:0\nPYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_11.py --config configs/dynamics_11.yaml --stage train --device cuda:0\nPYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_11.py --config configs/dynamics_11.yaml --stage report --device cuda:0\nPYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python -m pytest tests/dynamics/test_dynamics_11_goal_alignment.py -q\n")
    (out/"files_changed.txt").write_text("\n".join(["configs/dynamics_11.yaml","prompts/dynamics_11.md","scripts/dynamics/run_dynamics_11.py","tests/dynamics/test_dynamics_11_goal_alignment.py","reports/dynamics_11_results.md","RESEARCH_LOG.md","NEXT_EXPERIMENT.md",config["experiment"]["output_root"]+"/"])+"\n")
    print(json.dumps({"stage":"report","M1":"SUPPORTED_FOR_INTERVENTION","selection":"NO_CANDIDATE_PASSED","C11":"NOT_TESTED","heldout_opened":False}),flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--stage", choices=("prepare", "phasea", "train", "final", "report", "all"), default="all"); parser.add_argument("--device"); args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text()); device = torch.device(args.device or config["runtime"]["device"]); torch.set_num_threads(int(config["runtime"]["torch_cpu_threads"]))
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("Registered Wave23 run requires CUDA")
    stages = ("prepare", "phasea", "train", "final", "report") if args.stage == "all" else (args.stage,)
    for stage in stages:
        print(json.dumps({"stage": stage, "started_at": now()}), flush=True)
        if stage == "report": report(config, device)
        else: {"prepare": prepare, "phasea": phase_a, "train": train, "final": evaluate_final}[stage](config, device)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run Wave 24 state/horizon-conditioned displacement-family experiments.

Purpose
-------
Reconstruct exact paired CALVIN transitions, compare static, horizon-specific,
and source-conditioned transition supports on development, and only if M2 is
authorized train/evaluate LCT-TD with one displacement-matching loss.

Parameters
----------
--config: Wave 24 YAML configuration.
--stage: ``prepare``, ``phasea``, ``train``, ``final``, ``report``, or ``all``.
--device: Optional torch device override; the registered run uses ``cuda:0``.

Usage
-----
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/run_dynamics_12.py --config configs/dynamics_12.yaml \
  --stage all --device cuda:0

Outputs
-------
Writes paired Parquet inventory, frozen support manifests, diagnostics,
checkpoints when authorized, raw tables/figure data, reports, claims, and
reproducibility records under
``results/dynamics/twenty_fourth_wave/2026-08-14_dynamics_12``. The report
stage also updates ``reports/dynamics_12_results.md``, ``RESEARCH_LOG.md``, and
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
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from scipy import stats
from torch.utils.data import DataLoader, TensorDataset

try:
    from scripts.dynamics.run_dynamics_9 import (
        LCT, cluster_bootstrap, dataset_tensors, decode_continuous,
        load_representation, normalize, predict_ensemble, read_json,
        region_metrics, sha256, write_json,
    )
    from scripts.dynamics.run_dynamics_10 import cycle_numpy, distribution
    from scripts.dynamics.run_dynamics_11 import goal_geometry, knn_distances
except ModuleNotFoundError:
    from run_dynamics_9 import (
        LCT, cluster_bootstrap, dataset_tensors, decode_continuous,
        load_representation, normalize, predict_ensemble, read_json,
        region_metrics, sha256, write_json,
    )
    from run_dynamics_10 import cycle_numpy, distribution
    from run_dynamics_11 import goal_geometry, knn_distances


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


def positive_ci(values: np.ndarray, sessions: np.ndarray, config: dict, seed_offset: int = 0) -> dict[str, Any]:
    return cluster_bootstrap(values, sessions, int(config["evaluation"]["bootstrap_replicates"]), int(config["evaluation"]["bootstrap_seed"]) + seed_offset)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-8)


def macro_accuracy(prediction: np.ndarray, target: np.ndarray, goal_count: int) -> float:
    return float(np.mean([np.mean(prediction[target == goal] == goal) for goal in range(goal_count)]))


def source_neighbors(query: np.ndarray, train_current: np.ndarray, train_ids: np.ndarray, goal: int, k: int, execution: bool = True) -> tuple[np.ndarray, np.ndarray]:
    eligible = np.flatnonzero(train_ids == goal)
    sl = slice(16, None) if execution else slice(None)
    distance = np.linalg.norm(train_current[eligible, sl] - query[None, sl], axis=1)
    count = min(k, len(eligible)); local = np.argpartition(distance, count - 1)[:count]
    order = local[np.argsort(distance[local])]
    return eligible[order], distance[order]


def paired_predictors(train: dict[str, np.ndarray], query_current: np.ndarray, query_ids: np.ndarray, horizon_index: int, tau_by_goal: dict[int, float], k: int) -> dict[str, np.ndarray]:
    train_current, train_ids = train["z_current"], train["goal_id"]
    train_delta = train["future_latents"][:, horizon_index] - train_current
    mean_values = []; weighted_values = []; nearest_values = []; goal_values = []
    for current, goal in zip(query_current, query_ids):
        indices, distances = source_neighbors(current, train_current, train_ids, int(goal), k, True)
        delta = train_delta[indices]; tau = max(float(tau_by_goal[int(goal)]), 1e-8)
        weights = np.exp(-(distances ** 2) / (tau ** 2)); weights /= max(float(weights.sum()), 1e-12)
        mean_values.append(delta.mean(0)); weighted_values.append(np.sum(delta * weights[:, None], axis=0)); nearest_values.append(delta[0]); goal_values.append(train_delta[train_ids == goal].mean(0))
    return {"D1_mean_local": np.stack(mean_values), "D2_weighted_local": np.stack(weighted_values), "D3_nearest": np.stack(nearest_values), "goal_horizon_mean": np.stack(goal_values)}


def compute_tau(train_current: np.ndarray, train_ids: np.ndarray, goal_count: int, k: int) -> dict[int, float]:
    result = {}
    for goal in range(goal_count):
        values = train_current[train_ids == goal, 16:]
        distance = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=-1)
        np.fill_diagonal(distance, np.inf); count = min(k, len(values) - 1)
        selected = np.partition(distance, count - 1, axis=1)[:, :count]
        result[goal] = float(np.median(selected))
    return result


def load_numpy_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def prepare(config: dict, device: torch.device) -> None:
    out = out_path(config); out.mkdir(parents=True, exist_ok=True)
    ctx = load_context(config, device); wave21 = ctx["wave21"]
    frozen21 = read_json(wave21 / "wave21_frozen_representation_manifest.json")
    wave23 = ROOT / config["experiment"]["wave23_root"]
    manifest = {
        "created_before_phaseA_and_optimizer": True, "created_at": now(),
        "historical_results_unchanged": {"Wave21_C7": "REJECTED", "Wave21_C8": "REJECTED", "Wave22_M0": "REJECTED", "Wave23_M1": "SUPPORTED_FOR_INTERVENTION", "Wave23_C11": "NOT_TESTED", "Wave23_C12": "NOT_TESTED", "Wave23_heldout": "UNOPENED"},
        "representation_checkpoint": frozen21["checkpoint"], "representation_sha256": frozen21["checkpoint_sha256"], "encoder_sha256": frozen21["action_encoder_sha256"], "decoder_sha256": frozen21["decoder_sha256"], "semantic_projection_sha256": frozen21["semantic_projection_sha256"], "text_feature_archive_sha256": frozen21["text_feature_archive_sha256"], "normalization": frozen21["normalization"], "normalization_sha256": frozen21["normalization_sha256"],
        "Wave21_B1_hashes": {str(seed): sha256(wave21 / "checkpoints" / "B1_correct_language" / f"seed_{seed}.pt") for seed in config["model"]["seeds"]}, "Wave21_B0_hashes": {str(seed): sha256(wave21 / "checkpoints" / "B0_unconditional" / f"seed_{seed}.pt") for seed in config["model"]["seeds"]},
        "session_split_sha256": sha256(wave21 / "wave21_session_split_manifest.json"), "transition_inventory_sha256": sha256(wave21 / "wave21_transition_inventory.csv"), "train_dataset_sha256": sha256(wave21 / "datasets" / "train.npz"), "development_dataset_sha256": sha256(wave21 / "datasets" / "development.npz"), "historical_test_dataset_sha256": sha256(wave21 / "datasets" / "test.npz"),
        "Wave23_static_core_manifest_sha256": sha256(wave23 / "wave23_goal_core_manifest.json"), "Wave23_static_core_archive_sha256": sha256(wave23 / "wave23_goal_cores.npz"),
        "representation_optimizer_steps": 0, "encoder_optimizer_steps": 0, "decoder_optimizer_steps": 0, "text_encoder_optimizer_steps": 0, "Wave21_LCT_optimizer_steps_phaseA": 0, "wave24_heldout_arrays_opened": False,
    }
    write_json(out / "wave24_frozen_manifest.json", manifest)
    split_manifest = read_json(wave21 / "wave21_session_split_manifest.json")
    write_json(out / "wave24_split_freeze.json", {"source": "exact Wave21 source-session split", "sessions": split_manifest["sessions"], "session_names": split_manifest["session_names"], "disjoint": split_manifest["disjoint"], "train_support_only": True, "development_mechanism_only": True, "heldout_arrays_opened": False, "boundary": "next annotation true start", "chunk_frames": 16, "horizons": {"H1": 16, "H2": 32, "H4": 64}, "sparse_annotation_gaps_preserved": True})

    inventory = list(csv.DictReader((wave21 / "wave21_transition_inventory.csv").open()))
    loaded = {name: load_numpy_dataset(wave21 / "datasets" / f"{name}.npz") for name in ("train", "development")}
    lookups = {name: {(int(data["session_row"][i]), int(data["boundary_frame"][i])): i for i in range(len(data["goal_id"]))} for name, data in loaded.items()}
    records = []
    for row in inventory:
        split = row["split"]; key = (int(row["session_row"]), int(row["boundary_frame"])); materialized = split in loaded
        data = loaded.get(split); index = lookups[split][key] if materialized else None
        record: dict[str, Any] = {"source_session": row["source_session"], "session_row": int(row["session_row"]), "boundary_id": row["boundary_id"], "boundary_frame": int(row["boundary_frame"]), "previous_goal": row["previous_label"], "next_goal": row["next_label"], "split": split, "source_frame_contiguous": row["source_frame_contiguous"] == "True", "reset_or_discontinuity": row["reset_or_discontinuity"] == "True", "annotation_gap_frames": int(row["annotation_gap_frames"]), "annotation_relation": row["annotation_relation"], "latent_arrays_materialized": materialized}
        for name in ("z_previous", "z_current"):
            record[name] = data[name][index].astype(np.float32).tolist() if materialized else None
        for horizon, hindex in ((1, 0), (2, 1), (4, 3)):
            future = data["future_latents"][index, hindex].astype(np.float32) if materialized else None
            record[f"z_future_H{horizon}"] = future.tolist() if materialized else None
            record[f"delta_H{horizon}"] = (future - data["z_current"][index]).astype(np.float32).tolist() if materialized else None
            record[f"future_actions_H{horizon}"] = data["future_actions"][index, hindex].astype(np.float32).reshape(-1).tolist() if materialized else None
        records.append(record)
    parquet_path = out / "wave24_paired_transition_inventory.parquet"; pq.write_table(pa.Table.from_pylist(records), parquet_path, compression="zstd")
    report_counts = {split: sum(row["split"] == split for row in records) for split in ("train", "development", "test")}
    (out / "wave24_paired_transition_inventory_report.md").write_text("# Wave 24 paired transition inventory\n\n" + f"Inventory rows: **{len(records)}** across **{len(set(row['source_session'] for row in records))}** sessions; split counts `{report_counts}`. Train/development paired arrays were reconstructed exactly; 164 test rows contain metadata with null latent/action fields so held-out arrays remain unopened.\n\nEvery row is physically contiguous, crosses no reset, uses the next annotation's true start, and retains the original annotation gap without synthetic labels. H1/H2/H4 are 16/32/64 frames.\n")

    train = loaded["train"]; horizons = list(config["transition_family"]["horizons"]); goal_count = len(ctx["vocab"]); k = int(config["transition_family"]["neighbors"])
    tau = compute_tau(train["z_current"], train["goal_id"], goal_count, k)
    horizon_arrays: dict[str, np.ndarray] = {}; family_stats: dict[str, Any] = {}; eligible = []
    for goal, task in enumerate(ctx["vocab"]):
        for horizon, hindex in ((1, 0), (2, 1), (4, 3)):
            mask = train["goal_id"] == goal; endpoints = train["future_latents"][mask, hindex]; deltas = endpoints - train["z_current"][mask]
            horizon_arrays[f"{task}__H{horizon}"] = endpoints
            norms = np.linalg.norm(deltas, axis=1); normalized = deltas / np.maximum(norms[:, None], 1e-8); pair_cos = normalized @ normalized.T; upper = pair_cos[np.triu_indices(len(deltas), 1)]
            pair_mag = np.linalg.norm(deltas[:, None, :] - deltas[None, :, :], axis=-1); pair_mag = pair_mag[np.triu_indices(len(deltas), 1)]
            covariance = np.cov(deltas, rowvar=False); eig = np.maximum(np.linalg.eigvalsh(covariance), 0); weights = eig / max(float(eig.sum()), 1e-12); effective_rank = float(np.exp(-np.sum(weights * np.log(np.maximum(weights, 1e-12)))))
            adequate = len(deltas) >= int(config["transition_family"]["minimum_cell_support"])
            if adequate: eligible.append(f"{task}__H{horizon}")
            family_stats[f"{task}__H{horizon}"] = {"goal": task, "horizon": horizon, "train_transition_count": len(deltas), "adequate": adequate, "K_used": min(k, len(deltas)), "tau_train_only": tau[goal], "mean_displacement_norm": float(norms.mean()), "displacement_norm": distribution(norms), "covariance": covariance.tolist(), "effective_rank": effective_rank, "pairwise_cosine": distribution(upper), "pairwise_magnitude_difference": distribution(pair_mag), "within_goal_variation": float(np.var(deltas, axis=0).mean())}
    horizon_path = out / "wave24_horizon_cores.npz"; np.savez_compressed(horizon_path, **horizon_arrays)
    write_json(out / "wave24_static_core_manifest.json", {"source": "Wave23 exact train-only static 75th-percentile cores", "manifest_sha256": manifest["Wave23_static_core_manifest_sha256"], "archive_sha256": manifest["Wave23_static_core_archive_sha256"], "test_used": False})
    write_json(out / "wave24_horizon_core_manifest.json", {"created_before_development_metrics": True, "source_split": "train_only", "horizons_separate": True, "archive": horizon_path.relative_to(ROOT).as_posix(), "archive_sha256": sha256(horizon_path), "cells": {key: {field: value[field] for field in ("goal", "horizon", "train_transition_count", "adequate", "K_used")} for key, value in family_stats.items()}, "test_used": False})
    write_json(out / "wave24_transition_family_manifest.json", {"created_before_development_metrics": True, "source_split": "train_only", "neighbor_selection": "execution L2 between query current latent and train source current latent; endpoint never used", "K": k, "minimum_support": config["transition_family"]["minimum_cell_support"], "tau_rule": config["transition_family"]["tau_rule"], "tau_by_goal": {ctx["vocab"][goal]: value for goal, value in tau.items()}, "eligible_cells": eligible, "cell_count": len(family_stats), "adequate_goal_count": sum(all(f"{task}__H{h}" in eligible for h in horizons) for task in ctx["vocab"]), "statistics": family_stats, "development_used": False, "test_used": False})
    write_json(out / "wave24_seed_preregistration.json", {"created_before_training": True, "seeds": config["model"]["seeds"], "paired_with_Wave21": True, "no_replacement": True})
    write_json(out / "wave24_model_preregistration.json", {"created_before_phaseA_training_decision": True, "architecture": "exact Wave21 B1 LCT", "only_new_factor_if_M2_passes": "horizon-specific source-conditioned transition-displacement softmin", "horizons": horizons, "horizon_weights": config["model"]["horizon_weights"], "K": k, "lambda_TM_candidates": config["model"]["lambda_TM_candidates"], "forbidden": {"endpoint_attraction": 0.0, "goal_core_softmin": 0.0, "cycle_loss": 0.0, "classification_loss": 0.0, "prototype_loss": 0.0, "PCA_loss": 0.0, "F2": False, "DEL": False}})
    write_json(out / "wave24_phaseA_preregistration.json", {"created_before_development_metrics": True, "train_support": True, "development_gate": True, "heldout_arrays_opened": False, "primary": {"source_distance": "execution L2", "K": 20, "predictor": "D2", "tau": "train-only per-goal/horizon median K-neighbor source distance"}, "M2": {"A1": "HorizonCoreGain clustered lower95>0", "A2": "D2 full and execution cosine clustered lower95>0", "A3": "goal+horizon mean minus D2 full and execution MSE clustered lower95>0", "A4": "D2 H2 full MSE and H4 decoded MSE each below both static-core retrieval and language prototype", "A5": "D2 aggregate macro endpoint and decoded/reencoded identity each >= B1 under identical train-region classifier", "A6": "D2 aggregate continuity error <= B1"}, "bootstrap": {"cluster": "source_session", "replicates": 10000, "seed": 240824}})
    print(json.dumps({"stage": "prepare", "inventory": len(records), "materialized": report_counts["train"] + report_counts["development"], "test_masked": report_counts["test"], "eligible_cells": len(eligible)}), flush=True)


def phase_a(config: dict, device: torch.device) -> None:
    out = out_path(config)
    if not (out / "wave24_phaseA_preregistration.json").exists(): raise RuntimeError("prepare must freeze M2 before Phase A")
    ctx = load_context(config, device); wave21 = ctx["wave21"]; train = load_numpy_dataset(wave21 / "datasets" / "train.npz"); dev_np = load_numpy_dataset(wave21 / "datasets" / "development.npz")
    dev = dataset_tensors(wave21 / "datasets" / "development.npz", ctx["goals"], device); ids = dev_np["goal_id"]; sessions = dev_np["session_row"]; current = dev_np["z_current"]; k = int(config["transition_family"]["neighbors"]); goal_count = len(ctx["vocab"])
    tau_by_goal = {goal: compute_tau(train["z_current"], train["goal_id"], goal_count, k)[goal] for goal in range(goal_count)}
    with np.load(ROOT / config["experiment"]["wave23_root"] / "wave23_goal_cores.npz") as archive: static_cores = {task: archive[task].copy() for task in ctx["vocab"]}
    with np.load(out / "wave24_horizon_cores.npz") as archive: horizon_cores = {(goal, horizon): archive[f"{task}__H{horizon}"].copy() for goal, task in enumerate(ctx["vocab"]) for horizon in (1, 2, 4)}
    b1_prediction, _ = predict_ensemble(ctx["wcfg"], "B1_correct_language", dev, dev["goal"], device, wave21)

    predictors: dict[str, dict[int, np.ndarray]] = defaultdict(dict); rows = []; horizon_gain_values = []; repeated_sessions = []
    direction = {key: [] for key in ("D1_full_cosine", "D2_full_cosine", "D3_full_cosine", "goal_mean_full_cosine", "D2_execution_cosine", "D2_semantic_cosine", "D2_norm_ratio", "D2_angular_error")}
    model_metrics: dict[str, dict[int, dict[str, np.ndarray]]] = defaultdict(dict)
    for horizon, hindex in ((1, 0), (2, 1), (4, 3)):
        true_endpoint = dev_np["future_latents"][:, hindex]; true_delta = true_endpoint - current
        predicted_delta = paired_predictors(train, current, ids, hindex, tau_by_goal, k)
        for name, delta in predicted_delta.items(): predictors[name][horizon] = current + delta
        static_retrieval = []; horizon_retrieval = []; prototype = []
        for z, goal in zip(current, ids):
            static = static_cores[ctx["vocab"][goal]]; static_retrieval.append(static[np.argmin(np.linalg.norm(static[:, 16:] - z[None, 16:], axis=1))])
            hcore = horizon_cores[(int(goal), horizon)]; horizon_retrieval.append(hcore[np.argmin(np.linalg.norm(hcore[:, 16:] - z[None, 16:], axis=1))])
            prototype.append(ctx["regions"][ctx["vocab"][goal]].mean(0))
        predictors["static_core_retrieval"][horizon] = np.stack(static_retrieval); predictors["horizon_core_retrieval"][horizon] = np.stack(horizon_retrieval); predictors["language_prototype"][horizon] = np.stack(prototype); predictors["Wave21_B1"][horizon] = b1_prediction[:, hindex]
        static_distance = np.asarray([knn_distances(true_endpoint[i:i+1], static_cores[ctx["vocab"][ids[i]]], min(k, len(static_cores[ctx["vocab"][ids[i]]])))[0] for i in range(len(ids))])
        horizon_distance = np.asarray([knn_distances(true_endpoint[i:i+1], horizon_cores[(int(ids[i]), horizon)], min(k, len(horizon_cores[(int(ids[i]), horizon)])))[0] for i in range(len(ids))])
        gain = static_distance - horizon_distance; horizon_gain_values.append(gain); repeated_sessions.append(sessions)
        for label, delta in (("D1", predicted_delta["D1_mean_local"]), ("D2", predicted_delta["D2_weighted_local"]), ("D3", predicted_delta["D3_nearest"]), ("goal_mean", predicted_delta["goal_horizon_mean"])):
            direction[f"{label}_full_cosine" if label != "goal_mean" else "goal_mean_full_cosine"].append(cosine(delta, true_delta))
        d2_delta = predicted_delta["D2_weighted_local"]
        direction["D2_execution_cosine"].append(cosine(d2_delta[:, 16:], true_delta[:, 16:])); direction["D2_semantic_cosine"].append(cosine(d2_delta[:, :16], true_delta[:, :16])); direction["D2_norm_ratio"].append(np.linalg.norm(d2_delta, axis=1) / np.maximum(np.linalg.norm(true_delta, axis=1), 1e-8)); direction["D2_angular_error"].append(np.arccos(np.clip(cosine(d2_delta, true_delta), -1, 1)))
        for name, endpoint in predictors.items():
            if horizon not in endpoint: continue
            value = endpoint[horizon]; decoded = decode_continuous(ctx["representation"], value, ctx["mean"], ctx["std"], device); decoded_error = np.mean((decoded - dev_np["future_actions"][:, hindex, :, :6]) ** 2, axis=(1, 2)); _, reencoded, correction = cycle_numpy(ctx["representation"], value, device); rm = region_metrics(value, ctx["regions"], ctx["vocab"], ids, k); rr = region_metrics(reencoded, ctx["regions"], ctx["vocab"], ids, k)
            current_last = dev_np["current_action"][:, -1, :6]; gt_jump = np.linalg.norm(dev_np["future_actions"][:, hindex, 0, :6] - current_last, axis=1); pred_jump = np.linalg.norm(decoded[:, 0] - current_last, axis=1)
            hcores = {task: horizon_cores[(goal, horizon)] for goal, task in enumerate(ctx["vocab"])}; hgeom = goal_geometry(value, hcores, ctx["vocab"], ids, k); sgeom = goal_geometry(value, static_cores, ctx["vocab"], ids, k)
            model_metrics[name][horizon] = {"full_mse": np.mean((value - true_endpoint) ** 2, axis=1), "semantic_mse": np.mean((value[:, :16] - true_endpoint[:, :16]) ** 2, axis=1), "execution_mse": np.mean((value[:, 16:] - true_endpoint[:, 16:]) ** 2, axis=1), "decoded_mse": decoded_error, "endpoint_prediction": rm["prediction"], "reencoded_prediction": rr["prediction"], "cycle_residual": np.linalg.norm(correction, axis=1), "continuity_error": np.abs(pred_jump - gt_jump), "static_margin": sgeom["margin"], "horizon_margin": hgeom["margin"]}
        for i in range(len(ids)):
            rows.append({"sample": i, "session": int(sessions[i]), "goal": ctx["vocab"][ids[i]], "horizon": horizon, "HorizonCoreGain": float(gain[i]), "D2_full_cosine": float(direction["D2_full_cosine"][-1][i]), "D2_execution_cosine": float(direction["D2_execution_cosine"][-1][i]), "D2_full_mse": float(model_metrics["D2_weighted_local"][horizon]["full_mse"][i]), "goal_mean_full_mse": float(model_metrics["goal_horizon_mean"][horizon]["full_mse"][i]), "D2_execution_mse": float(model_metrics["D2_weighted_local"][horizon]["execution_mse"][i]), "goal_mean_execution_mse": float(model_metrics["goal_horizon_mean"][horizon]["execution_mse"][i])})

    flat_sessions = np.concatenate(repeated_sessions); gain_ci = positive_ci(np.concatenate(horizon_gain_values), flat_sessions, config, 0)
    d2_full_cos = np.concatenate(direction["D2_full_cosine"]); d2_exec_cos = np.concatenate(direction["D2_execution_cosine"])
    full_cos_ci = positive_ci(d2_full_cos, flat_sessions, config, 1); exec_cos_ci = positive_ci(d2_exec_cos, flat_sessions, config, 2)
    d2_full_error = np.concatenate([model_metrics["D2_weighted_local"][h]["full_mse"] for h in (1, 2, 4)]); mean_full_error = np.concatenate([model_metrics["goal_horizon_mean"][h]["full_mse"] for h in (1, 2, 4)])
    d2_exec_error = np.concatenate([model_metrics["D2_weighted_local"][h]["execution_mse"] for h in (1, 2, 4)]); mean_exec_error = np.concatenate([model_metrics["goal_horizon_mean"][h]["execution_mse"] for h in (1, 2, 4)])
    full_gain_ci = positive_ci(mean_full_error - d2_full_error, flat_sessions, config, 3); exec_gain_ci = positive_ci(mean_exec_error - d2_exec_error, flat_sessions, config, 4)
    d2_identity = macro_accuracy(np.concatenate([model_metrics["D2_weighted_local"][h]["endpoint_prediction"] for h in (1, 2, 4)]), np.tile(ids, 3), goal_count); b1_identity = macro_accuracy(np.concatenate([model_metrics["Wave21_B1"][h]["endpoint_prediction"] for h in (1, 2, 4)]), np.tile(ids, 3), goal_count)
    d2_recoded = macro_accuracy(np.concatenate([model_metrics["D2_weighted_local"][h]["reencoded_prediction"] for h in (1, 2, 4)]), np.tile(ids, 3), goal_count); b1_recoded = macro_accuracy(np.concatenate([model_metrics["Wave21_B1"][h]["reencoded_prediction"] for h in (1, 2, 4)]), np.tile(ids, 3), goal_count)
    d2_continuity = float(np.mean(np.concatenate([model_metrics["D2_weighted_local"][h]["continuity_error"] for h in (1, 2, 4)]))); b1_continuity = float(np.mean(np.concatenate([model_metrics["Wave21_B1"][h]["continuity_error"] for h in (1, 2, 4)])))
    a1 = gain_ci["lower_95"] > 0; a2 = full_cos_ci["lower_95"] > 0 and exec_cos_ci["lower_95"] > 0; a3 = full_gain_ci["lower_95"] > 0 and exec_gain_ci["lower_95"] > 0
    a4 = float(model_metrics["D2_weighted_local"][2]["full_mse"].mean()) < min(float(model_metrics["static_core_retrieval"][2]["full_mse"].mean()), float(model_metrics["language_prototype"][2]["full_mse"].mean())) and float(model_metrics["D2_weighted_local"][4]["decoded_mse"].mean()) < min(float(model_metrics["static_core_retrieval"][4]["decoded_mse"].mean()), float(model_metrics["language_prototype"][4]["decoded_mse"].mean()))
    a5 = d2_identity >= b1_identity and d2_recoded >= b1_recoded; a6 = d2_continuity <= b1_continuity; gates = {"A1": a1, "A2": a2, "A3": a3, "A4": a4, "A5": a5, "A6": a6}; authorized = all(gates.values())
    summary_metrics = {name: {str(h): {key: float(value.mean()) if isinstance(value, np.ndarray) and value.dtype.kind == "f" else (macro_accuracy(value, ids, goal_count) if key in ("endpoint_prediction", "reencoded_prediction") else None) for key, value in metrics.items()} for h, metrics in horizons.items()} for name, horizons in model_metrics.items()}
    results = {"M2_state_horizon_conditioned_displacement_family": "SUPPORTED_FOR_INTERVENTION" if authorized else "REJECTED", "gates": gates, "test_arrays_opened": False, "optimizer_steps_before_decision": 0, "development_samples": len(ids), "eligible_cells": read_json(out / "wave24_transition_family_manifest.json")["eligible_cells"], "A1_HorizonCoreGain": gain_ci, "A2_D2_cosine": {"full": full_cos_ci, "execution": exec_cos_ci}, "A3_D2_minus_goal_mean": {"full_MSE_improvement": full_gain_ci, "execution_MSE_improvement": exec_gain_ci}, "A4_static_baselines": {"D2_H2_full_MSE": float(model_metrics["D2_weighted_local"][2]["full_mse"].mean()), "static_H2_full_MSE": float(model_metrics["static_core_retrieval"][2]["full_mse"].mean()), "prototype_H2_full_MSE": float(model_metrics["language_prototype"][2]["full_mse"].mean()), "D2_H4_decoded_MSE": float(model_metrics["D2_weighted_local"][4]["decoded_mse"].mean()), "static_H4_decoded_MSE": float(model_metrics["static_core_retrieval"][4]["decoded_mse"].mean()), "prototype_H4_decoded_MSE": float(model_metrics["language_prototype"][4]["decoded_mse"].mean())}, "A5_identity": {"D2_endpoint_macro": d2_identity, "B1_endpoint_macro": b1_identity, "D2_decode_reencode_macro": d2_recoded, "B1_decode_reencode_macro": b1_recoded}, "A6_continuity": {"D2": d2_continuity, "B1": b1_continuity}, "direction_metrics": {key: {str(h): distribution(values[i]) for i, h in enumerate((1, 2, 4))} for key, values in direction.items()}, "model_metrics": summary_metrics}
    write_json(out / "wave24_phaseA_results.json", results); write_json(out / "wave24_mechanism_gate.json", {"M2_state_horizon_conditioned_displacement_family": results["M2_state_horizon_conditioned_displacement_family"], "gates": gates, "definitions_frozen_before_development": True, "test_arrays_opened": False})
    figures = out / "publication_figures_data"; figures.mkdir(exist_ok=True)
    with (figures / "phaseA_displacement_per_sample.csv").open("w", newline="") as handle: writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    write_json(figures / "phaseA_model_metrics.json", summary_metrics)
    (out / "wave24_phaseA_horizon_core_diagnosis.md").write_text(f"# Wave 24 Phase A1: horizon cores\n\nHorizonCoreGain={gain_ci['mean']:.6f}, clustered 95% CI [{gain_ci['lower_95']:.6f}, {gain_ci['upper_95']:.6f}]. A1={a1}. All S1/S2 supports are train-only and H1/H2/H4 remain separate.\n")
    (out / "wave24_phaseA_source_conditioned_displacement.md").write_text(f"# Wave 24 Phase A2: source-conditioned displacement\n\nD2 full cosine={full_cos_ci['mean']:.6f} [{full_cos_ci['lower_95']:.6f}, {full_cos_ci['upper_95']:.6f}]; execution cosine={exec_cos_ci['mean']:.6f} [{exec_cos_ci['lower_95']:.6f}, {exec_cos_ci['upper_95']:.6f}]. Goal-mean minus D2 full-MSE={full_gain_ci['mean']:.6f} [{full_gain_ci['lower_95']:.6f}, {full_gain_ci['upper_95']:.6f}], execution={exec_gain_ci['mean']:.6f} [{exec_gain_ci['lower_95']:.6f}, {exec_gain_ci['upper_95']:.6f}]. M2=**{results['M2_state_horizon_conditioned_displacement_family']}**, gates `{json.dumps(gates,sort_keys=True)}`; no optimizer or held-out access occurred.\n")
    if not authorized: (out / "wave24_displacement_family_mechanism_rejected.md").write_text("# Wave 24 displacement-family mechanism rejected\n\nAt least one prospectively frozen development gate failed. No LCT-TD training, held-out inference, new K, new tau, extra lambda, or rescue loss is permitted.\n")
    print(json.dumps({"stage": "phasea", "M2": results["M2_state_horizon_conditioned_displacement_family"], "gates": gates}), flush=True)


def train(config: dict, device: torch.device) -> None:
    gate = read_json(out_path(config) / "wave24_mechanism_gate.json")
    if gate["M2_state_horizon_conditioned_displacement_family"] != "SUPPORTED_FOR_INTERVENTION": raise RuntimeError("STOP: M2 rejected; LCT-TD training forbidden")
    raise NotImplementedError("training is activated only after M2 authorization")


def evaluate_final(config: dict, device: torch.device) -> None:
    raise NotImplementedError("final is activated only after development model selection")


def report(config: dict, device: torch.device) -> None:
    out=out_path(config);phase=read_json(out/"wave24_phaseA_results.json")
    if phase["M2_state_horizon_conditioned_displacement_family"]!="REJECTED":raise RuntimeError("This report branch expects the registered M2 stop")
    ctx=load_context(config,device);wave21=ctx["wave21"];train=load_numpy_dataset(wave21/"datasets"/"train.npz");dev_np=load_numpy_dataset(wave21/"datasets"/"development.npz");dev=dataset_tensors(wave21/"datasets"/"development.npz",ctx["goals"],device)
    ids=dev_np["goal_id"];sessions=dev_np["session_row"];current=dev_np["z_current"];n=len(ids);goal_count=len(ctx["vocab"]);k=int(config["transition_family"]["neighbors"]);tau=compute_tau(train["z_current"],train["goal_id"],goal_count,k)
    with np.load(ROOT/config["experiment"]["wave23_root"]/"wave23_goal_cores.npz") as archive:static_cores={task:archive[task].copy() for task in ctx["vocab"]}
    with np.load(out/"wave24_horizon_cores.npz") as archive:horizon_cores={(g,h):archive[f"{task}__H{h}"].copy() for g,task in enumerate(ctx["vocab"]) for h in (1,2,4)}

    def family_geometry(predicted_delta:np.ndarray,query_current:np.ndarray,hindex:int,target:np.ndarray,execution:bool=False)->dict[str,np.ndarray]:
        distances=np.empty((len(target),goal_count),np.float32);sl=slice(16,None) if execution else slice(None)
        train_delta=train["future_latents"][:,hindex]-train["z_current"]
        for i,(delta,z) in enumerate(zip(predicted_delta,query_current)):
            for goal in range(goal_count):
                indices,_=source_neighbors(z,train["z_current"],train["goal_id"],goal,k,True)
                distances[i,goal]=np.linalg.norm(train_delta[indices,sl]-delta[None,sl],axis=1).mean()
        target_distance=distances[np.arange(len(target)),target];competing=distances.copy();competing[np.arange(len(target)),target]=np.inf
        return {"distances":distances,"target_distance":target_distance,"margin":competing.min(1)-target_distance,"prediction":distances.argmin(1)}

    # Development observed-query family diagnostics and explanatory rows.
    explanatory=[];observed_family={};source_rows=[]
    for horizon,hindex in ((1,0),(2,1),(4,3)):
        true_endpoint=dev_np["future_latents"][:,hindex];true_delta=true_endpoint-current;preds=paired_predictors(train,current,ids,hindex,tau,k);d2=preds["D2_weighted_local"]
        family=family_geometry(d2,current,hindex,ids);family_exec=family_geometry(d2,current,hindex,ids,True);_,_,gt_correction=cycle_numpy(ctx["representation"],true_endpoint,device)
        static_distance=np.asarray([knn_distances(true_endpoint[i:i+1],static_cores[ctx["vocab"][ids[i]]],k)[0] for i in range(n)]);horizon_distance=np.asarray([knn_distances(true_endpoint[i:i+1],horizon_cores[(int(ids[i]),horizon)],k)[0] for i in range(n)])
        endpoint=current+d2;rm=region_metrics(endpoint,ctx["regions"],ctx["vocab"],ids,k)
        observed_family[str(horizon)]={"full_family_margin":positive_ci(family["margin"],sessions,config,horizon*10),"execution_family_margin":positive_ci(family_exec["margin"],sessions,config,horizon*10+1),"full_family_identity":macro_accuracy(family["prediction"],ids,goal_count),"execution_family_identity":macro_accuracy(family_exec["prediction"],ids,goal_count)}
        for i in range(n):
            indices,distances=source_neighbors(current[i],train["z_current"],train["goal_id"],int(ids[i]),k,True);neighbor_delta=train["future_latents"][indices,hindex]-train["z_current"][indices];neighbor_similarity=float(cosine(neighbor_delta,true_delta[i:i+1]).mean())
            source_rows.append({"sample":i,"session":int(sessions[i]),"goal":ctx["vocab"][ids[i]],"horizon":horizon,"mean_source_distance":float(distances.mean()),"mean_neighbor_displacement_cosine":neighbor_similarity,"D2_cosine":float(cosine(d2[i:i+1],true_delta[i:i+1])[0])})
            explanatory.append({"session":int(sessions[i]),"static_distance":float(static_distance[i]),"horizon_distance":float(horizon_distance[i]),"source_distance":float(distances.mean()),"paired_cosine":float(cosine(d2[i:i+1],true_delta[i:i+1])[0]),"family_margin":float(family["margin"][i]),"global_cycle_residual":float(np.linalg.norm(gt_correction[i])),"endpoint_correct":float(rm["prediction"][i]==ids[i]),"D2_full_MSE":float(np.mean((d2[i]-true_delta[i])**2))})
    figures_data=out/"publication_figures_data";figures_data.mkdir(exist_ok=True)
    write_json(figures_data/"development_family_metrics.json",observed_family)
    with (figures_data/"source_conditioning_rows.csv").open("w",newline="") as handle:writer=csv.DictWriter(handle,fieldnames=list(source_rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(source_rows)

    # Standardized explanatory regression with source-session clustered bootstrap.
    x_names=["static_distance","horizon_distance","source_distance","paired_cosine","family_margin","global_cycle_residual"];X=np.asarray([[row[key] for key in x_names] for row in explanatory],np.float64);y=np.asarray([row["endpoint_correct"] for row in explanatory]);reg_sessions=np.asarray([row["session"] for row in explanatory]);unique=np.unique(reg_sessions);rng=np.random.default_rng(int(config["evaluation"]["bootstrap_seed"])+90)
    def fit_reg(idx:np.ndarray)->tuple[np.ndarray,float,float]:
        xx=X[idx];xx=(xx-xx.mean(0))/np.maximum(xx.std(0),1e-8);design=np.column_stack((np.ones(len(idx)),xx));beta=np.linalg.lstsq(design,y[idx],rcond=None)[0];pred=design@beta;den=max(float(np.sum((y[idx]-y[idx].mean())**2)),1e-12);r2=1-float(np.sum((y[idx]-pred)**2))/den
        base=design[:,[0,1,6]];base_pred=base@np.linalg.lstsq(base,y[idx],rcond=None)[0];base_r2=1-float(np.sum((y[idx]-base_pred)**2))/den
        return beta[1:],r2,r2-base_r2
    point=fit_reg(np.arange(len(y)));samples=np.empty((10000,len(x_names)+2))
    for b in range(10000):
        chosen=rng.choice(unique,len(unique),replace=True);idx=np.concatenate([np.flatnonzero(reg_sessions==session) for session in chosen]);beta,r2,inc=fit_reg(idx);samples[b]=np.r_[beta,r2,inc]
    regression={"outcome":"development D2 endpoint correctness","predictors":{name:{"standardized_coefficient":float(point[0][i]),"lower_95":float(np.quantile(samples[:,i],.025)),"upper_95":float(np.quantile(samples[:,i],.975)),"spearman_with_D2_MSE":float(stats.spearmanr(X[:,i],np.asarray([row['D2_full_MSE'] for row in explanatory])).statistic)} for i,name in enumerate(x_names)},"full_R2":point[1],"incremental_R2_over_static_distance_plus_cycle":point[2],"cluster":"source_session","replicates":10000,"seed":int(config["evaluation"]["bootstrap_seed"])+90,"causal_interpretation":False}
    write_json(figures_data/"development_explanatory_regression.json",regression)

    # Development same-state six-way D2 diagnostic and frozen B1 control.
    goals_t=torch.from_numpy(ctx["goals"]).float().to(device);same_state={};b1=[]
    for goal in range(goal_count):b1.append(predict_ensemble(ctx["wcfg"],"B1_correct_language",dev,goals_t[goal].expand(n,-1),device,wave21)[0])
    same_state["Wave21_B1"]=np.stack(b1,axis=1)
    d2_traj=np.empty((n,goal_count,3,32),np.float32)
    for goal in range(goal_count):
        requested_ids=np.full(n,goal);pred_by_h=[]
        for hindex in (0,1,3):pred_by_h.append(current+paired_predictors(train,current,requested_ids,hindex,tau,k)["D2_weighted_local"])
        d2_traj[:,goal]=np.stack(pred_by_h,axis=1)
    same_state["D2_diagnostic"]=d2_traj
    np.savez_compressed(figures_data/"development_same_state_displacements.npz",**same_state,z_current=current,goal_id=ids,session_row=sessions)
    same_metrics={};same_per_goal={}
    for name,trajectory in same_state.items():
        endpoint=trajectory[:,:,-1];flat=endpoint.reshape(-1,32);requested=np.tile(np.arange(goal_count),n);rm=region_metrics(flat,ctx["regions"],ctx["vocab"],requested,k);_,recoded,_=cycle_numpy(ctx["representation"],flat,device);rr=region_metrics(recoded,ctx["regions"],ctx["vocab"],requested,k)
        target_ep=endpoint[np.arange(n),ids];wrong_ep=np.stack([np.mean(np.delete(endpoint[i],ids[i],axis=0),axis=0) for i in range(n)]);td=region_metrics(target_ep,ctx["regions"],ctx["vocab"],ids,k)["target_distance"];wd=region_metrics(wrong_ep,ctx["regions"],ctx["vocab"],ids,k)["target_distance"];te=region_metrics(target_ep,ctx["regions"],ctx["vocab"],ids,k,slice(16,None))["target_distance"];we=region_metrics(wrong_ep,ctx["regions"],ctx["vocab"],ids,k,slice(16,None))["target_distance"]
        same_metrics[name]={"RedirectGain":float(np.mean(wd-td)),"Execution_RedirectGain":float(np.mean(we-te)),"endpoint_macro_accuracy":macro_accuracy(rm["prediction"],requested,goal_count),"decode_reencode_macro_accuracy":macro_accuracy(rr["prediction"],requested,goal_count)}
        same_per_goal[name]={task:float(np.mean(rm["prediction"][requested==goal]==goal)) for goal,task in enumerate(ctx["vocab"])}
    write_json(figures_data/"development_same_state_metrics.json",same_metrics);write_json(figures_data/"development_same_state_per_goal.json",same_per_goal)

    # All eligible development lift->place records; test remains masked.
    inventory=list(csv.DictReader((wave21/"wave21_transition_inventory.csv").open()));lookup={(int(sessions[i]),int(dev_np["boundary_frame"][i])):i for i in range(n)};cases=[row for row in inventory if row["split"]=="development" and row["previous_label"]=="lift_blue_block_slider" and row["next_label"]=="place_in_slider" and (int(row["session_row"]),int(row["boundary_frame"])) in lookup];place=ctx["vocab"].index("place_in_slider");lift_rows=[]
    for row in cases:
        i=lookup[(int(row["session_row"]),int(row["boundary_frame"]))]
        for horizon,hindex in ((1,0),(2,1),(4,3)):
            true_delta=dev_np["future_latents"][i,hindex]-current[i];requested=np.full(1,place);d2=paired_predictors(train,current[i:i+1],requested,hindex,tau,k)["D2_weighted_local"][0];goalmean=paired_predictors(train,current[i:i+1],requested,hindex,tau,k)["goal_horizon_mean"][0]
            for name,delta in (("D2",d2),("goal_horizon_mean",goalmean),("ground_truth",true_delta)):
                endpoint=current[i]+delta;decoded=decode_continuous(ctx["representation"],endpoint[None],ctx["mean"],ctx["std"],device)[0];fam=family_geometry(delta[None],current[i:i+1],hindex,requested);fam_exec=family_geometry(delta[None],current[i:i+1],hindex,requested,True)
                lift_rows.append({"sample":i,"session":int(sessions[i]),"boundary_frame":int(dev_np["boundary_frame"][i]),"horizon":horizon,"model":name,"displacement_cosine":float(cosine(delta[None],true_delta[None])[0]),"execution_cosine":float(cosine(delta[None,16:],true_delta[None,16:])[0]),"norm_ratio":float(np.linalg.norm(delta)/max(np.linalg.norm(true_delta),1e-8)),"future_latent_MSE":float(np.mean((delta-true_delta)**2)),"decoded_action_MSE":float(np.mean((decoded-dev_np["future_actions"][i,hindex,:,:6])**2)),"family_margin":float(fam["margin"][0]),"execution_family_margin":float(fam_exec["margin"][0])})
    with (figures_data/"development_lift_to_place.csv").open("w",newline="") as handle:
        fields=list(lift_rows[0]) if lift_rows else ["sample","session","boundary_frame","horizon","model","displacement_cosine","execution_cosine","norm_ratio","future_latent_MSE","decoded_action_MSE","family_margin","execution_family_margin"];writer=csv.DictWriter(handle,fieldnames=fields,lineterminator="\n");writer.writeheader();writer.writerows(lift_rows)

    write_json(out/"wave24_transition_weight_selection.json",{"status":"NOT_RUN_M2_REJECTED","selected_lambda_TM":None,"candidate_set":config["model"]["lambda_TM_candidates"],"development_training_started":False,"heldout_opened":False,"no_extra_lambda":True})
    write_json(out/"wave24_final_test_preregistration.json",{"status":"NOT_ACTIVATED_M2_REJECTED","M2":"REJECTED","selected_lambda_TM":None,"heldout_arrays_opened":False,"masked_test_rows":164,"bootstrap":{"cluster":"source_session","replicates":10000,"seed":240824},"post_stop_rescue":False})
    claim={"M2_state_horizon_conditioned_displacement_family":"REJECTED","C13_language_selects_state_conditioned_transition_family":"NOT_TESTED","C14_language_as_state_horizon_conditioned_executable_transition_selector":"NOT_TESTED","horizon_specific_support_better_than_static":False,"source_state_conditioning_matters":True,"paired_displacement_predicts_future":True,"full_redirect_preserved":"inconclusive","execution_redirect_preserved":"inconclusive","endpoint_identity_repaired":False,"decode_reencode_identity_repaired":"inconclusive","continuity_preserved":False,"gates":phase["gates"],"heldout_arrays_opened":False,"scientific_summary":"Source-conditioned paired displacements predict direction and beat goal+horizon means, but weighted averaging shrinks magnitude and fails static baselines, endpoint identity, and continuity."}
    write_json(out/"wave24_claim_decision.json",claim)

    # Tables A-F.
    tables=out/"publication_tables";figures=out/"publication_figures";tables.mkdir(exist_ok=True);figures.mkdir(exist_ok=True);parquet=pq.read_table(out/"wave24_paired_transition_inventory.parquet").to_pylist()
    with (tables/"table_A_paired_inventory.csv").open("w",newline="") as handle:
        writer=csv.writer(handle,lineterminator="\n");writer.writerow(["goal","horizon","train","development","test_metadata","adequate"])
        for task in ctx["vocab"]:
            counts={split:sum(row["next_goal"]==task and row["split"]==split for row in parquet) for split in ("train","development","test")}
            for h in (1,2,4):writer.writerow([task,h,counts["train"],counts["development"],counts["test"],counts["train"]>=8])
    with (tables/"table_B_support_diagnostic.csv").open("w",newline="") as handle:
        writer=csv.writer(handle,lineterminator="\n");writer.writerow(["structure","H2_full_MSE","H4_decoded_MSE","endpoint_macro","continuity"])
        for name in ("static_core_retrieval","horizon_core_retrieval","D2_weighted_local","language_prototype","Wave21_B1"):writer.writerow([name,phase["model_metrics"][name]["2"]["full_mse"],phase["model_metrics"][name]["4"]["decoded_mse"],np.mean([phase["model_metrics"][name][str(h)]["endpoint_prediction"] for h in (1,2,4)]),np.mean([phase["model_metrics"][name][str(h)]["continuity_error"] for h in (1,2,4)])])
    with (tables/"table_C_displacement_direction.csv").open("w",newline="") as handle:
        writer=csv.writer(handle,lineterminator="\n");writer.writerow(["predictor","horizon","full_cosine","execution_cosine","norm_ratio"])
        for h in (1,2,4):writer.writerow(["D2",h,phase["direction_metrics"]["D2_full_cosine"][str(h)]["mean"],phase["direction_metrics"]["D2_execution_cosine"][str(h)]["mean"],phase["direction_metrics"]["D2_norm_ratio"][str(h)]["mean"]])
    with (tables/"table_D_M2_gate.csv").open("w",newline="") as handle:writer=csv.writer(handle,lineterminator="\n");writer.writerow(["gate","pass"]);[writer.writerow([key,value]) for key,value in phase["gates"].items()];writer.writerow(["M2","REJECTED"])
    with (tables/"table_E_LCT_TD_heldout.csv").open("w",newline="") as handle:csv.writer(handle,lineterminator="\n").writerows([["metric","LCT_TD"],["all","NOT_TESTED_M2_REJECTED"]])
    with (tables/"table_F_claims.csv").open("w",newline="") as handle:csv.writer(handle,lineterminator="\n").writerows([["claim","decision"],["M2","REJECTED"],["C13","NOT_TESTED"],["C14","NOT_TESTED"]])

    # Figures 1-7, all train/development only.
    try:
        import matplotlib.pyplot as plt
        rng_fig=np.random.default_rng(24);fig,ax=plt.subplots(figsize=(7,3));ax.scatter([0],[0],s=700,alpha=.2,label="static C_g");ax.scatter([2],[0],s=450,alpha=.3,label="horizon C_g,h");ax.quiver([4,4,4],[0,-.4,.4],[1,1.1,.9],[.2,.5,-.1],angles="xy",scale_units="xy",scale=1);ax.text(4,-.8,"source-conditioned displacement family");ax.legend();ax.set_axis_off();fig.tight_layout();fig.savefig(figures/"figure_1_support_structures.png",dpi=160);plt.close(fig)
        family_manifest=read_json(out/"wave24_transition_family_manifest.json");fig,ax=plt.subplots(figsize=(9,3));x=np.arange(goal_count);width=.25
        for j,h in enumerate((1,2,4)):ax.bar(x+(j-1)*width,[family_manifest["statistics"][f"{task}__H{h}"]["mean_displacement_norm"] for task in ctx["vocab"]],width,label=f"H{h}")
        ax.set_xticks(x,ctx["vocab"],rotation=35,ha="right",fontsize=6);ax.legend();ax.set_title("Train displacement norms by goal/horizon");fig.tight_layout();fig.savefig(figures/"figure_2_horizon_distributions.png",dpi=160);plt.close(fig)
        sx=np.asarray([row["mean_source_distance"] for row in source_rows]);sy=np.asarray([row["mean_neighbor_displacement_cosine"] for row in source_rows]);fig,ax=plt.subplots();ax.scatter(sx,sy,s=8,alpha=.4);ax.set(xlabel="mean source-state distance",ylabel="neighbor displacement cosine");fig.tight_layout();fig.savefig(figures/"figure_3_source_proximity.png",dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,3));xx=np.arange(3);width=.35;ax.bar(xx-width/2,[phase["model_metrics"]["D2_weighted_local"][str(h)]["full_mse"] for h in (1,2,4)],width,label="D2");ax.bar(xx+width/2,[phase["model_metrics"]["goal_horizon_mean"][str(h)]["full_mse"] for h in (1,2,4)],width,label="goal+h mean");ax.set_xticks(xx,["H1","H2","H4"]);ax.legend();ax.set_title("Development full MSE");fig.tight_layout();fig.savefig(figures/"figure_4_D2_vs_goal_mean.png",dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,3));xx=np.arange(goal_count);width=.35;ax.bar(xx-width/2,[same_per_goal["Wave21_B1"][task] for task in ctx["vocab"]],width,label="B1");ax.bar(xx+width/2,[same_per_goal["D2_diagnostic"][task] for task in ctx["vocab"]],width,label="D2");ax.set_xticks(xx,ctx["vocab"],rotation=35,ha="right",fontsize=6);ax.legend();ax.set_title("Development same-state H4 identity");fig.tight_layout();fig.savefig(figures/"figure_5_same_state_sixway.png",dpi=160);plt.close(fig)
        all_train=np.concatenate([train["future_latents"][:,i]-train["z_current"] for i in (0,1,3)]);center=all_train.mean(0);_,_,vt=np.linalg.svd(all_train-center,full_matrices=False);basis=vt[:2].T;goal=ctx["vocab"].index("place_in_slider");fig,ax=plt.subplots();
        for h,hindex in ((1,0),(2,1),(4,3)):
            values=train["future_latents"][train["goal_id"]==goal,hindex]-train["z_current"][train["goal_id"]==goal];p=(values-center)@basis;ax.scatter(p[:,0],p[:,1],s=8,alpha=.25,label=f"train H{h}")
        dev_indices=np.flatnonzero(ids==goal)[:3]
        for h,hindex in ((1,0),(2,1),(4,3)):
            d2=paired_predictors(train,current[dev_indices],ids[dev_indices],hindex,tau,k)["D2_weighted_local"];p=(d2-center)@basis;ax.scatter(p[:,0],p[:,1],marker="x",s=45)
        ax.legend(fontsize=6);ax.set_title("Same goal, distinct current states");fig.tight_layout();fig.savefig(figures/"figure_6_state_conditioned_transitions.png",dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,3));
        if lift_rows:
            for name in ("D2","goal_horizon_mean","ground_truth"):ax.plot((1,2,4),[np.mean([row["displacement_cosine"] for row in lift_rows if row["model"]==name and row["horizon"]==h]) for h in (1,2,4)],marker="o",label=name)
            ax.legend();ax.set(xlabel="horizon",ylabel="displacement cosine");ax.set_title("Development lift→place: all eligible")
        else:ax.text(.5,.5,"No eligible development case\nheld-out masked",ha="center",va="center");ax.set_axis_off()
        fig.tight_layout();fig.savefig(figures/"figure_7_lift_to_place.png",dpi=160);plt.close(fig)
    except ImportError:write_json(figures/"matplotlib_unavailable.json",{"raw_figure_data_complete":True})

    # Focused deliverables and final narrative.
    (out/"wave24_training_report.md").write_text("# Wave 24 training report\n\nNo LCT-TD training ran. M2 failed A1/A4/A5/A6 before any Wave24 optimizer step; lambda_TM candidates and six seeds remain unused.\n")
    (out/"wave24_main_comparison.md").write_text("# Wave 24 main comparison\n\nDevelopment-only D1/D2/D3, goal+horizon mean, static core retrieval, horizon core retrieval, prototype, and frozen B1 are reported. D2 beat goal+horizon mean but not B1/prototype, so no held-out LCT-TD comparison exists.\n")
    (out/"wave24_same_state_language_swap.md").write_text(f"# Wave 24 same-state language swap\n\nDevelopment-only six-way diagnostic: B1 RedirectGain={same_metrics['Wave21_B1']['RedirectGain']:.6f}, D2={same_metrics['D2_diagnostic']['RedirectGain']:.6f}. Only requested language/family changed for fixed current/history. No learned LCT-TD exists.\n")
    (out/"wave24_transition_family_results.md").write_text(f"# Wave 24 transition-family results\n\nD2 predicts direction: full cosine={phase['A2_D2_cosine']['full']['mean']:.6f}, execution={phase['A2_D2_cosine']['execution']['mean']:.6f}. It beats goal+horizon mean by full MSE={phase['A3_D2_minus_goal_mean']['full_MSE_improvement']['mean']:.6f} and execution MSE={phase['A3_D2_minus_goal_mean']['execution_MSE_improvement']['mean']:.6f}. However mean norm ratios are H1/H2/H4={phase['direction_metrics']['D2_norm_ratio']['1']['mean']:.3f}/{phase['direction_metrics']['D2_norm_ratio']['2']['mean']:.3f}/{phase['direction_metrics']['D2_norm_ratio']['4']['mean']:.3f}, consistent with averaging/cancellation.\n")
    (out/"wave24_decode_reencode_results.md").write_text(f"# Wave 24 decode/re-encode results\n\nDevelopment aggregate D2 identity={phase['A5_identity']['D2_decode_reencode_macro']:.6f} versus B1={phase['A5_identity']['B1_decode_reencode_macro']:.6f}; this submetric improved, but endpoint identity degraded and joint A5 failed. No held-out result exists.\n")
    (out/"wave24_continuity_results.md").write_text(f"# Wave 24 continuity results\n\nDevelopment D2 continuity error={phase['A6_continuity']['D2']:.6f} versus B1={phase['A6_continuity']['B1']:.6f}; A6 failed. No learned or held-out result exists.\n")
    (out/"wave24_lift_to_place_case.md").write_text(f"# Wave 24 lift-to-place case\n\nHeld-out remained masked after M2 rejection. All {len(cases)} eligible development `lift_blue_block_slider -> place_in_slider` cases were analyzed at H1/H2/H4 without cherry-picking; raw results are in `publication_figures_data/development_lift_to_place.csv`.\n")
    (out/"wave24_statistical_report.md").write_text(f"# Wave 24 statistical report\n\nIndependent unit: source session; development n=6 sessions. Bootstrap=10,000, seed family 240824. HorizonCoreGain={phase['A1_HorizonCoreGain']['mean']:.6f} [{phase['A1_HorizonCoreGain']['lower_95']:.6f}, {phase['A1_HorizonCoreGain']['upper_95']:.6f}]; full cosine={phase['A2_D2_cosine']['full']['mean']:.6f} [{phase['A2_D2_cosine']['full']['lower_95']:.6f}, {phase['A2_D2_cosine']['full']['upper_95']:.6f}]; goal-mean minus D2 full MSE={phase['A3_D2_minus_goal_mean']['full_MSE_improvement']['mean']:.6f} [{phase['A3_D2_minus_goal_mean']['full_MSE_improvement']['lower_95']:.6f}, {phase['A3_D2_minus_goal_mean']['full_MSE_improvement']['upper_95']:.6f}]. No test/C13/C14 statistics were computed.\n")
    (out/"wave24_failure_taxonomy.md").write_text("# Wave 24 failure taxonomy\n\nFrozen categories: no horizon dependence; no source-state dependence; goal-only displacement sufficient; poor displacement direction; wrong displacement magnitude; semantic-only transition family; execution-family overlap; endpoint identity failure; decode/reencode failure; continuity failure; sparse-data cell; long-horizon accumulation; other.\n\nActivated: no horizon dependence under the registered core metric, wrong displacement magnitude, endpoint identity failure, continuity failure, and long-horizon accumulation. Source-state dependence and displacement direction did not fail; no sparse-data cell activated.\n")

    q=[
        "560 metadata rows were reconstructed for H1/H2/H4; 396 train/development records contain paired arrays and 164 test rows remain masked.","31 source sessions in metadata; Phase A materialized 25 train/development sessions.","All 18 goal/horizon cells are adequate; train counts are at least 25 and K=20 is available.",f"No. Aggregate HorizonCoreGain={phase['A1_HorizonCoreGain']['mean']:.6f}; static Wave23 cores were closer.",f"No. Its clustered lower95={phase['A1_HorizonCoreGain']['lower_95']:.6f}.","Yes directionally. Both full and execution cosine are positive with lower bounds above zero.",f"Full cosine={phase['A2_D2_cosine']['full']['mean']:.6f} [{phase['A2_D2_cosine']['full']['lower_95']:.6f}, {phase['A2_D2_cosine']['full']['upper_95']:.6f}].",f"Execution cosine={phase['A2_D2_cosine']['execution']['mean']:.6f} [{phase['A2_D2_cosine']['execution']['lower_95']:.6f}, {phase['A2_D2_cosine']['execution']['upper_95']:.6f}].",f"Yes. Goal-mean minus D2 full MSE={phase['A3_D2_minus_goal_mean']['full_MSE_improvement']['mean']:.6f}, lower95={phase['A3_D2_minus_goal_mean']['full_MSE_improvement']['lower_95']:.6f}.",f"Yes. Execution improvement={phase['A3_D2_minus_goal_mean']['execution_MSE_improvement']['mean']:.6f}, lower95={phase['A3_D2_minus_goal_mean']['execution_MSE_improvement']['lower_95']:.6f}.","Yes for predicting displacement relative to a goal+horizon mean, but not strongly enough to authorize the complete executable-transition mechanism.",f"No. D2 H2 full MSE={phase['A4_static_baselines']['D2_H2_full_MSE']:.6f} exceeded prototype={phase['A4_static_baselines']['prototype_H2_full_MSE']:.6f}; H4 decoded MSE also exceeded prototype.",f"No jointly. Endpoint macro {phase['A5_identity']['D2_endpoint_macro']:.6f} < B1 {phase['A5_identity']['B1_endpoint_macro']:.6f}, although decode/reencode improved.",f"No. D2 continuity={phase['A6_continuity']['D2']:.6f} > B1={phase['A6_continuity']['B1']:.6f}.","No. M2 is REJECTED (A1/A4/A5/A6 failed).","None; LCT-TD training was forbidden.","Not tested; no LCT-TD exists.","Not tested; no LCT-TD exists.","Not tested; no LCT-TD exists.","Not tested; no LCT-TD exists.","Development D2 family margins are reported, but held-out lower95 was not opened.","Not tested held-out.","Not tested held-out.","Direction remained positive at H1/H2/H4, but the full mechanism failed jointly.","Source-state directional benefit is broad descriptively; C13 breadth was not tested.",f"Held-out was masked. {len(cases)} development cases were analyzed at all horizons.","C13 is NOT_TESTED.","C14 is NOT_TESTED.","Waves21–23 are best explained by a multimodal, state-dependent transition distribution: language changes direction and local source states inform it, but deterministic neighborhood averaging cancels modes and underestimates magnitude; static/horizon endpoint sets do not repair this.","Defensible claim: source-conditioned paired train transitions predict development displacement direction and outperform a goal+horizon mean, but deterministic averaged displacement is insufficient for executable identity and continuity.","If C13 had passed, the next step would be matched-state closed-loop CALVIN execution with receding-horizon decoding, frozen B1/LCT-TD controls, and no rescue.","The causal language-vector-field effect and execution redirection remain supported; Wave24 adds that current-state neighbors carry predictive directional information, while an executable state/horizon selector remains unsupported."]
    lines=["# Twenty-fourth wave results: transition displacement families","",f"Run date: {now()}","","## Outcome","","- M2 state/horizon-conditioned displacement family: **REJECTED**","- C13/C14: **NOT_TESTED**","- LCT-TD optimizer steps: **0**","- Held-out arrays materialized: **false**","","A2/A3 passed: state-conditioned paired displacements predict direction and beat a goal+horizon mean. A1/A4/A5/A6 failed: horizon cores were not better, D2 lost to prototype/B1 on required errors, endpoint identity degraded, and continuity worsened. The characteristic norm shrinkage indicates that weighted averaging is canceling heterogeneous displacement modes.","","## Required questions",""]+[f"{i}. {answer}" for i,answer in enumerate(q,1)]+["","## Scientific decision","","Do not train LCT-TD with a softmin-to-mean displacement target. The surviving signal is distributional: current state narrows the transition family, but a single averaged vector is not an adequate estimator of its executable mode.","","## Discipline disclosure","","Train built S1/S2/S3 and tau; development alone decided M2. Test rows remained null in Parquet. No new K/tau, lambda sweep, loss, seed, closed-loop rollout, F2, DEL, cycle rescue, or endpoint attraction was introduced."]
    result_text="\n".join(lines)+"\n";(out/"twenty_fourth_wave_results.md").write_text(result_text);report_path=ROOT/config["experiment"]["report_path"];report_path.parent.mkdir(parents=True,exist_ok=True);report_path.write_text(result_text)
    next_text="""# Twenty-fourth wave next experiment

## Decision from Wave 24

Wave 25 should remain on the language-conditioned latent-dynamics research line, but replace deterministic neighborhood averaging with a compact conditional distribution over displacement modes. Wave24 found reliable state-conditioned direction (full cosine 0.627, execution cosine 0.648) and large gains over a goal+horizon mean, while D2 produced only 56–66% of true displacement magnitude and failed endpoint/continuity gates. This is the signature expected when heterogeneous local displacement modes are averaged.

## Recommended experiment: conditional mode diagnosis before a new model

First, use train only to cluster normalized displacement direction and log-magnitude separately within each `(goal,horizon)` cell. Condition mode probabilities on `(z_previous,z_current,language,horizon)`. On development, compare the deterministic mean, nearest mode, a compact mixture-density head, and an oracle best mode. Authorize a learned model only if a non-oracle mode selector improves H2 full MSE, H4 decoded MSE, endpoint identity, and continuity while preserving the Wave21 language effect. This directly tests whether multimodality—not missing language or source state—is causing mean cancellation.

If authorized, train a small `LCT-MD` mixture-displacement head rather than a large end-to-end policy: predict categorical mode probabilities plus per-mode direction and log-norm residuals, then decode the selected/sampled latent trajectory. Keep the representation, decoder, text projection, Wave21 split, six seeds, source-session bootstrap, and all historical rejection decisions frozen. Do not add endpoint classification, cycle loss, F2, DEL, or closed-loop execution.

## Relation to recent methods

The distributional direction is consistent with [Diffusion Policy](https://arxiv.org/abs/2303.04137), which models multimodal high-dimensional robot actions rather than regressing their mean. More recent flow approaches make the same issue explicit: [VFP](https://arxiv.org/abs/2508.01622) adds a variational prior, optimal-transport alignment, and mixture-of-experts specialization for task/path multimodality; [LG-Flow Policy](https://arxiv.org/abs/2601.23087) performs flow matching in a temporally regularized latent action space to improve smoothness; and [Latent Action Guided Flow Matching](https://arxiv.org/abs/2606.23420) replaces one global Gaussian with state-selected learned priors for fragmented, heteroscedastic action spaces. These papers motivate distributional modeling, but the present dataset has only 257 train transitions, so Wave25 should begin with a compact mixture head and an oracle/non-oracle mode diagnostic rather than immediately adopting a high-capacity diffusion or flow model.

If the compact mixture diagnostic fails, the next conclusion should be that the frozen latent representation lacks sufficient phase/contact information. Only then should the project add a phase variable or learn a temporally regularized latent action representation; it should not keep stacking geometric attraction losses onto the current coordinates.
"""
    (out/"twenty_fourth_wave_next_experiment.md").write_text(next_text);(ROOT/"NEXT_EXPERIMENT.md").write_text(next_text)
    log_path=ROOT/"RESEARCH_LOG.md";log_text=log_path.read_text();marker="## Wave 24 — State/horizon-conditioned displacement families"
    if marker not in log_text:
        log_text+=f"\n{marker} ({datetime.now().date()})\n\n- Reconstructed 560 transition metadata rows; materialized 396 train/dev paired records and kept 164 test rows masked.\n- All 18 goal/horizon train cells adequate; K=20, tau train-only.\n- M2 **REJECTED**: A2/A3 passed, A1/A4/A5/A6 failed.\n- D2 cosine full/execution={phase['A2_D2_cosine']['full']['mean']:.6f}/{phase['A2_D2_cosine']['execution']['mean']:.6f}; it beat goal+horizon means but shrank displacement magnitude and failed identity/continuity.\n- C13/C14 **NOT_TESTED**; no optimizer or held-out materialization.\n- Full artifacts: `{out.relative_to(ROOT)}`.\n";log_path.write_text(log_text)
    (out/"updated_RESEARCH_LOG.md").write_text(log_path.read_text());(out/"updated_NEXT_EXPERIMENT.md").write_text((ROOT/"NEXT_EXPERIMENT.md").read_text())
    (out/"environment_freeze.txt").write_text("\n".join([f"timestamp={now()}",f"python={' '.join(sys.version.split())}",f"platform={platform.platform()}",f"torch={torch.__version__}",f"numpy={np.__version__}",f"pyarrow={pa.__version__}",f"cuda_available={torch.cuda.is_available()}",f"cuda_device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}"])+"\n")
    (out/"exact_commands.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n# The first direct report invocation lacked PYTHONPATH and stopped at ModuleNotFoundError; no experiment code ran.\n/home/jinjaguo/anaconda3/envs/libero/bin/python -m pip install pyarrow==17.0.0\nPYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_12.py --config configs/dynamics_12.yaml --stage prepare --device cuda:0\nPYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_12.py --config configs/dynamics_12.yaml --stage phasea --device cuda:0\nPYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_12.py --config configs/dynamics_12.yaml --stage report --device cuda:0\nPYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python -m pytest tests/dynamics/test_dynamics_12_displacement_family.py -q\n")
    (out/"files_changed.txt").write_text("\n".join(["configs/dynamics_12.yaml","prompts/dynamics_12.md","scripts/dynamics/run_dynamics_12.py","tests/dynamics/test_dynamics_12_displacement_family.py","reports/dynamics_12_results.md","RESEARCH_LOG.md","NEXT_EXPERIMENT.md",config["experiment"]["output_root"]+"/"])+"\n")
    print(json.dumps({"stage":"report","M2":"REJECTED","C13":"NOT_TESTED","heldout_materialized":False}),flush=True)


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--config",type=Path,required=True);parser.add_argument("--stage",choices=("prepare","phasea","train","final","report","all"),default="all");parser.add_argument("--device");args=parser.parse_args()
    config=yaml.safe_load((ROOT/args.config).read_text());device=torch.device(args.device or config["runtime"]["device"]);torch.set_num_threads(int(config["runtime"]["torch_cpu_threads"]))
    if device.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("Registered Wave24 run requires CUDA")
    stages=("prepare","phasea","train","final","report") if args.stage=="all" else (args.stage,)
    for stage in stages:
        print(json.dumps({"stage":stage,"started_at":now()}),flush=True)
        if stage=="report":report(config,device)
        else:{"prepare":prepare,"phasea":phase_a,"train":train,"final":evaluate_final}[stage](config,device)


if __name__=="__main__":main()

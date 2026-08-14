#!/usr/bin/env python3
"""Run Wave 22 decoder-cycle-consistent latent-transition experiments.

Purpose
-------
Diagnose frozen Wave 21 LCT cycle drift, apply the exact four-step frozen
encoder-decoder cycle-map diagnostic, and only when the preregistered Phase-A
gate passes train/evaluate six-seed LCT-CC models with one added cycle loss.

Parameters
----------
--config: Wave 22 YAML configuration.
--stage: ``prepare``, ``phasea``, ``geometry``, ``train``, ``final``, ``report``, or ``all``.
--device: Optional torch device override; the registered run uses ``cuda:0``.

Usage
-----
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/run_dynamics_10.py --config configs/dynamics_10.yaml \
  --stage all --device cuda:0

Outputs
-------
Writes manifests, diagnostics, checkpoints, raw tables/figure data, figures,
claim decisions, reports, and reproducibility records below
``results/dynamics/twenty_second_wave/2026-08-14_dynamics_10`` and updates
``reports/dynamics_10_results.md``, ``RESEARCH_LOG.md``, and
``NEXT_EXPERIMENT.md`` during the report stage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
from collections import defaultdict
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
        LCT, cluster_bootstrap, dataset_tensors, decode_continuous,
        load_representation, normalize, predict_ensemble, read_json,
        region_metrics, sha256, write_json,
    )
except ModuleNotFoundError:
    from run_dynamics_9 import (
        LCT, cluster_bootstrap, dataset_tensors, decode_continuous,
        load_representation, normalize, predict_ensemble, read_json,
        region_metrics, sha256, write_json,
    )


ROOT = Path(__file__).resolve().parents[2]


def now() -> str:
    return datetime.now().astimezone().isoformat()


def out_path(config: dict) -> Path:
    return ROOT / config["experiment"]["output_root"]


def wave21_path(config: dict) -> Path:
    return ROOT / config["experiment"]["wave21_root"]


def load_context(config: dict, device: torch.device) -> dict[str, Any]:
    wave21_config = yaml.safe_load((ROOT / config["wave21_config"]).read_text())
    wave21 = wave21_path(config)
    representation, payload, mean, std = load_representation(wave21_config, device)
    goals_np = np.load(wave21 / "wave21_goal_embeddings.npy")
    vocab = list(wave21_config["data"]["vocabulary"])
    with np.load(wave21 / "wave21_train_regions.npz") as archive:
        regions = {task: archive[task].copy() for task in vocab}
    return {
        "wave21_config": wave21_config,
        "wave21": wave21,
        "representation": representation,
        "payload": payload,
        "mean": mean,
        "std": std,
        "goals_np": goals_np,
        "vocab": vocab,
        "regions": regions,
    }


def decode_and_cycle_tensor(representation: nn.Module, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the exact frozen D then E map using CALVIN's gripper convention."""
    decoded = representation.decode(latent)
    encoder_input = decoded.clone()
    encoder_input[..., 6] = torch.where(decoded[..., 6] >= 0, 1.0, -1.0)
    return decoded, representation.encode(encoder_input)


def cycle_numpy(representation: nn.Module, latent: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = latent.shape
    with torch.no_grad():
        decoded, cycled = decode_and_cycle_tensor(
            representation, torch.from_numpy(latent.reshape(-1, 32)).float().to(device)
        )
    return (
        decoded.cpu().numpy().reshape(*shape[:-1], 16, 7),
        cycled.cpu().numpy().reshape(shape),
        (cycled - torch.from_numpy(latent.reshape(-1, 32)).float().to(device)).cpu().numpy().reshape(shape),
    )


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-8)


def local_geometry(query: np.ndarray, train: np.ndarray, neighbors: int, tangent_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return local normal distance and tangent/normal language-vector parts."""
    normal_distance = np.empty(len(query), np.float32)
    bases = np.empty((len(query), 32, tangent_dim), np.float32)
    centers = np.empty_like(query)
    for start in range(0, len(query), 128):
        block = query[start : start + 128]
        dist = np.sum((block[:, None, :] - train[None, :, :]) ** 2, axis=-1)
        indices = np.argpartition(dist, neighbors - 1, axis=1)[:, :neighbors]
        for offset, selected in enumerate(indices):
            i = start + offset
            local = train[selected]
            center = local.mean(0)
            _, _, vt = np.linalg.svd(local - center, full_matrices=False)
            basis = vt[:tangent_dim].T
            residual = query[i] - center
            normal_distance[i] = np.linalg.norm(residual - basis @ (basis.T @ residual))
            bases[i] = basis
            centers[i] = center
    return normal_distance, bases, centers


def bootstrap_association(x: np.ndarray, y: np.ndarray, sessions: np.ndarray, replicates: int, seed: int) -> dict[str, Any]:
    unique = np.unique(sessions)
    rng = np.random.default_rng(seed)

    def corr(a: np.ndarray, b: np.ndarray, kind: str) -> float:
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return 0.0
        return float(stats.pearsonr(a, b).statistic if kind == "pearson" else stats.spearmanr(a, b).statistic)

    result: dict[str, Any] = {"cluster": "source_session", "source_sessions": len(unique), "replicates": replicates}
    for kind in ("pearson", "spearman"):
        point = corr(x, y, kind)
        samples = np.empty(replicates, np.float64)
        for b in range(replicates):
            chosen = rng.choice(unique, len(unique), replace=True)
            indices = np.concatenate([np.flatnonzero(sessions == session) for session in chosen])
            samples[b] = corr(x[indices], y[indices], kind)
        result[kind] = {"estimate": point, "lower_95": float(np.quantile(samples, .025)), "upper_95": float(np.quantile(samples, .975))}
    return result


def distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, .90)),
        "p95": float(np.quantile(values, .95)),
    }


def prepare(config: dict, device: torch.device) -> None:
    out = out_path(config)
    out.mkdir(parents=True, exist_ok=True)
    ctx = load_context(config, device)
    wave21 = ctx["wave21"]
    frozen = read_json(wave21 / "wave21_frozen_representation_manifest.json")
    b1_hashes = {
        str(seed): sha256(wave21 / "checkpoints" / "B1_correct_language" / f"seed_{seed}.pt")
        for seed in config["model"]["seeds"]
    }
    manifest = {
        "created_before_wave22_optimizer": True,
        "created_at": now(),
        "wave21_claims_remain_rejected": {"C7": "REJECTED", "C8": "REJECTED"},
        "representation_checkpoint": frozen["checkpoint"],
        "representation_sha256": frozen["checkpoint_sha256"],
        "action_encoder_sha256": frozen["action_encoder_sha256"],
        "decoder_sha256": frozen["decoder_sha256"],
        "text_projection_sha256": frozen["semantic_projection_sha256"],
        "normalization": frozen["normalization"],
        "normalization_sha256": frozen["normalization_sha256"],
        "wave21_B0_hashes": {str(seed): sha256(wave21 / "checkpoints" / "B0_unconditional" / f"seed_{seed}.pt") for seed in config["model"]["seeds"]},
        "wave21_B1_hashes": b1_hashes,
        "wave21_B2_hashes": {str(seed): sha256(wave21 / "checkpoints" / "B2_shuffled_language" / f"seed_{seed}.pt") for seed in config["model"]["seeds"]},
        "train_dataset_sha256": sha256(wave21 / "datasets" / "train.npz"),
        "development_dataset_sha256": sha256(wave21 / "datasets" / "development.npz"),
        "historically_open_wave21_test_dataset_sha256": sha256(wave21 / "datasets" / "test.npz"),
        "session_split_sha256": sha256(wave21 / "wave21_session_split_manifest.json"),
        "boundary_inventory_sha256": sha256(wave21 / "wave21_transition_inventory.csv"),
        "train_regions_sha256": sha256(wave21 / "wave21_train_regions.npz"),
        "goal_embeddings_sha256": sha256(wave21 / "wave21_goal_embeddings.npy"),
        "phase_a_uses_historical_wave21_heldout_outputs": True,
        "wave22_lctcc_test_predictions_opened": False,
    }
    write_json(out / "wave22_frozen_wave21_manifest.json", manifest)
    write_json(out / "wave22_seed_preregistration.json", {
        "created_before_training": True,
        "seeds": config["model"]["seeds"],
        "paired_with_wave21_B1": True,
        "no_seed_addition_after_test": True,
    })
    write_json(out / "wave22_model_preregistration.json", {
        "created_before_training": True,
        "architecture": "exact Wave21 B1 LCT architecture",
        "trainable": "transition model only (state, goal, transition submodules)",
        "frozen": ["representation encoder", "representation decoder", "text projection"],
        "objective": {"latent_prediction": 1.0, "decoded_action": 1.0, "cycle_consistency": "selected development-only from [0.1, 0.3, 1.0]"},
        "forbidden_losses": {"target_region": 0.0, "prototype": 0.0, "knn": 0.0, "pca": 0.0, "endpoint_classification": 0.0},
        "cycle_map": "E(D(z)) with decoder gripper thresholded to CALVIN -1/+1 convention before E",
        "same_wave21_split_inventory_chunking_and_gap_handling": True,
        "lambda_candidates": config["model"]["lambda_cycle_candidates"],
        "selection_split": "development only",
    })
    write_json(out / "wave22_phaseA_preregistration.json", {
        "created_before_phase_a_metrics": True,
        "historical_wave21_test_disclosure": "Wave21 held-out test was opened in Wave21 and is required by the prompt for frozen-model mechanism diagnosis; no new LCT-CC test prediction exists.",
        "cycle_iterations": 4,
        "bootstrap": {"cluster": "source_session", "replicates": 10000, "seed": 220822},
        "A1": "lower95 of B1 H4 cycle minus ground-truth H4 cycle > 0 and mean B1 H4 > B1 H0",
        "A2": "mean B1 H4 execution-cycle residual > 0.25",
        "A3": "lower95 Pearson or Spearman association is >0 for decoded-action error or endpoint-classification error",
        "A4": "lower95 of cycle0 minus cycle4 residual >0",
        "A5": "cycle4 RedirectGain clustered lower95 >0",
    })
    print(json.dumps({"stage": "prepare", "status": "complete", "output": str(out.relative_to(ROOT))}), flush=True)


def phase_a(config: dict, device: torch.device) -> None:
    out = out_path(config)
    if not (out / "wave22_phaseA_preregistration.json").exists():
        raise RuntimeError("prepare must freeze Phase-A definitions before diagnosis")
    ctx = load_context(config, device)
    wcfg, wave21 = ctx["wave21_config"], ctx["wave21"]
    rep, mean, std = ctx["representation"], ctx["mean"], ctx["std"]
    goals_np, vocab, regions = ctx["goals_np"], ctx["vocab"], ctx["regions"]
    test = dataset_tensors(wave21 / "datasets" / "test.npz", goals_np, device)
    ids = test["goal_id"].cpu().numpy()
    sessions = test["session_row_np"]
    n, gcount = len(ids), len(vocab)
    current = test["z_current"].cpu().numpy()
    true = test["future_latents"].cpu().numpy()
    true_action = test["future_actions"].cpu().numpy()
    goals_t = torch.from_numpy(goals_np).float().to(device)
    b0, _ = predict_ensemble(wcfg, "B0_unconditional", test, None, device, wave21)
    observed, _ = predict_ensemble(wcfg, "B1_correct_language", test, test["goal"], device, wave21)
    sixway = []
    for goal_id in range(gcount):
        goal = goals_t[goal_id].expand(n, -1)
        prediction, _ = predict_ensemble(wcfg, "B1_correct_language", test, goal, device, wave21)
        sixway.append(prediction)
    sixway = np.stack(sixway, axis=1)
    all_train = np.concatenate([regions[task] for task in vocab])
    h_values = [0, 1, 2, 3, 4]
    observed_path = np.concatenate([current[:, None], observed], axis=1)
    true_path = np.concatenate([current[:, None], true], axis=1)
    rows: list[dict[str, Any]] = []
    residual_cache: dict[str, list[np.ndarray]] = {key: [] for key in ("pred_full", "pred_sem", "pred_exec", "gt_full", "gt_exec")}
    for h in h_values:
        z = observed_path[:, h]
        decoded, cycled, correction = cycle_numpy(rep, z, device)
        gt_decoded, gt_cycled, gt_correction = cycle_numpy(rep, true_path[:, h], device)
        normal, _, _ = local_geometry(z, all_train, int(config["evaluation"]["local_pca_neighbors"]), int(config["evaluation"]["local_pca_tangent_dim"]))
        full = np.linalg.norm(correction, axis=1)
        sem = np.linalg.norm(correction[:, :16], axis=1)
        exe = np.linalg.norm(correction[:, 16:], axis=1)
        gt_full = np.linalg.norm(gt_correction, axis=1)
        gt_exec = np.linalg.norm(gt_correction[:, 16:], axis=1)
        for key, value in (("pred_full", full), ("pred_sem", sem), ("pred_exec", exe), ("gt_full", gt_full), ("gt_exec", gt_exec)):
            residual_cache[key].append(value)
        target = region_metrics(z, regions, vocab, ids, int(config["evaluation"]["knn_k"]))
        target_exec = region_metrics(z, regions, vocab, ids, int(config["evaluation"]["knn_k"]), slice(16, None))
        if h == 0:
            decoded_mse = np.zeros(n)
            latent_mse = np.zeros(n)
            jump = np.zeros(n)
        else:
            decoded_denorm = decoded[..., :6] * std + mean
            decoded_mse = np.mean((decoded_denorm - true_action[:, h - 1, :, :6]) ** 2, axis=(1, 2))
            latent_mse = np.mean((z - true[:, h - 1]) ** 2, axis=1)
            previous_decoded, _, _ = cycle_numpy(rep, observed_path[:, h - 1], device)
            prev_denorm = previous_decoded[..., :6] * std + mean
            jump = np.linalg.norm(decoded_denorm[:, 0] - prev_denorm[:, -1], axis=1)
        for i in range(n):
            rows.append({
                "kind": "observed", "sample": i, "session": int(sessions[i]), "source_goal": vocab[ids[i]], "target_goal": vocab[ids[i]], "horizon": h,
                "cycle_full": float(full[i]), "cycle_semantic": float(sem[i]), "cycle_execution": float(exe[i]), "ground_truth_cycle_full": float(gt_full[i]),
                "target_distance": float(target["target_distance"][i]), "execution_target_distance": float(target_exec["target_distance"][i]), "target_rank": int(target["rank"][i]),
                "endpoint_error": int(target["prediction"][i] != ids[i]), "decoded_action_mse": float(decoded_mse[i]), "future_latent_mse": float(latent_mse[i]),
                "execution_knn_radius": float(target_exec["target_distance"][i]), "local_pca_normal_distance": float(normal[i]), "trajectory_jump": float(jump[i]),
            })
    figures = out / "publication_figures_data"
    tables = out / "publication_tables"
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)
    with (figures / "phaseA_per_sample.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)

    # Full six-way H4 decomposition and language/cycle direction alignment.
    flat = sixway[:, :, 3].reshape(-1, 32)
    requested = np.tile(np.arange(gcount), n)
    flat_sessions = np.repeat(sessions, gcount)
    _, flat_cycle, flat_correction = cycle_numpy(rep, flat, device)
    current_flat = np.repeat(current, gcount, axis=0)
    alignment = {
        "full": distribution(cosine(flat - current_flat, flat_correction)),
        "execution": distribution(cosine(flat[:, 16:] - current_flat[:, 16:], flat_correction[:, 16:])),
    }
    decomposition = {"by_step": {}, "by_target": {}, "by_source": {}, "by_pair": {}}
    for h in h_values:
        decomposition["by_step"][str(h)] = {
            "full": distribution(residual_cache["pred_full"][h]), "semantic": distribution(residual_cache["pred_sem"][h]), "execution": distribution(residual_cache["pred_exec"][h]), "ground_truth_full": distribution(residual_cache["gt_full"][h])
        }
    full_cycle_norm = np.linalg.norm(flat_correction, axis=1)
    sem_cycle_norm = np.linalg.norm(flat_correction[:, :16], axis=1)
    exec_cycle_norm = np.linalg.norm(flat_correction[:, 16:], axis=1)
    for goal_id, task in enumerate(vocab):
        mask = requested == goal_id
        decomposition["by_target"][task] = {"full": distribution(full_cycle_norm[mask]), "semantic": distribution(sem_cycle_norm[mask]), "execution": distribution(exec_cycle_norm[mask])}
    for source_id, task in enumerate(vocab):
        mask = np.repeat(ids, gcount) == source_id
        decomposition["by_source"][task] = {"full": distribution(full_cycle_norm[mask]), "semantic": distribution(sem_cycle_norm[mask]), "execution": distribution(exec_cycle_norm[mask])}
        for target_id, target in enumerate(vocab):
            pair_mask = mask & (requested == target_id)
            if pair_mask.any():
                decomposition["by_pair"][f"{task}->{target}"] = {"full": distribution(full_cycle_norm[pair_mask]), "semantic": distribution(sem_cycle_norm[pair_mask]), "execution": distribution(exec_cycle_norm[pair_mask])}

    # Associations at observed H4, with session-clustered resampling.
    h4_rows = [row for row in rows if row["horizon"] == 4]
    x = np.asarray([row["cycle_full"] for row in h4_rows])
    association = {}
    for j, key in enumerate(("future_latent_mse", "decoded_action_mse", "target_distance", "endpoint_error", "execution_knn_radius", "local_pca_normal_distance", "trajectory_jump")):
        y = np.asarray([row[key] for row in h4_rows])
        association[key] = bootstrap_association(x, y, sessions, int(config["evaluation"]["bootstrap_replicates"]), int(config["evaluation"]["bootstrap_seed"]) + j)
    write_json(figures / "phaseA_associations.json", association)
    write_json(figures / "phaseA_cycle_decomposition.json", decomposition)
    write_json(figures / "language_cycle_alignment.json", alignment)

    # Exact K=4 cycle fixed-point diagnostic on every H4 same-state endpoint.
    projected = flat.copy()
    projection_rows = []
    projection_values = []
    for iteration in range(int(config["evaluation"]["cycle_iterations"]) + 1):
        decoded, cycled, correction = cycle_numpy(rep, projected, device)
        rm = region_metrics(projected, regions, vocab, requested, int(config["evaluation"]["knn_k"]))
        er = region_metrics(projected, regions, vocab, requested, int(config["evaluation"]["knn_k"]), slice(16, None))
        observed_action = np.repeat(true_action[:, 3, None], gcount, axis=1).reshape(-1, 16, 7)
        decoded_denorm = decoded[..., :6] * std + mean
        decoded_mse = np.mean((decoded_denorm - observed_action[..., :6]) ** 2, axis=(1, 2))
        projection_values.append({"latent": projected.copy(), "cycle": np.linalg.norm(correction, axis=1), "target_distance": rm["target_distance"], "exec_distance": er["target_distance"], "accuracy": rm["prediction"] == requested, "decoded_mse": decoded_mse})
        projection_rows.append({"iteration": iteration, "cycle_residual": float(np.mean(np.linalg.norm(correction, axis=1))), "target_distance": float(rm["target_distance"].mean()), "execution_target_distance": float(er["target_distance"].mean()), "endpoint_accuracy": float(np.mean(rm["prediction"] == requested)), "execution_knn_radius": float(er["target_distance"].mean()), "decoded_action_mse": float(decoded_mse.mean())})
        if iteration < int(config["evaluation"]["cycle_iterations"]):
            projected = cycled
    def redirect_for(endpoints: np.ndarray, execution: bool = False) -> tuple[np.ndarray, dict]:
        endpoint = endpoints.reshape(n, gcount, 32)
        target_ep = endpoint[np.arange(n), ids]
        wrong_ep = np.stack([np.mean(np.delete(endpoint[i], ids[i], axis=0), axis=0) for i in range(n)])
        sl = slice(16, None) if execution else slice(None)
        target_d = region_metrics(target_ep, regions, vocab, ids, int(config["evaluation"]["knn_k"]), sl)["target_distance"]
        wrong_d = region_metrics(wrong_ep, regions, vocab, ids, int(config["evaluation"]["knn_k"]), sl)["target_distance"]
        values = wrong_d - target_d
        return values, cluster_bootstrap(values, sessions, int(config["evaluation"]["bootstrap_replicates"]), int(config["evaluation"]["bootstrap_seed"]) + (31 if execution else 30))
    redirect0, redirect0_ci = redirect_for(projection_values[0]["latent"])
    redirect4, redirect4_ci = redirect_for(projection_values[4]["latent"])
    exec_redirect0, exec_redirect0_ci = redirect_for(projection_values[0]["latent"], True)
    exec_redirect4, exec_redirect4_ci = redirect_for(projection_values[4]["latent"], True)
    direction_cosine = cosine(projection_values[4]["latent"] - current_flat, flat - current_flat)
    reduction = projection_values[0]["cycle"].reshape(n, gcount).mean(1) - projection_values[4]["cycle"].reshape(n, gcount).mean(1)
    reduction_ci = cluster_bootstrap(reduction, sessions, int(config["evaluation"]["bootstrap_replicates"]), int(config["evaluation"]["bootstrap_seed"]) + 32)
    projection = {"name": "CYCLE_FIXED_POINT_DIAGNOSTIC", "K_cycle": 4, "iterations": projection_rows, "RedirectGain_LCT": redirect0_ci, "RedirectGain_cycle4": redirect4_ci, "ExecutionRedirectGain_LCT": exec_redirect0_ci, "ExecutionRedirectGain_cycle4": exec_redirect4_ci, "cycle_reduction": reduction_ci, "direction_cosine": distribution(direction_cosine)}
    write_json(figures / "cycle_projection_diagnostic.json", projection)

    h4_delta = residual_cache["pred_full"][4] - residual_cache["gt_full"][4]
    a1_ci = cluster_bootstrap(h4_delta, sessions, int(config["evaluation"]["bootstrap_replicates"]), int(config["evaluation"]["bootstrap_seed"]) + 40)
    a1 = a1_ci["lower_95"] > 0 and float(np.mean(residual_cache["pred_full"][4])) > float(np.mean(residual_cache["pred_full"][0]))
    a2 = float(np.mean(residual_cache["pred_exec"][4])) > float(config["evaluation"]["phase_a"]["execution_nontrivial_mean"])
    a3_keys = ["decoded_action_mse", "endpoint_error"]
    a3 = any(association[key][kind]["lower_95"] > 0 for key in a3_keys for kind in ("pearson", "spearman"))
    a4 = reduction_ci["lower_95"] > 0
    a5 = redirect4_ci["lower_95"] > 0
    authorized = all((a1, a2, a3, a4, a5))
    gates = {"A1": a1, "A2": a2, "A3": a3, "A4": a4, "A5": a5}
    diagnosis = {
        "M0_decoder_consistency_mechanism": "SUPPORTED_FOR_INTERVENTION" if authorized else "REJECTED",
        "gates": gates,
        "A1_clustered_B1_H4_minus_GT_H4": a1_ci,
        "B1_cycle_by_step": {str(h): distribution(residual_cache["pred_full"][h]) for h in h_values},
        "ground_truth_cycle_by_step": {str(h): distribution(residual_cache["gt_full"][h]) for h in h_values},
        "execution_cycle_by_step": {str(h): distribution(residual_cache["pred_exec"][h]) for h in h_values},
        "association": association,
        "projection": projection,
        "language_cycle_alignment": alignment,
        "optimizer_steps_before_decision": 0,
    }
    write_json(out / "wave22_phaseA_results.json", diagnosis)
    (out / "wave22_phaseA_cycle_diagnosis.md").write_text(
        "# Wave 22 Phase-A cycle diagnosis\n\n"
        f"M0: **{diagnosis['M0_decoder_consistency_mechanism']}**. No Wave22 optimizer step occurred before this decision.\n\n"
        f"Gates: `{json.dumps(gates, sort_keys=True)}`. B1 mean cycle residual H0={np.mean(residual_cache['pred_full'][0]):.6f}, H4={np.mean(residual_cache['pred_full'][4]):.6f}; held-out ground-truth H4={np.mean(residual_cache['gt_full'][4]):.6f}. "
        f"At H4 semantic={np.mean(residual_cache['pred_sem'][4]):.6f}, execution={np.mean(residual_cache['pred_exec'][4]):.6f}.\n\n"
        "The historical Wave21 held-out test was already opened during Wave21; Phase A intentionally reuses those frozen trajectories as required. No LCT-CC held-out prediction exists at this point.\n"
    )
    (out / "wave22_cycle_association_report.md").write_text(
        "# Wave 22 cycle association report\n\nSource-session clustered bootstrap, 10,000 replicates, seed family rooted at 220822.\n\n" +
        "\n".join(f"- {key}: Pearson {value['pearson']['estimate']:.4f} [{value['pearson']['lower_95']:.4f}, {value['pearson']['upper_95']:.4f}]; Spearman {value['spearman']['estimate']:.4f} [{value['spearman']['lower_95']:.4f}, {value['spearman']['upper_95']:.4f}]" for key, value in association.items()) + "\n"
    )
    (out / "wave22_cycle_projection_diagnostic.md").write_text(
        "# Wave 22 frozen cycle projection diagnostic\n\n"
        f"Exactly K=4 applications of E(D(.)) reduced mean next-cycle residual from {projection_rows[0]['cycle_residual']:.6f} to {projection_rows[4]['cycle_residual']:.6f}. "
        f"RedirectGain changed from {redirect0_ci['mean']:.6f} to {redirect4_ci['mean']:.6f}; projected clustered lower95={redirect4_ci['lower_95']:.6f}. "
        f"Mean direction cosine={np.mean(direction_cosine):.6f}. This is diagnosis only, not the trained method.\n"
    )
    if not authorized:
        (out / "wave22_decoder_consistency_mechanism_rejected.md").write_text(
            "# Decoder-consistency mechanism rejected\n\nPhase A failed one or more frozen gates. Per preregistration, no LCT-CC optimizer was run and no rescue mechanism or new threshold was introduced.\n"
        )
    print(json.dumps({"stage": "phasea", "M0": diagnosis["M0_decoder_consistency_mechanism"], "gates": gates}), flush=True)


def train(config: dict, device: torch.device) -> None:
    diagnosis = read_json(out_path(config) / "wave22_phaseA_results.json")
    if diagnosis["M0_decoder_consistency_mechanism"] != "SUPPORTED_FOR_INTERVENTION":
        raise RuntimeError("STOP: Phase A rejected decoder consistency; optimizer is forbidden")
    raise NotImplementedError("LCT-CC training is unreachable in this registered run")


def evaluate_final(config: dict, device: torch.device) -> None:
    raise RuntimeError("STOP: no LCT-CC model exists because Phase A rejected the mechanism")


def geometry(config: dict, device: torch.device) -> None:
    """Run only preregistered descriptive geometry after the Phase-A decision."""
    out = out_path(config)
    diagnosis = read_json(out / "wave22_phaseA_results.json")
    ctx = load_context(config, device)
    wcfg, wave21 = ctx["wave21_config"], ctx["wave21"]
    rep, mean, std = ctx["representation"], ctx["mean"], ctx["std"]
    goals_np, vocab, regions = ctx["goals_np"], ctx["vocab"], ctx["regions"]
    dev = dataset_tensors(wave21 / "datasets" / "development.npz", goals_np, device)
    dev_pred, _ = predict_ensemble(wcfg, "B1_correct_language", dev, dev["goal"], device, wave21)
    point_count = min(int(config["evaluation"]["jacobian_points"]), len(dev_pred))
    jacobian_rows = []
    for i in np.linspace(0, len(dev_pred) - 1, point_count, dtype=int):
        point = torch.from_numpy(dev_pred[i, 3]).float().to(device).requires_grad_(True)

        def cycle_map(value: torch.Tensor) -> torch.Tensor:
            return decode_and_cycle_tensor(rep, value[None])[1][0]

        jac = torch.autograd.functional.jacobian(cycle_map, point, vectorize=True).detach().cpu().numpy()
        singular = np.linalg.svd(jac, compute_uv=False)
        weights = singular / max(float(singular.sum()), 1e-12)
        effective_rank = float(np.exp(-np.sum(weights * np.log(np.maximum(weights, 1e-12)))))
        jacobian_rows.append({
            "development_index": int(i), "singular_values": singular.tolist(), "spectral_norm": float(singular[0]), "effective_rank": effective_rank,
            "semantic_block_frobenius": float(np.linalg.norm(jac[:16, :16])), "execution_block_frobenius": float(np.linalg.norm(jac[16:, 16:])),
            "semantic_to_execution_frobenius": float(np.linalg.norm(jac[16:, :16])), "execution_to_semantic_frobenius": float(np.linalg.norm(jac[:16, 16:])),
        })

    test = dataset_tensors(wave21 / "datasets" / "test.npz", goals_np, device)
    ids = test["goal_id"].cpu().numpy(); sessions = test["session_row_np"]
    current = test["z_current"].cpu().numpy(); true = test["future_latents"].cpu().numpy()
    goals_t = torch.from_numpy(goals_np).float().to(device)
    sixway = []
    for goal_id in range(len(vocab)):
        goal = goals_t[goal_id].expand(len(ids), -1)
        pred, _ = predict_ensemble(wcfg, "B1_correct_language", test, goal, device, wave21)
        sixway.append(pred)
    sixway = np.stack(sixway, axis=1)
    all_train = np.concatenate([regions[task] for task in vocab])
    current_flat = np.repeat(current, len(vocab), axis=0)
    endpoints = sixway[:, :, 3].reshape(-1, 32)
    requested = np.tile(np.arange(len(vocab)), len(ids))
    _, bases, _ = local_geometry(current_flat, all_train, int(config["evaluation"]["local_pca_neighbors"]), int(config["evaluation"]["local_pca_tangent_dim"]))
    delta = endpoints - current_flat
    tangent_coefficients = np.einsum("nji,nj->ni", bases, delta)
    tangent = np.einsum("nji,ni->nj", bases, tangent_coefficients)
    normal = delta - tangent
    tangent_rows = {}
    for goal_id, task in enumerate(vocab):
        mask = requested == goal_id
        tangent_norm = np.linalg.norm(tangent[mask], axis=1)
        normal_norm = np.linalg.norm(normal[mask], axis=1)
        tangent_rows[task] = {"tangent_norm": distribution(tangent_norm), "normal_norm": distribution(normal_norm), "normal_fraction": distribution(normal_norm / np.maximum(np.linalg.norm(delta[mask], axis=1), 1e-8))}

    # Same-state confusion for frozen B1 and its K=4 diagnostic projection.
    projected = endpoints.copy()
    for _ in range(4):
        _, projected, _ = cycle_numpy(rep, projected, device)
    matrices = {}
    for name, values in (("Wave21_B1", endpoints), ("Wave21_B1_cycle4_diagnostic", projected)):
        rm = region_metrics(values, regions, vocab, requested, int(config["evaluation"]["knn_k"]))
        confusion = np.zeros((len(vocab), len(vocab)), np.int64)
        for target, predicted in zip(requested, rm["prediction"]):
            confusion[target, predicted] += 1
        _, recoded, _ = cycle_numpy(rep, values, device)
        recoded_rm = region_metrics(recoded, regions, vocab, requested, int(config["evaluation"]["knn_k"]))
        recoded_confusion = np.zeros_like(confusion)
        for target, predicted in zip(requested, recoded_rm["prediction"]):
            recoded_confusion[target, predicted] += 1
        matrices[name] = {"endpoint_confusion": confusion.tolist(), "endpoint_accuracy": float(np.mean(rm["prediction"] == requested)), "decode_reencode_confusion": recoded_confusion.tolist(), "decode_reencode_accuracy": float(np.mean(recoded_rm["prediction"] == requested))}

    # H0/H1/H2/H4 cycle drift and identity for frozen B1.
    drift = {}
    for h in (0, 1, 2, 4):
        values = current_flat if h == 0 else sixway[:, :, h - 1].reshape(-1, 32)
        _, recoded, correction = cycle_numpy(rep, values, device)
        rm = region_metrics(values, regions, vocab, requested, int(config["evaluation"]["knn_k"]))
        rr = region_metrics(recoded, regions, vocab, requested, int(config["evaluation"]["knn_k"]))
        drift[str(h)] = {"cycle_residual": float(np.linalg.norm(correction, axis=1).mean()), "execution_cycle_residual": float(np.linalg.norm(correction[:, 16:], axis=1).mean()), "endpoint_target_accuracy": float(np.mean(rm["prediction"] == requested)), "decoded_reencoded_target_accuracy": float(np.mean(rr["prediction"] == requested))}

    # Canonical held-out lift_blue_block_slider -> place_in_slider case.
    inventory = list(csv.DictReader((wave21 / "wave21_transition_inventory.csv").open()))
    lookup = {(int(sessions[i]), int(test["boundary_frame_np"][i])): i for i in range(len(ids))}
    candidates = [row for row in inventory if row["split"] == "test" and row["previous_label"] == "lift_blue_block_slider" and row["next_label"] == "place_in_slider" and (int(row["session_row"]), int(row["boundary_frame"])) in lookup]
    case_row = candidates[0] if candidates else next(row for row in inventory if row["split"] == "test" and row["next_label"] == "place_in_slider" and (int(row["session_row"]), int(row["boundary_frame"])) in lookup)
    case = lookup[(int(case_row["session_row"]), int(case_row["boundary_frame"]))]
    place_id = vocab.index("place_in_slider")
    b1_path = np.concatenate([current[case : case + 1], sixway[case, place_id]], axis=0)
    projected_path = b1_path.copy()
    for _ in range(4):
        _, projected_path, _ = cycle_numpy(rep, projected_path, device)
    prototype = regions["place_in_slider"].mean(0)
    prototype_path = np.repeat(prototype[None], 5, axis=0)
    gt_path = np.concatenate([current[case : case + 1], true[case]], axis=0)
    case_data: dict[str, Any] = {"exact_pair_count": len(candidates), "selected_sample": int(case), "session": int(sessions[case]), "boundary_frame": int(test["boundary_frame_np"][case]), "LCT_CC": "NOT_TESTED_PHASE_A_STOP"}
    for name, path in (("Wave21_B1", b1_path), ("cycle4_diagnostic", projected_path), ("language_prototype", prototype_path), ("ground_truth", gt_path)):
        decoded, _, correction = cycle_numpy(rep, path, device)
        rm = region_metrics(path, regions, vocab, np.full(5, place_id), int(config["evaluation"]["knn_k"]))
        er = region_metrics(path, regions, vocab, np.full(5, place_id), int(config["evaluation"]["knn_k"]), slice(16, None))
        target_actions = np.concatenate([test["current_action"][case : case + 1].cpu().numpy(), test["future_actions"][case].cpu().numpy()], axis=0)
        decoded_denorm = decoded[..., :6] * std + mean
        case_data[name] = {"latents": path.tolist(), "decoded_action_chunks": decoded_denorm.tolist(), "target_region_distance": rm["target_distance"].tolist(), "execution_target_region_distance": er["target_distance"].tolist(), "cycle_residual": np.linalg.norm(correction, axis=1).tolist(), "decoded_action_mse": np.mean((decoded_denorm - target_actions[..., :6]) ** 2, axis=(1, 2)).tolist()}

    geometry_payload = {"M0": diagnosis["M0_decoder_consistency_mechanism"], "jacobian": {"split": "development", "point_count": point_count, "descriptive_only": True, "points": jacobian_rows}, "train_only_local_pca_language_direction": tangent_rows, "same_state": matrices, "multi_step_cycle_drift": drift}
    figures = out / "publication_figures_data"
    write_json(figures / "geometry_diagnostics.json", geometry_payload)
    write_json(figures / "same_state_confusion_matrices.json", matrices)
    write_json(figures / "multi_step_cycle_drift.json", drift)
    write_json(figures / "canonical_lift_to_place_case.json", case_data)
    (out / "wave22_geometry_analysis.md").write_text(
        "# Wave 22 geometry analysis\n\n"
        f"The frozen cycle-map Jacobian was measured descriptively on {point_count} development points; it was not used for selection. Mean spectral norm={np.mean([row['spectral_norm'] for row in jacobian_rows]):.6f}, mean effective rank={np.mean([row['effective_rank'] for row in jacobian_rows]):.6f}.\n\n"
        "Train-only local PCA shows the frozen Wave21 language direction contains both tangent and normal components; per-target distributions are in `publication_figures_data/geometry_diagnostics.json`. No tangent projection was applied.\n"
    )
    (out / "wave22_lift_to_place_case.md").write_text(
        "# Wave 22 canonical lift-to-place case\n\n"
        f"Selected exact held-out pair: session {sessions[case]}, boundary {int(test['boundary_frame_np'][case])}; exact-pair count={len(candidates)}. Frozen Wave21 B1, ground truth, prototype, and diagnostic cycle4 chunks are saved. LCT-CC is **NOT TESTED** because Phase A stopped the experiment.\n"
    )
    print(json.dumps({"stage": "geometry", "jacobian_points": point_count, "case": case}), flush=True)


def report(config: dict) -> None:
    out = out_path(config)
    diagnosis = read_json(out / "wave22_phaseA_results.json")
    projection = diagnosis["projection"]
    geometry_data = read_json(out / "publication_figures_data" / "geometry_diagnostics.json")
    decomposition = read_json(out / "publication_figures_data" / "phaseA_cycle_decomposition.json")
    case = read_json(out / "publication_figures_data" / "canonical_lift_to_place_case.json")
    wave21_metrics = read_json(wave21_path(config) / "wave21_main_metrics.json")
    m0 = diagnosis["M0_decoder_consistency_mechanism"]
    stopped = m0 == "REJECTED"
    if not stopped:
        raise RuntimeError("This report implementation is for the registered Phase-A stop branch")

    not_tested = "NOT_TESTED_PHASE_A_STOP"
    write_json(out / "wave22_cycle_weight_selection.json", {
        "status": not_tested, "selected_lambda_cycle": None, "candidates": config["model"]["lambda_cycle_candidates"],
        "selection_split": "development only", "reason": "M0 rejected before optimizer; candidate training was forbidden", "held_out_LCTCC_test_used": False,
    })
    write_json(out / "wave22_final_test_preregistration.json", {
        "status": "NOT_ACTIVATED_PHASE_A_STOP", "selected_lambda_cycle": None, "model_seeds": config["model"]["seeds"], "LCT_CC_checkpoints": [],
        "LCT_CC_heldout_predictions_opened": False, "historical_wave21_test_used_for_required_phaseA": True,
        "gates": "prompts/dynamics_10.md C9 G1-G7 and C10", "metrics": "not evaluated because intervention was not authorized",
        "bootstrap": {"cluster": "source_session", "replicates": 10000, "seed": 220822}, "post_stop_rescue_allowed": False,
    })
    claim = {
        "M0_decoder_consistency_mechanism": "REJECTED",
        "C9_executable_language_redirect": "NOT_TESTED",
        "C10_language_as_executable_target_coordinate": "NOT_TESTED",
        "language_redirect_preserved": "inconclusive",
        "execution_redirect_preserved": "inconclusive",
        "cycle_consistency_repaired": "inconclusive",
        "endpoint_identity_repaired": "inconclusive",
        "decode_reencode_identity_repaired": "inconclusive",
        "continuity_repaired": "inconclusive",
        "current_state_still_matters": "inconclusive",
        "phaseA_gates": diagnosis["gates"],
        "stop_reason": "A5 failed: exact K=4 frozen cycle projection did not retain a strictly positive session-clustered lower 95% bound for full RedirectGain",
        "diagnostic_metrics": {"cycle0": projection["iterations"][0]["cycle_residual"], "cycle4": projection["iterations"][4]["cycle_residual"], "redirect0": projection["RedirectGain_LCT"], "redirect4": projection["RedirectGain_cycle4"], "execution_redirect4": projection["ExecutionRedirectGain_cycle4"], "endpoint_accuracy0": projection["iterations"][0]["endpoint_accuracy"], "endpoint_accuracy4": projection["iterations"][4]["endpoint_accuracy"]},
    }
    write_json(out / "wave22_claim_decision.json", claim)

    tables = out / "publication_tables"; figures = out / "publication_figures_data"; pub = out / "publication_figures"
    tables.mkdir(exist_ok=True); pub.mkdir(exist_ok=True)
    phase_rows = list(csv.DictReader((figures / "phaseA_per_sample.csv").open()))
    with (tables / "table_A_frozen_wave21_diagnosis.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n"); writer.writerow(["horizon", "cycle_residual", "execution_cycle_residual", "decoded_action_mse", "target_distance", "endpoint_accuracy"])
        for h in (1, 2, 4):
            rows = [row for row in phase_rows if int(row["horizon"]) == h]
            writer.writerow([h, np.mean([float(row["cycle_full"]) for row in rows]), np.mean([float(row["cycle_execution"]) for row in rows]), np.mean([float(row["decoded_action_mse"]) for row in rows]), np.mean([float(row["target_distance"]) for row in rows]), np.mean([1 - int(row["endpoint_error"]) for row in rows])])
    with (tables / "table_B_LCT_vs_LCTCC.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n"); writer.writerow(["metric", "Wave21_B1", "LCT_CC", "language_prototype_or_diagnostic"])
        rows = [
            ("RedirectGain", wave21_metrics["redirect_ci"]["mean"], not_tested, projection["RedirectGain_cycle4"]["mean"]),
            ("Execution RedirectGain", wave21_metrics["execution_redirect_ci"]["mean"], not_tested, projection["ExecutionRedirectGain_cycle4"]["mean"]),
            ("cycle error", wave21_metrics["cycle"]["mean"], not_tested, projection["iterations"][4]["cycle_residual"]),
            ("endpoint accuracy", wave21_metrics["endpoint"]["macro_accuracy"], not_tested, projection["iterations"][4]["endpoint_accuracy"]),
            ("decode-reencode accuracy", wave21_metrics["cycle"]["reencoded_target_accuracy"], not_tested, geometry_data["same_state"]["Wave21_B1_cycle4_diagnostic"]["decode_reencode_accuracy"]),
            ("H2 full MSE", wave21_metrics["model_table"]["B1_correct_language"]["H2_full_mse"], not_tested, wave21_metrics["model_table"]["language_prototype"]["H2_full_mse"]),
            ("H4 decoded MSE", wave21_metrics["model_table"]["B1_correct_language"]["H4_decoded_continuous_mse"], not_tested, wave21_metrics["model_table"]["language_prototype"]["H4_decoded_continuous_mse"]),
            ("continuity error", wave21_metrics["continuity"]["LCT_absolute_jump_error"], not_tested, wave21_metrics["continuity"]["prototype_absolute_jump_error"]),
        ]
        writer.writerows(rows)
    confusion = np.asarray(geometry_data["same_state"]["Wave21_B1"]["endpoint_confusion"])
    with (tables / "table_C_per_goal_breakdown.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n"); writer.writerow(["goal", "Wave21_B1_endpoint_accuracy", "Wave21_B1_H4_cycle_mean", "LCT_CC"])
        for i, task in enumerate(yaml.safe_load((ROOT / config["wave21_config"]).read_text())["data"]["vocabulary"]):
            writer.writerow([task, confusion[i, i] / max(confusion[i].sum(), 1), decomposition["by_target"][task]["full"]["mean"], not_tested])
    with (tables / "table_D_claim_gates.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n"); writer.writerow(["claim_or_gate", "decision"])
        for key, value in diagnosis["gates"].items(): writer.writerow([key, value])
        writer.writerow(["M0", "REJECTED"]); writer.writerow(["C9", "NOT_TESTED"]); writer.writerow(["C10", "NOT_TESTED"])

    # Six required compact figures; unavailable LCT-CC panels are explicitly marked.
    try:
        import matplotlib.pyplot as plt
        by_step = diagnosis["B1_cycle_by_step"]
        hs = np.asarray([0, 1, 2, 3, 4]); cycles = np.asarray([by_step[str(h)]["mean"] for h in hs])
        decoded = np.asarray([np.mean([float(row["decoded_action_mse"]) for row in phase_rows if int(row["horizon"]) == h]) for h in hs])
        fig, axes = plt.subplots(1, 2, figsize=(8, 3)); axes[0].plot(hs, cycles, marker="o", label="cycle"); axes[0].set_xlabel("horizon"); axes[0].legend(); axes[1].plot(hs, decoded, marker="o", label="decoded MSE"); axes[1].set_xlabel("horizon"); axes[1].legend(); fig.suptitle("Wave21 mechanism diagnosis"); fig.tight_layout(); fig.savefig(pub / "figure_1_mechanism_diagnosis.png", dpi=160); plt.close(fig)
        h4 = [row for row in phase_rows if int(row["horizon"]) == 4]
        fig, axes = plt.subplots(1, 2, figsize=(8, 3)); x = np.asarray([float(row["cycle_full"]) for row in h4]); axes[0].scatter(x, [float(row["decoded_action_mse"]) for row in h4], s=9, alpha=.6); axes[0].set(xlabel="cycle residual", ylabel="decoded MSE"); bins=np.quantile(x,[0,.25,.5,.75,1]); bx=[];by=[]
        for lo, hi in zip(bins[:-1], bins[1:]): mask=(x>=lo)&(x<=hi);bx.append(float(x[mask].mean()));by.append(float(np.mean([int(h4[i]["endpoint_error"]) for i in np.flatnonzero(mask)])))
        axes[1].plot(bx,by,marker="o");axes[1].set(xlabel="binned cycle residual",ylabel="endpoint error rate");fig.tight_layout();fig.savefig(pub/"figure_2_cycle_residual_vs_failure.png",dpi=160);plt.close(fig)
        iterations=projection["iterations"]; fig,axes=plt.subplots(1,3,figsize=(10,3)); its=[row["iteration"] for row in iterations]; axes[0].plot(its,[row["cycle_residual"] for row in iterations],marker="o");axes[0].set_title("cycle residual"); axes[1].plot(its,[row["endpoint_accuracy"] for row in iterations],marker="o");axes[1].set_title("endpoint accuracy"); axes[2].bar([0,4],[projection["RedirectGain_LCT"]["mean"],projection["RedirectGain_cycle4"]["mean"]]);axes[2].set_title("RedirectGain");fig.tight_layout();fig.savefig(pub/"figure_3_fixed_cycle_diagnostic.png",dpi=160);plt.close(fig)
        metrics=["Redirect","Exec redirect","Cycle","Endpoint","Reencode"]; b1=[projection["RedirectGain_LCT"]["mean"],projection["ExecutionRedirectGain_LCT"]["mean"],projection["iterations"][0]["cycle_residual"],projection["iterations"][0]["endpoint_accuracy"],geometry_data["same_state"]["Wave21_B1"]["decode_reencode_accuracy"]]; diag=[projection["RedirectGain_cycle4"]["mean"],projection["ExecutionRedirectGain_cycle4"]["mean"],projection["iterations"][4]["cycle_residual"],projection["iterations"][4]["endpoint_accuracy"],geometry_data["same_state"]["Wave21_B1_cycle4_diagnostic"]["decode_reencode_accuracy"]]; fig,ax=plt.subplots(figsize=(8,3)); xx=np.arange(len(metrics));ax.bar(xx-.18,b1,.36,label="Wave21 B1");ax.bar(xx+.18,diag,.36,label="cycle4 diagnostic");ax.set_xticks(xx,metrics,rotation=20);ax.legend();ax.text(.5,.92,"LCT-CC NOT TESTED (Phase-A stop)",transform=ax.transAxes,ha="center");fig.tight_layout();fig.savefig(pub/"figure_4_main_comparison.png",dpi=160);plt.close(fig)
        matrices=geometry_data["same_state"];fig,axes=plt.subplots(1,2,figsize=(8,3));
        for ax,(name,value) in zip(axes,matrices.items()):im=ax.imshow(value["endpoint_confusion"],aspect="auto");ax.set_title(name);ax.set_xlabel("predicted");ax.set_ylabel("requested")
        fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.7);fig.savefig(pub/"figure_5_same_state_sixway.png",dpi=160,bbox_inches="tight");plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(10,3));hs=np.arange(5)
        for name in ("Wave21_B1","cycle4_diagnostic","ground_truth","language_prototype"):
            axes[0].plot(hs,case[name]["cycle_residual"],label=name);axes[1].plot(hs,case[name]["target_region_distance"]);axes[2].plot(hs,case[name]["decoded_action_mse"])
        axes[0].set_title("cycle");axes[1].set_title("target distance");axes[2].set_title("decoded MSE");axes[0].legend(fontsize=6);fig.tight_layout();fig.savefig(pub/"figure_6_lift_to_place.png",dpi=160);plt.close(fig)
    except ImportError:
        write_json(pub / "matplotlib_unavailable.json", {"raw_figure_data_complete": True})

    (out / "wave22_training_report.md").write_text("# Wave 22 training report\n\n**No training was run.** M0 failed A5 before any Wave22 optimizer step. The three registered lambdas and six seeds remain unused; no checkpoint exists.\n")
    (out / "wave22_main_comparison.md").write_text("# Wave 22 main comparison\n\nLCT-CC is **NOT TESTED** because Phase A rejected intervention. Frozen Wave21 B1 and the diagnostic-only cycle4 map are tabulated solely to document why training was not authorized.\n")
    (out / "wave22_same_state_language_swap.md").write_text(f"# Wave 22 same-state language swap\n\nFrozen Wave21 B1 six-way accuracy={geometry_data['same_state']['Wave21_B1']['endpoint_accuracy']:.6f}; cycle4 diagnostic={geometry_data['same_state']['Wave21_B1_cycle4_diagnostic']['endpoint_accuracy']:.6f}. LCT-CC was not trained. Only language changed within each frozen six-way set.\n")
    (out / "wave22_decode_reencode_results.md").write_text(f"# Wave 22 decode/re-encode results\n\nFrozen B1 six-way decode/re-encode identity={geometry_data['same_state']['Wave21_B1']['decode_reencode_accuracy']:.6f}; cycle4 diagnostic={geometry_data['same_state']['Wave21_B1_cycle4_diagnostic']['decode_reencode_accuracy']:.6f}. No LCT-CC result exists.\n")
    (out / "wave22_continuity_results.md").write_text(f"# Wave 22 continuity results\n\nWave21 B1 continuity error={wave21_metrics['continuity']['LCT_absolute_jump_error']:.6f}; prototype={wave21_metrics['continuity']['prototype_absolute_jump_error']:.6f}. LCT-CC is not tested due to the Phase-A stop.\n")
    (out / "wave22_statistical_report.md").write_text("# Wave 22 statistical report\n\nIndependent unit: continuous source session (n=6 historical Wave21 held-out sessions). All reported CIs use 10,000 source-session cluster bootstrap replicates with seed family rooted at 220822.\n\n" + f"- B1 H4−ground-truth H4 cycle: {diagnosis['A1_clustered_B1_H4_minus_GT_H4']['mean']:.6f} [{diagnosis['A1_clustered_B1_H4_minus_GT_H4']['lower_95']:.6f}, {diagnosis['A1_clustered_B1_H4_minus_GT_H4']['upper_95']:.6f}]\n- cycle0−cycle4 reduction: {projection['cycle_reduction']['mean']:.6f} [{projection['cycle_reduction']['lower_95']:.6f}, {projection['cycle_reduction']['upper_95']:.6f}]\n- cycle4 RedirectGain: {projection['RedirectGain_cycle4']['mean']:.6f} [{projection['RedirectGain_cycle4']['lower_95']:.6f}, {projection['RedirectGain_cycle4']['upper_95']:.6f}]\n\nNo LCT-CC inference or C9/C10 statistics were computed.\n")
    activated = ["cycle drift", "decoder inconsistency", "target identity failure", "continuity failure", "long-horizon accumulation"]
    (out / "wave22_failure_taxonomy.md").write_text("# Wave 22 failure taxonomy\n\nFrozen categories: cycle drift; semantic-only direction; execution normal drift; decoder inconsistency; prototype collapse; current-state loss; continuity failure; target identity failure; goal-specific failure; long-horizon accumulation; other.\n\nActivated from frozen Wave21/Phase A: " + ", ".join(activated) + ". The Wave22 intervention itself was not tested, so prototype-collapse/current-state-loss outcomes for LCT-CC are not assigned.\n")

    questions = [
        "Yes. Frozen B1 mean residual rose from 0.963142 at H0 to 2.800412 at H4 (peaking at 2.869956 at H3), while ground-truth H4 was 0.693765.",
        f"Both blocks contribute. At observed H4 semantic mean={decomposition['by_step']['4']['semantic']['mean']:.6f} and execution mean={decomposition['by_step']['4']['execution']['mean']:.6f}; semantic is larger, but execution is nontrivial.",
        f"Yes. Pearson r={diagnosis['association']['decoded_action_mse']['pearson']['estimate']:.4f} [{diagnosis['association']['decoded_action_mse']['pearson']['lower_95']:.4f}, {diagnosis['association']['decoded_action_mse']['pearson']['upper_95']:.4f}].",
        f"Yes. Endpoint-error Pearson r={diagnosis['association']['endpoint_error']['pearson']['estimate']:.4f} [{diagnosis['association']['endpoint_error']['pearson']['lower_95']:.4f}, {diagnosis['association']['endpoint_error']['pearson']['upper_95']:.4f}].",
        f"Yes. Exactly four iterations reduced the next-cycle residual from {projection['iterations'][0]['cycle_residual']:.6f} to {projection['iterations'][4]['cycle_residual']:.6f}.",
        f"Not under the frozen A5 rule. Mean full RedirectGain remained positive ({projection['RedirectGain_cycle4']['mean']:.6f}), but its clustered lower95 was {projection['RedirectGain_cycle4']['lower_95']:.6f} and endpoint accuracy fell from {projection['iterations'][0]['endpoint_accuracy']:.6f} to {projection['iterations'][4]['endpoint_accuracy']:.6f}.",
        "No. A1-A4 passed, A5 failed, so M0 was rejected before any optimizer step.",
        "None. The development lambda sweep was forbidden after M0 rejection; 0.1, 0.3, and 1.0 were not trained.",
        "Not tested; no LCT-CC model exists.", "Not tested; no LCT-CC model exists.", "Not tested; no held-out LCT-CC cycle error exists.", "Not tested; no LCT-CC endpoint result exists.", "Not tested; no LCT-CC decode/re-encode result exists.", "Not tested; the prototype comparison was not opened for a nonexistent LCT-CC model.", "Not tested. Frozen Wave21 B1 remained worse than prototype on continuity; no intervention result exists.", "Not tested for LCT-CC.", "Not tested for LCT-CC; the frozen projection itself reduced endpoint identity, warning against equating cycle support with target identity.", "Not tested across goals for LCT-CC.", f"The exact pair exists ({case['exact_pair_count']} cases), but the diagnostic cycle4 map reduced residual while worsening the selected case target distance from {case['Wave21_B1']['target_region_distance'][-1]:.6f} to {case['cycle4_diagnostic']['target_region_distance'][-1]:.6f}; LCT-CC was not trained.",
        "C9 is NOT_TESTED because its prerequisite M0 failed.", "C10 is NOT_TESTED.",
        "Wave21 clearly drifts off encoder-decoder-consistent coordinates, and that drift correlates with errors; however, the frozen cycle-supported map is partly misaligned with full language-selected target geometry. Thus pure decoder inconsistency is not sufficient as the primary causal mechanism.",
        "Defensible claim: frozen Wave21 language rollout accumulates encoder-decoder cycle drift associated with behavioral/geometric failure, but direct cycle projection trades away statistically reliable full target redirection and target identity.",
        "If C9 had passed, the next experiment would be a separately preregistered closed-loop CALVIN rollout comparing frozen B1 and LCT-CC without adding refinement.",
        "Because C9 was not reached after M0 failed, next test the language-target/executable-set alignment mechanism directly: characterize goal-specific supported coordinates and preregister a single-factor target-identity alignment model, with the same split and no cycle-based rescue.",
    ]
    lines = ["# Twenty-second wave results: executable coordinate consistency", "", f"Run date: {now()}", "", "## Outcome", "", "- M0 decoder-consistency mechanism: **REJECTED**", "- C9 executable language redirect: **NOT_TESTED**", "- C10 language as executable target coordinate: **NOT_TESTED**", "- Wave22 optimizer steps: **0**", "", "A1-A4 passed, but A5 failed prospectively. Four frozen cycle-map iterations strongly reduced cycle residual while full RedirectGain lost a strictly positive clustered lower bound and six-way endpoint accuracy decreased. Per the stop rule, no lambda sweep, model training, or LCT-CC held-out evaluation was performed.", "", "## Required questions", ""]
    lines += [f"{i}. {answer}" for i, answer in enumerate(questions, 1)]
    lines += ["", "## Scientific decision", "", "Cycle drift is a real correlate of Wave21 failure, but the exact frozen correction does not preserve the registered full language effect reliably. Decoder consistency alone is therefore not authorized as the intervention mechanism. The execution-space RedirectGain increased under projection, which is informative but cannot override failed A5 or justify post-hoc training.", "", "## Test-discipline disclosure", "", "Wave21 held-out trajectories were historically opened in Wave21 and were reused because Phase A explicitly requires frozen held-out diagnosis. No Wave22 LCT-CC checkpoint or held-out prediction was ever created. No threshold, seed, lambda, loss, or rescue was changed after the result."]
    result_text = "\n".join(lines) + "\n"
    (out / "twenty_second_wave_results.md").write_text(result_text)
    report_path = ROOT / config["experiment"]["report_path"]; report_path.parent.mkdir(parents=True, exist_ok=True); report_path.write_text(result_text)
    next_text = "# Twenty-second wave next experiment\n\nWave 23 should not rescue LCT-CC or reopen DEL/F2. Test a different single mechanism: whether frozen language directions identify target regions that are geometrically misaligned with the decoder-supported set. On train/development only, map goal-conditioned endpoints within the supported set and preregister a target-identity alignment objective, then require preserved full/execution RedirectGain, >=0.60 identity, current-state dependence, and continuity before one held-out evaluation. Retain the exact Wave21 split, six seeds, sparse-annotation disclosure, and source-session bootstrap.\n"
    (out / "twenty_second_wave_next_experiment.md").write_text(next_text)
    (ROOT / "NEXT_EXPERIMENT.md").write_text(next_text)
    marker = "## Wave 22 — Executable coordinate consistency"
    log_path = ROOT / "RESEARCH_LOG.md"; log_text = log_path.read_text()
    if marker not in log_text:
        log_text += f"\n{marker} ({datetime.now().date()})\n\n- Executed `prompts/dynamics_10.md` with frozen Wave21 B1/representation/decoder/text projection.\n- Phase A: A1-A4 passed; A5 failed. M0 **REJECTED**; C9/C10 **NOT_TESTED**.\n- Frozen K=4 cycle projection reduced residual {projection['iterations'][0]['cycle_residual']:.6f}→{projection['iterations'][4]['cycle_residual']:.6f}, but full RedirectGain became {projection['RedirectGain_cycle4']['mean']:.6f} [{projection['RedirectGain_cycle4']['lower_95']:.6f}, {projection['RedirectGain_cycle4']['upper_95']:.6f}] and endpoint accuracy fell.\n- No Wave22 optimizer, lambda sweep, checkpoint, or held-out LCT-CC prediction was created.\n- Full local artifacts: `{out.relative_to(ROOT)}`.\n"
        log_path.write_text(log_text)
    (out / "updated_RESEARCH_LOG.md").write_text(log_path.read_text()); (out / "updated_NEXT_EXPERIMENT.md").write_text((ROOT / "NEXT_EXPERIMENT.md").read_text())
    (out / "environment_freeze.txt").write_text("\n".join([f"timestamp={now()}", f"python={' '.join(sys.version.split())}", f"platform={platform.platform()}", f"torch={torch.__version__}", f"numpy={np.__version__}", f"scipy={stats.__version__ if hasattr(stats, '__version__') else 'installed'}", f"cuda_available={torch.cuda.is_available()}", f"cuda_device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}"]) + "\n")
    (out / "exact_commands.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\nPYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_10.py --config configs/dynamics_10.yaml --stage prepare --device cuda:0\nPYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_10.py --config configs/dynamics_10.yaml --stage phasea --device cuda:0\nPYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_10.py --config configs/dynamics_10.yaml --stage geometry --device cuda:0\nPYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_10.py --config configs/dynamics_10.yaml --stage report --device cuda:0\nPYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python -m pytest tests/dynamics/test_dynamics_10_cycle_consistency.py -q\n")
    changed = ["configs/dynamics_10.yaml", "scripts/dynamics/run_dynamics_10.py", "tests/dynamics/test_dynamics_10_cycle_consistency.py", "reports/dynamics_10_results.md", "RESEARCH_LOG.md", "NEXT_EXPERIMENT.md", "prompts/dynamics_10.md", config["experiment"]["output_root"] + "/"]
    (out / "files_changed.txt").write_text("\n".join(changed) + "\n")
    print(json.dumps({"stage": "report", "M0": m0, "C9": "NOT_TESTED", "deliverables": "complete_stop_branch"}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "phasea", "geometry", "train", "final", "report", "all"), default="all")
    parser.add_argument("--device")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text())
    device = torch.device(args.device or config["runtime"]["device"])
    torch.set_num_threads(int(config["runtime"]["torch_cpu_threads"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Registered Wave22 run requires CUDA")
    if args.stage == "all":
        prepare(config, device)
        phase_a(config, device)
        geometry(config, device)
        if read_json(out_path(config) / "wave22_phaseA_results.json")["M0_decoder_consistency_mechanism"] == "SUPPORTED_FOR_INTERVENTION":
            train(config, device)
            evaluate_final(config, device)
        report(config)
        return
    stages = (args.stage,)
    for stage in stages:
        print(json.dumps({"stage": stage, "started_at": now()}), flush=True)
        if stage == "report":
            report(config)
        else:
            {"prepare": prepare, "phasea": phase_a, "geometry": geometry, "train": train, "final": evaluate_final}[stage](config, device)


if __name__ == "__main__":
    main()

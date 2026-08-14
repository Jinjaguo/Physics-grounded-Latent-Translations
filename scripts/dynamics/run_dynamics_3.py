#!/usr/bin/env python3
"""Run the complete third dynamics experiment (PGLT wave 15).

Purpose
-------
Test factorized dynamics on the frozen 16-D executable subspace: train one
shared semantic predictor, F1 execution MLP, F2 matched generic refinement,
F3 free execution-only DEL, and F4 decoder-geometry execution DEL; perform the
development compatibility gate and exactly one frozen official validation.

Parameters
----------
``--config`` selects the preregistered YAML, ``--stage`` is one of prepare,
validate_metric, train, development, preflight, freeze, validation, finalize,
or all, and ``--device`` is a PyTorch device string (default ``cpu``).

Usage
-----
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/run_dynamics_3.py \
  --config configs/dynamics_3.yaml --stage all --device cpu

Outputs
-------
All checkpoints, audits, metrics, gates, manifests, reports, and exact command
records are saved under ``results/dynamics/fifteenth_wave/...``.  The final
report is mirrored to ``reports/dynamics_3_results.md`` and the repository
``RESEARCH_LOG.md`` / ``NEXT_EXPERIMENT.md`` handoff files are updated.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from pglt.dynamics.dynamics_data import (
    DynamicsSequence,
    horizon_starts,
    load_frozen_representation,
    sha256_file,
    transition_records,
    write_json,
)
from pglt.dynamics.factorized import (
    DecoderGeometryDEL,
    ExecutionMLP,
    ExecutionMatchedRefinement,
    FreeExecutionDEL,
    SemanticPredictor,
    factorized_model_spec,
)
from pglt.dynamics.runner import knn_distances, load_sequences, off_manifold_threshold


ROOT = Path(__file__).resolve().parents[2]
MODEL_ORDER = ("F1_execution_mlp", "F2_matched_refinement", "F3_free_execution_del", "F4_decoder_geometry_del")


@dataclass
class FactorBatch:
    """Aligned frozen transition tensors without any future action input."""

    s_previous: torch.Tensor
    s_current: torch.Tensor
    s_target: torch.Tensor
    e_previous: torch.Tensor
    e_current: torch.Tensor
    e_target: torch.Tensor
    context: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--stage", required=True,
        choices=("prepare", "validate_metric", "train", "development", "preflight", "freeze", "validation", "finalize", "all"),
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def out_root(config: Mapping[str, Any]) -> Path:
    return ROOT / config["experiment"]["output_root"]


def disk_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    """Enforce the user-requested 20 GiB free-space floor."""

    usage = shutil.disk_usage(ROOT)
    required = int(config["storage"]["minimum_filesystem_available_gb"]) * 1024 ** 3
    if usage.free < required:
        raise RuntimeError(f"Only {usage.free} free bytes; required floor is {required}")
    return {
        "filesystem_total_bytes": usage.total,
        "filesystem_used_bytes": usage.used,
        "filesystem_available_bytes": usage.free,
        "minimum_available_bytes": required,
        "passed": True,
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_representation(config: Mapping[str, Any]) -> tuple[nn.Module, dict[str, Any], Path, dict[str, Any]]:
    manifest = read_json(ROOT / config["representation"]["checkpoint_manifest"])
    entry = next(item for item in manifest["checkpoints"] if item["condition"] == config["representation"]["expected_condition"])
    if int(entry["seed_base"]) != int(config["representation"]["expected_seed_base"]):
        raise RuntimeError("Frozen representation manifest selection changed")
    path = ROOT / entry["path"]
    if sha256_file(path) != entry["sha256"]:
        raise RuntimeError("Frozen representation checkpoint hash mismatch")
    model, payload = load_frozen_representation(read_yaml(ROOT / config["representation"]["config"]), path)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload, path, entry


def tensor_hashes(model: nn.Module) -> dict[str, str]:
    return {
        name: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
        for name, value in model.state_dict().items()
    }


def arrays(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    with np.load(ROOT / config["data"]["frozen_latents"], allow_pickle=False) as saved:
        result = {key: saved[key].copy() for key in saved.files}
    if result["latents"].shape[1] != 32:
        raise RuntimeError("Expected exact frozen 32-D latent")
    if not np.array_equal(result["latents"][:, :16], result["semantic_latents"]):
        raise RuntimeError("Semantic slice is not the exact frozen prefix")
    if not np.array_equal(result["latents"][:, 16:], result["execution_latents"]):
        raise RuntimeError("Execution slice is not the exact frozen suffix")
    return result


def sequences(config: Mapping[str, Any], split: str) -> list[DynamicsSequence]:
    return load_sequences(ROOT / config["data"][f"{split}_sequences"])


def make_batch(
    saved: Mapping[str, np.ndarray], seqs: Sequence[DynamicsSequence], indices: Sequence[int], device: torch.device
) -> FactorBatch:
    records = transition_records(seqs)
    selected = [records[index] for index in indices]
    previous = np.asarray([item.previous_index for item in selected], dtype=np.int64)
    current = np.asarray([item.current_index for item in selected], dtype=np.int64)
    target = np.asarray([item.target_index for item in selected], dtype=np.int64)
    take = lambda name, ids: torch.from_numpy(saved[name][ids]).float().to(device)
    return FactorBatch(
        take("semantic_latents", previous), take("semantic_latents", current), take("semantic_latents", target),
        take("execution_latents", previous), take("execution_latents", current), take("execution_latents", target),
        take("contexts", current),
    )


def make_initial_models(config: Mapping[str, Any]) -> dict[str, nn.Module]:
    values = config["models"]
    common = {"context_dim": int(values["context_dim"]), "hidden_dim": int(values["hidden_dim"]), "depth": int(values["depth"])}
    semantic = SemanticPredictor(**common)
    f1 = ExecutionMLP(context_dim=32, hidden_dim=int(values["hidden_dim"]), depth=int(values["depth"]))
    f3 = FreeExecutionDEL(
        context_dim=32, mass_hidden_dim=int(values["free_mass_hidden_dim"]),
        potential_hidden_dim=int(values["potential_hidden_dim"]), depth=int(values["depth"]),
        solver_iterations=int(values["refinement_iterations"]),
        solver_step_size=float(values["free_del_step_size"]),
        solver_tolerance=float(values["solver_tolerance"]),
        mass_epsilon=float(values["metric_epsilon"]),
    )
    return {"semantic": semantic, "F1_execution_mlp": f1, "F3_free_execution_del": f3}


def attach_refinement_models(
    config: Mapping[str, Any], initial: dict[str, nn.Module], representation: nn.Module
) -> dict[str, nn.Module]:
    values = config["models"]
    f1 = initial["F1_execution_mlp"]
    initial["F2_matched_refinement"] = ExecutionMatchedRefinement(
        f1, context_dim=32, hidden_dim=int(values["hidden_dim"]), depth=int(values["depth"]),
        iterations=int(values["refinement_iterations"]), step_size=float(values["generic_refinement_step_size"]),
    )
    initial["F4_decoder_geometry_del"] = DecoderGeometryDEL(
        f1, deepcopy(representation.decoder), context_dim=int(values["context_dim"]),
        potential_hidden_dim=int(values["potential_hidden_dim"]), depth=int(values["depth"]),
        iterations=int(values["refinement_iterations"]), step_size=float(values["geometry_del_step_size"]),
        tolerance=float(values["solver_tolerance"]), metric_epsilon=float(values["metric_epsilon"]),
    )
    return initial


def forward_execution(
    name: str, model: nn.Module, batch: FactorBatch, physical_step: float
) -> tuple[torch.Tensor, dict[str, Any]]:
    combined = torch.cat((batch.s_current, batch.context), dim=-1)
    if name == "F1_execution_mlp":
        return model(batch.e_previous, batch.e_current, combined), {}
    if name == "F2_matched_refinement":
        prediction, info = model(batch.e_previous, batch.e_current, combined)
        return prediction, {"refinement_info": info}
    if name == "F3_free_execution_del":
        prediction, info = model(batch.e_previous, batch.e_current, combined, physical_step)
        return prediction, {"del_info": info}
    if name == "F4_decoder_geometry_del":
        prediction, info = model(
            batch.e_previous, batch.e_current, batch.s_current, batch.context, physical_step
        )
        return prediction, {"del_info": info}
    raise KeyError(name)


def checkpoint_payload(path: Path, model: nn.Module, name: str, epoch: int, score: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_name": name, "model_state_dict": model.state_dict(),
        "best_epoch": epoch, "development_normalized_rollout_auc": score,
    }, path)


def semantic_development_mse(
    model: nn.Module, saved: Mapping[str, np.ndarray], seqs: Sequence[DynamicsSequence], device: torch.device
) -> float:
    batch = make_batch(saved, seqs, range(len(transition_records(seqs))), device)
    model.eval()
    with torch.no_grad():
        prediction = model(batch.s_previous, batch.s_current, batch.context)
    return float(F.mse_loss(prediction, batch.s_target).cpu())


def execution_selection_score(
    name: str, model: nn.Module, saved: Mapping[str, np.ndarray], seqs: Sequence[DynamicsSequence],
    semantic_model: nn.Module, variance: float, physical_step: float, device: torch.device,
) -> tuple[float, dict[str, Any]]:
    points = []
    details = {}
    model.eval()
    for horizon in (1, 2):
        starts = horizon_starts(seqs, horizon)
        predictions = []
        targets = []
        for sequence_index, offset in starts:
            ids = seqs[sequence_index].latent_indices
            sp = torch.from_numpy(saved["semantic_latents"][ids[offset]:ids[offset] + 1]).float().to(device)
            sc = torch.from_numpy(saved["semantic_latents"][ids[offset + 1]:ids[offset + 1] + 1]).float().to(device)
            ep = torch.from_numpy(saved["execution_latents"][ids[offset]:ids[offset] + 1]).float().to(device)
            ec = torch.from_numpy(saved["execution_latents"][ids[offset + 1]:ids[offset + 1] + 1]).float().to(device)
            context = torch.from_numpy(saved["contexts"][ids[offset + 1]:ids[offset + 1] + 1]).float().to(device)
            for step in range(horizon):
                target_id = ids[offset + 2 + step]
                batch = FactorBatch(sp, sc, sc, ep, ec, torch.from_numpy(saved["execution_latents"][target_id:target_id + 1]).float().to(device), context)
                with torch.enable_grad():
                    following, _ = forward_execution(name, model, batch, physical_step)
                with torch.no_grad():
                    semantic_following = semantic_model(sp, sc, context)
                ep, ec = ec.detach(), following.detach()
                sp, sc = sc.detach(), semantic_following.detach()
            predictions.append(ec.cpu())
            targets.append(torch.from_numpy(saved["execution_latents"][ids[offset + 1 + horizon]:ids[offset + 1 + horizon] + 1]))
        mse = float(F.mse_loss(torch.cat(predictions), torch.cat(targets)).item())
        details[str(horizon)] = {"samples": len(starts), "execution_mse": mse}
        points.append((horizon, mse / variance))
    score = float(np.trapz([value for _, value in points], [h for h, _ in points]))
    return score, details


def prepare(config: Mapping[str, Any]) -> None:
    """Write the wave-15 preregistration and immutable-source audits pre-training."""

    out = out_root(config)
    out.mkdir(parents=True, exist_ok=True)
    storage = disk_audit(config)
    representation, _, checkpoint, entry = select_representation(config)
    saved = arrays(config)
    split_path = ROOT / config["data"]["split_preregistration"]
    split = read_json(split_path)
    source_files = {
        "frozen_latents": ROOT / config["data"]["frozen_latents"],
        "train_sequences": ROOT / config["data"]["train_sequences"],
        "development_sequences": ROOT / config["data"]["development_sequences"],
        "validation_sequences": ROOT / config["data"]["validation_sequences"],
        "split_preregistration": split_path,
    }
    historical_files = list((ROOT / config["experiment"]["wave13_root"] / "checkpoints").glob("*.pt")) + [
        ROOT / config["experiment"]["wave14_root"] / "final_del_failure_mechanism.json",
        ROOT / config["experiment"]["wave14_root"] / "fourteenth_wave_results.md",
    ]
    preregistration = {
        "created_at": now(),
        "written_before_wave15_training": True,
        "hypothesis": "decoder-grounded variational dynamics is useful only on frozen z_exec",
        "immutable_facts": {
            "historical_gate_A": "FAIL", "prospective_R_gate": "PASS",
            "representation_ready": True, "full_latent_unforced_DEL": "not_supported",
            "full_latent_forced_DEL": "not_supported", "full_latent_variational_model_mismatch": "supported",
        },
        "latent_slicing": {"total": 32, "z_sem": [0, 16], "z_exec": [16, 32]},
        "shared_semantic_predictor": True,
        "primary_models": list(MODEL_ORDER),
        "equal_information_fields": ["e_previous", "e_current", "s_current", "context"],
        "future_target_actions": False,
        "decoder_metric": {
            "method": "JVP pullback", "W_a": "identity over all 112 normalized continuous decoder outputs including gripper logits",
            "gripper_threshold_differentiated": False, "epsilon": config["models"]["metric_epsilon"],
            "interpretation": "decoder-induced executable geometry, not physical inertia",
        },
        "matched_fairness": {
            "same_F1_initializer": True,
            "F2_iterations": config["models"]["refinement_iterations"],
            "F4_iterations": config["models"]["refinement_iterations"],
            "line_search_evaluations": 0,
        },
        "optimization": config["optimization"],
        "hard_gate": [
            "F4 rollout AUC < F1", "F4 rollout AUC < F2",
            "F4 one-step decoded continuous MSE <= 1.05 * F1",
            "F4 two-step execution kNN radius < F1", "true-next F4 residual mean < F1 residual mean",
            "Spearman(F4 residual, execution squared error) > 0", "F4 nonfinite rate == 0",
            "F4 convergence rate > historical full-latent rate 0",
        ],
        "development_only_selection": True,
        "official_validation_one_shot_after_manifest": True,
        "config": config,
    }
    write_json(out / "factorized_dynamics_preregistration.json", preregistration)
    write_json(out / "frozen_representation_model_audit.json", {
        "created_at": now(), "checkpoint": entry, "checkpoint_path": checkpoint.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint), "all_parameters_require_grad_false": all(not p.requires_grad for p in representation.parameters()),
        "representation_optimizer_steps": 0, "representation_backward_calls": 0, "ema_updates": 0,
        "initial_state_tensor_hashes": tensor_hashes(representation),
        "source_artifact_sha256": {name: sha256_file(path) for name, path in source_files.items()},
        "historical_negative_artifact_sha256": {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in historical_files},
        "latent_shape": list(saved["latents"].shape), "exact_16_16_slicing": True,
    })
    write_json(out / "exact_dynamics_split_audit.json", {
        "reused_wave13_split_without_resampling": True,
        "source": split_path.relative_to(ROOT).as_posix(), "source_sha256": sha256_file(split_path),
        "train_episode_rows": split["train_episode_rows"], "development_episode_rows": split["development_episode_rows"],
        "reserved_validation_rows": split["reserved_official_validation_rows"],
        "train_transitions": split["train"]["transitions"], "development_transitions": split["development"]["transitions"],
        "validation_transitions_pre_metric": split["reserved_validation_pre_metric_counts"]["transitions"],
    })
    write_json(out / "environment_provenance.json", {
        "created_at": now(), "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(), "storage_at_prepare": storage,
    })
    commands = """PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_3.py --config configs/dynamics_3.yaml --stage prepare --device cpu
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/jinjaguo/anaconda3/envs/libero/bin/python -m pytest tests/representation tests/dynamics -q --junitxml=results/dynamics/fifteenth_wave/2026-08-12_dynamics_3/pytest_pretraining.xml
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_3.py --config configs/dynamics_3.yaml --stage validate_metric --device cpu
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_3.py --config configs/dynamics_3.yaml --stage train --device cpu
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_3.py --config configs/dynamics_3.yaml --stage development --device cpu
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_3.py --config configs/dynamics_3.yaml --stage preflight --device cpu
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_3.py --config configs/dynamics_3.yaml --stage freeze --device cpu
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_3.py --config configs/dynamics_3.yaml --stage validation --device cpu
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/jinjaguo/anaconda3/envs/libero/bin/python -m pytest tests/representation tests/dynamics -q --junitxml=results/dynamics/fifteenth_wave/2026-08-12_dynamics_3/pytest_results.xml
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_3.py --config configs/dynamics_3.yaml --stage finalize --device cpu
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/audit_dynamics_3.py --config configs/dynamics_3.yaml
df -BG .
"""
    (out / "executed_commands.txt").write_text(commands, encoding="utf-8")
    print(json.dumps({"stage": "prepare", "output": str(out), "available_gib": storage["filesystem_available_bytes"] / 1024 ** 3}))


def validate_metric(config: Mapping[str, Any], device: torch.device) -> None:
    """Validate decoder JVP, pullback PSD, differentiability, and freezing."""

    out = out_root(config)
    if not (out / "factorized_dynamics_preregistration.json").is_file():
        raise RuntimeError("Prepare must precede metric validation")
    disk_audit(config)
    set_seed(int(config["experiment"]["seed"]))
    representation, _, _, _ = select_representation(config)
    representation.to(device)
    initial_hashes = tensor_hashes(representation)
    models = attach_refinement_models(config, make_initial_models(config), representation)
    f4 = models["F4_decoder_geometry_del"].to(device)
    saved = arrays(config)
    semantic = torch.from_numpy(saved["semantic_latents"][:4]).float().to(device)
    execution = torch.from_numpy(saved["execution_latents"][:4]).float().to(device).requires_grad_(True)
    tangent = torch.randn_like(execution)
    jvp = f4.lagrangian.decoder_jvp(semantic, execution, tangent)
    delta = 1e-3
    with torch.no_grad():
        plus = representation.decode(torch.cat((semantic, execution + delta * tangent), dim=-1)).flatten(1)
        minus = representation.decode(torch.cat((semantic, execution - delta * tangent), dim=-1)).flatten(1)
        finite_difference = (plus - minus) / (2 * delta)
    relative_error = float((jvp.detach() - finite_difference).norm().cpu() / finite_difference.norm().clamp_min(1e-12).cpu())
    quadratic = f4.lagrangian.metric_quadratic(semantic, execution, tangent)
    batch = FactorBatch(semantic[:2], semantic[1:3], semantic[2:4], execution[:2], execution[1:3], execution[2:4], torch.from_numpy(saved["contexts"][1:3]).float().to(device))
    prediction, extras = forward_execution("F4_decoder_geometry_del", f4, batch, 16 / 30)
    loss = F.mse_loss(prediction, batch.e_target) + 1e-4 * extras["del_info"].residual_norm.square().mean()
    loss.backward()
    decoder_grad_count = sum(p.grad is not None for p in f4.lagrangian.decoder.parameters())
    potential_grads_finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in f4.lagrangian.potential_network.parameters())
    report = {
        "created_at": now(), "automatic_differentiation": "torch.autograd.functional.jvp",
        "deterministic_sample_indices": [0, 1, 2, 3], "finite_difference_delta": delta,
        "jacobian_jvp_relative_error": relative_error, "jacobian_correct_within_0.01": relative_error < 0.01,
        "metric_quadratic_values": [float(value) for value in quadratic.detach().cpu()],
        "metric_psd_tolerance": -1e-7, "metric_psd_passed": bool(torch.all(quadratic >= -1e-7)),
        "gripper_metric_input": "continuous decoder logit included; threshold excluded",
        "decoder_parameter_gradients_non_none": decoder_grad_count,
        "decoder_gradients_zero": decoder_grad_count == 0,
        "potential_gradients_finite": bool(potential_grads_finite), "prediction_finite": bool(torch.isfinite(prediction).all()),
        "representation_state_unchanged": initial_hashes == tensor_hashes(representation),
    }
    report["all_passed"] = all((report["jacobian_correct_within_0.01"], report["metric_psd_passed"], report["decoder_gradients_zero"], report["potential_gradients_finite"], report["prediction_finite"], report["representation_state_unchanged"]))
    write_json(out / "decoder_jacobian_metric_validation_report.json", report)
    if not report["all_passed"]:
        raise RuntimeError(f"Decoder metric validation failed: {report}")
    preregistration = read_json(out / "factorized_dynamics_preregistration.json")
    preregistration["pretraining_numerical_amendment"] = {
        "written_before_training_or_development_metrics": True,
        "reason": "initial unpreconditioned decoder-metric residual update produced infinite residual loss in the formal metric preflight",
        "change": "use a frozen per-sample decoder-metric Rayleigh quotient as scalar JVP preconditioner; retain all preregistered epochs, iteration counts, step sizes, epsilon, objectives, and gates",
        "validation_after_change": "decoder_jacobian_metric_validation_report.json all_passed=true",
    }
    write_json(out / "factorized_dynamics_preregistration.json", preregistration)
    print(json.dumps({"stage": "validate_metric", "all_passed": True, "jvp_relative_error": relative_error}))


def fit_semantic(
    config: Mapping[str, Any], model: nn.Module, saved: Mapping[str, np.ndarray], train: Sequence[DynamicsSequence],
    development: Sequence[DynamicsSequence], path: Path, device: torch.device,
) -> dict[str, Any]:
    opt_cfg = config["optimization"]
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(opt_cfg["learning_rate"]), weight_decay=float(opt_cfg["weight_decay"]))
    records = transition_records(train)
    best = math.inf
    best_state = None
    best_epoch = 0
    log = []
    for epoch in range(1, int(opt_cfg["epochs"]) + 1):
        model.train()
        order = np.random.default_rng(int(config["experiment"]["seed"]) + epoch).permutation(len(records))
        losses = []
        for offset in range(0, len(order), int(opt_cfg["batch_size"])):
            batch = make_batch(saved, train, order[offset:offset + int(opt_cfg["batch_size"])].tolist(), device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(batch.s_previous, batch.s_current, batch.context), batch.s_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(opt_cfg["gradient_clip_norm"]))
            optimizer.step()
            losses.append(float(loss.detach()))
        item = {"epoch": epoch, "training_semantic_mse": float(np.mean(losses))}
        if epoch % int(opt_cfg["evaluation_interval"]) == 0 or epoch == int(opt_cfg["epochs"]):
            value = semantic_development_mse(model, saved, development, device)
            item["development_semantic_mse"] = value
            if value < best:
                best, best_epoch, best_state = value, epoch, deepcopy(model.state_dict())
        log.append(item)
    if best_state is None:
        raise RuntimeError("No semantic checkpoint selected")
    model.load_state_dict(best_state)
    checkpoint_payload(path, model, "semantic", best_epoch, best)
    return {"best_epoch": best_epoch, "development_semantic_mse": best, "training_log": log}


def fit_execution(
    config: Mapping[str, Any], name: str, model: nn.Module, semantic_model: nn.Module,
    saved: Mapping[str, np.ndarray], train: Sequence[DynamicsSequence], development: Sequence[DynamicsSequence],
    execution_variance: float, path: Path, device: torch.device,
) -> dict[str, Any]:
    opt_cfg = config["optimization"]
    physical_step = int(config["data"]["chunk_length"]) / float(config["data"]["control_frequency_hz"])
    model.to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(opt_cfg["learning_rate"]), weight_decay=float(opt_cfg["weight_decay"]))
    records = transition_records(train)
    best = math.inf
    best_state = None
    best_epoch = 0
    log = []
    started = time.perf_counter()
    for epoch in range(1, int(opt_cfg["epochs"]) + 1):
        model.train()
        order = np.random.default_rng(int(config["experiment"]["seed"]) + epoch).permutation(len(records))
        losses = []
        prediction_losses = []
        residual_losses = []
        for offset in range(0, len(order), int(opt_cfg["batch_size"])):
            batch = make_batch(saved, train, order[offset:offset + int(opt_cfg["batch_size"])].tolist(), device)
            optimizer.zero_grad(set_to_none=True)
            with torch.enable_grad():
                prediction, extras = forward_execution(name, model, batch, physical_step)
                prediction_loss = F.mse_loss(prediction, batch.e_target)
                residual_loss = prediction.new_zeros(())
                if "del_info" in extras:
                    residual_loss = extras["del_info"].residual_norm.square().mean()
                loss = prediction_loss + float(opt_cfg["lambda_del"]) * residual_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {name} loss at epoch {epoch}")
            loss.backward()
            gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
            if not all(torch.isfinite(gradient).all() for gradient in gradients):
                raise FloatingPointError(f"Nonfinite {name} gradient at epoch {epoch}")
            torch.nn.utils.clip_grad_norm_(parameters, float(opt_cfg["gradient_clip_norm"]))
            optimizer.step()
            losses.append(float(loss.detach()))
            prediction_losses.append(float(prediction_loss.detach()))
            residual_losses.append(float(residual_loss.detach()))
        item = {
            "epoch": epoch, "training_loss": float(np.mean(losses)),
            "training_prediction_mse": float(np.mean(prediction_losses)),
            "training_del_residual_loss": float(np.mean(residual_losses)),
        }
        if epoch % int(opt_cfg["evaluation_interval"]) == 0 or epoch == int(opt_cfg["epochs"]):
            score, rollout = execution_selection_score(
                name, model, saved, development, semantic_model, execution_variance, physical_step, device
            )
            item["development_normalized_execution_rollout_auc"] = score
            item["development_rollout"] = rollout
            if score < best:
                best, best_epoch, best_state = score, epoch, deepcopy(model.state_dict())
        log.append(item)
    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {name}")
    model.load_state_dict(best_state)
    checkpoint_payload(path, model, name, best_epoch, best)
    return {
        "best_epoch": best_epoch, "development_normalized_execution_rollout_auc": best,
        "training_wall_clock_seconds": time.perf_counter() - started, "training_log": log,
    }


def train(config: Mapping[str, Any], device: torch.device) -> None:
    """Train shared semantic and all four execution models using train/dev only."""

    out = out_root(config)
    metric_report = read_json(out / "decoder_jacobian_metric_validation_report.json")
    if not metric_report["all_passed"]:
        raise RuntimeError("Metric validation must pass before training")
    disk_audit(config)
    set_seed(int(config["experiment"]["seed"]))
    saved = arrays(config)
    train_sequences = sequences(config, "train")
    development_sequences = sequences(config, "development")
    train_ids = np.asarray([index for sequence in train_sequences for index in sequence.latent_indices], dtype=np.int64)
    execution_variance = float(np.var(saved["execution_latents"][train_ids], axis=0).mean())
    representation, _, _, _ = select_representation(config)
    representation.to(device)
    representation_hashes = tensor_hashes(representation)
    models = make_initial_models(config)
    checkpoint_dir = out / "checkpoints"
    summaries = {
        "semantic": fit_semantic(config, models["semantic"], saved, train_sequences, development_sequences, checkpoint_dir / "semantic.pt", device)
    }
    summaries["F1_execution_mlp"] = fit_execution(
        config, "F1_execution_mlp", models["F1_execution_mlp"], models["semantic"], saved,
        train_sequences, development_sequences, execution_variance, checkpoint_dir / "F1_execution_mlp.pt", device,
    )
    models = attach_refinement_models(config, models, representation)
    for name in ("F2_matched_refinement", "F3_free_execution_del", "F4_decoder_geometry_del"):
        summaries[name] = fit_execution(
            config, name, models[name], models["semantic"], saved, train_sequences,
            development_sequences, execution_variance, checkpoint_dir / f"{name}.pt", device,
        )
    specs = {name: factorized_model_spec(model) for name, model in models.items()}
    f1_state = models["F1_execution_mlp"].state_dict()
    f2_init_state = models["F2_matched_refinement"].initializer.state_dict()
    f4_init_state = models["F4_decoder_geometry_del"].initializer.state_dict()
    init_equal = all(torch.equal(f1_state[key].cpu(), f2_init_state[key].cpu()) and torch.equal(f1_state[key].cpu(), f4_init_state[key].cpu()) for key in f1_state)
    semantic_checkpoint_hash = sha256_file(checkpoint_dir / "semantic.pt")
    write_json(out / "parameter_count_table.json", specs)
    write_json(out / "semantic_predictor_specification_results.json", {
        "specification": specs["semantic"], "shared_by_models": list(MODEL_ORDER),
        "single_checkpoint_sha256": semantic_checkpoint_hash, "standalone_results": summaries["semantic"],
        "retrained_per_execution_model": False,
    })
    for name in MODEL_ORDER:
        write_json(out / f"{name}_specification_training_results.json", {"specification": specs[name], "training_selection": summaries[name]})
    write_json(out / "training_and_development_selection.json", {
        "created_at": now(), "execution_training_variance_mean": execution_variance,
        "development_only_selection": True, "official_validation_metrics_read": False,
        "models": summaries,
    })
    write_json(out / "information_and_fairness_audit.json", {
        "all_execution_information_fields": {name: specs[name]["information_fields"] for name in MODEL_ORDER},
        "future_target_actions_accessed": False,
        "shared_semantic_predictor_checkpoint": semantic_checkpoint_hash,
        "shared_semantic_predictor_object_used_by_all_evaluations": True,
        "F2_F4_initializer_state_exactly_equal_to_F1": init_equal,
        "F2_iterations": models["F2_matched_refinement"].iterations,
        "F4_iterations": models["F4_decoder_geometry_del"].iterations,
        "identical_iteration_counts": models["F2_matched_refinement"].iterations == models["F4_decoder_geometry_del"].iterations,
        "line_search_evaluations_each": 0,
    })
    if not init_equal:
        raise RuntimeError("F2/F4 initializers differ from the selected F1 checkpoint")
    if representation_hashes != tensor_hashes(representation):
        raise RuntimeError("Frozen representation changed during wave-15 training")
    write_json(out / "representation_post_training_audit.json", {
        "state_unchanged": True, "representation_optimizer_steps": 0,
        "representation_backward_calls": 0, "ema_updates": 0,
        "decoder_parameter_gradients_non_none": sum(p.grad is not None for p in representation.parameters()),
        "state_tensor_hashes": tensor_hashes(representation),
    })
    disk_audit(config)
    print(json.dumps({"stage": "train", "models": list(summaries), "initializer_fairness": init_equal}))


def load_models(config: Mapping[str, Any], device: torch.device) -> tuple[dict[str, nn.Module], nn.Module, dict[str, Any]]:
    out = out_root(config)
    representation, payload, _, _ = select_representation(config)
    representation.to(device)
    models = make_initial_models(config)
    for name in ("semantic", "F1_execution_mlp", "F3_free_execution_del"):
        saved = torch.load(out / "checkpoints" / f"{name}.pt", map_location=device, weights_only=False)
        models[name].load_state_dict(saved["model_state_dict"])
    models = attach_refinement_models(config, models, representation)
    for name in ("F2_matched_refinement", "F4_decoder_geometry_del"):
        saved = torch.load(out / "checkpoints" / f"{name}.pt", map_location=device, weights_only=False)
        models[name].load_state_dict(saved["model_state_dict"])
    return {name: model.to(device).eval() for name, model in models.items()}, representation.eval(), payload


def task_prototypes(saved: Mapping[str, np.ndarray], train_ids: np.ndarray) -> tuple[list[str], np.ndarray]:
    tasks = sorted(set(str(value) for value in saved["task"][train_ids]))
    vectors = []
    for task in tasks:
        vector = saved["contexts"][train_ids][saved["task"][train_ids] == task].mean(axis=0)
        vectors.append(vector / max(float(np.linalg.norm(vector)), 1e-12))
    return tasks, np.asarray(vectors, dtype=np.float32)


def predict_rollout(
    name: str, models: Mapping[str, nn.Module], saved: Mapping[str, np.ndarray], seqs: Sequence[DynamicsSequence],
    horizon: int, physical_step: float, device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]], float]:
    predictions = []
    targets = []
    target_ids = []
    solver_records = []
    starts = horizon_starts(seqs, horizon)
    started = time.perf_counter()
    for sequence_index, offset in starts:
        ids = seqs[sequence_index].latent_indices
        sp = torch.from_numpy(saved["semantic_latents"][ids[offset]:ids[offset] + 1]).float().to(device)
        sc = torch.from_numpy(saved["semantic_latents"][ids[offset + 1]:ids[offset + 1] + 1]).float().to(device)
        ep = torch.from_numpy(saved["execution_latents"][ids[offset]:ids[offset] + 1]).float().to(device)
        ec = torch.from_numpy(saved["execution_latents"][ids[offset + 1]:ids[offset + 1] + 1]).float().to(device)
        context = torch.from_numpy(saved["contexts"][ids[offset + 1]:ids[offset + 1] + 1]).float().to(device)
        for step in range(horizon):
            target_id = ids[offset + 2 + step]
            batch = FactorBatch(
                sp, sc, sc, ep, ec,
                torch.from_numpy(saved["execution_latents"][target_id:target_id + 1]).float().to(device), context,
            )
            with torch.no_grad():
                sn = models["semantic"](sp, sc, context)
            with torch.enable_grad():
                en, extras = forward_execution(name, models[name], batch, physical_step)
            if "del_info" in extras:
                info = extras["del_info"]
                solver_records.append({
                    "residual_norm": float(info.residual_norm.detach().mean().cpu()),
                    "iterations": float(info.iterations),
                    "converged": float(info.converged.float().detach().mean().cpu()),
                    "nonfinite": float(info.failed.float().detach().mean().cpu()),
                })
            elif "refinement_info" in extras:
                solver_records.append({
                    "iterations": float(extras["refinement_info"]["iterations"].detach().cpu()),
                    "nonfinite": float(not torch.isfinite(en).all()),
                })
            sp, sc = sc.detach(), sn.detach()
            ep, ec = ec.detach(), en.detach()
        prediction = torch.cat((sc, ec), dim=-1)
        final_id = ids[offset + 1 + horizon]
        predictions.append(prediction.cpu().numpy()[0])
        targets.append(saved["latents"][final_id])
        target_ids.append(final_id)
    elapsed = time.perf_counter() - started
    return np.asarray(predictions, np.float32), np.asarray(targets, np.float32), np.asarray(target_ids, np.int64), solver_records, elapsed


def evaluate_factorized_model(
    config: Mapping[str, Any], name: str, models: Mapping[str, nn.Module], representation: nn.Module,
    representation_payload: Mapping[str, Any], saved: Mapping[str, np.ndarray], seqs: Sequence[DynamicsSequence],
    train_ids: np.ndarray, thresholds: Mapping[str, Any], execution_variance: float, device: torch.device,
) -> dict[str, Any]:
    physical_step = int(config["data"]["chunk_length"]) / float(config["data"]["control_frequency_hz"])
    normalization = representation_payload["resolved_config"]["normalization"]
    action_mean = np.asarray(normalization["action_mean"], dtype=np.float32)
    action_std = np.asarray(normalization["action_std"], dtype=np.float32)
    prototype_tasks, prototypes = task_prototypes(saved, train_ids)
    result: dict[str, Any] = {"model": name, "horizons": {}}
    auc_points = []
    for horizon in [int(value) for value in config["evaluation"]["rollout_horizons"]]:
        prediction, target, target_ids, solver, elapsed = predict_rollout(name, models, saved, seqs, horizon, physical_step, device)
        if len(target_ids) == 0:
            result["horizons"][str(horizon)] = {"supported": False, "sample_count": 0}
            continue
        errors = prediction - target
        execution_prediction = prediction[:, 16:]
        execution_target = target[:, 16:]
        execution_errors = execution_prediction - execution_target
        execution_cosine = np.sum(execution_prediction * execution_target, axis=1) / np.maximum(np.linalg.norm(execution_prediction, axis=1) * np.linalg.norm(execution_target, axis=1), 1e-12)
        with torch.no_grad():
            decoded = representation.decode(torch.from_numpy(prediction).float().to(device)).cpu().numpy()
        decoded_raw = decoded.copy()
        decoded_raw[:, :, :6] = decoded_raw[:, :, :6] * action_std.reshape(1, 1, -1) + action_mean.reshape(1, 1, -1)
        decoded_gripper = np.where(decoded_raw[:, :, 6] >= 0, 1.0, -1.0)
        target_actions = saved["raw_actions"][target_ids]
        continuous_errors = (decoded_raw[:, :, :6] - target_actions[:, :, :6]) ** 2
        gripper_correct = decoded_gripper == target_actions[:, :, 6]
        semantic = prediction[:, :16]
        semantic_norm = semantic / np.maximum(np.linalg.norm(semantic, axis=1, keepdims=True), 1e-12)
        scores = semantic_norm @ prototypes.T
        predicted_tasks = np.asarray([prototype_tasks[index] for index in scores.argmax(axis=1)])
        correct_tasks = saved["task"][target_ids]
        correct_indices = np.asarray([prototype_tasks.index(str(task)) for task in correct_tasks])
        correct_cosines = scores[np.arange(len(target_ids)), correct_indices]
        full_nearest, full_radius = knn_distances(saved["latents"][train_ids], prediction, int(config["evaluation"]["knn_k"]))
        _, full_target_radius = knn_distances(saved["latents"][train_ids], target, int(config["evaluation"]["knn_k"]))
        exec_nearest, exec_radius = knn_distances(saved["execution_latents"][train_ids], execution_prediction, int(config["evaluation"]["knn_k"]))
        _, exec_target_radius = knn_distances(saved["execution_latents"][train_ids], execution_target, int(config["evaluation"]["knn_k"]))
        execution_mse = float(np.mean(execution_errors ** 2))
        metrics: dict[str, Any] = {
            "supported": True, "sample_count": len(target_ids), "physical_duration_seconds": horizon * physical_step,
            "execution_state": {
                "mse": execution_mse, "cosine_similarity": float(execution_cosine.mean()),
                "cosine_error": float((1 - execution_cosine).mean()),
            },
            "full_latent": {
                "mse": float(np.mean(errors ** 2)), "semantic_mse": float(np.mean(errors[:, :16] ** 2)),
                "execution_mse": execution_mse,
            },
            "decoded_actions": {
                "continuous_mse": float(continuous_errors.mean()), "gripper_accuracy": float(gripper_correct.mean()),
                "per_action_dimension_error": [float(continuous_errors[:, :, dimension].mean()) for dimension in range(6)] + [float((~gripper_correct).mean())],
                "dimension_7_metric": "gripper classification error; dimensions 1-6 are MSE",
            },
            "semantic_retention": {
                "predicted_latent_to_text_retrieval_accuracy": float(np.mean(predicted_tasks == correct_tasks)),
                "semantic_cosine_to_task_prototype": float(correct_cosines.mean()),
            },
            "off_manifold": {
                "full_latent_knn_radius": float(full_radius.mean()),
                "full_latent_ground_truth_knn_radius": float(full_target_radius.mean()),
                "full_latent_fraction_beyond_train_quantile": float(np.mean(full_radius > thresholds["full"]["threshold"])),
                "execution_nearest_training_distance": float(exec_nearest.mean()),
                "execution_knn_radius": float(exec_radius.mean()),
                "execution_ground_truth_knn_radius": float(exec_target_radius.mean()),
                "execution_knn_radius_ratio_to_ground_truth": float(exec_radius.mean() / max(float(exec_target_radius.mean()), 1e-12)),
                "execution_fraction_beyond_train_quantile": float(np.mean(exec_radius > thresholds["execution"]["threshold"])),
                "full_threshold": thresholds["full"]["threshold"], "execution_threshold": thresholds["execution"]["threshold"],
            },
            "runtime": {"wall_clock_seconds": elapsed, "seconds_per_rollout_start": elapsed / len(target_ids)},
        }
        if solver:
            keys = sorted(set().union(*(record.keys() for record in solver)))
            metrics["solver"] = {key: float(np.mean([record[key] for record in solver if key in record])) for key in keys}
            if "converged" in metrics["solver"]:
                metrics["solver"]["convergence_rate"] = metrics["solver"].pop("converged")
            if "nonfinite" in metrics["solver"]:
                metrics["solver"]["nonfinite_rate"] = metrics["solver"].pop("nonfinite")
        result["horizons"][str(horizon)] = metrics
        auc_points.append((horizon, execution_mse / execution_variance))
    result["normalized_execution_rollout_auc"] = float(np.trapz([value for _, value in auc_points], [horizon for horizon, _ in auc_points]))
    result["auc_supported_horizons"] = [horizon for horizon, _ in auc_points]
    return result


def evaluate_split(config: Mapping[str, Any], split: str, device: torch.device) -> dict[str, Any]:
    out = out_root(config)
    saved = arrays(config)
    split_sequences = sequences(config, split)
    train_sequences = sequences(config, "train")
    train_ids = np.asarray([index for sequence in train_sequences for index in sequence.latent_indices], dtype=np.int64)
    models, representation, payload = load_models(config, device)
    training = read_json(out / "training_and_development_selection.json")
    thresholds = read_json(out / "off_manifold_thresholds.json")
    evaluations = {
        name: evaluate_factorized_model(
            config, name, models, representation, payload, saved, split_sequences, train_ids,
            thresholds, float(training["execution_training_variance_mean"]), device,
        )
        for name in MODEL_ORDER
    }
    ranking = sorted(
        ({"model": name, "normalized_execution_rollout_auc": evaluations[name]["normalized_execution_rollout_auc"]} for name in MODEL_ORDER),
        key=lambda item: item["normalized_execution_rollout_auc"],
    )
    return {"split": split, "models": evaluations, "primary_ranking": ranking}


def development(config: Mapping[str, Any], device: torch.device) -> None:
    """Evaluate all selected checkpoints on development and write required tables."""

    out = out_root(config)
    disk_audit(config)
    saved = arrays(config)
    train_sequences = sequences(config, "train")
    train_ids = np.asarray([index for sequence in train_sequences for index in sequence.latent_indices], dtype=np.int64)
    thresholds = {
        "full": off_manifold_threshold(saved["latents"][train_ids], int(config["evaluation"]["knn_k"]), float(config["evaluation"]["off_manifold_quantile"])),
        "execution": off_manifold_threshold(saved["execution_latents"][train_ids], int(config["evaluation"]["knn_k"]), float(config["evaluation"]["off_manifold_quantile"])),
    }
    write_json(out / "off_manifold_thresholds.json", thresholds)
    result = evaluate_split(config, "development", device)
    write_json(out / "development_rollout_metrics.json", result)
    write_json(out / "development_decoded_action_metrics.json", {name: {h: value["decoded_actions"] for h, value in model["horizons"].items() if value.get("supported")} for name, model in result["models"].items()})
    write_json(out / "development_semantic_retention_metrics.json", {name: {h: value["semantic_retention"] for h, value in model["horizons"].items() if value.get("supported")} for name, model in result["models"].items()})
    write_json(out / "development_full_and_execution_off_manifold_metrics.json", {name: {h: value["off_manifold"] for h, value in model["horizons"].items() if value.get("supported")} for name, model in result["models"].items()})
    print(json.dumps({"stage": "development", "ranking": result["primary_ranking"]}))


def residual_for(
    name: str, model: nn.Module, batch: FactorBatch, candidate: torch.Tensor, physical_step: float
) -> torch.Tensor:
    if name == "F3_free_execution_del":
        combined = torch.cat((batch.s_current, batch.context), dim=-1)
        return model.residual(
            batch.e_previous, batch.e_current, candidate, combined, physical_step,
            torch.zeros_like(candidate), create_graph=False,
        )
    if name == "F4_decoder_geometry_del":
        return model.residual(
            batch.e_previous, batch.e_current, candidate, batch.s_current,
            batch.context, physical_step, create_graph=False,
        )
    raise KeyError(name)


def bootstrap_spearman(residual: np.ndarray, error: np.ndarray, seed: int, draws: int = 500) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        ids = rng.integers(0, len(residual), size=len(residual))
        value = float(spearmanr(residual[ids], error[ids]).statistic)
        if math.isfinite(value):
            values.append(value)
    return {
        "draws": draws, "finite_draws": len(values),
        "lower_95": float(np.quantile(values, 0.025)), "upper_95": float(np.quantile(values, 0.975)),
    }


def compatibility_preflight(config: Mapping[str, Any], device: torch.device) -> None:
    """Run the preregistered development residual compatibility/alignment audit."""

    out = out_root(config)
    if not (out / "development_rollout_metrics.json").is_file():
        raise RuntimeError("Development evaluation must precede compatibility preflight")
    disk_audit(config)
    saved = arrays(config)
    dev = sequences(config, "development")
    models, _, _ = load_models(config, device)
    physical_step = int(config["data"]["chunk_length"]) / float(config["data"]["control_frequency_hz"])
    record_count = len(transition_records(dev))
    collected: dict[str, dict[str, list[np.ndarray]]] = {
        name: {candidate: [] for candidate in ("true_next", "execution_mlp", "matched_refinement", "del_prediction")}
        for name in ("F3_free_execution_del", "F4_decoder_geometry_del")
    }
    errors: dict[str, list[np.ndarray]] = {candidate: [] for candidate in ("true_next", "execution_mlp", "matched_refinement", "F3_free_execution_del", "F4_decoder_geometry_del")}
    batch_size = 64
    for offset in range(0, record_count, batch_size):
        batch = make_batch(saved, dev, range(offset, min(offset + batch_size, record_count)), device)
        with torch.no_grad():
            f1, _ = forward_execution("F1_execution_mlp", models["F1_execution_mlp"], batch, physical_step)
        with torch.enable_grad():
            f2, _ = forward_execution("F2_matched_refinement", models["F2_matched_refinement"], batch, physical_step)
            f3, _ = forward_execution("F3_free_execution_del", models["F3_free_execution_del"], batch, physical_step)
            f4, _ = forward_execution("F4_decoder_geometry_del", models["F4_decoder_geometry_del"], batch, physical_step)
        candidates = {
            "true_next": batch.e_target, "execution_mlp": f1.detach(),
            "matched_refinement": f2.detach(),
        }
        for del_name, own in (("F3_free_execution_del", f3.detach()), ("F4_decoder_geometry_del", f4.detach())):
            local = dict(candidates)
            local["del_prediction"] = own
            for candidate_name, candidate in local.items():
                with torch.enable_grad():
                    residual = residual_for(del_name, models[del_name], batch, candidate, physical_step)
                collected[del_name][candidate_name].append(residual.detach().norm(dim=-1).cpu().numpy())
        all_predictions = dict(candidates)
        all_predictions["F3_free_execution_del"] = f3.detach()
        all_predictions["F4_decoder_geometry_del"] = f4.detach()
        for candidate_name, candidate in all_predictions.items():
            errors[candidate_name].append((candidate - batch.e_target).square().mean(dim=-1).detach().cpu().numpy())
    residual_report: dict[str, Any] = {}
    alignment_report: dict[str, Any] = {}
    for del_name in collected:
        residual_arrays = {name: np.concatenate(values) for name, values in collected[del_name].items()}
        residual_report[del_name] = {
            name: {
                "mean": float(values.mean()), "median": float(np.median(values)),
                "p90": float(np.quantile(values, 0.9)), "sample_count": len(values),
            }
            for name, values in residual_arrays.items()
        }
        residual_report[del_name]["compatibility_true_lower_than_execution_mlp"] = bool(residual_arrays["true_next"].mean() < residual_arrays["execution_mlp"].mean())
        residual_concat = np.concatenate([residual_arrays["true_next"], residual_arrays["execution_mlp"], residual_arrays["matched_refinement"], residual_arrays["del_prediction"]])
        error_names = ["true_next", "execution_mlp", "matched_refinement", del_name]
        error_concat = np.concatenate([np.concatenate(errors[name]) for name in error_names])
        rho = float(spearmanr(residual_concat, error_concat).statistic)
        alignment_report[del_name] = {
            "spearman_residual_norm_vs_execution_mse": rho,
            "positive_alignment": rho > 0,
            "bootstrap_95_interval": bootstrap_spearman(residual_concat, error_concat, int(config["experiment"]["seed"])),
            "points": len(residual_concat), "candidate_groups": error_names,
        }
    f4_compatible = residual_report["F4_decoder_geometry_del"]["compatibility_true_lower_than_execution_mlp"]
    f4_aligned = alignment_report["F4_decoder_geometry_del"]["positive_alignment"]
    write_json(out / "compatibility_preflight_report.json", {
        "created_at": now(), "split": "development_only", "official_validation_metrics_read": False,
        "residuals": residual_report, "F4_compatibility_passed": f4_compatible,
        "F4_alignment_passed": f4_aligned, "positive_DEL_claim_allowed_to_proceed": f4_compatible and f4_aligned,
    })
    write_json(out / "residual_vs_error_report.json", alignment_report)
    dev_result = read_json(out / "development_rollout_metrics.json")
    f1 = dev_result["models"]["F1_execution_mlp"]
    f2 = dev_result["models"]["F2_matched_refinement"]
    f4 = dev_result["models"]["F4_decoder_geometry_del"]
    worsening = float(config["evaluation"]["decoded_action_material_worsening_relative"])
    conditions = {
        "beats_F1_rollout_auc": f4["normalized_execution_rollout_auc"] < f1["normalized_execution_rollout_auc"],
        "beats_F2_rollout_auc": f4["normalized_execution_rollout_auc"] < f2["normalized_execution_rollout_auc"],
        "one_step_decoded_action_not_materially_worse": f4["horizons"]["1"]["decoded_actions"]["continuous_mse"] <= (1 + worsening) * f1["horizons"]["1"]["decoded_actions"]["continuous_mse"],
        "lowers_two_step_execution_off_manifold_drift": f4["horizons"]["2"]["off_manifold"]["execution_knn_radius"] < f1["horizons"]["2"]["off_manifold"]["execution_knn_radius"],
        "residual_compatibility": f4_compatible,
        "residual_error_positive_alignment": f4_aligned,
        "solver_nonfinite_rate_zero": f4["horizons"]["1"]["solver"]["nonfinite_rate"] == 0,
        "solver_convergence_materially_above_historical_zero": f4["horizons"]["1"]["solver"]["convergence_rate"] > 0,
    }
    write_json(out / "hard_variational_claim_gate.json", {
        "created_at": now(), "split": "development", "conditions": conditions,
        "all_conditions_passed": all(conditions.values()), "gate_relaxed_after_results": False,
        "C3b_development_decision": "SUPPORTED" if all(conditions.values()) else "REJECTED",
    })
    print(json.dumps({"stage": "preflight", "F4_compatibility": f4_compatible, "F4_alignment": f4_aligned, "hard_gate": all(conditions.values())}))


def freeze(config: Mapping[str, Any]) -> None:
    """Freeze all model/settings/metric hashes before official validation."""

    out = out_root(config)
    if not (out / "hard_variational_claim_gate.json").is_file():
        raise RuntimeError("Development hard gate must be adjudicated before freeze")
    storage = disk_audit(config)
    code_paths = [
        ROOT / "scripts/dynamics/run_dynamics_3.py", ROOT / "src/pglt/dynamics/factorized.py",
        ROOT / "src/pglt/dynamics/runner.py", ROOT / "src/pglt/dynamics/dynamics_data.py",
    ]
    manifest = {
        "created_at": now(), "frozen_before_official_validation_metrics": True,
        "official_validation_metrics_read": False,
        "checkpoints": {path.name: sha256_file(path) for path in sorted((out / "checkpoints").glob("*.pt"))},
        "settings": config, "metric_code_sha256": {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in code_paths},
        "development_gate": read_json(out / "hard_variational_claim_gate.json"),
        "compatibility_preflight": read_json(out / "compatibility_preflight_report.json"),
        "off_manifold_thresholds": read_json(out / "off_manifold_thresholds.json"),
        "storage_at_freeze": storage,
    }
    write_json(out / "factorized_dynamics_confirmation_manifest.json", manifest)
    print(json.dumps({"stage": "freeze", "checkpoints": len(manifest["checkpoints"]), "available_gib": storage["filesystem_available_bytes"] / 1024 ** 3}))


def validation(config: Mapping[str, Any], device: torch.device) -> None:
    """Perform the sole official-validation metric read after manifest freeze."""

    out = out_root(config)
    manifest = read_json(out / "factorized_dynamics_confirmation_manifest.json")
    if not manifest["frozen_before_official_validation_metrics"]:
        raise RuntimeError("Official validation requires frozen manifest")
    result_path = out / "one_shot_official_validation_results.json"
    if result_path.exists():
        raise RuntimeError("Official validation results already exist; refusing a second read")
    disk_audit(config)
    result = evaluate_split(config, "validation", device)
    result["one_shot"] = True
    result["manifest_created_at"] = manifest["created_at"]
    result["evaluated_at"] = now()
    write_json(result_path, result)
    write_json(out / "official_validation_decoded_action_metrics.json", {name: {h: value["decoded_actions"] for h, value in model["horizons"].items() if value.get("supported")} for name, model in result["models"].items()})
    write_json(out / "official_validation_semantic_retention_metrics.json", {name: {h: value["semantic_retention"] for h, value in model["horizons"].items() if value.get("supported")} for name, model in result["models"].items()})
    write_json(out / "official_validation_full_and_execution_off_manifold_metrics.json", {name: {h: value["off_manifold"] for h, value in model["horizons"].items() if value.get("supported")} for name, model in result["models"].items()})
    print(json.dumps({"stage": "validation", "one_shot": True, "ranking": result["primary_ranking"]}))


def value(result: Mapping[str, Any], model: str, horizon: str, section: str, metric: str) -> float:
    return float(result["models"][model]["horizons"][horizon][section][metric])


def fmt(number: Any) -> str:
    return f"{number:.6g}" if isinstance(number, float) else str(number)


def finalize(config: Mapping[str, Any]) -> None:
    """Adjudicate claims and write the complete scientific report and handoff."""

    out = out_root(config)
    development_result = read_json(out / "development_rollout_metrics.json")
    validation_result = read_json(out / "one_shot_official_validation_results.json")
    compatibility = read_json(out / "compatibility_preflight_report.json")
    alignment = read_json(out / "residual_vs_error_report.json")
    gate = read_json(out / "hard_variational_claim_gate.json")
    semantic = read_json(out / "semantic_predictor_specification_results.json")
    initial_audit = read_json(out / "frozen_representation_model_audit.json")
    representation, _, checkpoint, _ = select_representation(config)
    current_historical = {path: sha256_file(ROOT / path) for path in initial_audit["historical_negative_artifact_sha256"]}
    historical_unchanged = current_historical == initial_audit["historical_negative_artifact_sha256"]
    representation_unchanged = sha256_file(checkpoint) == initial_audit["checkpoint_sha256"] and tensor_hashes(representation) == initial_audit["initial_state_tensor_hashes"]
    dev_auc = {name: float(development_result["models"][name]["normalized_execution_rollout_auc"]) for name in MODEL_ORDER}
    val_auc = {name: float(validation_result["models"][name]["normalized_execution_rollout_auc"]) for name in MODEL_ORDER}
    validation_direction_agrees = val_auc["F4_decoder_geometry_del"] < val_auc["F1_execution_mlp"] and val_auc["F4_decoder_geometry_del"] < val_auc["F2_matched_refinement"]
    c3b = "SUPPORTED" if gate["all_conditions_passed"] and validation_direction_agrees else "REJECTED"
    generic_dev = dev_auc["F2_matched_refinement"] < dev_auc["F1_execution_mlp"]
    generic_val = val_auc["F2_matched_refinement"] < val_auc["F1_execution_mlp"]
    c3c = "SUPPORTED" if generic_dev and generic_val else "REJECTED"
    dev_order = [item["model"] for item in development_result["primary_ranking"]]
    val_order = [item["model"] for item in validation_result["primary_ranking"]]
    ordering_preserved = dev_order == val_order
    historical_dev = read_json(ROOT / config["experiment"]["wave13_root"] / "development_evaluation.json")
    historical_full_exec_mse = float(historical_dev["models"]["unforced_del"]["horizons"]["1"]["latent"]["execution_mse"])
    f3_exec_mse = value(development_result, "F3_free_execution_del", "1", "execution_state", "mse")
    f3_beats_historical = f3_exec_mse < historical_full_exec_mse
    f4_compat = bool(compatibility["F4_compatibility_passed"])
    f4_align = bool(compatibility["F4_alignment_passed"])
    f3_compat = bool(compatibility["residuals"]["F3_free_execution_del"]["compatibility_true_lower_than_execution_mlp"])
    f4_off = value(development_result, "F4_decoder_geometry_del", "2", "off_manifold", "execution_knn_radius")
    f1_off = value(development_result, "F1_execution_mlp", "2", "off_manifold", "execution_knn_radius")
    f4_decode = value(development_result, "F4_decoder_geometry_del", "1", "decoded_actions", "continuous_mse")
    f1_decode = value(development_result, "F1_execution_mlp", "1", "decoded_actions", "continuous_mse")
    semantic_values = [value(development_result, name, "2", "semantic_retention", "predicted_latent_to_text_retrieval_accuracy") for name in MODEL_ORDER]
    semantic_stable = max(semantic_values) - min(semantic_values) <= float(config["evaluation"]["semantic_retention_tolerance_absolute"])
    f4_convergence = value(development_result, "F4_decoder_geometry_del", "1", "solver", "convergence_rate")
    if c3b == "SUPPORTED":
        story = "Language anchors what an action means; variational executable geometry constrains how it evolves."
        story_cn = "语言锚定动作 latent 的语义成分，而由可执行解码几何约束的变分动力学负责其执行成分的连续演化。"
        next_experiment = "Expose annotation-consistent trajectories with at least 10 non-overlapping H=16 windows per valid segment; prospectively compare F1 execution MLP, F2 matched refinement, and F4 decoder-geometry DEL at horizons 1/2/4/8."
        carry = ["F1_execution_mlp", "F2_matched_refinement", "F4_decoder_geometry_del"]
        expose_longer = True
    elif c3c == "SUPPORTED":
        story = "Language defines meaningful action coordinates; structured refinement keeps learned transitions near executable regions."
        story_cn = "语言定义有意义且可执行的动作坐标；通用结构化 refinement 改善局部转移，而 DEL 仅保留为负机制基线。"
        next_experiment = "Expose longer annotation-consistent trajectories and make F1 execution MLP versus F2 matched generic refinement the sole primary comparison at horizons 1/2/4/8; retain DEL only as a frozen negative baseline and do not attempt another DEL rescue."
        carry = ["F1_execution_mlp", "F2_matched_refinement", "historical_DEL_negative_baseline"]
        expose_longer = True
    else:
        story = "Language-grounded action coordinates are semantically addressable and executable, but the present data do not support a stable structured-dynamics claim."
        story_cn = "语言落地动作坐标具有语义可寻址性与可执行性，但当前证据不支持稳定的结构化动力学主张。"
        next_experiment = "Do not open another structured-dynamics rescue wave for this submission. Consolidate the representation and learned-prediction evidence; collect longer trajectories only for a future separately preregistered study, carrying F1 as the predictive baseline."
        carry = ["F1_execution_mlp"]
        expose_longer = False
    decisions = {
        "created_at": now(), "C1_language_addressable_action_coordinate": "SUPPORTED",
        "C2_executable_continuous_action_coordinate": "SUPPORTED",
        "C3a_full_latent_useful_DEL_dynamics": "REJECTED",
        "C3b_executable_decoder_grounded_variational_dynamics": c3b,
        "C3c_generic_structured_refinement": c3c,
        "development_hard_gate_passed": gate["all_conditions_passed"],
        "official_validation_direction_agrees_for_F4": validation_direction_agrees,
        "exact_defensible_paper_story": story, "exact_defensible_paper_story_chinese": story_cn,
        "next_wave_expose_longer_trajectories": expose_longer, "models_to_carry_forward": carry,
    }
    write_json(out / "paper_claim_decision.json", decisions)
    answers = [
        f"1. Restricting DEL to `z_exec` removes the residual mismatch: **{'yes' if f3_compat else 'no'}** (F3 true-next compatibility={f3_compat}).",
        f"2. Decoder metric makes true-next lower-residual than F1: **{'yes' if f4_compat else 'no'}** (means {fmt(compatibility['residuals']['F4_decoder_geometry_del']['true_next']['mean'])} vs {fmt(compatibility['residuals']['F4_decoder_geometry_del']['execution_mlp']['mean'])}).",
        f"3. F4 residual is positively aligned with execution error: **{'yes' if f4_align else 'no'}** (Spearman={fmt(alignment['F4_decoder_geometry_del']['spearman_residual_norm_vs_execution_mse'])}, bootstrap 95% [{fmt(alignment['F4_decoder_geometry_del']['bootstrap_95_interval']['lower_95'])}, {fmt(alignment['F4_decoder_geometry_del']['bootstrap_95_interval']['upper_95'])}]).",
        f"4. Execution-only free DEL outperforms historical full-latent DEL: **{'yes' if f3_beats_historical else 'no'}** on development one-step execution MSE ({fmt(f3_exec_mse)} vs {fmt(historical_full_exec_mse)}).",
        f"5. F4 outperforms F1: **{'yes' if dev_auc['F4_decoder_geometry_del'] < dev_auc['F1_execution_mlp'] else 'no'}** on development AUC ({fmt(dev_auc['F4_decoder_geometry_del'])} vs {fmt(dev_auc['F1_execution_mlp'])}).",
        f"6. F4 outperforms matched F2: **{'yes' if dev_auc['F4_decoder_geometry_del'] < dev_auc['F2_matched_refinement'] else 'no'}** ({fmt(dev_auc['F4_decoder_geometry_del'])} vs {fmt(dev_auc['F2_matched_refinement'])}).",
        f"7. F4 reduces execution off-manifold drift: **{'yes' if f4_off < f1_off else 'no'}** (two-step kNN radius {fmt(f4_off)} vs {fmt(f1_off)}).",
        f"8. F4 hybrid latent decodes more accurately than F1: **{'yes' if f4_decode < f1_decode else 'no'}** (one-step continuous MSE {fmt(f4_decode)} vs {fmt(f1_decode)}).",
        f"9. Semantic retention remains stable: **{'yes' if semantic_stable else 'no'}**; every execution model uses the identical shared semantic checkpoint and two-step retrieval range is {fmt(min(semantic_values))}–{fmt(max(semantic_values))}.",
        f"10. F4 solver converges materially better than historical rate 0: **{'yes' if f4_convergence > 0 else 'no'}** (development one-step convergence={fmt(f4_convergence)}).",
        f"11. One-shot validation preserves the exact development ordering: **{'yes' if ordering_preserved else 'no'}** (dev {dev_order}; validation {val_order}).",
        f"12. C3b is **{c3b}**.",
        f"13. Generic structured refinement C3c is **{c3c}** (F2<F1 development={generic_dev}, validation={generic_val}).",
        f"14. Defensible paper story: **{story}**",
        f"15. Next-wave longer trajectories: **{'yes' if expose_longer else 'no for this submission'}**; carry {carry}.",
    ]
    report = "# PGLT 第十五轮 / 第三次动力学实验报告\n\n"
    report += "## 结论\n\n"
    report += f"C3b（执行子空间 decoder-grounded variational dynamics）最终为 **{c3b}**；C3c（generic structured refinement）为 **{c3c}**。开发 hard gate={'PASS' if gate['all_conditions_passed'] else 'FAIL'}，official validation 的 F4 相对方向一致={validation_direction_agrees}。冻结表示 optimizer/backward/EMA 步数均为 0，历史 full-latent DEL 负结论保持不变。\n\n"
    report += "## 实验设计与执行完整性\n\n"
    report += f"沿用 wave-13 完全相同的 train/development/official-validation episode split 与非重叠 H=16、stride=16 窗口。共享 semantic predictor 参数量为 {read_json(out / 'parameter_count_table.json')['semantic']['trainable_parameters']}，standalone development MSE={fmt(semantic['standalone_results']['development_semantic_mse'])}。F2/F4 使用同一冻结 F1 初始化与相同 4 次迭代，均无未来 target action。F4 的 metric 为冻结 decoder 连续输出（含 gripper logit、不含阈值）的 JVP pullback，epsilon=1e-3，仅 potential 可训练。\n\n"
    for title, result in (("Development", development_result), ("One-shot official validation", validation_result)):
        report += f"## {title} 主结果\n\n| model | H1 exec MSE | H2 exec MSE | H1 decoded MSE | H2 exec kNN radius | normalized rollout AUC |\n|---|---:|---:|---:|---:|---:|\n"
        for name in MODEL_ORDER:
            model = result["models"][name]
            report += f"| {name} | {fmt(value(result,name,'1','execution_state','mse'))} | {fmt(value(result,name,'2','execution_state','mse'))} | {fmt(value(result,name,'1','decoded_actions','continuous_mse'))} | {fmt(value(result,name,'2','off_manifold','execution_knn_radius'))} | {fmt(model['normalized_execution_rollout_auc'])} |\n"
        report += "\n"
    report += "## Compatibility preflight 与 hard gate\n\n"
    report += f"F4 true/F1/F2/F4 residual mean 分别为 {fmt(compatibility['residuals']['F4_decoder_geometry_del']['true_next']['mean'])} / {fmt(compatibility['residuals']['F4_decoder_geometry_del']['execution_mlp']['mean'])} / {fmt(compatibility['residuals']['F4_decoder_geometry_del']['matched_refinement']['mean'])} / {fmt(compatibility['residuals']['F4_decoder_geometry_del']['del_prediction']['mean'])}。\n\n"
    for condition, passed in gate["conditions"].items():
        report += f"- {condition}: **{'PASS' if passed else 'FAIL'}**\n"
    report += "\n## 15 个明确回答\n\n" + "\n".join(answers) + "\n\n"
    report += "## 科学上可辩护的故事与下一实验\n\n"
    report += f"{story}\n\n中文：{story_cn}\n\n下一实验：{next_experiment}\n\n"
    report += "## 可复现性与存储\n\n所有 exact commands、预注册、checkpoint、开发/验证 raw aggregate、residual/error、decoded/semantic/off-manifold 表、参数量、manifest、gate、claim JSON、pytest XML、环境和文件审计均保存在本 wave 目录。最终磁盘审计见 `final_integrity_check.json`。\n"
    (out / "fifteenth_wave_results.md").write_text(report, encoding="utf-8")
    report_path = ROOT / config["experiment"]["report_path"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    next_text = f"# Fifteenth-wave next experiment\n\n{next_experiment}\n\nCarry forward: {', '.join(carry)}.\n"
    (out / "fifteenth_wave_next_experiment.md").write_text(next_text, encoding="utf-8")
    (ROOT / "NEXT_EXPERIMENT.md").write_text(next_text, encoding="utf-8")
    log_path = ROOT / "RESEARCH_LOG.md"
    previous_log = log_path.read_text(encoding="utf-8") if log_path.exists() else "# RESEARCH_LOG\n"
    entry = f"\n## {now()} — dynamics_3 / wave 15\n\nCompleted the full factorized executable-subspace experiment. Development hard gate: {'PASS' if gate['all_conditions_passed'] else 'FAIL'}; C3b: {c3b}; C3c: {c3c}. Official validation was read exactly once after manifest freeze. See `{config['experiment']['report_path']}`.\n"
    log_path.write_text(previous_log.rstrip() + "\n" + entry, encoding="utf-8")
    storage = disk_audit(config)
    final_integrity = {
        "created_at": now(), "representation_unchanged": representation_unchanged,
        "historical_full_latent_DEL_artifacts_unchanged": historical_unchanged,
        "representation_optimizer_steps": 0, "representation_backward_calls": 0, "ema_updates": 0,
        "official_validation_one_shot": validation_result["one_shot"],
        "manifest_preceded_validation": validation_result["manifest_created_at"] < validation_result["evaluated_at"],
        "storage": storage, "all_passed": representation_unchanged and historical_unchanged and validation_result["one_shot"] and storage["passed"],
    }
    write_json(out / "final_integrity_check.json", final_integrity)
    files = sorted(path for path in out.rglob("*") if path.is_file()) + [report_path, ROOT / "NEXT_EXPERIMENT.md", ROOT / "RESEARCH_LOG.md"]
    write_json(out / "files_changed_report.json", {
        "created_or_updated_files": [{"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files]
    })
    print(json.dumps({"stage": "finalize", "C3b": c3b, "C3c": c3c, "report": str(report_path), "available_gib": storage["filesystem_available_bytes"] / 1024 ** 3}))


def main() -> None:
    args = parse_args()
    config = read_yaml(args.config)
    device = torch.device(args.device)
    stages = ("prepare", "validate_metric", "train", "development", "preflight", "freeze", "validation", "finalize") if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "prepare":
            prepare(config)
        elif stage == "validate_metric":
            validate_metric(config, device)
        elif stage == "train":
            train(config, device)
        elif stage == "development":
            development(config, device)
        elif stage == "preflight":
            compatibility_preflight(config, device)
        elif stage == "freeze":
            freeze(config)
        elif stage == "validation":
            validation(config, device)
        elif stage == "finalize":
            finalize(config)


if __name__ == "__main__":
    main()

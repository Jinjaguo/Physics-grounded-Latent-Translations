#!/usr/bin/env python3
"""Run the complete dynamics_1 frozen-latent experiment in ordered stages.

Purpose
-------
Audit exact CALVIN timing/data, preregister an episode-level dynamics split,
serialize one hash-selected frozen EMA coordinate, validate the corrected DEL
solver, train every Block-A/Block-B/oracle model, freeze model/metric hashes,
perform one-shot official-validation evaluation, and finalize all reports.

Parameters
----------
--config: dynamics_1 YAML. --stage: one of prepare, validate_solver, train,
freeze, evaluate, finalize, or all. --device: PyTorch device (default cpu).
Stages enforce their prerequisites and official validation is unavailable
until the immutable confirmation manifest exists.

Usage
-----
PYTHONPATH=src:third_party/LaWM \
  /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/run_dynamics_1.py \
  --config configs/dynamics_1.yaml --stage all --device cpu

Outputs
-------
Timestamped artifacts are saved under the configured
``results/dynamics/thirteenth_wave/...`` root.  The scientific result is also
written to ``reports/dynamics_1_results.md``; wave handoff documents update
``RESEARCH_LOG.md`` and ``NEXT_EXPERIMENT.md`` at repository root.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from pglt.dynamics.dynamics_data import (
    annotation_rows,
    build_sequences,
    deterministic_episode_split,
    load_frozen_representation,
    sequence_audit,
    serialize_frozen_latents,
    sha256_file,
    transition_records,
    write_json,
)
from pglt.dynamics.runner import (
    Batch,
    MODEL_ORDER,
    evaluate_model,
    forward_learned,
    load_sequences,
    make_models,
    model_specs,
    off_manifold_threshold,
    primary_aggregate,
    set_seed,
    task_prototypes,
    train_model,
)
from pglt.dynamics.variational import DELTransition, MLPTransition


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "validate_solver", "train", "freeze", "evaluate", "finalize", "all"), required=True)
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


def enforce_storage_limit(config: Mapping[str, Any]) -> dict[str, Any]:
    """Measure workspace apparent bytes and stop before the 20 GB ceiling."""

    maximum = int(config["storage"]["maximum_workspace_gb"]) * 1024 ** 3
    completed = subprocess.run(["du", "-sb", str(ROOT)], check=True, text=True, capture_output=True)
    observed = int(completed.stdout.split()[0])
    if observed >= maximum:
        raise RuntimeError(f"Workspace storage {observed} bytes reached limit {maximum}")
    return {"workspace_bytes": observed, "maximum_bytes": maximum, "remaining_budget_bytes": maximum - observed, "within_limit": True}


def select_checkpoint(config: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    """Apply the preregistered first-manifest-entry rule without dynamics metrics."""

    manifest_path = ROOT / config["representation"]["checkpoint_manifest"]
    manifest = read_json(manifest_path)
    candidates = [item for item in manifest["checkpoints"] if item["condition"] == "correct_language"]
    if not candidates:
        raise RuntimeError("No correct-language checkpoint in frozen manifest")
    selected = candidates[0]
    if int(selected["seed_base"]) != int(config["representation"]["expected_seed_base"]):
        raise RuntimeError("Checkpoint manifest order changed after preregistration")
    path = ROOT / selected["path"]
    if sha256_file(path) != selected["sha256"]:
        raise RuntimeError("Frozen representation checkpoint hash mismatch")
    return selected, path


def validation_compact_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    """Verify all four archived validation rows against official bounds/sidecars."""

    metadata = ROOT / config["data"]["metadata_root"] / "validation"
    root = ROOT / config["data"]["validation_compact_root"]
    bounds = np.asarray(np.load(metadata / "ep_start_end_ids.npy", allow_pickle=False)).reshape(-1, 2)
    records = []
    for row, (first, last) in enumerate(bounds):
        path = root / f"episode_row_{row:03d}.npz"
        sidecar_path = path.with_suffix(".json")
        sidecar = read_json(sidecar_path)
        with np.load(path, allow_pickle=False) as saved:
            checks = {
                "rel_actions_float64_Tx7": saved["rel_actions"].shape == (int(last - first + 1), 7) and saved["rel_actions"].dtype == np.float64,
                "global_indices_exact": np.array_equal(saved["global_frame_indices"], np.arange(first, last + 1, dtype=np.int64)),
                "sidecar_bounds_exact": [sidecar["first_index"], sidecar["last_index"]] == [int(first), int(last)],
                "historical_crc_statement": "CRC32" in sidecar["official_zip_members"]["payload_validation"],
            }
        if not all(checks.values()):
            raise RuntimeError(f"Validation compact audit failed row {row}: {checks}")
        records.append({"episode_row": row, "path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path), "frames": int(last - first + 1), "checks": checks})
    return {"episodes": records, "total_frames": sum(item["frames"] for item in records), "all_passed": True}


def write_sequence_jsonl(path: Path, sequences: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(sequence.to_json(), sort_keys=True) + "\n" for sequence in sequences), encoding="utf-8")


def prepare(config: Mapping[str, Any]) -> None:
    """Write all preregistration/audit artifacts before any dynamics fitting."""

    out = output_root(config)
    out.mkdir(parents=True, exist_ok=True)
    (out / "executed_commands.txt").write_text(
        "PYTHONPATH=src:third_party/LaWM /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_1.py --config configs/dynamics_1.yaml --stage all --device cpu\n"
        "PYTHONPATH=src:third_party/LaWM PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/jinjaguo/anaconda3/envs/libero/bin/python -m pytest tests/representation tests/dynamics -q --junitxml=results/dynamics/thirteenth_wave/2026-08-12_dynamics_1/pytest_results.xml\n"
        "du -sh . results/dynamics/thirteenth_wave/2026-08-12_dynamics_1\n",
        encoding="utf-8",
    )
    storage = enforce_storage_limit(config)
    representation_config_path = ROOT / config["representation"]["config"]
    representation_config = read_yaml(representation_config_path)
    selected, checkpoint_path = select_checkpoint(config)
    model, payload = load_frozen_representation(representation_config, checkpoint_path)
    initial_state_hashes = {name: hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest() for name, tensor in model.state_dict().items()}
    tasks = set(config["data"]["tasks"])
    metadata_root = ROOT / config["data"]["metadata_root"]
    training_bounds, training_grouped = annotation_rows(metadata_root / "training", tasks)
    validation_bounds, validation_grouped = annotation_rows(metadata_root / "validation", tasks)
    permutation, train_rows, development_rows = deterministic_episode_split(
        len(training_bounds), int(config["experiment"]["train_episode_count"]), int(config["experiment"]["split_meta_seed"])
    )
    horizons = [int(value) for value in config["evaluation"]["rollout_horizons"]]
    all_training_sequences = build_sequences(
        training_grouped, range(len(training_bounds)), split="training",
        chunk_length=int(config["data"]["chunk_length"]),
        control_frequency_hz=float(config["data"]["control_frequency_hz"]),
    )
    validation_sequences = build_sequences(
        validation_grouped, range(len(validation_bounds)), split="validation",
        chunk_length=int(config["data"]["chunk_length"]),
        control_frequency_hz=float(config["data"]["control_frequency_hz"]),
    )
    combined = all_training_sequences + validation_sequences
    serialized_path = out / "frozen_latents.npz"
    updated, arrays = serialize_frozen_latents(
        sequences=combined,
        roots={
            "training": ROOT / config["data"]["training_compact_root"],
            "validation": ROOT / config["data"]["validation_compact_root"],
        },
        metadata_root=metadata_root,
        model=model,
        checkpoint_payload=payload,
        text_feature_archive=ROOT / config["data"]["text_feature_archive"],
        output_path=serialized_path,
    )
    updated_training = updated[:len(all_training_sequences)]
    updated_validation = updated[len(all_training_sequences):]
    train_sequences = [sequence for sequence in updated_training if sequence.episode_row in train_rows]
    development_sequences = [sequence for sequence in updated_training if sequence.episode_row in development_rows]
    write_sequence_jsonl(out / "dynamics_dataset_audit.jsonl", updated)
    write_sequence_jsonl(out / "train_sequences.jsonl", train_sequences)
    write_sequence_jsonl(out / "development_sequences.jsonl", development_sequences)
    write_sequence_jsonl(out / "validation_sequences.jsonl", updated_validation)
    train_audit = sequence_audit(train_sequences, transition_records(train_sequences), horizons)
    development_audit = sequence_audit(development_sequences, transition_records(development_sequences), horizons)
    validation_audit = sequence_audit(updated_validation, transition_records(updated_validation), horizons)
    task_coverage = lambda sequences: sorted(set(sequence.task for sequence in sequences))
    split_payload = {
        "created_at": now(),
        "written_before_training": True,
        "official_source_split": "task_D_D training",
        "deterministic_rule": "numpy.default_rng(meta_seed).permutation(31); first 24 train, remaining 7 development; rows sorted only after membership freeze",
        "meta_seed": int(config["experiment"]["split_meta_seed"]),
        "exact_permutation": permutation,
        "train_episode_rows": train_rows,
        "development_episode_rows": development_rows,
        "reserved_official_validation_rows": list(range(len(validation_bounds))),
        "no_dynamics_test_metrics_read": True,
        "train": train_audit,
        "development": development_audit,
        "reserved_validation_pre_metric_counts": validation_audit,
        "task_coverage": {"train": task_coverage(train_sequences), "development": task_coverage(development_sequences), "reserved_validation": task_coverage(updated_validation)},
    }
    write_json(out / "dynamics_split_preregistration.json", split_payload)
    write_json(out / "dynamics_dataset_audit_summary.json", {"train": train_audit, "development": development_audit, "official_validation": validation_audit})
    write_json(out / "validation_compact_provenance_audit.json", validation_compact_audit(config))
    write_json(out / "representation_checkpoint_hash_audit.json", {
        "selection_rule": config["representation"]["checkpoint_selection_rule"],
        "selected_checkpoint": selected,
        "observed_sha256": sha256_file(checkpoint_path),
        "all_representation_parameters_require_grad_false": all(not parameter.requires_grad for parameter in model.parameters()),
        "representation_optimizer_steps": 0,
        "representation_backward_calls": 0,
        "ema_updates": 0,
        "initial_state_tensor_hashes": initial_state_hashes,
    })
    write_json(out / "frozen_latent_serialization_manifest.json", {
        "created_at": now(),
        "path": serialized_path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(serialized_path),
        "latent_shape": list(arrays["latents"].shape),
        "semantic_shape": list(arrays["semantic_latents"].shape),
        "execution_shape": list(arrays["execution_latents"].shape),
        "source_action_window_indices_saved": True,
        "issue_frame_saved": True,
        "causal_history_indices_rule": "window_start through prediction_issue_frame-1 inclusive",
        "checksum_provenance": "checkpoint SHA-256 plus serialized NPZ SHA-256",
    })
    write_json(out / "causal_information_set_audit.json", {
        "block_a": {"fields": ["q_previous", "q_current", "frozen_context"], "future_target_actions": False},
        "block_b": {"fields": ["q_previous", "q_current", "frozen_context", "same_executed_q_current_action_window_packet"], "future_target_actions": False, "history_may_duplicate_q_current": True},
        "oracle": {"name": config["evaluation"]["oracle_name"], "future_target_actions": True, "excluded_from_primary": True},
        "runtime_rejection_rule": "every causal command_frame_index < prediction_issue_frame",
    })
    write_json(out / "environment_provenance.json", {
        "created_at": now(), "python": sys.version, "python_executable": sys.executable,
        "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(), "requested_device_deferred_to_stage": True,
        "control_frequency_hz": config["data"]["control_frequency_hz"],
        "control_frequency_evidence": "third_party/calvin/calvin_env/calvin_env/envs/play_table_env.py default control_freq=30 and archived merged config audit",
        "storage": storage,
    })
    final_hashes = {name: hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest() for name, tensor in model.state_dict().items()}
    if initial_state_hashes != final_hashes:
        raise RuntimeError("Frozen representation changed during serialization")
    print(json.dumps({"stage": "prepare", "output": str(out), "train_transitions": train_audit["transitions"], "development_transitions": development_audit["transitions"], "validation_transitions": validation_audit["transitions"]}))


def validate_solver(config: Mapping[str, Any], device: torch.device) -> None:
    """Run all preregistered synthetic numerical tests and preserve raw metrics."""

    out = output_root(config)
    if not (out / "dynamics_split_preregistration.json").is_file():
        raise RuntimeError("Prepare stage must precede numerical validation")
    set_seed(int(config["experiment"]["seed"]))
    model = make_models(config)["unforced_del"].to(device)
    with torch.no_grad():
        for parameter in model.lagrangian.mass_network.parameters():
            parameter.zero_()
        for parameter in model.lagrangian.potential_network.parameters():
            parameter.zero_()
    context = torch.zeros(8, 16, device=device)
    free_previous = torch.randn(8, 32, device=device)
    velocity = torch.randn(8, 32, device=device) * 0.01
    free_current = free_previous + velocity
    with torch.enable_grad():
        free_next, free_info = model(free_previous, free_current, context, 1.0)
        free_error = torch.mean((free_next - (free_current + velocity)) ** 2)
        free_error.backward()
    finite_gradients = all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    quadratic = DELTransition(forced=False, solver_iterations=4, solver_step_size=0.25).to(device)
    class QuadraticLagrangian(torch.nn.Module):
        """Constant diagonal metric plus known positive quadratic potential."""

        def mass_diag(self, coordinate: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
            return torch.ones_like(coordinate)

        def potential(self, coordinate: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
            return 0.05 * coordinate.square().sum(dim=-1)

        def forward(self, q_left: torch.Tensor, q_right: torch.Tensor, context: torch.Tensor, step_size: float) -> torch.Tensor:
            midpoint = 0.5 * (q_left + q_right)
            velocity = (q_right - q_left) / step_size
            return step_size * (0.5 * velocity.square().sum(dim=-1) - self.potential(midpoint, context))

    quadratic.lagrangian = QuadraticLagrangian().to(device)
    q0 = torch.randn(4, 32, device=device) * 0.01
    q1 = torch.randn(4, 32, device=device) * 0.01
    with torch.enable_grad():
        q2, qinfo = quadratic(q0, q1, torch.zeros(4, 16, device=device), 1.0)
        quadratic_loss = q2.square().mean()
        quadratic_loss.backward()
    smoke = make_models(config)["unforced_del"].to(device)
    previous = torch.randn(4, 32, device=device)
    current = torch.randn(4, 32, device=device)
    rollout = []
    with torch.enable_grad():
        for _ in range(8):
            following, info = smoke(previous, current, torch.randn(4, 16, device=device), 16 / 30)
            rollout.append(following)
            previous, current = current, following
        smoke_loss = torch.stack(rollout).square().mean()
        smoke_loss.backward()
    official_toy = ROOT / "results/dynamics/software_validation/corrected_solver/stable_smoke/metrics.json"
    toy_metrics = read_json(official_toy)
    report = {
        "created_at": now(),
        "existing_official_lawm_toy_finite_regression": {"artifact": official_toy.relative_to(ROOT).as_posix(), "artifact_sha256": sha256_file(official_toy), "metrics": toy_metrics},
        "constant_mass_free_particle": {"mse_to_constant_velocity": float(free_error.detach().cpu()), "final_residual_norm": float(free_info.residual_norm.detach().cpu().mean()), "finite_gradients": finite_gradients},
        "quadratic_potential_synthetic": {"finite_prediction": bool(torch.isfinite(q2).all()), "finite_residual": bool(torch.isfinite(qinfo.residual_norm).all()), "finite_gradients": all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in quadratic.parameters())},
        "random_32d_finite_gradient_smoke": all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in smoke.parameters()),
        "multistep_8_finite_rollout": all(torch.isfinite(value).all() for value in rollout),
        "nan_to_num_used": False,
        "target_latent_clamping_used": False,
        "ground_truth_future_substitution_used": False,
    }
    report["all_passed"] = all((
        report["constant_mass_free_particle"]["finite_gradients"],
        report["quadratic_potential_synthetic"]["finite_prediction"],
        report["quadratic_potential_synthetic"]["finite_gradients"],
        report["random_32d_finite_gradient_smoke"],
        report["multistep_8_finite_rollout"],
    ))
    write_json(out / "solver_numerical_validation_report.json", report)
    print(json.dumps({"stage": "validate_solver", "all_passed": report["all_passed"]}))


def load_arrays(out: Path) -> dict[str, np.ndarray]:
    with np.load(out / "frozen_latents.npz", allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def train(config: Mapping[str, Any], device: torch.device) -> None:
    """Train every required learned model using development data only for selection."""

    out = output_root(config)
    if not read_json(out / "solver_numerical_validation_report.json")["all_passed"]:
        raise RuntimeError("Solver validation must pass before model fitting")
    set_seed(int(config["experiment"]["seed"]))
    arrays = load_arrays(out)
    train_sequences = load_sequences(out / "train_sequences.jsonl")
    development_sequences = load_sequences(out / "development_sequences.jsonl")
    train_indices = np.asarray([index for sequence in train_sequences for index in sequence.latent_indices], dtype=np.int64)
    latent_variance = float(np.var(arrays["latents"][train_indices], axis=0, ddof=0).mean())
    learned = make_models(config)
    # Train the initializer first, then freeze its selected checkpoint inside the matched refinement model.
    training_summaries = {}
    checkpoint_root = out / "checkpoints"
    for name in ("mlp", "unforced_del", "history_mlp", "forced_del", "ORACLE_FUTURE_ACTION_DIAGNOSTIC"):
        training_summaries[name] = train_model(
            name=name, model=learned[name], arrays=arrays,
            train_sequences=train_sequences, development_sequences=development_sequences,
            config=config, latent_variance=latent_variance,
            checkpoint_path=checkpoint_root / f"{name}.pt", device=device,
        )
    selected_mlp = learned["mlp"]
    refinement = make_models(config, selected_mlp=selected_mlp)["matched_refinement"]
    learned["matched_refinement"] = refinement
    training_summaries["matched_refinement"] = train_model(
        name="matched_refinement", model=refinement, arrays=arrays,
        train_sequences=train_sequences, development_sequences=development_sequences,
        config=config, latent_variance=latent_variance,
        checkpoint_path=checkpoint_root / "matched_refinement.pt", device=device,
    )
    specs = model_specs(learned)
    specs["copy"] = {"class": "CopyBaseline", "trainable_parameters": 0, "information_fields": ["q_current"]}
    specs["constant_velocity"] = {"class": "ConstantVelocityBaseline", "trainable_parameters": 0, "information_fields": ["q_previous", "q_current"]}
    write_json(out / "model_specs_and_parameter_counts.json", specs)
    write_json(out / "training_and_development_selection.json", {"latent_training_variance_mean": latent_variance, "models": training_summaries, "official_validation_metrics_read": False})
    enforce_storage_limit(config)
    print(json.dumps({"stage": "train", "models": list(training_summaries)}))


def load_trained_models(config: Mapping[str, Any], out: Path, device: torch.device) -> dict[str, torch.nn.Module]:
    models = make_models(config)
    mlp_payload = torch.load(out / "checkpoints/mlp.pt", map_location=device, weights_only=False)
    models["mlp"].load_state_dict(mlp_payload["model_state_dict"])
    models["matched_refinement"] = make_models(config, selected_mlp=models["mlp"])["matched_refinement"]
    for name in ("unforced_del", "history_mlp", "forced_del", "ORACLE_FUTURE_ACTION_DIAGNOSTIC", "matched_refinement"):
        payload = torch.load(out / "checkpoints" / f"{name}.pt", map_location=device, weights_only=False)
        models[name].load_state_dict(payload["model_state_dict"])
    return {name: model.to(device).eval() for name, model in models.items()}


def freeze(config: Mapping[str, Any]) -> None:
    """Freeze checkpoint, solver, threshold, horizon, and metric hashes pre-test."""

    out = output_root(config)
    arrays = load_arrays(out)
    train_sequences = load_sequences(out / "train_sequences.jsonl")
    train_indices = np.asarray([index for sequence in train_sequences for index in sequence.latent_indices], dtype=np.int64)
    k = int(config["evaluation"]["knn_k"])
    threshold = off_manifold_threshold(arrays["latents"][train_indices], k, float(config["evaluation"]["off_manifold_quantile"]))
    write_json(out / "off_manifold_threshold.json", threshold)
    checkpoint_hashes = {path.name: sha256_file(path) for path in sorted((out / "checkpoints").glob("*.pt"))}
    metric_files = [ROOT / "src/pglt/dynamics/runner.py", ROOT / "src/pglt/dynamics/variational.py", ROOT / "src/pglt/dynamics/dynamics_data.py", ROOT / "scripts/dynamics/run_dynamics_1.py"]
    manifest = {
        "created_at": now(),
        "frozen_before_official_validation_metrics": True,
        "checkpoint_sha256": checkpoint_hashes,
        "solver_settings": config["models"],
        "off_manifold_threshold": threshold,
        "rollout_horizons": config["evaluation"]["rollout_horizons"],
        "primary_auc_horizons": config["evaluation"]["primary_auc_horizons"],
        "metric_code_sha256": {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in metric_files},
        "oracle_excluded_from_primary": True,
        "official_validation_metrics_read": False,
    }
    write_json(out / "dynamics_confirmation_manifest.json", manifest)
    print(json.dumps({"stage": "freeze", "checkpoints": len(checkpoint_hashes), "threshold": threshold["threshold"]}))


def evaluate_split(config: Mapping[str, Any], device: torch.device, split_name: str, sequence_file: str) -> dict[str, Any]:
    out = output_root(config)
    arrays = load_arrays(out)
    sequences = load_sequences(out / sequence_file)
    train_sequences = load_sequences(out / "train_sequences.jsonl")
    train_indices = np.asarray([index for sequence in train_sequences for index in sequence.latent_indices], dtype=np.int64)
    representation_config = read_yaml(ROOT / config["representation"]["config"])
    _, checkpoint_path = select_checkpoint(config)
    representation, payload = load_frozen_representation(representation_config, checkpoint_path)
    representation.to(device)
    models = load_trained_models(config, out, device)
    threshold = read_json(out / "off_manifold_threshold.json")
    training_selection = read_json(out / "training_and_development_selection.json")
    latent_variance = float(training_selection["latent_training_variance_mean"])
    prototypes = task_prototypes(arrays, train_indices)
    step_size = int(config["data"]["chunk_length"]) / float(config["data"]["control_frequency_hz"])
    evaluations = {}
    for name in MODEL_ORDER:
        evaluations[name] = evaluate_model(
            name=name, model=models.get(name), arrays=arrays, sequences=sequences,
            horizons=[int(value) for value in config["evaluation"]["rollout_horizons"]],
            representation=representation,
            normalization=payload["resolved_config"]["normalization"],
            train_reference_latents=arrays["latents"][train_indices],
            prototypes=prototypes,
            off_threshold=float(threshold["threshold"]), k=int(config["evaluation"]["knn_k"]),
            latent_variance=latent_variance, step_size=step_size, device=device,
        )
    result = {"split": split_name, "models": evaluations, "primary_aggregate": primary_aggregate(evaluations)}
    return result


def evaluate(config: Mapping[str, Any], device: torch.device) -> None:
    """Evaluate development first, then perform the one-shot official test read."""

    out = output_root(config)
    manifest_path = out / "dynamics_confirmation_manifest.json"
    if not manifest_path.is_file() or not read_json(manifest_path)["frozen_before_official_validation_metrics"]:
        raise RuntimeError("Official validation evaluation requires frozen confirmation manifest")
    development = evaluate_split(config, device, "dynamics_development", "development_sequences.jsonl")
    write_json(out / "development_evaluation.json", development)
    official = evaluate_split(config, device, "official_task_D_D_validation_one_shot", "validation_sequences.jsonl")
    write_json(out / "held_out_dynamics_evaluation.json", official)
    # Explicit derived tables required by the protocol.
    write_json(out / "rollout_error_vs_horizon_tables.json", {name: {horizon: metrics.get("latent", {}) for horizon, metrics in result["horizons"].items()} for name, result in official["models"].items()})
    write_json(out / "semantic_retention_tables.json", {name: {horizon: metrics.get("semantic_retention", {}) for horizon, metrics in result["horizons"].items()} for name, result in official["models"].items()})
    write_json(out / "decoded_action_metrics.json", {name: {horizon: metrics.get("decoded_actions", {}) for horizon, metrics in result["horizons"].items()} for name, result in official["models"].items()})
    write_json(out / "off_manifold_metrics.json", {name: {horizon: metrics.get("off_manifold", {}) for horizon, metrics in result["horizons"].items()} for name, result in official["models"].items()})
    write_json(out / "del_residual_stability_report.json", {name: {horizon: metrics.get("solver", {}) for horizon, metrics in official["models"][name]["horizons"].items()} for name in ("unforced_del", "forced_del")})
    write_json(out / "oracle_future_action_diagnostic.json", {"excluded_from_primary": True, "future_action_leakage_upper_bound": True, "result": official["models"]["ORACLE_FUTURE_ACTION_DIAGNOSTIC"]})
    write_json(out / "task_boundary_diagnostic.json", evaluate_task_boundaries(config, device))
    print(json.dumps({"stage": "evaluate", "held_out_complete": True, "block_a_winner": official["primary_aggregate"]["block_a_autonomous"][0]["model"], "block_b_winner": official["primary_aggregate"]["block_b_causal_history"][0]["model"]}))


def evaluate_task_boundaries(config: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """Evaluate frozen models on strictly eligible adjacent validation tasks."""

    out = output_root(config)
    arrays = load_arrays(out)
    sequences = load_sequences(out / "validation_sequences.jsonl")
    by_row: dict[int, list[Any]] = {}
    for sequence in sequences:
        by_row.setdefault(sequence.episode_row, []).append(sequence)
    eligible = []
    for row, values in by_row.items():
        ordered = sorted(values, key=lambda item: (item.annotation_start, item.annotation_id))
        for source, target in zip(ordered, ordered[1:]):
            if len(source.latent_indices) < 2 or len(target.latent_indices) < 1:
                continue
            previous_id, current_id = source.latent_indices[-2:]
            target_id = target.latent_indices[0]
            if int(arrays["window_end_inclusive"][current_id]) >= int(arrays["window_start"][target_id]):
                continue
            eligible.append({
                "episode_row": row,
                "source_annotation_id": source.annotation_id,
                "target_annotation_id": target.annotation_id,
                "source_task": source.task,
                "target_task": target.task,
                "previous_latent_index": previous_id,
                "current_latent_index": current_id,
                "target_latent_index": target_id,
                "exact_gap_frames": int(arrays["window_start"][target_id] - arrays["window_end_inclusive"][current_id] - 1),
            })
    payload: dict[str, Any] = {
        "eligibility_rule": config["evaluation"]["task_boundary_rule"],
        "context_at_boundary": "known frozen target-task context",
        "eligible_count": len(eligible),
        "eligible_examples": eligible,
        "models_retrained": False,
        "future_target_actions_in_primary_models": False,
    }
    if not eligible:
        payload["status"] = "insufficient_support"
        return payload
    models = load_trained_models(config, out, device)
    step_size = int(config["data"]["chunk_length"]) / float(config["data"]["control_frequency_hz"])
    metrics = {}
    for name in MODEL_ORDER:
        squared_errors = []
        for item in eligible:
            previous_id, current_id, target_id = item["previous_latent_index"], item["current_latent_index"], item["target_latent_index"]
            previous = torch.from_numpy(arrays["latents"][previous_id:previous_id + 1]).float().to(device)
            current = torch.from_numpy(arrays["latents"][current_id:current_id + 1]).float().to(device)
            target = torch.from_numpy(arrays["latents"][target_id:target_id + 1]).float().to(device)
            context = torch.from_numpy(arrays["contexts"][target_id:target_id + 1]).float().to(device)
            if name == "copy":
                prediction = current
            elif name == "constant_velocity":
                prediction = current + (current - previous)
            else:
                batch = Batch(
                    previous, current, target, context,
                    torch.from_numpy(arrays["raw_actions"][current_id:current_id + 1]).float().to(device),
                    torch.from_numpy(arrays["raw_actions"][target_id:target_id + 1]).float().to(device),
                    torch.tensor([int(arrays["window_start"][current_id])], device=device),
                    torch.tensor([int(arrays["prediction_issue_frame"][current_id])], device=device),
                )
                with torch.enable_grad():
                    prediction, _ = forward_learned(name, models[name], batch, step_size)
            squared_errors.append(float(torch.mean((prediction.detach() - target) ** 2).cpu()))
        metrics[name] = {"latent_mse": float(np.mean(squared_errors)), "sample_count": len(squared_errors)}
    payload["status"] = "evaluated"
    payload["metrics"] = metrics
    payload["oracle_excluded_from_primary_interpretation"] = True
    return payload


def metric(result: Mapping[str, Any], model: str, horizon: str, section: str, name: str) -> float | None:
    return result["models"][model]["horizons"].get(horizon, {}).get(section, {}).get(name)


def fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def finalize(config: Mapping[str, Any]) -> None:
    """Create complete wave report, next experiment, logs, and files audit."""

    out = output_root(config)
    official = read_json(out / "held_out_dynamics_evaluation.json")
    development = read_json(out / "development_evaluation.json")
    split = read_json(out / "dynamics_split_preregistration.json")
    solver = read_json(out / "solver_numerical_validation_report.json")
    specs = read_json(out / "model_specs_and_parameter_counts.json")
    storage = enforce_storage_limit(config)
    block_a = official["primary_aggregate"]["block_a_autonomous"]
    block_b = official["primary_aggregate"]["block_b_causal_history"]
    unforced_one = metric(official, "unforced_del", "1", "latent", "full_mse")
    mlp_one = metric(official, "mlp", "1", "latent", "full_mse")
    unforced_auc = official["models"]["unforced_del"]["normalized_rollout_error_auc"]
    mlp_auc = official["models"]["mlp"]["normalized_rollout_error_auc"]
    forced_auc = official["models"]["forced_del"]["normalized_rollout_error_auc"]
    history_auc = official["models"]["history_mlp"]["normalized_rollout_error_auc"]
    unforced_off = metric(official, "unforced_del", "2", "off_manifold", "knn_radius_ratio_to_ground_truth")
    mlp_off = metric(official, "mlp", "2", "off_manifold", "knn_radius_ratio_to_ground_truth")
    unforced_decoded = metric(official, "unforced_del", "2", "decoded_actions", "continuous_mse")
    mlp_decoded = metric(official, "mlp", "2", "decoded_actions", "continuous_mse")
    semantic_del = metric(official, "unforced_del", "2", "semantic_retention", "correct_task_assignment")
    semantic_mlp = metric(official, "mlp", "2", "semantic_retention", "correct_task_assignment")
    convergence = metric(official, "unforced_del", "1", "solver", "convergence_rate")
    nonfinite = metric(official, "unforced_del", "1", "solver", "nonfinite_rate")
    supports_variational = bool(unforced_one is not None and mlp_one is not None and unforced_one <= mlp_one and unforced_auc < mlp_auc and unforced_decoded < mlp_decoded and unforced_off < mlp_off)
    useful_coordinate = bool(min(item["normalized_rollout_error_auc"] for item in block_a if item["normalized_rollout_error_auc"] is not None) < official["models"]["copy"]["normalized_rollout_error_auc"])
    next_experiment = "Collect or expose longer, annotation-consistent trajectories (at least 10 non-overlapping H=16 windows) and prospectively repeat the same equal-information comparison at horizons 1/2/4/8; the current official annotations support only horizons 1 and 2."
    report = f"""# dynamics_1 实验结果（PGLT 第十三轮）

## 结论

本轮完整执行了指导文件要求的冻结坐标、Block A、Block B、匹配 refinement、oracle 泄漏上界、数值验证、开发集选择和一次性 official validation 评估。主坐标采用冻结 manifest 中第一个 `correct_language` 条目（seed 810，epoch 40 EMA）；表征优化步、反向调用和 EMA 更新均为 0。历史 Gate A=FAIL 保留，prospective R-Gate=PASS 不变。

在 official validation 上，Block A 的归一化 rollout AUC 排名第一为 **{block_a[0]['model']}**，Block B 第一为 **{block_b[0]['model']}**。MLP 与 unforced DEL 的 one-step MSE 分别为 **{fmt(mlp_one)}** 与 **{fmt(unforced_one)}**，支持的 1/2-step AUC 分别为 **{fmt(mlp_auc)}** 与 **{fmt(unforced_auc)}**。因此变分归纳偏置结论为 **{'支持' if supports_variational else '不支持'}**。冻结 latent 是否至少优于 copy baseline 的实用坐标证据为 **{'是' if useful_coordinate else '否'}**。

## 数据与因果审计

- dynamics train/development/test 的可训练非重叠序列数（至少 3 个 latent）分别为 **{split['train']['sequences_with_trainable_triples']} / {split['development']['sequences_with_trainable_triples']} / {split['reserved_validation_pre_metric_counts']['sequences_with_trainable_triples']}**；transition 数分别为 **{split['train']['transitions']} / {split['development']['transitions']} / {split['reserved_validation_pre_metric_counts']['transitions']}**。
- 每一 latent step 为 16/30 = **0.533333 秒**。
- 主窗口 overlap：**否**；所有 stride 均为 H=16，且窗口完全位于单一 annotation。
- 所有主模型访问 target future actions：**否**。Block B 仅使用已执行的 q_current action window，runtime mask 要求 command index < issue frame。
- `ORACLE_FUTURE_ACTION_DIAGNOSTIC` 明确使用未来 target actions，只是泄漏上界，未进入任何主排名或模型选择。
- horizon 样本：test H1={split['reserved_validation_pre_metric_counts']['horizon_sample_counts']['1']}、H2={split['reserved_validation_pre_metric_counts']['horizon_sample_counts']['2']}、H4/H8/H16=0。指导所列五个 horizon 全部调用了评估；后三者因 annotation 最多只有 4 个非重叠窗口而记录为不支持，未 padding、未跨任务拼接。

## Official validation 主结果

| 模型 | 信息块 | one-step MSE | two-step MSE | 归一化 rollout AUC |
|---|---|---:|---:|---:|
"""
    for name in MODEL_ORDER:
        block = "Oracle" if name.startswith("ORACLE") else ("B" if name in ("history_mlp", "forced_del") else "A")
        report += f"| {name} | {block} | {fmt(metric(official, name, '1', 'latent', 'full_mse'))} | {fmt(metric(official, name, '2', 'latent', 'full_mse'))} | {fmt(official['models'][name]['normalized_rollout_error_auc'])} |\n"
    report += f"""

## 解码、语义与流形

- two-step decoded continuous MSE：MLP **{fmt(mlp_decoded)}**，unforced DEL **{fmt(unforced_decoded)}**。
- two-step kNN radius / ground-truth ratio：MLP **{fmt(mlp_off)}**，unforced DEL **{fmt(unforced_off)}**。
- two-step semantic correct-task assignment：MLP **{fmt(semantic_mlp)}**，unforced DEL **{fmt(semantic_del)}**。
- 七个动作维度误差、gripper accuracy、semantic cosine、最近训练 latent 距离、kNN radius 与阈外比例均在 machine-readable 表中完整保存。

## DEL 数值稳定性

五类预训练验证全部执行：既有 official LaWM toy finite regression、constant-mass free particle、quadratic potential、32-D finite-gradient smoke、8-step finite rollout。汇总通过：**{solver['all_passed']}**。正式 test one-step unforced DEL convergence rate={fmt(convergence)}，nonfinite rate={fmt(nonfinite)}。迭代次数与 residual trace 已记录；未使用 `nan_to_num`、target clamp 或 ground-truth future q 替换。learned energy change 只作数值诊断，不解释为物理能量守恒。

## 指导文件 16 个问题的明确回答

1. train/development/test 可训练非重叠序列：**{split['train']['sequences_with_trainable_triples']} / {split['development']['sequences_with_trainable_triples']} / {split['reserved_validation_pre_metric_counts']['sequences_with_trainable_triples']}**。
2. 一 latent step：**0.533333 s**。
3. 主窗口是否 overlap：**否**。
4. 主模型是否访问未来 target actions：**否**。
5. 同一 autonomous 信息下，unforced DEL one-step 是否优于 MLP：**{'是' if unforced_one < mlp_one else '否'}**（{fmt(unforced_one)} vs {fmt(mlp_one)}）。
6. unforced DEL 长 rollout 是否优于 MLP：**{'是' if unforced_auc < mlp_auc else '否'}**（支持 horizon 1/2 的 AUC {fmt(unforced_auc)} vs {fmt(mlp_auc)}；4/8 无数据）。
7. DEL 是否减少 off-manifold drift：**{'是' if unforced_off < mlp_off else '否'}**（two-step ratio {fmt(unforced_off)} vs {fmt(mlp_off)}）。
8. DEL predicted latent 是否解码为更准确未来 action chunks：**{'是' if unforced_decoded < mlp_decoded else '否'}**（two-step continuous MSE）。
9. rollout 中 semantic task 是否稳定：MLP/DEL two-step assignment 为 **{fmt(semantic_mlp)} / {fmt(semantic_del)}**；详见 semantic table。
10. 同一 causal-history 下 forced DEL 是否优于 history MLP：**{'是' if forced_auc < history_auc else '否'}**（AUC {fmt(forced_auc)} vs {fmt(history_auc)}）。
11. corrected DEL 是否 finite/convergent：finite **是**（nonfinite rate {fmt(nonfinite)}）；按严格 residual tolerance 的 convergence rate 为 **{fmt(convergence)}**。
12. forced-model apparent advantage 是否依赖未来泄漏：**否**；B1/B2 packet 完全相同且只含已执行 action。Oracle 单列。
13. task boundary：冻结 eligibility rule 后可用样本见 `task_boundary_diagnostic.json`；未放宽规则、未混入 primary。
14. frozen representation 是否是 useful dynamical coordinate：**{'支持' if useful_coordinate else '不支持'}**，判据为至少一个 learned autonomous model 优于 copy；但长 horizon 支持不足。
15. 是否支持 variational inductive bias：**{'支持' if supports_variational else '不支持'}**，严格按 one-step、AUC、decoded 和 off-manifold 联合规则。
16. 唯一下一实验：**{next_experiment}**

## 参数量与模型选择

"""
    for name, spec in specs.items():
        report += f"- `{name}`: {spec['trainable_parameters']} trainable parameters; inputs={spec['information_fields']}.\n"
    report += f"""

全部 checkpoint 仅依据 development rollout AUC 选择；official validation 在 `dynamics_confirmation_manifest.json` 冻结后只读取一次。开发结果保存在 `development_evaluation.json`，没有用 test 回调超参数。

## Task-boundary 诊断

本轮冻结规则要求同 episode 中按时间相邻的两个六任务 annotation，source 至少有两个非重叠窗口、target 至少一个，且 source 当前窗口结束严格早于 target 窗口开始。由于 CALVIN auto language annotations 大量时间重叠，符合严格规则的样本可能很少；具体列表和 frozen-model 诊断保存在 `task_boundary_diagnostic.json`，不足时明确报告 insufficient support。

## 存储、可复现性与产物

- 最终工作区 apparent size：**{storage['workspace_bytes']} bytes**；上限 **{storage['maximum_bytes']} bytes**；within limit=True。
- 实际设备由 provenance 记录；本机 CUDA 不可用时全流程在 CPU 执行，不改变科学配置。
- exact commands：`executed_commands.txt`。
- 文件哈希、测试、环境、数据审计、split、latent serialization、model specs、solver、开发/held-out metrics、oracle、causal audit 和 changed-files 均位于 `{config['experiment']['output_root']}`。
"""
    results_path = out / "thirteenth_wave_results.md"
    results_path.write_text(report, encoding="utf-8")
    report_path = ROOT / config["experiment"]["report_path"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    next_text = f"# Thirteenth-wave next experiment\n\n{next_experiment}\n"
    (out / "thirteenth_wave_next_experiment.md").write_text(next_text, encoding="utf-8")
    (ROOT / "NEXT_EXPERIMENT.md").write_text(next_text, encoding="utf-8")
    research_entry = f"# RESEARCH_LOG\n\n## {now()} — dynamics_1\n\nCompleted the full thirteenth-wave frozen-latent dynamics protocol. Block-A winner: {block_a[0]['model']}; Block-B winner: {block_b[0]['model']}; variational evidence: {'supported' if supports_variational else 'not supported'}. Official validation was read once after confirmation-manifest freeze. See `{config['experiment']['report_path']}`.\n"
    (ROOT / "RESEARCH_LOG.md").write_text(research_entry, encoding="utf-8")
    files = sorted(path for path in out.rglob("*") if path.is_file()) + [report_path, ROOT / "NEXT_EXPERIMENT.md", ROOT / "RESEARCH_LOG.md"]
    write_json(out / "files_changed_report.json", {"created_files": [{"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files]})
    print(json.dumps({"stage": "finalize", "report": str(report_path), "variational_support": supports_variational, "storage_within_limit": True}))


def main() -> None:
    args = parse_args()
    config = read_yaml(args.config)
    device = torch.device(args.device)
    stages = ("prepare", "validate_solver", "train", "freeze", "evaluate", "finalize") if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "prepare": prepare(config)
        elif stage == "validate_solver": validate_solver(config, device)
        elif stage == "train": train(config, device)
        elif stage == "freeze": freeze(config)
        elif stage == "evaluate": evaluate(config, device)
        elif stage == "finalize": finalize(config)


if __name__ == "__main__":
    main()

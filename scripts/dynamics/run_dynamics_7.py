#!/usr/bin/env python3
"""Prepare and certify the wave-19 independent official LIBERO-10 replication.

Purpose
-------
Resolve the installed official LIBERO-10 suite, freeze OpenPI / π0.5 source
policy provenance and the cross-domain protocol, audit the exact action and
time interfaces, and run the development-only repeated exact-state restore
gate before any source-trajectory collection or model inference.

Parameters
----------
``--config`` selects the frozen wave-19 YAML. ``--stage audit`` writes suite,
source-policy, collection, representation, dynamics, and closed-loop
preregistrations. ``--stage audit-resume`` only completes environment metadata
after an interrupted audit. ``--stage dev-certify`` runs 100 simulator-only
repeated restore/replay trials and freezes empirical certification tolerances.

Usage
-----
PYTHONPATH=src:/home/jinjaguo/LIBERO \
  /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/run_dynamics_7.py --config configs/dynamics_7.yaml --stage audit

PYTHONPATH=src:/home/jinjaguo/LIBERO MUJOCO_GL=egl \
  /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/run_dynamics_7.py --config configs/dynamics_7.yaml --stage dev-certify

Outputs
-------
Audit and gate artifacts are written below
``results/dynamics/nineteenth_wave/2026-08-14_dynamics_7``. Prospective source
data and its provenance/audits are written below
``data/wave19_libero_branchable``.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("audit", "audit-resume", "dev-certify"), required=True)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    resolved = path if path.is_absolute() else ROOT / path
    return yaml.safe_load(resolved.read_text(encoding="utf-8"))


def out_root(config: Mapping[str, Any]) -> Path:
    return ROOT / config["experiment"]["output_root"]


def data_root(config: Mapping[str, Any]) -> Path:
    return ROOT / config["experiment"]["data_root"]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def append_command(config: Mapping[str, Any], command: str, phase: str) -> None:
    path = out_root(config) / "exact_commands.sh"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n# {now()} phase={phase}\n{command}\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_record(path: Path) -> dict[str, Any]:
    commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(path), "status", "--short"], text=True)
    return {"path": str(path), "commit": commit, "status_short": status.splitlines()}


def freeze_packages(python: str) -> str:
    """Freeze distributions without assuming a uv-managed venv contains pip."""

    code = (
        "import importlib.metadata as m; "
        "rows=sorted(f\"{d.metadata['Name']}=={d.version}\" for d in m.distributions() if d.metadata['Name']); "
        "print('\\n'.join(rows))"
    )
    return subprocess.check_output([python, "-c", code], text=True).rstrip()


def write_environment_freeze(config: Mapping[str, Any]) -> str:
    sources = config["sources"]
    freeze_parts = []
    for label, python in (("LIBERO", sources["libero_python"]), ("OpenPI", sources["openpi_python"])):
        freeze_parts.append(f"## {label} environment\n{freeze_packages(python)}")
    environment = (
        f"recorded_at={now()}\nplatform={platform.platform()}\nrunner_python={' '.join(sys.version.split())}\n\n"
        + "\n\n".join(freeze_parts)
    )
    write_text(out_root(config) / "environment_freeze.txt", environment)
    write_text(data_root(config) / "provenance/environment_freeze.txt", environment)
    return environment


def disk_record(config: Mapping[str, Any], phase: str) -> dict[str, Any]:
    usage = shutil.disk_usage(ROOT)
    record = {
        "recorded_at": now(),
        "phase": phase,
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "minimum_free_bytes": int(config["runtime"]["minimum_free_disk_bytes"]),
        "preferred_free_bytes": int(config["runtime"]["preferred_free_disk_bytes"]),
    }
    record["minimum_pass"] = record["free_bytes"] >= record["minimum_free_bytes"]
    record["preferred_pass"] = record["free_bytes"] >= record["preferred_free_bytes"]
    append_jsonl(data_root(config) / "audits/disk_usage_log.jsonl", record)
    if not record["minimum_pass"]:
        raise RuntimeError(f"Free disk {record['free_bytes']} is below required minimum")
    return record


def goal_section(text: str) -> str:
    marker = "(:goal"
    start = text.lower().find(marker)
    if start < 0:
        raise ValueError("BDDL has no (:goal section")
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError("Unbalanced BDDL goal section")


def audit(config: Mapping[str, Any]) -> None:
    out = out_root(config)
    data = data_root(config)
    if (out / "wave19_collection_preregistration.json").exists():
        raise RuntimeError("Wave-19 audit is already frozen; refusing to rewrite preregistration")
    for path in (
        out,
        out / "publication_tables",
        out / "publication_figures_data",
        data / "provenance",
        data / "raw_collection",
        data / "certified",
        data / "derived/representation",
        data / "derived/dynamics",
        data / "derived/manifests",
        data / "audits",
    ):
        path.mkdir(parents=True, exist_ok=True)

    disk = disk_record(config, "audit")
    sources = config["sources"]
    libero_root = Path(sources["libero_root"])
    openpi_root = Path(sources["openpi_root"])
    libero_git = git_record(libero_root)
    openpi_git = git_record(openpi_root)
    write_text(data / "provenance/libero_git_commit.txt", libero_git["commit"])
    write_text(data / "provenance/openpi_git_commit.txt", openpi_git["commit"])

    from libero.libero import benchmark

    mapping = benchmark.get_benchmark_dict()
    requested = sources["official_suite"]
    if requested not in mapping:
        raise RuntimeError(f"Requested suite {requested!r} not in installed mapping {sorted(mapping)}")
    suite = mapping[requested]()
    if suite.get_num_tasks() != 10:
        raise RuntimeError(f"Official target must contain exactly 10 tasks, got {suite.get_num_tasks()}")

    source_root = libero_root / "libero/libero"
    task_rows = []
    for task_id in range(suite.get_num_tasks()):
        task = suite.get_task(task_id)
        bddl = source_root / "bddl_files" / task.problem_folder / task.bddl_file
        init_state = source_root / "init_files" / task.problem_folder / task.init_states_file
        if not bddl.is_file() or not init_state.is_file():
            raise FileNotFoundError(f"Missing official task source: {bddl} or {init_state}")
        task_rows.append(
            {
                "task_id": task_id,
                "task_name": task.name,
                "language_instruction": task.language,
                "bddl_file": str(bddl),
                "init_states_file": str(init_state),
                "success_predicate_bddl": goal_section(bddl.read_text(encoding="utf-8")),
            }
        )

    suite_md = [
        "# Wave-19 LIBERO suite audit",
        "",
        "```text",
        "requested_internal_name = LIBERO-Long",
        f"resolved_official_suite_name = {requested}",
        f"number_of_tasks = {len(task_rows)}",
        "exact_task_list = [",
    ]
    suite_md.extend(f"  {row['task_id']}: {row['language_instruction']}" for row in task_rows)
    suite_md.extend(["]", "```", "", f"Available installed suites: {sorted(mapping)}", ""])
    for row in task_rows:
        suite_md.extend(
            [
                f"## Task {row['task_id']:02d}",
                "",
                f"- BDDL: `{row['bddl_file']}`",
                f"- language: {row['language_instruction']}",
                f"- success predicate: `{row['success_predicate_bddl']}`",
                "- controller: `OSC_POSE`",
                "",
            ]
        )
    write_text(out / "wave19_libero_suite_audit.md", "\n".join(suite_md))
    write_json(out / "wave19_libero_suite_tasks.json", task_rows)

    checkpoint_dir = Path(sources["checkpoint_dir"])
    model_path = checkpoint_dir / "model.safetensors"
    norm_path = Path(sources["norm_assets_dir"]) / "physical-intelligence/libero/norm_stats.json"
    required_checkpoint_files = [
        model_path,
        checkpoint_dir / "config.json",
        checkpoint_dir / "policy_preprocessor.json",
        checkpoint_dir / "policy_postprocessor.json",
        norm_path,
    ]
    missing = [str(path) for path in required_checkpoint_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Official π0.5 checkpoint assets missing: {missing}")

    policy_manifest = {
        "created_at": now(),
        "role": "fixed_source_trajectory_generator_only",
        "checkpoint_uri": sources["checkpoint_uri"],
        "local_checkpoint_path": str(checkpoint_dir),
        "model_file_bytes": model_path.stat().st_size,
        "model_file_sha256": sha256_file(model_path),
        "openpi": openpi_git,
        "libero": libero_git,
        "config_name": "pi05_libero",
        "normalization_assets": str(norm_path),
        "normalization_assets_sha256": sha256_file(norm_path),
        "policy_preprocessor": json.loads((checkpoint_dir / "policy_preprocessor.json").read_text()),
        "policy_postprocessor": json.loads((checkpoint_dir / "policy_postprocessor.json").read_text()),
        "observation_preprocessing": "agentview and wrist rotated 180 degrees; resize_with_pad 224x224; uint8",
        "network_action_horizon": int(config["action_interface"]["policy_action_horizon"]),
        "executed_replan_steps": int(config["action_interface"]["policy_replan_steps"]),
        "environment_action_dim": int(config["action_interface"]["action_dim"]),
        "server": "scripts/dynamics/start_wave19_pi05_server.py, websocket localhost:8000",
        "random_seed": int(config["experiment"]["seed"]),
        "fine_tuned_during_wave19": False,
        "pi05_hidden_states_used": False,
    }
    write_json(out / "wave19_pi05_source_policy_manifest.json", policy_manifest)
    write_json(data / "provenance/pi05_source_policy_manifest.json", policy_manifest)

    commands = f"""# Official OpenPI reference evaluation resolved against the installed README.
cd {openpi_root}
source .venv/bin/activate
uv run scripts/serve_policy.py --env LIBERO policy:checkpoint \\
  --policy.config pi05_libero --policy.dir {sources['checkpoint_uri']}

# In a second terminal, using the official LIBERO client environment:
cd {openpi_root}
MUJOCO_GL=egl python examples/libero/main.py --args.task-suite-name libero_10

# Exact Wave-19 raw-action server used for prospective branchable collection.
{sources['openpi_python']} {ROOT}/scripts/dynamics/start_wave19_pi05_server.py \\
  --checkpoint_dir {sources['checkpoint_dir']} \\
  --norm_assets_dir {sources['norm_assets_dir']} \\
  --seed {config['experiment']['seed']} --port 8000

# In a second terminal, exact Wave-19 official LIBERO-10 collector.
cd {ROOT}
PYTHONPATH=src:{libero_root}:{openpi_root}/packages/openpi-client/src MUJOCO_GL=egl \\
  {sources['libero_python']} scripts/dynamics/collect_wave19_libero.py \\
  --config configs/dynamics_7.yaml --host localhost --port 8000
"""
    write_text(data / "provenance/pi05_download_and_eval_commands.sh", commands)

    action_manifest = {
        "frozen_at": now(),
        **config["action_interface"],
        "action_repeat": 1,
        "physics_steps_per_control_step": 25,
        "clipping": "OSC clips each of first six normalized components to [-1,1] before scaling",
        "network_vs_executed_boundary": (
            "policy_output_actions stores websocket output after OpenPI unnormalization; executed_actions stores an "
            "independent copy passed to env.step; no additional client clipping"
        ),
        "immutable_action_boundary": "safe_env_step passes action.copy() and checks caller bytes unchanged",
    }
    write_json(out / "wave19_libero_action_interface.json", action_manifest)
    write_text(
        out / "wave19_libero_action_interface.md",
        "# Wave-19 official LIBERO action interface\n\n"
        + "\n".join(f"- {key}: `{value}`" for key, value in action_manifest.items()),
    )
    write_json(
        out / "wave19_timebase_preregistration.json",
        {"frozen_at": now(), "uses_final_test_outputs": False, **config["timebase"]},
    )
    write_json(data / "provenance/collection_config.json", config["collection"])
    write_json(
        out / "wave19_collection_preregistration.json",
        {
            "frozen_at": now(),
            "written_before_collection": True,
            "suite": requested,
            "task_ids": list(range(10)),
            "task_balanced": True,
            "source_outcomes_retained": ["successes", "failures"],
            "fresh_official_initial_state_per_attempt": True,
            "source_policy": "fixed official pi05_libero",
            **config["collection"],
        },
    )
    write_json(
        out / "wave19_representation_preregistration.json",
        {
            "frozen_at": now(),
            "independent_from_calvin": True,
            "calvin_checkpoints_forbidden": True,
            **config["representation"],
            "gate": {
                "semantic_delta_positive_both_directions": True,
                "episode_clustered_bootstrap_lower_95_gt_zero": True,
                "motor_fidelity_not_destroyed": True,
                "minimum_seeds": 6,
            },
        },
    )
    write_json(
        out / "wave19_dynamics_preregistration.json",
        {
            "frozen_at": now(),
            "independent_from_calvin": True,
            "future_inputs_forbidden": True,
            **config["dynamics"],
            "offline_horizons": config["timebase"]["rollout_horizons"],
            "cluster": config["statistics"]["cluster"],
        },
    )
    write_json(
        out / "wave19_closed_loop_preregistration.json",
        {
            "frozen_at": now(),
            "authorized_only_after_snapshot_representation_offline_gates": True,
            "branch_fractions": config["collection"]["branch_fractions"],
            "horizons": config["timebase"]["rollout_horizons"],
            "methods": ["source_pi05", "F1", "F2", "norm_matched_random", "shuffled", "negative"],
            "refinement_iterations": 4,
            "observation_feedback_to_F1_F2": False,
            "cluster": "source_episode",
            "bootstrap_replicates": config["statistics"]["bootstrap_replicates"],
            "bootstrap_seed": config["statistics"]["bootstrap_seed"],
        },
    )
    write_text(
        data / "audits/action_mutation_audit.md",
        "# Action mutation audit\n\nAll Wave-19 environment actions cross `safe_env_step`, which passes a private "
        "copy to LIBERO and verifies the caller-owned byte sequence is unchanged. The corresponding unit test uses "
        "an intentionally mutating environment to prove the boundary isolates the source array.",
    )

    write_environment_freeze(config)
    write_json(
        out / "wave19_audit_summary.json",
        {
            "created_at": now(),
            "suite_resolved": True,
            "official_task_count": len(task_rows),
            "checkpoint_available": True,
            "libero_git": libero_git,
            "openpi_git": openpi_git,
            "disk": disk,
            "global_libero_config_is_usable": False,
            "global_libero_config_issue": (
                "/home/jinjaguo/.libero/config.yaml points to missing pi0-text-latent/third_party/modified_libero; "
                "Wave 19 freezes explicit /home/jinjaguo/LIBERO paths instead"
            ),
        },
    )
    append_command(
        config,
        "PYTHONPATH=src:/home/jinjaguo/LIBERO /home/jinjaguo/anaconda3/envs/libero/bin/python "
        "scripts/dynamics/run_dynamics_7.py --config configs/dynamics_7.yaml --stage audit",
        "audit",
    )


def resume_audit(config: Mapping[str, Any]) -> None:
    """Complete only metadata that follows the frozen preregistrations."""

    out = out_root(config)
    data = data_root(config)
    if not (out / "wave19_collection_preregistration.json").is_file():
        raise RuntimeError("No interrupted audit preregistration is available to resume")
    if (out / "wave19_audit_summary.json").exists():
        raise RuntimeError("Audit summary already exists; refusing redundant audit-resume")
    write_environment_freeze(config)
    sources = config["sources"]
    disk = disk_record(config, "audit-resume")
    task_rows = json.loads((out / "wave19_libero_suite_tasks.json").read_text(encoding="utf-8"))
    write_json(
        out / "wave19_audit_summary.json",
        {
            "created_at": now(),
            "suite_resolved": True,
            "official_task_count": len(task_rows),
            "checkpoint_available": True,
            "libero_git": git_record(Path(sources["libero_root"])),
            "openpi_git": git_record(Path(sources["openpi_root"])),
            "disk": disk,
            "initial_audit_interruption": (
                "OpenPI uv venv intentionally has no pip module; preregistrations were already frozen and were "
                "not rewritten. Package freeze resumed via importlib.metadata."
            ),
            "global_libero_config_is_usable": False,
            "global_libero_config_issue": (
                "/home/jinjaguo/.libero/config.yaml points to missing pi0-text-latent/third_party/modified_libero; "
                "Wave 19 freezes explicit /home/jinjaguo/LIBERO paths instead"
            ),
        },
    )
    append_command(
        config,
        "# FAILED after preregistration write: OpenPI uv venv has no pip module\n"
        "PYTHONPATH=src:/home/jinjaguo/LIBERO /home/jinjaguo/anaconda3/envs/libero/bin/python "
        "scripts/dynamics/run_dynamics_7.py --config configs/dynamics_7.yaml --stage audit",
        "audit-interrupted",
    )
    append_command(
        config,
        "PYTHONPATH=src:/home/jinjaguo/LIBERO /home/jinjaguo/anaconda3/envs/libero/bin/python "
        "scripts/dynamics/run_dynamics_7.py --config configs/dynamics_7.yaml --stage audit-resume",
        "audit-resume",
    )


def build_env(bddl: Path, resolution: int, *, use_camera_obs: bool = True):
    from libero.libero.envs import OffScreenRenderEnv

    return OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=resolution,
        camera_widths=resolution,
        use_camera_obs=use_camera_obs,
    )


def dev_certify(config: Mapping[str, Any]) -> None:
    out = out_root(config)
    prereg = out / "wave19_collection_preregistration.json"
    if not prereg.is_file():
        raise RuntimeError("Run audit before dev-certify")
    result_path = out / "development_snapshot_determinism_results.json"
    if result_path.exists():
        raise RuntimeError("Development certification already exists; refusing to rerun frozen selection")
    disk_record(config, "dev-certify")

    import torch
    from libero.libero import benchmark
    from pglt.libero.snapshot import (
        capture_integration_state,
        capture_snapshot,
        restore_snapshot,
        safe_env_step,
    )

    cert = config["snapshot_certification"]
    source_root = Path(config["sources"]["libero_root"]) / "libero/libero"
    suite = benchmark.get_benchmark_dict()[config["sources"]["official_suite"]]()
    rng = np.random.default_rng(int(config["experiment"]["seed"]))
    rows = []
    for task_id in range(suite.get_num_tasks()):
        task = suite.get_task(task_id)
        bddl = source_root / "bddl_files" / task.problem_folder / task.bddl_file
        init_path = source_root / "init_files" / task.problem_folder / task.init_states_file
        initial_states = torch.load(init_path)
        # Development certification compares the exact MuJoCo integration
        # state and official success predicate. Camera rendering cannot change
        # either result, so leave it disabled for this simulator-only gate.
        source = build_env(bddl, int(cert["camera_resolution"]), use_camera_obs=False)
        twin_a = build_env(bddl, int(cert["camera_resolution"]), use_camera_obs=False)
        twin_b = build_env(bddl, int(cert["camera_resolution"]), use_camera_obs=False)
        try:
            for local_trial in range(int(cert["trials_per_task"])):
                init_index = local_trial
                for env in (source, twin_a, twin_b):
                    env.seed(int(config["experiment"]["seed"]) + task_id * 100 + local_trial)
                    env.reset()
                    env.set_init_state(initial_states[init_index])
                pre = rng.uniform(-0.10, 0.10, size=(int(cert["pre_branch_steps"]), 7))
                replay = rng.uniform(-0.10, 0.10, size=(int(cert["replay_steps"]), 7))
                pre[:, 6] = np.where(np.arange(len(pre)) % 7 < 4, -1.0, 1.0)
                replay[:, 6] = np.where(np.arange(len(replay)) % 9 < 5, -1.0, 1.0)
                for action in pre:
                    safe_env_step(source, action)
                snapshot = capture_snapshot(source)
                source_states = []
                source_predicates = []
                for action in replay:
                    safe_env_step(source, action)
                    source_states.append(capture_integration_state(source.sim))
                    source_predicates.append(bool(source.check_success()))

                twin_errors = []
                terminal_predicates = []
                all_finite = True
                for twin in (twin_a, twin_b):
                    restore_snapshot(twin, snapshot)
                    step_errors = []
                    for action, expected in zip(replay, source_states):
                        safe_env_step(twin, action)
                        state = capture_integration_state(twin.sim)
                        all_finite = all_finite and bool(np.isfinite(state).all())
                        step_errors.append(float(np.max(np.abs(state - expected), initial=0.0)))
                    twin_errors.append(step_errors)
                    terminal_predicates.append(bool(twin.check_success()))
                rows.append(
                    {
                        "trial_id": len(rows),
                        "task_id": task_id,
                        "task": task.language,
                        "init_state_index": init_index,
                        "source_vs_twin_a_max": max(twin_errors[0]),
                        "source_vs_twin_b_max": max(twin_errors[1]),
                        "twin_a_vs_twin_b_terminal_state_max": float(
                            np.max(
                                np.abs(
                                    capture_integration_state(twin_a.sim)
                                    - capture_integration_state(twin_b.sim)
                                ),
                                initial=0.0,
                            )
                        ),
                        "source_terminal_predicate": source_predicates[-1],
                        "twin_terminal_predicates": terminal_predicates,
                        "terminal_predicate_agreement": bool(
                            terminal_predicates[0] == terminal_predicates[1] == source_predicates[-1]
                        ),
                        "all_finite": all_finite,
                    }
                )
                print(
                    f"dev-certify task={task_id:02d} trial={local_trial:02d} "
                    f"max_error={max(twin_errors):.3e}",
                    flush=True,
                )
        finally:
            source.close()
            twin_a.close()
            twin_b.close()

    if len(rows) != int(cert["development_trials"]):
        raise RuntimeError(f"Expected {cert['development_trials']} trials, got {len(rows)}")
    discrepancies = np.asarray(
        [
            max(row["source_vs_twin_a_max"], row["source_vs_twin_b_max"], row["twin_a_vs_twin_b_terminal_state_max"])
            for row in rows
        ],
        dtype=np.float64,
    )
    median = float(np.median(discrepancies))
    p95 = float(np.quantile(discrepancies, 0.95))
    predicate_agreement = float(np.mean([row["terminal_predicate_agreement"] for row in rows]))
    all_finite = all(row["all_finite"] for row in rows)
    gate = predicate_agreement == 1.0 and all_finite
    results = {
        "created_at": now(),
        "development_only": True,
        "model_inference_used": False,
        "trials": len(rows),
        "tasks": suite.get_num_tasks(),
        "state_spec": "mujoco.mjtState.mjSTATE_INTEGRATION",
        "median_max_abs_discrepancy": median,
        "p95_max_abs_discrepancy": p95,
        "maximum_abs_discrepancy": float(discrepancies.max(initial=0.0)),
        "terminal_predicate_agreement": predicate_agreement,
        "all_values_finite": all_finite,
        "development_determinism_gate": "PASS" if gate else "FAIL",
        "rows": rows,
    }
    write_json(result_path, results)
    write_json(
        out / "wave19_snapshot_certification_preregistration.json",
        {
            "frozen_at": now(),
            "frozen_before_source_collection": True,
            "tolerance_uses_model_outputs": False,
            "development_trials": len(rows),
            "median_tolerance": median,
            "p95_tolerance": p95,
            "required_terminal_predicate_agreement": 1.0,
            "required_all_finite": True,
            "candidate_branch_rule": "source episode fractions 0.25/0.50/0.75 with >=128 future steps",
            "certified_branch_rule": (
                "100% twin and restored-vs-source terminal-predicate agreement; finite states; episode median and "
                "P95 max-absolute integration-state discrepancy <= frozen development tolerances"
            ),
        },
    )
    report = f"""# Wave-19 development snapshot certification

The simulator-only development gate used {len(rows)} restore-and-replay trials across all 10 official
`libero_10` tasks. No π0.5, representation, F1, or F2 output was used.

- median segment maximum discrepancy: `{median:.17g}`
- P95 segment maximum discrepancy: `{p95:.17g}`
- maximum discrepancy: `{discrepancies.max(initial=0.0):.17g}`
- official terminal-predicate agreement: `{predicate_agreement:.3f}`
- all values finite: `{all_finite}`
- gate: `{'PASS' if gate else 'FAIL'}`

The frozen certification tolerances are empirical simulator-determinism tolerances and cannot be changed after
source-policy or F1/F2 outputs are observed.
"""
    write_text(out / "wave19_snapshot_certification_report.md", report)
    if not gate:
        write_text(
            out / "wave19_reconstruction_gate_failure.md",
            "# Wave-19 reconstruction gate failure\n\nThe development repeated-restore gate failed before π0.5 collection. "
            "Per protocol, source collection and all model training are forbidden.",
        )
        raise RuntimeError("Development snapshot determinism gate failed")
    append_command(
        config,
        "PYTHONPATH=src:/home/jinjaguo/LIBERO MUJOCO_GL=egl "
        "/home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_7.py "
        "--config configs/dynamics_7.yaml --stage dev-certify",
        "dev-certify",
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.stage == "audit":
        audit(config)
    elif args.stage == "audit-resume":
        resume_audit(config)
    else:
        dev_certify(config)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Collect and certify prospective π0.5 trajectories on official LIBERO-10.

Purpose
-------
Run the fixed official ``pi05_libero`` source policy on every task in the
installed official ``libero_10`` suite, retain every success and failure, save
an exact MuJoCo/controller snapshot at every control boundary, certify fixed
25/50/75-percent branches by twin replay, and freeze an episode-disjoint
60/20/20 dataset manifest.

Parameters
----------
``--config`` selects the frozen Wave-19 YAML. ``--host`` and ``--port`` select
the already-running Wave-19 π0.5 websocket server. Collection scale, seeds,
tasks, horizons, and branch rules come only from the frozen YAML/preregistration.

Usage
-----
PYTHONPATH=src:/home/jinjaguo/LIBERO:/home/jinjaguo/openpi/packages/openpi-client/src \
  MUJOCO_GL=egl /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/collect_wave19_libero.py \
  --config configs/dynamics_7.yaml --host localhost --port 8000

Outputs
-------
Immutable raw episodes are finalized below
``data/wave19_libero_branchable/raw_collection``. Certified episodes/branches
are written below ``data/wave19_libero_branchable/certified``. Collection,
snapshot-certification, and split manifests/reports are written below
``results/dynamics/nineteenth_wave/2026-08-14_dynamics_7``.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy

from pglt.libero.snapshot import (
    LiberoSnapshot,
    capture_snapshot,
    physical_state,
    restore_snapshot,
    safe_env_step,
)


ROOT = Path(__file__).resolve().parents[2]
DUMMY_ACTION = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    resolved = path if path.is_absolute() else ROOT / path
    return yaml.safe_load(resolved.read_text(encoding="utf-8"))


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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def disk_guard(config: Mapping[str, Any], phase: str) -> None:
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
    append_jsonl(ROOT / config["experiment"]["data_root"] / "audits/disk_usage_log.jsonl", record)
    if not record["minimum_pass"]:
        raise RuntimeError(f"Free disk {record['free_bytes']} is below the frozen 300 GB minimum")


def quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    value = np.asarray(quat, dtype=np.float64).copy()
    value[3] = np.clip(value[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - value[3] * value[3])
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3, dtype=np.float64)
    return value[:3] * 2.0 * math.acos(float(value[3])) / denominator


def processed_image(image: np.ndarray, size: int) -> np.ndarray:
    rotated = np.ascontiguousarray(image[::-1, ::-1])
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(rotated, size, size))


def policy_observation(obs: Mapping[str, np.ndarray], instruction: str, size: int) -> dict[str, Any]:
    return {
        "observation/image": processed_image(obs["agentview_image"], size),
        "observation/wrist_image": processed_image(obs["robot0_eye_in_hand_image"], size),
        "observation/state": np.concatenate(
            (
                np.asarray(obs["robot0_eef_pos"]),
                quat_to_axisangle(np.asarray(obs["robot0_eef_quat"])),
                np.asarray(obs["robot0_gripper_qpos"]),
            )
        ),
        "prompt": instruction,
    }


def build_env(bddl: Path, resolution: int, *, use_camera_obs: bool) -> OffScreenRenderEnv:
    return OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=resolution,
        camera_widths=resolution,
        use_camera_obs=use_camera_obs,
    )


@dataclass
class EpisodeRuntime:
    task_id: int
    attempt: int
    episode_id: str
    instruction: str
    bddl: Path
    init_path: Path
    init_index: int
    env_seed: int
    task_seed: int
    env: Any
    obs: Mapping[str, np.ndarray]
    initial_state_hash: str
    snapshots: list[LiberoSnapshot] = field(default_factory=list)
    physical: list[dict[str, np.ndarray]] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    policy_issue_for_action: list[int] = field(default_factory=list)
    policy_offset_for_action: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    successes: list[bool] = field(default_factory=list)
    action_plan: deque[tuple[np.ndarray, int, int]] = field(default_factory=deque)
    issue_steps: list[int] = field(default_factory=list)
    raw_agent_images: list[np.ndarray] = field(default_factory=list)
    raw_wrist_images: list[np.ndarray] = field(default_factory=list)
    processed_agent_images: list[np.ndarray] = field(default_factory=list)
    processed_wrist_images: list[np.ndarray] = field(default_factory=list)
    policy_states: list[np.ndarray] = field(default_factory=list)
    raw_model_chunks: list[np.ndarray] = field(default_factory=list)
    policy_action_chunks: list[np.ndarray] = field(default_factory=list)
    policy_infer_ms: list[float] = field(default_factory=list)
    batch_sizes: list[int] = field(default_factory=list)
    padding_requests: int = 0
    terminal_success: bool = False
    failure_reason: str | None = None
    exception_traceback: str | None = None

    @property
    def step(self) -> int:
        return len(self.actions)


def begin_episode(
    *,
    task_id: int,
    attempt: int,
    task: Any,
    source_root: Path,
    initial_states: torch.Tensor,
    config: Mapping[str, Any],
) -> EpisodeRuntime:
    collection = config["collection"]
    bddl = source_root / "bddl_files" / task.problem_folder / task.bddl_file
    init_path = source_root / "init_files" / task.problem_folder / task.init_states_file
    init_index = attempt
    if init_index >= len(initial_states):
        raise RuntimeError(f"Attempt {attempt} exceeds official init-state count {len(initial_states)}")
    env_seed = int(config["experiment"]["seed"]) + task_id * 10_000 + attempt
    task_seed = int(config["experiment"]["seed"]) + 1_000_000 + task_id * 10_000 + attempt
    env = build_env(bddl, int(collection["camera_resolution"]), use_camera_obs=True)
    env.seed(env_seed)
    env.reset()
    init_value = initial_states[init_index]
    obs = env.set_init_state(init_value)
    runtime = EpisodeRuntime(
        task_id=task_id,
        attempt=attempt,
        episode_id=f"task{task_id:02d}_attempt{attempt:03d}",
        instruction=task.language,
        bddl=bddl,
        init_path=init_path,
        init_index=init_index,
        env_seed=env_seed,
        task_seed=task_seed,
        env=env,
        obs=obs,
        initial_state_hash=sha256_bytes(np.asarray(init_value).tobytes()),
    )
    runtime.snapshots.append(capture_snapshot(env))
    runtime.physical.append(physical_state(env))
    return runtime


def execute_action(runtime: EpisodeRuntime, action: np.ndarray, issue: int, offset: int) -> None:
    saved = np.asarray(action, dtype=np.float64).copy()
    obs, reward, done, _, success, snapshot, physical = step_predicate_snapshot(runtime.env, saved)
    runtime.actions.append(saved)
    runtime.policy_issue_for_action.append(issue)
    runtime.policy_offset_for_action.append(offset)
    runtime.rewards.append(float(reward))
    runtime.dones.append(bool(done))
    runtime.successes.append(success)
    runtime.obs = obs
    runtime.snapshots.append(snapshot)
    runtime.physical.append(physical)
    runtime.terminal_success = success


def step_predicate_snapshot(env: Any, action: np.ndarray) -> tuple[Any, float, bool, dict[str, Any], bool, Any, dict[str, np.ndarray]]:
    """Step once and preserve the source ordering of predicate, canonical snapshot, and diagnostics."""

    obs, reward, done, info = safe_env_step(env, action)
    success = bool(env.check_success())
    snapshot = capture_snapshot(env)
    physical = physical_state(env)
    return obs, float(reward), bool(done), info, success, snapshot, physical


def add_policy_result(
    runtime: EpisodeRuntime,
    request: Mapping[str, Any],
    result: Mapping[str, Any],
    replan_steps: int,
) -> None:
    actions = np.asarray(result["actions"], dtype=np.float64)
    raw = np.asarray(result["raw_model_actions"], dtype=np.float32)
    if actions.shape != (10, 7) or raw.shape != (10, 32):
        raise RuntimeError(f"Unexpected π0.5 output shapes: actions={actions.shape}, raw={raw.shape}")
    if not np.isfinite(actions).all() or not np.isfinite(raw).all():
        raise FloatingPointError("Nonfinite π0.5 source-policy output")
    issue_index = len(runtime.issue_steps)
    runtime.issue_steps.append(runtime.step)
    runtime.raw_agent_images.append(np.asarray(runtime.obs["agentview_image"]).copy())
    runtime.raw_wrist_images.append(np.asarray(runtime.obs["robot0_eye_in_hand_image"]).copy())
    runtime.processed_agent_images.append(np.asarray(request["observation/image"]).copy())
    runtime.processed_wrist_images.append(np.asarray(request["observation/wrist_image"]).copy())
    runtime.policy_states.append(np.asarray(request["observation/state"]).copy())
    runtime.raw_model_chunks.append(raw.copy())
    runtime.policy_action_chunks.append(actions.copy())
    timing = result.get("policy_timing", {})
    runtime.policy_infer_ms.append(float(timing.get("infer_ms", np.nan)))
    runtime.batch_sizes.append(int(timing.get("batch_size", 1)))
    for offset in range(replan_steps):
        runtime.action_plan.append((actions[offset].copy(), issue_index, offset))


def stack_or_empty(values: Sequence[np.ndarray], shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    if not values:
        return np.empty((0, *shape), dtype=dtype)
    return np.stack(values).astype(dtype, copy=False)


def stacked_physical(physical: Sequence[Mapping[str, np.ndarray]], keys: Sequence[str]) -> dict[str, np.ndarray]:
    return {key: np.stack([np.asarray(state[key]) for state in physical]) for key in keys}


ROBOT_KEYS = (
    "qpos",
    "qvel",
    "ctrl",
    "robot_joint_qpos",
    "robot_joint_qvel",
    "eef_pos",
    "eef_ori",
    "gripper_qpos",
    "gripper_qvel",
)
OBJECT_KEYS = ("body_xpos", "body_xquat", "body_cvel")


def finalize_raw(runtime: EpisodeRuntime, config: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    data = ROOT / config["experiment"]["data_root"]
    task_root = data / "raw_collection" / f"task_{runtime.task_id:02d}"
    partial = task_root / ".partial" / runtime.episode_id
    outcome = "successes" if runtime.terminal_success else "failures"
    final = task_root / outcome / runtime.episode_id
    if final.exists() or partial.exists():
        raise RuntimeError(f"Raw episode path already exists: {final} or {partial}")
    partial.mkdir(parents=True)
    actions = stack_or_empty(runtime.actions, (7,), np.float64)
    integration = np.stack([snapshot.integration_state for snapshot in runtime.snapshots])
    metadata = {
        "finalized_at": now(),
        "episode_id": runtime.episode_id,
        "task_id": runtime.task_id,
        "attempt": runtime.attempt,
        "instruction": runtime.instruction,
        "bddl_path": str(runtime.bddl),
        "init_states_path": str(runtime.init_path),
        "official_init_state_index": runtime.init_index,
        "environment_seed": runtime.env_seed,
        "task_seed": runtime.task_seed,
        "policy_seed": int(config["experiment"]["seed"]),
        "initial_official_state_sha256": runtime.initial_state_hash,
        "initial_exact_integration_state_sha256": sha256_bytes(integration[0].tobytes()),
        "steps": len(actions),
        "wait_steps": int(config["collection"]["wait_steps"]),
        "policy_issue_count": len(runtime.issue_steps),
        "terminal_done": bool(runtime.dones[-1]) if runtime.dones else False,
        "terminal_official_success": runtime.terminal_success,
        "failure_reason": runtime.failure_reason,
        "exception_traceback": runtime.exception_traceback,
        "raw_collection_immutable_after_finalization": True,
        "simulator_state_spec": "mujoco.mjtState.mjSTATE_INTEGRATION",
        "policy_observation_camera_archive_saved_only_at_policy_issue_times": True,
        "exact_snapshot_includes_complete_observable_payload": True,
    }
    write_json(partial / "episode_metadata.json", metadata)
    write_text(partial / "instruction.txt", runtime.instruction)
    np.save(partial / "actions.npy", actions)
    np.save(partial / "integration_states.npy", integration)
    np.save(partial / "policy_issue_for_action.npy", np.asarray(runtime.policy_issue_for_action, dtype=np.int32))
    np.save(partial / "policy_offset_for_action.npy", np.asarray(runtime.policy_offset_for_action, dtype=np.int16))
    np.savez_compressed(
        partial / "step_outcomes.npz",
        rewards=np.asarray(runtime.rewards, dtype=np.float64),
        done=np.asarray(runtime.dones, dtype=np.bool_),
        success=np.asarray(runtime.successes, dtype=np.bool_),
    )
    np.savez_compressed(partial / "robot_states.npz", **stacked_physical(runtime.physical, ROBOT_KEYS))
    np.savez_compressed(partial / "object_states.npz", **stacked_physical(runtime.physical, OBJECT_KEYS))
    np.savez_compressed(
        partial / "policy_observations.npz",
        issue_steps=np.asarray(runtime.issue_steps, dtype=np.int32),
        raw_agentview=stack_or_empty(
            runtime.raw_agent_images,
            (int(config["collection"]["camera_resolution"]), int(config["collection"]["camera_resolution"]), 3),
            np.uint8,
        ),
        raw_wrist=stack_or_empty(
            runtime.raw_wrist_images,
            (int(config["collection"]["camera_resolution"]), int(config["collection"]["camera_resolution"]), 3),
            np.uint8,
        ),
        processed_agentview=stack_or_empty(
            runtime.processed_agent_images,
            (int(config["collection"]["policy_resize"]), int(config["collection"]["policy_resize"]), 3),
            np.uint8,
        ),
        processed_wrist=stack_or_empty(
            runtime.processed_wrist_images,
            (int(config["collection"]["policy_resize"]), int(config["collection"]["policy_resize"]), 3),
            np.uint8,
        ),
        state=stack_or_empty(runtime.policy_states, (8,), np.float64),
    )
    np.save(partial / "raw_model_action_chunks.npy", stack_or_empty(runtime.raw_model_chunks, (10, 32), np.float32))
    np.save(
        partial / "postprocessed_policy_action_chunks.npy",
        stack_or_empty(runtime.policy_action_chunks, (10, 7), np.float64),
    )
    np.savez_compressed(
        partial / "policy_timing.npz",
        infer_ms=np.asarray(runtime.policy_infer_ms, dtype=np.float64),
        batch_size=np.asarray(runtime.batch_sizes, dtype=np.int16),
        padding_requests=np.asarray(runtime.padding_requests, dtype=np.int32),
    )
    with (partial / "exact_snapshots.pkl").open("wb") as handle:
        pickle.dump(runtime.snapshots, handle, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(
        partial / "FINALIZED.json",
        {
            "finalized_at": metadata["finalized_at"],
            "episode_id": runtime.episode_id,
            "outcome": outcome,
            "files_complete": True,
        },
    )
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, final)
    return final, metadata


def numeric_discrepancy(left: Any, right: Any) -> float:
    if isinstance(left, Mapping):
        if set(left) != set(right):
            return float("inf")
        return max((numeric_discrepancy(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, (tuple, list)):
        if len(left) != len(right):
            return float("inf")
        return max((numeric_discrepancy(a, b) for a, b in zip(left, right)), default=0.0)
    if isinstance(left, (np.ndarray, np.generic, float, int, bool)):
        a = np.asarray(left)
        b = np.asarray(right)
        if a.shape != b.shape:
            return float("inf")
        if a.dtype.kind in "OUS" or b.dtype.kind in "OUS":
            return 0.0 if np.array_equal(a, b) else float("inf")
        return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)), initial=0.0))
    return 0.0 if left == right else float("inf")


def branch_steps(total_steps: int, wait_steps: int, fractions: Sequence[float], minimum_future: int) -> list[dict[str, Any]]:
    policy_steps = total_steps - wait_steps
    rows = []
    for fraction in fractions:
        step = wait_steps + int(math.floor(policy_steps * float(fraction)))
        rows.append(
            {
                "fraction": float(fraction),
                "step": step,
                "past_policy_steps": step - wait_steps,
                "future_steps": total_steps - step,
                "eligible": step - wait_steps >= 32 and total_steps - step >= minimum_future,
            }
        )
    return rows


def certify_branch(
    runtime: EpisodeRuntime,
    row: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    step = int(row["step"])
    resolution = int(config["collection"]["camera_resolution"])
    twin_a = build_env(runtime.bddl, resolution, use_camera_obs=True)
    twin_b = build_env(runtime.bddl, resolution, use_camera_obs=True)
    state_errors: list[float] = []
    controller_errors: list[float] = []
    object_errors: list[float] = []
    predicate_agreement: list[bool] = []
    terminal_predicates: list[bool] = []
    nonfinite = 0
    try:
        for twin in (twin_a, twin_b):
            twin.seed(runtime.env_seed)
            twin.reset()
            restore_snapshot(twin, runtime.snapshots[step])
            last_predicate = False
            for action_index in range(step, len(runtime.actions)):
                _, _, _, _, last_predicate, actual, actual_physical = step_predicate_snapshot(
                    twin, runtime.actions[action_index]
                )
                expected = runtime.snapshots[action_index + 1]
                state_error = float(
                    np.max(np.abs(actual.integration_state - expected.integration_state), initial=0.0)
                )
                controller_error = numeric_discrepancy(actual.controller_state, expected.controller_state)
                expected_physical = runtime.physical[action_index + 1]
                object_error = max(
                    float(np.max(np.abs(actual_physical[key] - expected_physical[key]), initial=0.0))
                    for key in OBJECT_KEYS
                )
                state_errors.append(state_error)
                controller_errors.append(controller_error)
                object_errors.append(object_error)
                finite = all(
                    np.isfinite(value).all()
                    for value in (actual.integration_state, *actual_physical.values())
                )
                nonfinite += int(not finite)
                predicate_agreement.append(last_predicate == runtime.successes[action_index])
            terminal_predicates.append(last_predicate)
    finally:
        twin_a.close()
        twin_b.close()
    frozen = json.loads(
        (ROOT / config["experiment"]["output_root"] / "wave19_snapshot_certification_preregistration.json").read_text(
            encoding="utf-8"
        )
    )
    state_array = np.asarray(state_errors, dtype=np.float64)
    median = float(np.median(state_array))
    p95 = float(np.quantile(state_array, 0.95))
    passed = bool(
        nonfinite == 0
        and all(predicate_agreement)
        and terminal_predicates == [runtime.terminal_success, runtime.terminal_success]
        and median <= float(frozen["median_tolerance"])
        and p95 <= float(frozen["p95_tolerance"])
        and max(controller_errors, default=0.0) == 0.0
        and max(object_errors, default=0.0) <= float(frozen["p95_tolerance"])
    )
    result = {
        **dict(row),
        "episode_id": runtime.episode_id,
        "task_id": runtime.task_id,
        "state_discrepancy_median": median,
        "state_discrepancy_p95": p95,
        "state_discrepancy_max": float(state_array.max(initial=0.0)),
        "controller_discrepancy_max": max(controller_errors, default=0.0),
        "object_discrepancy_max": max(object_errors, default=0.0),
        "predicate_agreement": bool(all(predicate_agreement)),
        "terminal_predicates": terminal_predicates,
        "source_terminal_success": runtime.terminal_success,
        "nonfinite_count": nonfinite,
        "replay_success_agreement": terminal_predicates == [runtime.terminal_success, runtime.terminal_success],
        "certified": passed,
    }
    return passed, result


def materialize_certified(
    runtime: EpisodeRuntime,
    raw_path: Path,
    metadata: Mapping[str, Any],
    branch_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Path:
    data = ROOT / config["experiment"]["data_root"]
    final = data / "certified" / f"task_{runtime.task_id:02d}" / runtime.episode_id
    partial = final.parent / f".{runtime.episode_id}.partial"
    if final.exists() or partial.exists():
        raise RuntimeError(f"Certified episode path already exists: {final} or {partial}")
    partial.mkdir(parents=True)
    certified_metadata = {
        **dict(metadata),
        "raw_source_path": str(raw_path.relative_to(ROOT)),
        "certified_at": now(),
        "eligible_branch_count": len(branch_rows),
        "all_eligible_branches_certified": True,
        "action_step_offset": int(config["collection"]["wait_steps"]),
    }
    certified_metadata.pop("camera_observations_saved_only_at_policy_issue_times", None)
    certified_metadata["policy_observation_camera_archive_saved_only_at_policy_issue_times"] = True
    certified_metadata["exact_snapshot_includes_complete_observable_payload"] = True
    write_json(partial / "episode_metadata.json", certified_metadata)
    write_text(partial / "instruction.txt", runtime.instruction)
    wait = int(config["collection"]["wait_steps"])
    np.save(partial / "actions.npy", np.stack(runtime.actions[wait:]))
    np.savez_compressed(partial / "robot_states.npz", **stacked_physical(runtime.physical[wait:], ROBOT_KEYS))
    np.savez_compressed(partial / "object_states.npz", **stacked_physical(runtime.physical[wait:], OBJECT_KEYS))
    source = partial / "source_continuation"
    source.mkdir()
    np.save(source / "actions.npy", np.stack(runtime.actions[wait:]))
    np.save(source / "integration_states.npy", np.stack([s.integration_state for s in runtime.snapshots[wait:]]))
    write_json(source / "terminal_success.json", {"official_success": runtime.terminal_success})
    for result in branch_rows:
        percent = int(round(float(result["fraction"]) * 100))
        branch = partial / "branches" / f"branch_{percent:03d}"
        causal = branch / "causal"
        reference = branch / "reference_only"
        causal.mkdir(parents=True)
        reference.mkdir(parents=True)
        step = int(result["step"])
        branch_metadata = {
            **dict(result),
            "source_episode_id": runtime.episode_id,
            "selection_rule": "wait_steps + floor((terminal_steps - wait_steps) * frozen_fraction)",
            "selected_without_F1_F2_outputs": True,
            "official_success_predicate_used": True,
        }
        write_json(branch / "branch_metadata.json", branch_metadata)
        np.savez_compressed(branch / "exact_sim_state.npz", integration_state=runtime.snapshots[step].integration_state)
        with (branch / "controller_state.pkl").open("wb") as handle:
            pickle.dump(runtime.snapshots[step].controller_state, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with (branch / "exact_snapshot.pkl").open("wb") as handle:
            pickle.dump(runtime.snapshots[step], handle, protocol=pickle.HIGHEST_PROTOCOL)
        np.save(causal / "past_actions.npy", np.stack(runtime.actions[wait:step]))
        write_text(causal / "current_instruction.txt", runtime.instruction)
        write_json(causal / "issue_step.json", {"environment_step": step, "policy_step": step - wait})
        write_json(
            causal / "causal_availability.json",
            {
                "available": ["past_actions", "current_instruction", "issue_step"],
                "forbidden": ["future_actions", "future_state", "future_success", "reference_only"],
            },
        )
        np.save(reference / "future_actions.npy", np.stack(runtime.actions[step:]))
        np.savez_compressed(
            reference / "future_robot_states.npz", **stacked_physical(runtime.physical[step + 1 :], ROBOT_KEYS)
        )
        np.savez_compressed(
            reference / "future_object_states.npz", **stacked_physical(runtime.physical[step + 1 :], OBJECT_KEYS)
        )
        write_json(reference / "source_terminal_success.json", {"official_success": runtime.terminal_success})
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, final)
    return final


def certify_success(
    runtime: EpisodeRuntime,
    raw_path: Path,
    metadata: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[bool, list[dict[str, Any]]]:
    candidates = branch_steps(
        len(runtime.actions),
        int(config["collection"]["wait_steps"]),
        config["collection"]["branch_fractions"],
        int(config["collection"]["minimum_future_steps"]),
    )
    eligible = [row for row in candidates if row["eligible"]]
    if not eligible:
        return False, [{**row, "episode_id": runtime.episode_id, "task_id": runtime.task_id} for row in candidates]
    results = []
    for candidate in eligible:
        passed, result = certify_branch(runtime, candidate, config)
        results.append(result)
        print(
            f"certify task={runtime.task_id:02d} attempt={runtime.attempt:03d} "
            f"fraction={candidate['fraction']:.2f} state_max={result['state_discrepancy_max']:.3e} "
            f"controller_max={result['controller_discrepancy_max']:.3e} pass={passed}",
            flush=True,
        )
        if not passed:
            out = ROOT / config["experiment"]["output_root"]
            write_json(out / "wave19_snapshot_certification_failure.json", result)
            write_text(
                out / "wave19_reconstruction_gate_failure.md",
                "# Wave-19 reconstruction gate failure\n\nA prospective source branch failed the frozen exact-state "
                "certification. Per protocol, collection and all downstream model work stopped.",
            )
            raise RuntimeError(f"Snapshot certification failed for {runtime.episode_id}")
    materialize_certified(runtime, raw_path, metadata, results, config)
    return True, results


def existing_attempts(data: Path, task_id: int) -> set[int]:
    attempts = set()
    root = data / "raw_collection" / f"task_{task_id:02d}"
    for outcome in ("successes", "failures"):
        for metadata in (root / outcome).glob("*/episode_metadata.json"):
            attempts.add(int(json.loads(metadata.read_text(encoding="utf-8"))["attempt"]))
    return attempts


def existing_certified(data: Path, task_id: int) -> list[str]:
    root = data / "certified" / f"task_{task_id:02d}"
    return sorted(path.name for path in root.glob("task*_attempt*") if (path / "episode_metadata.json").is_file())


def load_finalized_runtime(raw_path: Path) -> tuple[EpisodeRuntime, dict[str, Any]]:
    """Load immutable raw source data for certification-only crash recovery."""

    metadata = json.loads((raw_path / "episode_metadata.json").read_text(encoding="utf-8"))
    with (raw_path / "exact_snapshots.pkl").open("rb") as handle:
        snapshots = pickle.load(handle)
    actions = np.load(raw_path / "actions.npy")
    outcomes = np.load(raw_path / "step_outcomes.npz")
    robot = np.load(raw_path / "robot_states.npz")
    objects = np.load(raw_path / "object_states.npz")
    physical = [
        {
            **{key: np.asarray(robot[key][index]).copy() for key in ROBOT_KEYS},
            **{key: np.asarray(objects[key][index]).copy() for key in OBJECT_KEYS},
        }
        for index in range(len(snapshots))
    ]
    runtime = EpisodeRuntime(
        task_id=int(metadata["task_id"]),
        attempt=int(metadata["attempt"]),
        episode_id=str(metadata["episode_id"]),
        instruction=str(metadata["instruction"]),
        bddl=Path(metadata["bddl_path"]),
        init_path=Path(metadata["init_states_path"]),
        init_index=int(metadata["official_init_state_index"]),
        env_seed=int(metadata["environment_seed"]),
        task_seed=int(metadata["task_seed"]),
        env=None,
        obs={},
        initial_state_hash=str(metadata["initial_official_state_sha256"]),
        snapshots=snapshots,
        physical=physical,
        actions=[np.asarray(action).copy() for action in actions],
        successes=[bool(value) for value in outcomes["success"]],
        terminal_success=bool(metadata["terminal_official_success"]),
    )
    return runtime, metadata


def run_pair(
    *,
    task_id: int,
    attempts: Sequence[int],
    task: Any,
    source_root: Path,
    initial_states: torch.Tensor,
    client: WebsocketClientPolicy,
    config: Mapping[str, Any],
) -> list[EpisodeRuntime]:
    runtimes = [
        begin_episode(
            task_id=task_id,
            attempt=attempt,
            task=task,
            source_root=source_root,
            initial_states=initial_states,
            config=config,
        )
        for attempt in attempts
    ]
    collection = config["collection"]
    wait_steps = int(collection["wait_steps"])
    max_total_steps = wait_steps + int(collection["max_steps_libero_10"])
    replan = int(collection["policy_replan_steps"])
    resize = int(collection["policy_resize"])
    active = list(runtimes)
    while active:
        for runtime in list(active):
            if runtime.step < wait_steps:
                try:
                    execute_action(runtime, DUMMY_ACTION, -1, -1)
                except Exception:
                    runtime.failure_reason = "exception_during_wait"
                    runtime.exception_traceback = traceback.format_exc()
                    active.remove(runtime)
        needing = [runtime for runtime in active if runtime.step >= wait_steps and not runtime.action_plan]
        if needing:
            requests = [policy_observation(runtime.obs, runtime.instruction, resize) for runtime in needing]
            request_runtimes = list(needing)
            if len(requests) == 1:
                requests.append(requests[0])
                request_runtimes[0].padding_requests += 1
            response = client.infer({"wave19_observation_batch": requests})
            results = response["wave19_batch_results"]
            if len(results) != len(requests):
                raise RuntimeError("π0.5 batch response length differs from request length")
            for runtime, request, result in zip(request_runtimes, requests, results):
                add_policy_result(runtime, request, result, replan)
        for runtime in list(active):
            if runtime.step >= max_total_steps:
                runtime.failure_reason = "official_horizon_exhausted"
                active.remove(runtime)
                continue
            if runtime.step < wait_steps:
                continue
            try:
                action, issue, offset = runtime.action_plan.popleft()
                execute_action(runtime, action, issue, offset)
            except Exception:
                runtime.failure_reason = "exception_during_policy_rollout"
                runtime.exception_traceback = traceback.format_exc()
                active.remove(runtime)
                continue
            if runtime.terminal_success:
                active.remove(runtime)
            elif runtime.step >= max_total_steps:
                runtime.failure_reason = "official_horizon_exhausted"
                active.remove(runtime)
    return runtimes


def directory_episode_hash(path: Path) -> tuple[str, dict[str, str]]:
    selected = sorted(item for item in path.iterdir() if item.is_file())
    hashes = {item.name: sha256_file(item) for item in selected}
    aggregate = sha256_bytes("\n".join(f"{name} {value}" for name, value in sorted(hashes.items())).encode())
    return aggregate, hashes


def freeze_dataset(config: Mapping[str, Any], collection_rows: Sequence[Mapping[str, Any]]) -> None:
    out = ROOT / config["experiment"]["output_root"]
    data = ROOT / config["experiment"]["data_root"]
    target = int(config["collection"]["successes_per_task_target"])
    certified_by_task = {task_id: existing_certified(data, task_id) for task_id in range(10)}
    primary_per_task = min(target, *(len(values) for values in certified_by_task.values()))
    minimum = int(config["collection"]["successes_per_task_minimum"])
    if primary_per_task < minimum:
        raise RuntimeError(f"Only {primary_per_task} balanced certified successes per task; minimum is {minimum}")
    rng = np.random.default_rng(int(config["experiment"]["seed"]))
    episodes = []
    splits = {"train": [], "development": [], "test": []}
    for task_id, values in certified_by_task.items():
        selected = values[:primary_per_task]
        shuffled = list(selected)
        rng.shuffle(shuffled)
        train_count = int(round(primary_per_task * float(config["split"]["train_fraction"])))
        dev_count = int(round(primary_per_task * float(config["split"]["development_fraction"])))
        assignment = {
            "train": shuffled[:train_count],
            "development": shuffled[train_count : train_count + dev_count],
            "test": shuffled[train_count + dev_count :],
        }
        for split, episode_ids in assignment.items():
            for episode_id in episode_ids:
                certified_path = data / "certified" / f"task_{task_id:02d}" / episode_id
                metadata = json.loads((certified_path / "episode_metadata.json").read_text(encoding="utf-8"))
                raw_path = ROOT / metadata["raw_source_path"]
                episode_hash, file_hashes = directory_episode_hash(raw_path)
                branch_hashes = {
                    branch.parent.name: sha256_file(branch)
                    for branch in sorted((certified_path / "branches").glob("*/branch_metadata.json"))
                }
                row = {
                    "episode_id": episode_id,
                    "task_id": task_id,
                    "split": split,
                    "raw_path": str(raw_path.relative_to(ROOT)),
                    "certified_path": str(certified_path.relative_to(ROOT)),
                    "source_episode_sha256": episode_hash,
                    "source_file_sha256": file_hashes,
                    "branch_metadata_sha256": branch_hashes,
                }
                episodes.append(row)
                splits[split].append(episode_id)
    if len({row["episode_id"] for row in episodes}) != len(episodes):
        raise RuntimeError("Episode ID appears more than once in dataset split")
    write_json(
        out / "wave19_dataset_manifest.json",
        {
            "frozen_at": now(),
            "suite": "libero_10",
            "primary_balanced_episodes_per_task": primary_per_task,
            "total_primary_episodes": len(episodes),
            "episodes": episodes,
            "raw_collection_outcomes": list(collection_rows),
        },
    )
    write_json(
        out / "wave19_dataset_split_manifest.json",
        {
            "frozen_at": now(),
            "unit": "source_episode",
            "stratified_by_task": True,
            "episode_disjoint": True,
            "branch_inherits_source_episode_split": True,
            "final_test_unread_for_model_selection": True,
            "seed": int(config["experiment"]["seed"]),
            "counts": {key: len(value) for key, value in splits.items()},
            "assignments": splits,
            "episodes": episodes,
        },
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    out = ROOT / config["experiment"]["output_root"]
    data = ROOT / config["experiment"]["data_root"]
    if not (out / "wave19_snapshot_certification_preregistration.json").is_file():
        raise RuntimeError("Development snapshot gate must be frozen before source collection")
    if (out / "wave19_dataset_split_manifest.json").exists():
        raise RuntimeError("Wave-19 dataset is already frozen; refusing additional collection")
    disk_guard(config, "source-collection-start")
    execution_manifest = {
        "frozen_at": now(),
        "written_before_formal_source_collection": True,
        "gpu_batch_size": 2,
        "batching_changes_only_tensor_scheduling": True,
        "policy_replan_steps": int(config["collection"]["policy_replan_steps"]),
        "branch_progress_origin": "first post-wait source-policy control step",
        "branch_step_rule": "wait_steps + floor((terminal_steps - wait_steps) * fraction)",
        "causal_history_minimum_steps": 32,
        "minimum_future_steps": int(config["collection"]["minimum_future_steps"]),
        "global_policy_sampling_seed": int(config["experiment"]["seed"]),
        "batch_padding_result_discarded_when_only_one_episode_active": True,
    }
    execution_path = out / "wave19_collection_execution_manifest.json"
    if not execution_path.exists():
        write_json(execution_path, execution_manifest)
    with (out / "exact_commands.sh").open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n# {now()} phase=source-collection-start-or-resume\n"
            "PYTHONPATH=src:/home/jinjaguo/LIBERO:/home/jinjaguo/openpi/packages/openpi-client/src "
            "MUJOCO_GL=egl /home/jinjaguo/anaconda3/envs/libero/bin/python "
            "scripts/dynamics/collect_wave19_libero.py --config configs/dynamics_7.yaml "
            f"--host {args.host} --port {args.port}\n"
        )
    client = WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata()
    if metadata.get("wave19_batch_inference") is not True:
        raise RuntimeError("Connected server does not expose frozen Wave-19 batch inference")
    if int(metadata.get("wave19_policy_seed", -1)) != int(config["experiment"]["seed"]):
        raise RuntimeError("Connected source-policy seed differs from frozen Wave-19 seed")
    suite = benchmark.get_benchmark_dict()[config["sources"]["official_suite"]]()
    source_root = Path(config["sources"]["libero_root"]) / "libero/libero"
    target = int(config["collection"]["successes_per_task_target"])
    minimum = int(config["collection"]["successes_per_task_minimum"])
    maximum_attempts = int(config["collection"]["attempts_per_task_maximum"])
    collection_rows: list[dict[str, Any]] = []
    certification_rows: list[dict[str, Any]] = []
    for task_id in range(suite.get_num_tasks()):
        task = suite.get_task(task_id)
        init_path = source_root / "init_files" / task.problem_folder / task.init_states_file
        initial_states = torch.load(init_path)
        attempted = existing_attempts(data, task_id)
        certified = existing_certified(data, task_id)
        raw_success_root = data / "raw_collection" / f"task_{task_id:02d}" / "successes"
        for raw_path in sorted(raw_success_root.glob("task*_attempt*")):
            if raw_path.name in certified:
                continue
            runtime, episode_metadata = load_finalized_runtime(raw_path)
            certified_episode, branch_results = certify_success(runtime, raw_path, episode_metadata, config)
            certification_rows.extend(branch_results)
            if certified_episode:
                certified.append(runtime.episode_id)
            if "camera_observations_saved_only_at_policy_issue_times" in episode_metadata:
                append_jsonl(
                    data / "audits/raw_metadata_corrections.jsonl",
                    {
                        "recorded_at": now(),
                        "episode_id": runtime.episode_id,
                        "immutable_raw_metadata_key": "camera_observations_saved_only_at_policy_issue_times",
                        "correction": (
                            "policy_observations.npz stores cameras only at issue times; exact_snapshots.pkl "
                            "also preserves the complete observable payload at every control boundary"
                        ),
                    },
                )
            row = {
                "recorded_at": now(),
                "task_id": task_id,
                "task": task.language,
                "attempt": runtime.attempt,
                "episode_id": runtime.episode_id,
                "steps": runtime.step,
                "raw_success": runtime.terminal_success,
                "certified_episode": certified_episode,
                "eligible_branches": sum(bool(value.get("eligible")) for value in branch_results),
                "failure_reason": runtime.failure_reason,
                "raw_path": str(raw_path.relative_to(ROOT)),
                "recovered_from_finalized_raw": True,
            }
            collection_rows.append(row)
            append_jsonl(data / "audits/collection_progress.jsonl", row)
            print(
                f"recovered task={task_id:02d} attempt={runtime.attempt:03d} "
                f"certified={certified_episode} certified_total={len(certified)}/{target}",
                flush=True,
            )
        while len(certified) < target and len(attempted) < maximum_attempts:
            available = [attempt for attempt in range(maximum_attempts) if attempt not in attempted]
            pair_attempts = available[:2]
            runtimes = run_pair(
                task_id=task_id,
                attempts=pair_attempts,
                task=task,
                source_root=source_root,
                initial_states=initial_states,
                client=client,
                config=config,
            )
            finalized = []
            for runtime in runtimes:
                raw_path, episode_metadata = finalize_raw(runtime, config)
                attempted.add(runtime.attempt)
                runtime.env.close()
                finalized.append((runtime, raw_path, episode_metadata))
            for runtime, raw_path, episode_metadata in finalized:
                certified_episode = False
                branch_results: list[dict[str, Any]] = []
                if runtime.terminal_success:
                    certified_episode, branch_results = certify_success(runtime, raw_path, episode_metadata, config)
                    certification_rows.extend(branch_results)
                    if certified_episode:
                        certified.append(runtime.episode_id)
                row = {
                    "recorded_at": now(),
                    "task_id": task_id,
                    "task": task.language,
                    "attempt": runtime.attempt,
                    "episode_id": runtime.episode_id,
                    "steps": runtime.step,
                    "raw_success": runtime.terminal_success,
                    "certified_episode": certified_episode,
                    "eligible_branches": sum(bool(value.get("eligible")) for value in branch_results),
                    "failure_reason": runtime.failure_reason,
                    "raw_path": str(raw_path.relative_to(ROOT)),
                }
                collection_rows.append(row)
                append_jsonl(data / "audits/collection_progress.jsonl", row)
                print(
                    f"collection task={task_id:02d} attempt={runtime.attempt:03d} steps={runtime.step:03d} "
                    f"success={runtime.terminal_success} certified={certified_episode} "
                    f"certified_total={len(certified)}/{target}",
                    flush=True,
                )
            disk_guard(config, f"source-collection-task-{task_id:02d}-attempt-{max(pair_attempts):03d}")
        if len(certified) < minimum:
            write_text(
                out / "wave19_source_collection_failure.md",
                f"# Wave-19 source collection failure\n\nTask {task_id} produced only {len(certified)} certified "
                f"successful source episodes in {len(attempted)} attempts; the frozen minimum is {minimum}. "
                "Per protocol, representation and dynamics training did not run.",
            )
            raise RuntimeError(f"Task {task_id} failed minimum certified success count")
    counts = {task_id: len(existing_certified(data, task_id)) for task_id in range(10)}
    all_progress = []
    progress_path = data / "audits/collection_progress.jsonl"
    if progress_path.is_file():
        all_progress = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines() if line]
    all_certification = []
    for row in all_progress:
        if row["certified_episode"]:
            certified_path = data / "certified" / f"task_{row['task_id']:02d}" / row["episode_id"]
            for metadata_path in certified_path.glob("branches/*/branch_metadata.json"):
                all_certification.append(json.loads(metadata_path.read_text(encoding="utf-8")))
    write_json(
        out / "wave19_snapshot_certification_results.json",
        {
            "created_at": now(),
            "development_gate": "PASS",
            "prospective_branch_count": len(all_certification),
            "all_prospective_branches_certified": all(row.get("certified") is True for row in all_certification),
            "rows": all_certification,
        },
    )
    write_json(
        out / "wave19_data_collection_summary.json",
        {
            "created_at": now(),
            "certified_successes_by_task": counts,
            "total_certified_successes": sum(counts.values()),
            "target_per_task": target,
            "minimum_per_task": minimum,
            "progress_rows": all_progress,
        },
    )
    report_lines = [
        "# Wave-19 π0.5 source-data collection",
        "",
        "The fixed official `pi05_libero` policy generated all source trajectories on the installed official",
        "`libero_10` suite. Every attempt was retained and every admitted episode has at least one eligible",
        "25/50/75-percent branch that passed exact twin/source replay under the frozen zero-error tolerance.",
        "",
        "| task | attempts | raw successes | certified successes |",
        "|---:|---:|---:|---:|",
    ]
    for task_id in range(10):
        rows = [row for row in all_progress if row["task_id"] == task_id]
        report_lines.append(
            f"| {task_id} | {len(rows)} | {sum(row['raw_success'] for row in rows)} | {counts[task_id]} |"
        )
    write_text(out / "wave19_data_collection_report.md", "\n".join(report_lines))
    freeze_dataset(config, all_progress)
    disk_guard(config, "source-collection-complete")
    with (out / "exact_commands.sh").open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n# {now()} phase=source-collection\n"
            "PYTHONPATH=src:/home/jinjaguo/LIBERO:/home/jinjaguo/openpi/packages/openpi-client/src "
            "MUJOCO_GL=egl /home/jinjaguo/anaconda3/envs/libero/bin/python "
            "scripts/dynamics/collect_wave19_libero.py --config configs/dynamics_7.yaml "
            f"--host {args.host} --port {args.port}\n"
        )


if __name__ == "__main__":
    main()

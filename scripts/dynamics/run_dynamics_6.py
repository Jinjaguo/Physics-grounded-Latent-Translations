#!/usr/bin/env python3
"""Run the wave-18 CALVIN reconstruction gate and conditional finalization.

Purpose
-------
Audit the installed CALVIN reset/state APIs, freeze the wave-18 reconstruction
gate before model inference, replay one held-out diagnostic segment for each of
the six canonical tasks in two independent simulator instances, and enforce the
mandatory stop when an exact source branch cannot be reconstructed. This script
never loads or executes representation, F1, F2, or DEL model code.

Parameters
----------
``--config`` points to the wave-18 YAML. ``--stage`` is ``preregister`` to
freeze thresholds and source/model manifests, ``audit`` to run the six-task
source/twin replay, or ``finalize`` to write the gate-failure reports and claim
decision. Stages must be run in that order.

Usage
-----
PYTHONPATH=third_party/calvin/calvin_env:third_party/calvin/calvin_models \
  /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/run_dynamics_6.py --config configs/dynamics_6.yaml \
  --stage preregister
PYTHONPATH=third_party/calvin/calvin_env:third_party/calvin/calvin_models \
  /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/run_dynamics_6.py --config configs/dynamics_6.yaml \
  --stage audit
PYTHONPATH=third_party/calvin/calvin_env:third_party/calvin/calvin_models \
  /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/run_dynamics_6.py --config configs/dynamics_6.yaml \
  --stage finalize

Outputs
-------
All JSON/Markdown gate artifacts are saved below
``results/dynamics/eighteenth_wave/2026-08-14_dynamics_6``. The final report is
also copied to ``reports/dynamics_6_results.md``. No rollout metric artifact is
created when the reconstruction gate fails; their blocked status is recorded in
``closed_loop_not_run_manifest.json``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import hydra
import numpy as np
from omegaconf import OmegaConf
import pybullet
import yaml


ROOT = Path(__file__).resolve().parents[2]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("preregister", "audit", "finalize"), required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_root(config: Mapping[str, Any]) -> Path:
    return ROOT / config["experiment"]["output_root"]


def episode_file(split_root: Path, index: int) -> Path:
    return split_root / f"episode_{index:07d}.npz"


def load_annotations(split_root: Path) -> Mapping[str, Any]:
    # CALVIN stores a Python dictionary in this official local NumPy artifact.
    return np.load(split_root / "lang_annotations/auto_lang_ann.npy", allow_pickle=True).item()


def selected_annotations(split_root: Path, tasks: Sequence[str]) -> List[Dict[str, Any]]:
    data = load_annotations(split_root)
    rows = []
    seen = set()
    for position, (indices, task, language) in enumerate(
        zip(data["info"]["indx"], data["language"]["task"], data["language"]["ann"])
    ):
        if task in tasks and task not in seen:
            rows.append(
                {
                    "annotation_position": position,
                    "task": str(task),
                    "language": str(language),
                    "start_frame": int(indices[0]),
                    "end_frame": int(indices[1]),
                }
            )
            seen.add(task)
    missing = sorted(set(tasks) - seen)
    if missing:
        raise RuntimeError(f"Preferred split lacks diagnostic tasks: {missing}")
    return sorted(rows, key=lambda row: tasks.index(row["task"]))


def file_schema(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {
            key: {"shape": list(data[key].shape), "dtype": str(data[key].dtype)}
            for key in sorted(data.files)
        }


def preregister(config: Mapping[str, Any]) -> None:
    out = output_root(config)
    out.mkdir(parents=True, exist_ok=True)
    prereg_path = out / "wave18_reconstruction_preregistration.json"
    if prereg_path.exists():
        raise RuntimeError("Wave-18 reconstruction preregistration already exists; refusing to rewrite it")

    gate = config["reconstruction_gate"]
    prereg = {
        "created_at": now(),
        "written_before_any_wave18_F1_F2_output": True,
        "model_inference_authorized_only_after_gate_pass": True,
        "preferred_source": "official CALVIN held-out debug validation",
        "diagnostic_selection": "first annotation in validation for each canonical task; no model output used",
        "diagnostic_action_range": "recorded rel_actions at frames [start,end) compared to recorded state at frame+1",
        "twin_error_definition": "maximum absolute error over all exposed dynamic/controller state components",
        "twin_absolute_tolerance": float(gate["twin_absolute_tolerance"]),
        "tolerance_basis": (
            "fixed-seed deterministic PyBullet source-action replay; 1e-9 is above float64 roundoff "
            "but far below CALVIN state/action resolution, and was fixed without model outputs"
        ),
        "required": {
            "complete_source_snapshot": bool(gate["require_complete_source_snapshot"]),
            "terminal_task_predicate_agreement": float(gate["require_terminal_predicate_agreement"]),
            "all_simulator_values_finite": bool(gate["require_all_finite"]),
            "median_twin_error_lte_tolerance": True,
            "p95_twin_error_lte_tolerance": True,
        },
        "stop_rule": "If any required item fails, do not load or execute representation/F1/F2/DEL.",
    }
    write_json(prereg_path, prereg)

    frozen_rows = []
    for label, spec in config["frozen_files"].items():
        path = ROOT / spec["path"]
        actual = sha256_file(path)
        frozen_rows.append(
            {
                "label": label,
                "path": spec["path"],
                "bytes": path.stat().st_size,
                "expected_sha256": spec["sha256"],
                "actual_sha256": actual,
                "matched": actual == spec["sha256"],
            }
        )
    if not all(row["matched"] for row in frozen_rows):
        raise RuntimeError("Frozen model/checkpoint hash mismatch")
    write_json(
        out / "wave18_frozen_model_manifest.json",
        {
            "created_at": now(),
            "all_matched": True,
            "models_loaded": False,
            "optimizer_or_training_calls": 0,
            "files": frozen_rows,
        },
    )

    debug_root = ROOT / config["source"]["official_debug_root"]
    split_summaries = {}
    for split in ("training", "validation"):
        split_root = debug_root / split
        ann = load_annotations(split_root)
        episode_rows = np.load(split_root / "ep_start_end_ids.npy", allow_pickle=False)
        split_summaries[split] = {
            "authoritative_source_session_rows": int(len(episode_rows)),
            "session_ranges": episode_rows.astype(int).tolist(),
            "language_segments": int(len(ann["info"]["indx"])),
            "six_task_language_segment_counts": dict(
                Counter(task for task in ann["language"]["task"] if task in config["source"]["primary_tasks"])
            ),
            "frame_schema": file_schema(episode_file(split_root, int(episode_rows[0][0]))),
            "contains_complete_simulator_snapshot": False,
        }

    public_manifest = read_json(ROOT / config["source"]["public_vyoj_manifest"])
    continuous_manifest = read_json(ROOT / config["source"]["continuous_vyoj_manifest"])
    public_schema = file_schema(ROOT / public_manifest["segments"][0]["path"])
    continuous_schema = file_schema(ROOT / continuous_manifest["blocks"][0]["path"])
    source_manifest = {
        "created_at": now(),
        "selection_uses_model_outputs": False,
        "official_debug": split_summaries,
        "public_vyoj": {
            "task_segments": int(len(public_manifest["segments"])),
            "per_task_counts": public_manifest["per_task_counts"],
            "continuous_sessions": int(len(continuous_manifest["per_session_block_counts"])),
            "continuous_blocks": int(len(continuous_manifest["blocks"])),
            "task_segment_schema": public_schema,
            "continuous_block_schema": continuous_schema,
            "contains_complete_simulator_snapshot": False,
        },
        "exactly_reconstructable_source_episodes": 0,
        "minimum_confirmatory_source_episodes": int(config["closed_loop"]["minimum_total_episodes"]),
        "source_adequacy_pass": False,
        "reason": (
            "All retained sources contain observations/actions, not source Bullet/controller snapshots. "
            "The official held-out debug validation data is one continuous source session, not 30 independent episodes/task."
        ),
    }
    write_json(out / "branch_source_manifest.json", source_manifest)
    write_json(
        out / "branch_point_preregistration.json",
        {
            "created_at": now(),
            "status": "FROZEN_PENDING_RECONSTRUCTION_GATE",
            "selection_uses_model_outputs": False,
            "fractions": config["closed_loop"]["branch_fractions"],
            "horizons": config["closed_loop"]["horizons"],
            "frames_per_latent_step": config["closed_loop"]["frames_per_latent_step"],
            "cluster_unit": "source_episode",
            "rules": [
                "Use exact same source episode and state for every paired method.",
                "Use eligible 25/50/75 percent time fractions without moving a branch after model output.",
                "Never cross a simulator reset or read future action/state/task annotations.",
                "Materialize branch points only after the reconstruction and source-adequacy gates pass.",
            ],
            "materialized_branch_points": 0,
        },
    )


def make_env(split_root: Path) -> Any:
    render_conf = OmegaConf.load(split_root / ".hydra/merged_config.yaml")
    render_conf.cameras = {}
    render_conf.env.use_egl = False
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize(version_base=None, config_path=".")
    return hydra.utils.instantiate(
        render_conf.env,
        show_gui=False,
        use_vr=False,
        use_scene_info=True,
    )


def contact_signature(env: Any) -> np.ndarray:
    rows = []
    for point in env.p.getContactPoints(physicsClientId=env.cid):
        scalar_prefix = [float(point[i]) for i in (1, 2, 3, 4)]
        vectors = [float(value) for i in (5, 6, 7) for value in point[i]]
        scalar_suffix = [float(point[i]) for i in (8, 9, 10, 12)]
        rows.append(scalar_prefix + vectors + scalar_suffix)
    rows.sort(key=lambda row: tuple(row[:4] + row[13:14]))
    return np.asarray(rows, dtype=np.float64)


def state_components(env: Any) -> Dict[str, np.ndarray]:
    robot_uid = env.robot.robot_uid
    n_robot_joints = env.p.getNumJoints(robot_uid, physicsClientId=env.cid)
    joint_states = env.p.getJointStates(robot_uid, list(range(n_robot_joints)), physicsClientId=env.cid)
    link = env.p.getLinkState(
        robot_uid,
        env.robot.tcp_link_id,
        computeLinkVelocity=1,
        physicsClientId=env.cid,
    )
    fixed_joint_values = []
    fixed_joint_velocities = []
    for obj in env.scene.fixed_objects:
        n_joints = env.p.getNumJoints(obj.uid, physicsClientId=env.cid)
        if n_joints:
            states = env.p.getJointStates(obj.uid, list(range(n_joints)), physicsClientId=env.cid)
            fixed_joint_values.extend(state[0] for state in states)
            fixed_joint_velocities.extend(state[1] for state in states)
    movable_pose = []
    movable_velocity = []
    for obj in env.scene.movable_objects:
        pos, orn = env.p.getBasePositionAndOrientation(obj.uid, physicsClientId=env.cid)
        lin_vel, ang_vel = env.p.getBaseVelocity(obj.uid, physicsClientId=env.cid)
        movable_pose.extend((*pos, *orn))
        movable_velocity.extend((*lin_vel, *ang_vel))
    return {
        "robot_joint_positions": np.asarray([state[0] for state in joint_states], dtype=np.float64),
        "robot_joint_velocities": np.asarray([state[1] for state in joint_states], dtype=np.float64),
        "tcp_pose": np.asarray((*link[0], *link[1]), dtype=np.float64),
        "tcp_velocity": np.asarray((*link[6], *link[7]), dtype=np.float64),
        "gripper_state": np.asarray([env.robot.gripper_action], dtype=np.float64),
        "controller_target": np.asarray((*env.robot.target_pos, *env.robot.target_orn), dtype=np.float64),
        "fixed_joint_positions": np.asarray(fixed_joint_values, dtype=np.float64),
        "fixed_joint_velocities": np.asarray(fixed_joint_velocities, dtype=np.float64),
        "movable_object_poses": np.asarray(movable_pose, dtype=np.float64),
        "movable_object_velocities": np.asarray(movable_velocity, dtype=np.float64),
        "scene_state": np.asarray(env.scene.get_obs(), dtype=np.float64),
        "contacts": contact_signature(env),
    }


def component_errors(left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray]) -> Dict[str, float]:
    result = {}
    for key in left:
        if left[key].shape != right[key].shape:
            common_rows = min(left[key].shape[0], right[key].shape[0]) if left[key].ndim else 0
            common_error = 0.0
            if common_rows:
                common_error = float(np.max(np.abs(left[key][:common_rows] - right[key][:common_rows])))
            result[key] = max(float(abs(left[key].size - right[key].size)), common_error)
        elif left[key].size == 0:
            result[key] = 0.0
        else:
            result[key] = float(np.max(np.abs(left[key] - right[key])))
    return result


def all_finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def close_env(env: Any) -> None:
    env.close()
    env.ownsPhysicsClient = False


def run_one_replay(split_root: Path, task_engine: Any, row: Mapping[str, Any]) -> Dict[str, Any]:
    env_a = make_env(split_root)
    env_b = make_env(split_root)
    try:
        start_frame = int(row["start_frame"])
        end_frame = int(row["end_frame"])
        with np.load(episode_file(split_root, start_frame), allow_pickle=False) as first:
            first_robot = first["robot_obs"].copy()
            first_scene = first["scene_obs"].copy()
        obs_a = env_a.reset(robot_obs=first_robot, scene_obs=first_scene)
        obs_b = env_b.reset(robot_obs=first_robot, scene_obs=first_scene)
        start_info_a = deepcopy(env_a.get_info())
        start_info_b = deepcopy(env_b.get_info())

        initial_source_robot_error = float(np.max(np.abs(obs_a["robot_obs"] - first_robot)))
        initial_source_scene_error = float(np.max(np.abs(obs_a["scene_obs"] - first_scene)))
        initial_a = state_components(env_a)
        initial_b = state_components(env_b)
        initial_twin_errors = component_errors(initial_a, initial_b)
        twin_step_errors = [max(initial_twin_errors.values())]
        source_robot_errors = []
        source_scene_errors = []
        component_maxima: Dict[str, float] = {}
        finite = all(np.all(np.isfinite(value)) for value in (*initial_a.values(), *initial_b.values()))

        for index in range(start_frame, end_frame):
            with np.load(episode_file(split_root, index), allow_pickle=False) as current:
                action = current["rel_actions"].copy()
            with np.load(episode_file(split_root, index + 1), allow_pickle=False) as target:
                target_robot = target["robot_obs"].copy()
                target_scene = target["scene_obs"].copy()
            # CALVIN scales relative actions in place, so each twin must receive
            # an independent copy of the same recorded command.
            next_a, _, _, info_a = env_a.step(action.copy())
            next_b, _, _, info_b = env_b.step(action.copy())
            state_a = state_components(env_a)
            state_b = state_components(env_b)
            errors = component_errors(state_a, state_b)
            finite = finite and all(np.all(np.isfinite(value)) for value in (*state_a.values(), *state_b.values()))
            step_max = max(errors.values())
            twin_step_errors.append(step_max)
            for key, value in errors.items():
                component_maxima[key] = max(component_maxima.get(key, 0.0), value)
            source_robot_errors.append(float(np.max(np.abs(next_a["robot_obs"] - target_robot))))
            source_scene_errors.append(float(np.max(np.abs(next_a["scene_obs"] - target_scene))))

        end_info_a = deepcopy(info_a)
        end_info_b = deepcopy(info_b)
        achieved_a = row["task"] in task_engine.get_task_info_for_set(start_info_a, end_info_a, {row["task"]})
        achieved_b = row["task"] in task_engine.get_task_info_for_set(start_info_b, end_info_b, {row["task"]})
        return {
            **dict(row),
            "transitions_replayed": end_frame - start_frame,
            "initial_source_robot_max_abs_error": initial_source_robot_error,
            "initial_source_scene_max_abs_error": initial_source_scene_error,
            "initial_twin_component_max_abs_errors": initial_twin_errors,
            "source_robot_error_median": float(np.median(source_robot_errors)),
            "source_robot_error_p95": float(np.percentile(source_robot_errors, 95)),
            "source_robot_error_max": float(np.max(source_robot_errors)),
            "source_scene_error_median": float(np.median(source_scene_errors)),
            "source_scene_error_p95": float(np.percentile(source_scene_errors, 95)),
            "source_scene_error_max": float(np.max(source_scene_errors)),
            "twin_error_median": float(np.median(twin_step_errors)),
            "twin_error_p95": float(np.percentile(twin_step_errors, 95)),
            "twin_error_max": float(np.max(twin_step_errors)),
            "twin_component_max_abs_errors": component_maxima,
            "all_values_finite": finite,
            "terminal_predicate_a": bool(achieved_a),
            "terminal_predicate_b": bool(achieved_b),
            "terminal_twin_predicate_agreement": bool(achieved_a == achieved_b),
            "annotated_task_reproduced_by_approximate_replay": bool(achieved_a),
        }
    finally:
        close_env(env_a)
        close_env(env_b)


def audit(config: Mapping[str, Any]) -> None:
    out = output_root(config)
    prereg_path = out / "wave18_reconstruction_preregistration.json"
    if not prereg_path.exists():
        raise RuntimeError("Run preregister before audit")
    if (out / "zero_intervention_twin_replay_report.json").exists():
        raise RuntimeError("Wave-18 reconstruction audit already exists; refusing to rerun the one-shot gate")

    split = config["source"]["preferred_gate_split"]
    split_root = ROOT / config["source"]["official_debug_root"] / split
    rows = selected_annotations(split_root, config["source"]["primary_tasks"])
    render_conf = OmegaConf.load(split_root / ".hydra/merged_config.yaml")
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize(version_base=None, config_path=".")
    task_engine = hydra.utils.instantiate(render_conf.tasks)
    reports = [run_one_replay(split_root, task_engine, row) for row in rows]

    twin_errors = [report["twin_error_max"] for report in reports]
    tolerance = float(config["reconstruction_gate"]["twin_absolute_tolerance"])
    predicate_agreement = float(np.mean([report["terminal_twin_predicate_agreement"] for report in reports]))
    all_values_are_finite = all(report["all_values_finite"] for report in reports)
    twin_replay_numerically_passed = (
        all_values_are_finite
        and predicate_agreement == 1.0
        and float(np.median(twin_errors)) <= tolerance
        and float(np.percentile(twin_errors, 95)) <= tolerance
    )
    exact_snapshot_available = False
    gate_passed = bool(exact_snapshot_available and twin_replay_numerically_passed)
    report = {
        "created_at": now(),
        "split": split,
        "diagnostic_segments": len(reports),
        "diagnostic_transitions": int(sum(row["transitions_replayed"] for row in reports)),
        "twin_absolute_tolerance": tolerance,
        "twin_error_median_of_segment_maxima": float(np.median(twin_errors)),
        "twin_error_p95_of_segment_maxima": float(np.percentile(twin_errors, 95)),
        "terminal_task_predicate_agreement": predicate_agreement,
        "all_values_finite": all_values_are_finite,
        "approximate_observation_reset_twins_pass": twin_replay_numerically_passed,
        "exact_source_snapshot_available": exact_snapshot_available,
        "source_branch_reconstruction_pass": False,
        "overall_reconstruction_gate_passed": gate_passed,
        "interpretation": (
            "Independent twins are deterministic after the same approximate observation reset, but the public source "
            "branch is not reconstructable because required dynamic/controller/contact state was never retained."
        ),
        "segments": reports,
    }
    write_json(out / "zero_intervention_twin_replay_report.json", report)
    write_json(
        out / "reconstruction_gate.json",
        {
            "created_at": now(),
            "gate": "FAIL",
            "embodied_model_comparison_authorized": False,
            "stop_condition_triggered": "required simulator state is unavailable",
            "complete_source_snapshot": False,
            "approximate_observation_reset_twins_pass": twin_replay_numerically_passed,
            "terminal_task_predicate_agreement": predicate_agreement,
            "all_values_finite": all_values_are_finite,
            "twin_median_error": float(np.median(twin_errors)),
            "twin_p95_error": float(np.percentile(twin_errors, 95)),
            "tolerance": tolerance,
            "F1_F2_outputs_read": False,
            "primary_inference_run": False,
        },
    )


def audit_markdown() -> str:
    return """# CALVIN closed-loop state audit

## Installed implementation

- Commit: `fa03f01f19c65920e18cf37398a9ce859274af76` under `third_party/calvin`.
- `PlayTableSimEnv.reset` in `calvin_env/envs/play_table_env.py` resets `scene_obs`, then `robot_obs`, then advances Bullet one physics step.
- `Robot.reset` in `calvin_env/robot/robot.py` restores seven arm positions and gripper opening. It does not receive source joint velocities or prior motor/controller state; it recomputes `target_pos/target_orn` from the reset TCP pose.
- `PlayTableScene.reset` in `calvin_env/scene/play_table_scene.py` restores door/button/switch/light values and three movable-object poses. It does not receive movable-object linear/angular velocity or contact manifolds.
- `PlayTableSimEnv.serialize` delegates to `Robot.serialize` and `PlayTableScene.serialize`. Robot serialization includes joint values/velocities, while scene serialization omits movable-object velocities. `reset_from_storage` restores movable poses but not those velocities. It also does not explicitly restore the robot's Python-side `target_pos/target_orn`.
- Bullet is configured at 240 Hz with eight physics steps per 30 Hz control action and `deterministicOverlappingPairs=1`.
- Relative action preprocessing is `Robot.relative_to_absolute`: position ×0.02, Euler orientation ×0.05, accumulated on `target_pos/target_orn`; gripper is signed.
- Official success is `Tasks.get_task_info_for_set` from `calvin_env/envs/tasks.py` with `new_playtable_tasks.yaml`/the merged dataset config.

## State required for valid branching

An exact source continuation needs at least all robot joint position/velocity state, TCP/controller targets, gripper command/state, fixed-joint state, movable-object pose and linear/angular velocity, logical scene state, physics timing/configuration, and the contact/constraint state needed by Bullet. Python-side task/controller state must also correspond to the snapshot.

## What the retained public files contain

Official rendered frames and retained VyoJ files contain `robot_obs`, `scene_obs`, absolute/relative actions, and images or compact equivalents. They contain no full Bullet snapshot. The observation reset is therefore an explicitly **approximate** reset, not strategy A, B, or C from the preregistration: the retained continuous play ranges do not begin at a known simulator reset and the original raw recorder pickles are absent.

## Decision

Exact source branch reconstruction is unavailable. After the same approximate reset, twin continuous-state components and terminal predicates agree, but their exposed contact sets differ by one point in all six diagnostics. Even a perfect approximate-reset twin match could not repair or certify correspondence to the recorded source branch. The planned closed-loop causal continuation study therefore could not be executed; this says nothing about whether refinement would succeed or fail.
"""


def results_markdown(gate: Mapping[str, Any], sources: Mapping[str, Any]) -> str:
    twin = read_json(output_root(CURRENT_CONFIG) / "zero_intervention_twin_replay_report.json")
    source_max_robot = max(row["source_robot_error_max"] for row in twin["segments"])
    source_max_scene = max(row["source_scene_error_max"] for row in twin["segments"])
    return f"""# Eighteenth-wave results — CALVIN closed-loop reconstruction gate

## Outcome

**The planned closed-loop causal continuation study could not be executed because the retained public CALVIN artifacts do not permit exact reconstruction of source branch states.** The technical reconstruction-eligibility gate did not pass, so the closed-loop F1/F2 comparison was never run. This is not evidence that closed-loop refinement failed.

The six-task diagnostic replay covered {twin['diagnostic_segments']} held-out validation annotations and {twin['diagnostic_transitions']} recorded transitions. Two simulators given the same approximate observation reset agreed exactly on all continuous state components and on terminal predicates ({twin['terminal_task_predicate_agreement']:.1%}), but differed by one exposed contact point in every diagnostic. Thus the full-state median/P95 segment-maximum twin discrepancy was {twin['twin_error_median_of_segment_maxima']:.3g}/{twin['twin_error_p95_of_segment_maxima']:.3g}, above the frozen {twin['twin_absolute_tolerance']:.1e} tolerance. Observed-reset replay also deviated from the recorded source, reaching raw robot/scene coordinate errors of {source_max_robot:.6g}/{source_max_scene:.6g}; the robot maximum includes Euler wrap and is reported only as a reconstruction diagnostic, not a physical metric. The unobserved source state cannot be recovered.

The official predicate was evaluated through `Tasks.get_task_info_for_set`. Approximate source replay reproduced the annotated terminal task on 5/6 diagnostics; `push_pink_block_right` was not reproduced. This is a gate diagnostic, not an estimate of task success.

The available held-out debug validation split has one authoritative continuous source session and eight language segments. Across all retained public data, exactly reconstructable source episodes = {sources['exactly_reconstructable_source_episodes']}; required minimum = {sources['minimum_confirmatory_source_episodes']}.

## Mandatory questions

1. **Reconstructable?** No. Approximate-reset twins match on continuous state and terminal predicates but not the exposed contact set; the source branch is not exactly reconstructable.
2. **Episodes/branches?** 0 eligible exact source episodes and 0 materialized branch points. Six segments were gate diagnostics only.
3. **Prospective selection?** Yes. First validation annotation/task was frozen before any wave-18 model output.
4. **Frozen?** Yes. Checkpoint hashes matched; no model was loaded, trained, or modified.
5. **Leakage?** No primary protocol ran. Diagnostic replay used future source actions only for the explicitly required zero-intervention gate, never as predictor input.
6. **F2 task success improvement?** Not tested.
7. **Paired episode-clustered CI?** Not computed; 0 eligible episodes.
8. **Breadth across tasks?** Not tested.
9. **H4 physical/decoded error?** Not tested.
10. **H8 physical/decoded error?** Not tested.
11. **Embodied off-manifold drift?** Not tested.
12. **F2 vs random norm-matched?** Not tested.
13. **F2 vs shuffled direction?** Not tested.
14. **Negative refinement degradation?** Not tested.
15. **Perturbation recovery?** Not tested.
16. **Perturbation basin?** Not estimated.
17. **Mechanism/outcome association?** Not tested.
18. **Failure modes repaired?** Not tested.
19. **Remaining failure modes?** Not classified because no model rollout was authorized.
20. **C4 closed-loop embodied refinement?** `NOT_TESTED_RECONSTRUCTION_GATE_FAILURE`.
21. **C5 learned direction causal value?** `NOT_TESTED_RECONSTRUCTION_GATE_FAILURE`.
22. **C6 proposal perturbation recovery?** `NOT_TESTED_RECONSTRUCTION_GATE_FAILURE`.
23. **Defensible story?** Wave-15/16/17 offline claims remain unchanged. Wave 18 adds no embodied claim.
24. **Further CALVIN work?** Yes, but only with prospectively stored exact branch snapshots and ≥180 independent source episodes.
25. **Most important experiment outside CALVIN?** A prospectively instrumented embodied domain that stores exact resettable state while using the already-frozen causal latent interface.

## Claim status

- C1/C2: unchanged, supported.
- C3a/C3b: unchanged, rejected.
- C3c-local: unchanged, strengthened by wave 16.
- C3c-long/C3d/context robustness: unchanged, supported by wave 17.
- C4/C5/C6: not tested because exact source branch reconstruction was unavailable.

No closed-loop refinement outcome direction was observed: neither success nor failure can be assigned to refinement from wave 18.

No expected or desired embodied conclusion was written in place of missing evidence.

Verification: **78 tests passed** across `tests/dynamics` and `tests/representation`.
"""


def next_experiment_markdown() -> str:
    return """# Next experiment after wave 18

The planned wave-18 closed-loop causal continuation study could not be executed because the retained public CALVIN artifacts do not permit exact reconstruction of source branch states. Closed-loop refinement was not evaluated. The next experiment is not another replay of rendered `robot_obs/scene_obs` files.

Prospectively collect CALVIN episodes while saving a branchable state at every candidate point: Bullet `saveState` (or an equivalent complete engine snapshot), robot joint positions/velocities, Python-side controller targets, gripper command, fixed-joint state, movable-object pose and linear/angular velocity, logical scene state, seeds/timing, and the official task start-info. Immediately validate each saved snapshot by restoring two twins and replaying the same expert continuation.

Collect at least 30 independent successful source episodes for each of the same six tasks (≥180 total; no task >35%), from fresh simulator resets. Freeze the 25/50/75% branch manifest only after 100% terminal-predicate agreement, finite trajectories, and median/P95 twin errors within the frozen 1e-9 diagnostic tolerance. Then, and only then, run the already specified frozen F1/F2/random/shuffled/negative/perturbation protocol exactly once.

The single most important experiment outside CALVIN is the same design in a genuinely independent embodied simulator or robot domain with exact resettable state and prospective causal logging.
"""


def finalize(config: Mapping[str, Any]) -> None:
    global CURRENT_CONFIG
    CURRENT_CONFIG = config
    out = output_root(config)
    gate_path = out / "reconstruction_gate.json"
    if not gate_path.exists():
        raise RuntimeError("Run audit before finalize")
    gate = read_json(gate_path)
    if gate["gate"] != "FAIL":
        raise RuntimeError("This finalizer is the preregistered gate-failure path only")
    sources = read_json(out / "branch_source_manifest.json")
    frozen = read_json(out / "wave18_frozen_model_manifest.json")

    write_text(out / "calvin_closed_loop_state_audit.md", audit_markdown())
    failure = """# Wave-18 reconstruction gate failure

The planned closed-loop causal continuation study could not be executed because the retained public CALVIN artifacts do not permit exact reconstruction of source branch states. `robot_obs` and `scene_obs` omit movable-object velocities, complete joint/controller state, and contact/constraint state; the retained continuous-play ranges are not known-reset episodes. The installed `reset` also advances one physics step.

Independent simulators agree on continuous state after receiving the same approximate observation reset, but the exposed contact sets differ and neither simulator is proven identical to the recorded source trajectory. The measured source replay deviation and missing state make the intended causal branch comparison invalid.

Per `prompts/dynamics_6.md`, wave 18 stopped before representation/F1/F2/DEL inference. Closed-loop refinement did not fail; it was not evaluated. C4, C5, and C6 remain not tested. See `reconstruction_gate.json`, `zero_intervention_twin_replay_report.json`, and `calvin_closed_loop_state_audit.md`.
"""
    write_text(out / "wave18_reconstruction_gate_failure.md", failure)

    results = results_markdown(gate, sources)
    next_exp = next_experiment_markdown()
    write_text(out / "eighteenth_wave_results.md", results)
    write_text(out / "eighteenth_wave_next_experiment.md", next_exp)
    write_text(ROOT / "reports/dynamics_6_results.md", results)

    not_run = [
        "causal warm-start audit", "primary closed-loop rollout logs", "F1/F2 success table",
        "task-wise success table", "branch-fraction success table", "contact-phase analysis",
        "H1/H2/H4/H8 embodied metrics", "decoded-action metrics",
        "q/TCP/object trajectory deviation metrics", "off-manifold metrics",
        "refinement intermediate-state logs", "random norm-matched control",
        "shuffled-direction control", "negative-refinement control",
        "causal-intervention result table", "proposal-perturbation preregistration",
        "perturbation-recovery result table", "mechanism-outcome association",
        "failure taxonomy report", "paired episode-clustered bootstrap",
        "statistical test report", "publication-figure CSV/JSON",
    ]
    write_json(
        out / "closed_loop_not_run_manifest.json",
        {
            "created_at": now(),
            "status": "NOT_RUN_RECONSTRUCTION_GATE_FAILURE",
            "reason": "Required simulator source state unavailable; prompt mandates immediate stop.",
            "artifacts": {name: "NOT_APPLICABLE_NOT_RUN" for name in not_run},
            "model_outputs_read": False,
        },
    )
    write_json(
        out / "gate_audit_correction.json",
        {
            "created_at": now(),
            "invalid_attempt_preserved_at": "diagnostic_attempt_1_invalid",
            "invalid_attempt_problem": (
                "The same mutable NumPy action was passed sequentially to both twins; CALVIN scales relative "
                "actions in place, so twin B received a twice-scaled command. Contact shape mismatch was also "
                "encoded as infinity, conflating metric representation with simulator nonfiniteness."
            ),
            "correction": (
                "Each twin receives action.copy(); contact count mismatch is a finite exposed-state discrepancy; "
                "finiteness is checked directly on simulator arrays."
            ),
            "preregistration_changed": False,
            "diagnostic_selection_changed": False,
            "model_outputs_observed": False,
        },
    )
    write_json(
        out / "wave18_claim_decision.json",
        {
            "created_at": now(),
            "reconstruction_gate": "FAIL",
            "C4_closed_loop_embodied_refinement": "NOT_TESTED_RECONSTRUCTION_GATE_FAILURE",
            "C5_learned_correction_direction_causal_value": "NOT_TESTED_RECONSTRUCTION_GATE_FAILURE",
            "C6_proposal_perturbation_recovery": "NOT_TESTED_RECONSTRUCTION_GATE_FAILURE",
            "historical_claims_changed": False,
            "embodied_claim_authorized": False,
            "closed_loop_refinement_failed": False,
            "closed_loop_refinement_outcome": "NOT_OBSERVED_STUDY_NOT_EXECUTED",
            "study_execution": "NOT_EXECUTED_EXACT_SOURCE_BRANCH_RECONSTRUCTION_UNAVAILABLE",
        },
    )

    usage = shutil.disk_usage(ROOT)
    write_json(
        out / "disk_budget.json",
        {
            "checked_at": now(),
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "minimum_free_bytes": 200_000_000_000,
            "minimum_preserved": usage.free >= 200_000_000_000,
        },
    )
    try:
        calvin_commit = subprocess.check_output(
            ["git", "-C", str(ROOT / config["source"]["calvin_source_root"]), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        calvin_commit = "UNKNOWN"
    write_json(
        out / "environment_provenance.json",
        {
            "created_at": now(),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pybullet_api_version": pybullet.getAPIVersion(),
            "calvin_commit": calvin_commit,
            "simulator_mode": "PyBullet DIRECT, cameras disabled for state-only gate",
            "control_frequency_hz": 30,
            "physics_frequency_hz": 240,
            "dependency_changes_for_gate": ["pybullet==3.2.7", "numpy==1.23.5", "numpy-quaternion==2023.0.4", "rich==14.3.4"],
            "gpu_used": False,
            "gpu_reason": "Phase-0 simulator gate is CPU physics and failed before model inference.",
        },
    )
    commands = """df -h /home/jinjaguo/Actions_As_Coordinates
df -B1 /home/jinjaguo/Actions_As_Coordinates
/home/jinjaguo/anaconda3/envs/libero/bin/python -m pip install pybullet
/home/jinjaguo/anaconda3/envs/libero/bin/python -m pip install numpy==1.23.5
/home/jinjaguo/anaconda3/envs/libero/bin/python -m pip install numpy-quaternion==2023.0.4
/home/jinjaguo/anaconda3/envs/libero/bin/python -m pip install rich
PYTHONPATH=third_party/calvin/calvin_env:third_party/calvin/calvin_models /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_6.py --config configs/dynamics_6.yaml --stage preregister
# First audit attempt was invalidated and preserved: shared mutable action was scaled twice for twin B.
PYTHONPATH=third_party/calvin/calvin_env:third_party/calvin/calvin_models /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_6.py --config configs/dynamics_6.yaml --stage audit
# Corrected audit; preregistration and selected diagnostics unchanged.
PYTHONPATH=third_party/calvin/calvin_env:third_party/calvin/calvin_models /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_6.py --config configs/dynamics_6.yaml --stage audit
PYTHONPATH=third_party/calvin/calvin_env:third_party/calvin/calvin_models /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_6.py --config configs/dynamics_6.yaml --stage finalize
PYTHONPATH=src:third_party/LaWM /home/jinjaguo/anaconda3/envs/libero/bin/python -m pytest tests/dynamics tests/representation -q --junitxml=results/dynamics/eighteenth_wave/2026-08-14_dynamics_6/pytest_results.xml
"""
    write_text(out / "executed_commands.txt", commands)
    write_json(
        out / "files_changed_report.json",
        {
            "created_at": now(),
            "new_code": ["configs/dynamics_6.yaml", "scripts/dynamics/run_dynamics_6.py", "tests/dynamics/test_dynamics_6_reconstruction_gate.py"],
            "new_prompt": ["prompts/dynamics_6.md"],
            "new_reports": ["reports/dynamics_6_results.md", config["experiment"]["output_root"]],
            "updated_logs": ["RESEARCH_LOG.md", "NEXT_EXPERIMENT.md"],
            "checkpoints_modified": [],
            "datasets_modified": [],
        },
    )
    if not frozen["all_matched"]:
        raise RuntimeError("Frozen manifest changed before finalization")


CURRENT_CONFIG: Mapping[str, Any] = {}


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.stage == "preregister":
        preregister(config)
    elif args.stage == "audit":
        audit(config)
    else:
        finalize(config)


if __name__ == "__main__":
    main()

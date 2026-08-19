"""Exact MuJoCo and robosuite state snapshots for prospective LIBERO branches.

The installed LIBERO wrapper's ``get_sim_state`` only returns time, qpos, and
qvel.  Wave 19 instead uses MuJoCo's official ``mjSTATE_INTEGRATION`` state
specification and separately preserves Python-side robosuite controller,
buffer, observable, and episode-counter state.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

import mujoco
import numpy as np


STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


@dataclass(frozen=True)
class LiberoSnapshot:
    """One exactly restorable control-boundary snapshot."""

    integration_state: np.ndarray
    model_state: Mapping[str, np.ndarray]
    environment_state: Mapping[str, Any]
    controller_state: tuple[Mapping[str, Any], ...]
    observable_state: Mapping[str, Mapping[str, Any]]


def _copy_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.generic):
        return value.item()
    return deepcopy(value)


def _capture_buffer(buffer: Any) -> dict[str, Any]:
    fields = {}
    for name in ("dim", "length", "_size", "ptr", "buf", "last", "current"):
        # ``RingBuffer.current`` is a read-only property derived from ``buf``
        # and ``ptr``. Preserve only state actually owned by the instance.
        if name in vars(buffer):
            fields[name] = _copy_value(getattr(buffer, name))
    return {"class": type(buffer).__name__, "fields": fields}


def _restore_buffer(buffer: Any, payload: Mapping[str, Any]) -> None:
    if type(buffer).__name__ != payload["class"]:
        raise TypeError(f"Buffer class mismatch: {type(buffer).__name__} != {payload['class']}")
    for name, value in payload["fields"].items():
        setattr(buffer, name, _copy_value(value))


def _capture_robot(robot: Any) -> dict[str, Any]:
    controller = robot.controller
    controller_fields = {}
    for name in (
        "new_update",
        "goal_pos",
        "goal_ori",
        "relative_ori",
        "ori_ref",
        "initial_joint",
        "initial_ee_pos",
        "initial_ee_ori_mat",
        "action_scale",
        "action_input_transform",
        "action_output_transform",
        "kp",
        "kd",
        "torques",
        "ee_pos",
        "ee_ori_mat",
        "ee_pos_vel",
        "ee_ori_vel",
        "joint_pos",
        "joint_vel",
        "J_pos",
        "J_ori",
        "J_full",
        "mass_matrix",
    ):
        if hasattr(controller, name):
            controller_fields[name] = _copy_value(getattr(controller, name))

    robot_fields = {}
    for name in ("torques",):
        if hasattr(robot, name):
            robot_fields[name] = _copy_value(getattr(robot, name))

    buffers = {}
    for name, value in vars(robot).items():
        if name.startswith("recent_") and hasattr(value, "push"):
            buffers[name] = _capture_buffer(value)

    return {
        "robot_class": type(robot).__name__,
        "controller_class": type(controller).__name__,
        "controller_fields": controller_fields,
        "robot_fields": robot_fields,
        "gripper_fields": {
            name: _copy_value(getattr(robot.gripper, name))
            for name in ("current_action",)
            if hasattr(robot.gripper, name)
        },
        "buffers": buffers,
    }


def _restore_robot(robot: Any, payload: Mapping[str, Any]) -> None:
    if type(robot).__name__ != payload["robot_class"]:
        raise TypeError(f"Robot class mismatch: {type(robot).__name__} != {payload['robot_class']}")
    controller = robot.controller
    if type(controller).__name__ != payload["controller_class"]:
        raise TypeError(
            f"Controller class mismatch: {type(controller).__name__} != {payload['controller_class']}"
        )
    for name, value in payload["controller_fields"].items():
        setattr(controller, name, _copy_value(value))
    for name, value in payload["robot_fields"].items():
        setattr(robot, name, _copy_value(value))
    for name, value in payload["gripper_fields"].items():
        setattr(robot.gripper, name, _copy_value(value))
    for name, value in payload["buffers"].items():
        _restore_buffer(getattr(robot, name), value)


def _capture_observables(base_env: Any) -> dict[str, dict[str, Any]]:
    result = {}
    for name, observable in base_env._observables.items():
        result[name] = {
            field: _copy_value(getattr(observable, field))
            for field in (
                "_time_since_last_sample",
                "_current_delay",
                "_current_observed_value",
                "_sampled",
            )
        }
    return result


def _restore_observables(base_env: Any, payload: Mapping[str, Mapping[str, Any]]) -> None:
    if set(payload) != set(base_env._observables):
        raise ValueError("Observable set differs between snapshot and target environment")
    for name, fields in payload.items():
        observable = base_env._observables[name]
        for field, value in fields.items():
            setattr(observable, field, _copy_value(value))


def capture_integration_state(sim: Any) -> np.ndarray:
    """Capture the official MuJoCo integration state as float64."""

    size = mujoco.mj_stateSize(sim.model._model, STATE_SPEC)
    state = np.empty(size, dtype=np.float64)
    mujoco.mj_getState(sim.model._model, sim.data._data, state, STATE_SPEC)
    if not np.isfinite(state).all():
        raise FloatingPointError("Nonfinite value in MuJoCo integration state")
    return state


def restore_integration_state(sim: Any, state: np.ndarray) -> None:
    """Restore a state captured by :func:`capture_integration_state`."""

    expected = mujoco.mj_stateSize(sim.model._model, STATE_SPEC)
    value = np.asarray(state, dtype=np.float64)
    if value.shape != (expected,):
        raise ValueError(f"Integration-state shape mismatch: {value.shape} != {(expected,)}")
    mujoco.mj_setState(sim.model._model, sim.data._data, value, STATE_SPEC)


def capture_snapshot(env: Any) -> LiberoSnapshot:
    """Capture a LIBERO environment at an environment-control boundary."""

    base = env.env
    # Canonicalize MuJoCo's derived acceleration / warm-start fields without
    # advancing time. The same forward pass is applied after restoration.
    env.sim.forward()
    return LiberoSnapshot(
        integration_state=capture_integration_state(env.sim),
        model_state={
            "body_pos": np.asarray(env.sim.model.body_pos).copy(),
            "body_quat": np.asarray(env.sim.model.body_quat).copy(),
        },
        environment_state={
            "timestep": int(base.timestep),
            "cur_time": float(base.cur_time),
            "done": bool(base.done),
        },
        controller_state=tuple(_capture_robot(robot) for robot in base.robots),
        observable_state=_capture_observables(base),
    )


def restore_snapshot(env: Any, snapshot: LiberoSnapshot) -> dict[str, np.ndarray]:
    """Restore simulator and Python-side controller state and regenerate observations."""

    base = env.env
    if len(base.robots) != len(snapshot.controller_state):
        raise ValueError("Robot count differs between snapshot and target environment")
    model_state = getattr(snapshot, "model_state", {})
    for name, value in model_state.items():
        target = np.asarray(getattr(env.sim.model, name))
        source = np.asarray(value)
        if target.shape != source.shape:
            raise ValueError(f"Model-state shape mismatch for {name}: {source.shape} != {target.shape}")
        target[...] = source
    restore_integration_state(env.sim, snapshot.integration_state)
    for name, value in snapshot.environment_state.items():
        setattr(base, name, _copy_value(value))
    legacy_controller_state = []
    for robot, payload in zip(base.robots, snapshot.controller_state):
        _restore_robot(robot, payload)
        legacy_controller_state.append("ee_pos" not in payload["controller_fields"])
    _restore_observables(base, snapshot.observable_state)
    # Rebuild qpos-dependent geometry for controller queries, then restore the
    # complete integration state a second time. In contact-rich states,
    # ``mj_forward`` legitimately rewrites qacc / solver warm-start fields;
    # leaving those rewritten values caused immediate branch divergence.
    env.sim.forward()
    restore_integration_state(env.sim, snapshot.integration_state)
    # Wave-19 snapshots created before EXP_G24 did not preserve the OSC
    # controller's derived kinematics. If ``new_update`` was false, the first
    # restored action used whichever ee_pos happened to remain in the target
    # env. Canonicalize those legacy payloads once from the restored MuJoCo
    # state, then restore integration fields that mj_forward may have changed.
    for robot, legacy in zip(base.robots, legacy_controller_state):
        if legacy:
            robot.controller.update(force=True)
    if any(legacy_controller_state):
        restore_integration_state(env.sim, snapshot.integration_state)
    env._post_process()
    env._update_observables(force=True)
    return base._get_observations()


def safe_env_step(env: Any, action: np.ndarray) -> tuple[Any, float, bool, dict[str, Any]]:
    """Execute a private action copy and prove that the caller-owned bytes did not change."""

    value = np.asarray(action)
    before = value.tobytes()
    result = env.step(value.copy())
    if value.tobytes() != before:
        raise RuntimeError("Caller-owned action mutated across the environment-step boundary")
    return result


def integration_state_max_abs_error(left: Any, right: Any) -> float:
    """Return an exact-shape finite maximum absolute integration-state discrepancy."""

    a = capture_integration_state(left.sim) if not isinstance(left, np.ndarray) else left
    b = capture_integration_state(right.sim) if not isinstance(right, np.ndarray) else right
    if a.shape != b.shape:
        raise ValueError(f"State shape mismatch: {a.shape} != {b.shape}")
    error = np.abs(a - b)
    if not np.isfinite(error).all():
        raise FloatingPointError("Nonfinite integration-state discrepancy")
    return float(error.max(initial=0.0))


def physical_state(env: Any) -> dict[str, np.ndarray]:
    """Capture physical diagnostics without using rendered observations."""

    data = env.sim.data
    robot = env.env.robots[0]
    contact_capacity = max(int(getattr(env.sim.model, "nconmax", 0)), int(data.ncon))
    contact_geom_pairs = np.full((contact_capacity, 2), -1, dtype=np.int32)
    contact_distance = np.zeros(contact_capacity, dtype=np.float64)
    contact_position = np.zeros((contact_capacity, 3), dtype=np.float64)
    contact_frame = np.zeros((contact_capacity, 9), dtype=np.float64)
    contact_force = np.zeros((contact_capacity, 6), dtype=np.float64)
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        contact_geom_pairs[index] = (int(contact.geom1), int(contact.geom2))
        contact_distance[index] = float(contact.dist)
        contact_position[index] = np.asarray(contact.pos)
        contact_frame[index] = np.asarray(contact.frame)
        mujoco.mj_contactForce(env.sim.model._model, data._data, index, contact_force[index])
    return {
        "qpos": np.asarray(data.qpos).copy(),
        "qvel": np.asarray(data.qvel).copy(),
        "ctrl": np.asarray(data.ctrl).copy(),
        "body_xpos": np.asarray(data.xpos).copy(),
        "body_xquat": np.asarray(data.xquat).copy(),
        "body_cvel": np.asarray(data.cvel).copy(),
        "robot_joint_qpos": np.asarray(robot._joint_positions).copy(),
        "robot_joint_qvel": np.asarray(robot._joint_velocities).copy(),
        "eef_pos": np.asarray(robot.controller.ee_pos).copy(),
        "eef_ori": np.asarray(robot.controller.ee_ori_mat).copy(),
        "gripper_qpos": np.asarray(data.qpos[robot._ref_gripper_joint_pos_indexes]).copy(),
        "gripper_qvel": np.asarray(data.qvel[robot._ref_gripper_joint_vel_indexes]).copy(),
        "contact_count": np.asarray([data.ncon], dtype=np.int32),
        "contact_geom_pairs": contact_geom_pairs,
        "contact_distance": contact_distance,
        "contact_position": contact_position,
        "contact_frame": contact_frame,
        "contact_force": contact_force,
    }

"""Protocol tests for Wave-19 official LIBERO-10 prospective collection.

Purpose
-------
Verify the immutable action boundary, deterministic branch selection and
future-support gate, exact MuJoCo/controller snapshot mechanics, and the fixed
π0.5 output dimensions used by the Wave-19 collector.

Parameters
----------
The tests take no command-line parameters. The integration test uses the local
official LIBERO checkout and its task-0 init state with ``MUJOCO_GL=egl``.

Usage
-----
PYTHONPATH=src:/home/jinjaguo/LIBERO:/home/jinjaguo/openpi/packages/openpi-client/src \
  MUJOCO_GL=egl /home/jinjaguo/anaconda3/envs/libero/bin/python -m pytest -q \
  tests/dynamics/test_dynamics_7_libero_protocol.py

Outputs
-------
Pytest reports pass/fail status only; the tests do not create experiment data.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

from pglt.libero.snapshot import capture_snapshot, physical_state, restore_snapshot, safe_env_step


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "collect_wave19_libero", ROOT / "scripts/dynamics/collect_wave19_libero.py"
)
assert SPEC is not None and SPEC.loader is not None
COLLECTOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = COLLECTOR
SPEC.loader.exec_module(COLLECTOR)
REP_SPEC = importlib.util.spec_from_file_location(
    "train_wave19_representation", ROOT / "scripts/dynamics/train_wave19_representation.py"
)
assert REP_SPEC is not None and REP_SPEC.loader is not None
REPRESENTATION = importlib.util.module_from_spec(REP_SPEC)
sys.modules[REP_SPEC.name] = REPRESENTATION
REP_SPEC.loader.exec_module(REPRESENTATION)
DYN_SPEC = importlib.util.spec_from_file_location(
    "train_wave19_dynamics", ROOT / "scripts/dynamics/train_wave19_dynamics.py"
)
assert DYN_SPEC is not None and DYN_SPEC.loader is not None
DYNAMICS = importlib.util.module_from_spec(DYN_SPEC)
sys.modules[DYN_SPEC.name] = DYNAMICS
DYN_SPEC.loader.exec_module(DYNAMICS)
CLOSED_SPEC = importlib.util.spec_from_file_location(
    "run_wave19_closed_loop", ROOT / "scripts/dynamics/run_wave19_closed_loop.py"
)
assert CLOSED_SPEC is not None and CLOSED_SPEC.loader is not None
CLOSED_LOOP = importlib.util.module_from_spec(CLOSED_SPEC)
sys.modules[CLOSED_SPEC.name] = CLOSED_LOOP
CLOSED_SPEC.loader.exec_module(CLOSED_LOOP)


class MutatingEnvironment:
    def step(self, action: np.ndarray):
        action[...] = 0.0
        return {}, 0.0, False, {}


def test_safe_env_step_isolates_caller_action() -> None:
    action = np.linspace(-1.0, 1.0, 7, dtype=np.float64)
    expected = action.copy()
    safe_env_step(MutatingEnvironment(), action)
    np.testing.assert_array_equal(action, expected)


def test_collection_reads_predicate_before_canonical_snapshot(monkeypatch) -> None:
    events = []

    class OrderedEnvironment:
        def check_success(self):
            events.append("predicate")
            return False

    monkeypatch.setattr(COLLECTOR, "safe_env_step", lambda env, action: (events.append("step") or ({}, 0.0, False, {})))
    monkeypatch.setattr(COLLECTOR, "capture_snapshot", lambda env: events.append("snapshot") or object())
    monkeypatch.setattr(COLLECTOR, "physical_state", lambda env: events.append("physical") or {})
    COLLECTOR.step_predicate_snapshot(OrderedEnvironment(), np.zeros(7))
    assert events == ["step", "predicate", "snapshot", "physical"]


def test_branch_selection_is_fixed_and_requires_causal_and_future_support() -> None:
    rows = COLLECTOR.branch_steps(410, 10, [0.25, 0.5, 0.75], 128)
    assert [row["step"] for row in rows] == [110, 210, 310]
    assert [row["future_steps"] for row in rows] == [300, 200, 100]
    assert [row["eligible"] for row in rows] == [True, True, False]
    assert COLLECTOR.branch_steps(170, 10, [0.25], 128)[0]["eligible"] is False


def test_finalized_episode_hash_covers_every_raw_file(tmp_path) -> None:
    (tmp_path / "actions.npy").write_bytes(b"actions")
    (tmp_path / "robot_states.npz").write_bytes(b"robot-v1")
    before, files = COLLECTOR.directory_episode_hash(tmp_path)
    assert set(files) == {"actions.npy", "robot_states.npz"}
    (tmp_path / "robot_states.npz").write_bytes(b"robot-v2")
    after, _ = COLLECTOR.directory_episode_hash(tmp_path)
    assert after != before


def test_libero_normalization_preserves_continuous_gripper_command() -> None:
    actions = np.arange(16 * 7, dtype=np.float64).reshape(16, 7) / 100.0
    actions[:, 6] = np.linspace(-1.006, 1.004, 16)
    chunk = REPRESENTATION.Chunk("episode", 0, 0, actions)
    mean = np.arange(6, dtype=np.float32) / 10.0
    std = np.full(6, 0.5, dtype=np.float32)
    normalized = REPRESENTATION.normalized_actions(chunk, mean, std).numpy()
    np.testing.assert_allclose(normalized[:, :6], (actions[:, :6] - mean) / std, atol=1e-6)
    np.testing.assert_allclose(normalized[:, 6], actions[:, 6], atol=1e-6)


def test_training_chunks_start_at_certified_policy_step_zero(tmp_path, monkeypatch) -> None:
    certified = tmp_path / "certified_episode"
    certified.mkdir()
    actions = np.repeat(np.arange(48, dtype=np.float32)[:, None], 7, axis=1)
    np.save(certified / "actions.npy", actions)
    row = {"episode_id": "episode", "task_id": 0, "certified_path": "certified_episode"}
    monkeypatch.setattr(REPRESENTATION, "ROOT", tmp_path)
    chunks, episodes = REPRESENTATION.load_chunks([row], chunk_length=16)
    assert [chunk.start for chunk in chunks] == [0, 16, 32]
    np.testing.assert_array_equal(episodes["episode"], actions)
    np.testing.assert_array_equal(chunks[0].actions, actions[:16])

    class IdentityRepresentation(torch.nn.Module):
        def forward(self, value):
            zeros = torch.zeros((len(value), 16), dtype=value.dtype, device=value.device)
            return {"semantic_latent": zeros, "execution_latent": zeros}

        def project_text(self, value):
            return value[:, :16]

    monkeypatch.setattr(DYNAMICS, "ROOT", tmp_path)
    encoded = DYNAMICS.encode_episodes(
        [row],
        IdentityRepresentation(),
        np.zeros((10, 768), dtype=np.float32),
        np.zeros(6, dtype=np.float32),
        np.ones(6, dtype=np.float32),
        chunk_length=16,
        device=torch.device("cpu"),
    )
    assert len(encoded) == 1 and encoded[0].length == 3
    np.testing.assert_array_equal(encoded[0].actions[0], actions[:16])


def test_representation_batches_have_unique_tasks_and_shuffle_is_deranged() -> None:
    chunks = [
        REPRESENTATION.Chunk(f"episode_{task}_{repeat}", task, 0, np.zeros((16, 7)))
        for repeat in range(3)
        for task in range(10)
    ]
    batches = REPRESENTATION.unique_task_batches(chunks, 190819)
    assert sorted(index for batch in batches for index in batch) == list(range(len(chunks)))
    assert all(len({chunks[index].task_id for index in batch}) == len(batch) for batch in batches)
    mapping = REPRESENTATION.derangement(190819)
    assert set(mapping) == set(range(10))
    assert set(mapping.values()) == set(range(10))
    assert all(task != replacement for task, replacement in mapping.items())


def test_f2_starts_from_exact_f1_and_runs_exactly_four_updates() -> None:
    torch.manual_seed(190819)
    f1 = DYNAMICS.ExecutionMLP(context_dim=32, hidden_dim=64, depth=3)
    f2 = DYNAMICS.ExecutionMatchedRefinement(
        f1, context_dim=32, hidden_dim=64, depth=3, iterations=4, step_size=0.01
    )
    assert f2.iterations == 4
    assert all(torch.equal(value, f2.initializer.state_dict()[key]) for key, value in f1.state_dict().items())
    previous = torch.randn(3, 16)
    current = torch.randn(3, 16)
    context = torch.randn(3, 32)
    with torch.no_grad():
        expected = f1(previous, current, context)
    result, states, gradients = DYNAMICS.refine_states(f2, previous, current, context)
    torch.testing.assert_close(states[0], expected, rtol=0, atol=0)
    assert len(states) == 5 and len(gradients) == 4 and result.shape == (3, 16)


def test_refinement_controls_are_norm_matched_unrelated_and_negative() -> None:
    direction = torch.randn(4, 16)
    target_norm = torch.tensor([[0.25], [0.5], [0.75], [1.0]])
    matched = CLOSED_LOOP.normalize_direction(direction, target_norm)
    torch.testing.assert_close(matched.norm(dim=-1, keepdim=True), target_norm)
    torch.manual_seed(190819)
    f1 = DYNAMICS.ExecutionMLP(context_dim=32, hidden_dim=64, depth=3)
    f2 = DYNAMICS.ExecutionMatchedRefinement(
        f1, context_dim=32, hidden_dim=64, depth=3, iterations=4, step_size=0.01
    )
    state = CLOSED_LOOP.ModelState(
        torch.randn(1, 16),
        torch.randn(1, 16),
        torch.randn(1, 16),
        torch.randn(1, 16),
        torch.randn(1, 16),
        3,
    )
    pool = np.zeros((2, 4, 16), dtype=np.float32)
    pool[0, :, 0] = 1.0
    pool[1, :, 1] = 1.0
    tasks = np.asarray([3, 4])
    shuffled, shuffled_info = CLOSED_LOOP.propose_execution(
        method="B4_shuffled",
        state=state,
        f1=f1,
        f2=f2,
        shuffled_pool=pool,
        shuffled_tasks=tasks,
        rng=np.random.default_rng(190819),
    )
    assert torch.count_nonzero(shuffled_info["applied_delta"][:, 0]) == 0
    assert torch.count_nonzero(shuffled_info["applied_delta"][:, 1]) == 1
    negative, negative_info = CLOSED_LOOP.propose_execution(
        method="B5_negative",
        state=state,
        f1=f1,
        f2=f2,
        shuffled_pool=pool,
        shuffled_tasks=tasks,
        rng=np.random.default_rng(190819),
    )
    expected = -(negative_info["learned_final"] - negative_info["initial"])
    torch.testing.assert_close(negative - negative_info["initial"], expected)

    noisy_initials = []
    for method in ("P_F1_noisy", "P_F2_noisy", "P_random_noisy", "P_negative_noisy"):
        _, info = CLOSED_LOOP.propose_execution(
            method=method,
            state=state,
            f1=f1,
            f2=f2,
            shuffled_pool=pool,
            shuffled_tasks=tasks,
            rng=np.random.default_rng(195819),
            noisy_scale=0.1,
            execution_std=np.linspace(0.5, 1.5, 16, dtype=np.float32),
        )
        noisy_initials.append(info["initial"])
    for value in noisy_initials[1:]:
        torch.testing.assert_close(value, noisy_initials[0], rtol=0, atol=0)


def test_closed_loop_bootstrap_clusters_nested_branches_by_source_episode() -> None:
    rows = []
    for episode, values in {
        "episode_a": {"B1_F1": [0, 0], "B2_F2": [1, 0]},
        "episode_b": {"B1_F1": [1], "B2_F2": [1]},
    }.items():
        for method, successes in values.items():
            for index, success in enumerate(successes):
                rows.append(
                    {
                        "episode_id": episode,
                        "method": method,
                        "terminal_success": bool(success),
                        "success_by_horizon": {"1": bool(success)},
                        "physical": {"tcp_position_deviation": float(index + success)},
                    }
                )
    result = CLOSED_LOOP.clustered_success(rows, "B1_F1", "B2_F2", "until", 100, 190819)
    assert result["episodes"] == 2
    assert result["mean_difference_right_minus_left"] == 0.25


def test_formal_minimum_history_camera_snapshot_restores_exactly() -> None:
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(0)
    source_root = Path("/home/jinjaguo/LIBERO/libero/libero")
    bddl = source_root / "bddl_files" / task.problem_folder / task.bddl_file
    initial = torch.load(source_root / "init_files" / task.problem_folder / task.init_states_file)[0]
    source = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=32,
        camera_widths=32,
        use_camera_obs=True,
    )
    twin = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=32,
        camera_widths=32,
        use_camera_obs=True,
    )
    try:
        for env in (source, twin):
            env.seed(190819)
            env.reset()
            env.set_init_state(initial)
        physical = physical_state(source)
        assert physical["contact_count"].shape == (1,)
        assert physical["contact_geom_pairs"].shape[0] == physical["contact_distance"].shape[0]
        assert int(physical["contact_count"][0]) <= physical["contact_geom_pairs"].shape[0]
        assert all(np.isfinite(value).all() for value in physical.values())
        rng = np.random.default_rng(190819)
        pre = rng.uniform(-0.1, 0.1, size=(42, 7))
        replay = rng.uniform(-0.1, 0.1, size=(16, 7))
        pre[:, 6] = np.where(np.arange(len(pre)) % 7 < 4, -1.0, 1.0)
        replay[:, 6] = np.where(np.arange(len(replay)) % 9 < 5, -1.0, 1.0)
        for action in pre:
            safe_env_step(source, action)
        branch = capture_snapshot(source)
        expected = []
        for action in replay:
            safe_env_step(source, action)
            expected.append(capture_snapshot(source))
        restore_snapshot(twin, branch)
        for action, source_snapshot in zip(replay, expected):
            safe_env_step(twin, action)
            twin_snapshot = capture_snapshot(twin)
            np.testing.assert_array_equal(twin_snapshot.integration_state, source_snapshot.integration_state)
            assert COLLECTOR.numeric_discrepancy(
                twin_snapshot.controller_state, source_snapshot.controller_state
            ) == 0.0
    finally:
        source.close()
        twin.close()

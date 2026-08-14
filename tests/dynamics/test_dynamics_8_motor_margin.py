"""Verify the preregistered Wave-20 LIBERO motor-margin experiment.

Purpose
-------
Check the frozen Wave-19 split hashes, new seed/loss constants, fresh 5/task
confirmation set, exact snapshot certification, leakage boundaries, paired
representation architecture, complete outputs, and scientific stop behavior.

Parameters
----------
No command-line parameters; paths are the frozen repository-relative Wave-19
and Wave-20 locations.

Usage
-----
PYTHONPATH=src:/home/jinjaguo/LIBERO \
  /home/jinjaguo/anaconda3/envs/libero/bin/python -m pytest -q \
  tests/dynamics/test_dynamics_8_motor_margin.py

Outputs
-------
Pytest status only; the tests do not modify experiment files.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts.dynamics.train_wave20_representation import CONDITIONS, representation_objective
from pglt.libero.snapshot import safe_env_step


ROOT = Path(__file__).resolve().parents[2]
W19 = ROOT / "results/dynamics/nineteenth_wave/2026-08-14_dynamics_7"
W20 = ROOT / "results/dynamics/twentieth_wave/2026-08-14_dynamics_8"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_wave19_split_freeze_is_unchanged_and_disjoint() -> None:
    frozen = read_json(W20 / "wave20_existing_split_freeze.json")
    split_path = W19 / "wave19_dataset_split_manifest.json"
    dataset_path = W19 / "wave19_dataset_manifest.json"
    assert sha256_file(split_path) == frozen["wave19_split_manifest_sha256"]
    assert sha256_file(dataset_path) == frozen["wave19_dataset_manifest_sha256"]
    split = read_json(split_path)
    assert split["counts"] == {"train": 140, "development": 50, "test": 50}
    groups = [set(split["assignments"][name]) for name in ("train", "development", "test")]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    assert frozen["wave20_test_payload_files_read"] is False


def test_seeds_and_single_factor_are_exactly_preregistered() -> None:
    config = yaml.safe_load((ROOT / "configs/dynamics_8.yaml").read_text(encoding="utf-8"))
    seeds = config["representation"]["seeds"]
    assert seeds == [200820, 201820, 202820, 203820, 204820, 205820]
    assert not set(seeds) & {190819, 191819, 192819, 193819, 194819, 195819}
    assert config["representation"]["reconstruction_weight"] == 2.0
    assert config["representation"]["epochs"] == 40
    assert config["representation"]["ema_decay"] == 0.999
    assert CONDITIONS == ("reconstruction_only", "correct_language", "shuffled_language_control")


def test_r0_and_r1_objectives_have_the_frozen_weights() -> None:
    rec = torch.tensor(3.0)
    sem = torch.tensor(5.0)
    assert representation_objective(rec, torch.tensor(0.0), "reconstruction_only", 2.0).item() == 3.0
    assert representation_objective(rec, sem, "correct_language", 2.0).item() == 11.0
    assert representation_objective(rec, sem, "shuffled_language_control", 2.0).item() == 11.0


def test_action_copy_boundary_protects_caller_array() -> None:
    class MutatingEnv:
        def step(self, action):
            action[:] = 42.0
            return None, 0.0, False, {}

    action = np.arange(7, dtype=np.float64)
    before = action.copy()
    safe_env_step(MutatingEnv(), action)
    np.testing.assert_array_equal(action, before)


def test_fresh_confirmation_is_exactly_five_per_task_and_disjoint() -> None:
    fresh = read_json(W20 / "wave20_fresh_confirmation_manifest.json")
    old = read_json(W19 / "wave19_dataset_split_manifest.json")
    rows = fresh["episodes"]
    assert len(rows) == 50
    assert Counter(row["task_id"] for row in rows) == Counter({task: 5 for task in range(10)})
    assert not {row["episode_id"] for row in rows} & {row["episode_id"] for row in old["episodes"]}
    assert all(row["terminal_official_success"] is True for row in rows)
    assert all(row["source_episode_sha256"] and row["branch_snapshot_sha256"] for row in rows)
    assert fresh["not_added_to_training"] is True
    assert fresh["wave19_final_test_read"] is False


def test_every_fresh_admitted_branch_is_exact() -> None:
    result = read_json(W20 / "wave20_snapshot_certification_results.json")
    assert result["all_prospective_branches_certified"] is True
    assert result["prospective_branch_count"] > 0
    for row in result["rows"]:
        assert row["certified"] is True
        assert row["state_discrepancy_max"] == 0.0
        assert row["controller_discrepancy_max"] == 0.0
        assert row["object_discrepancy_max"] == 0.0
        assert row["predicate_agreement"] is True
        assert row["replay_success_agreement"] is True
        assert row["nonfinite_count"] == 0


def test_representation_training_and_gate_are_complete_without_leakage() -> None:
    training = read_json(W20 / "wave20_representation_training_manifest.json")
    gate = read_json(W20 / "wave20_representation_gate.json")
    assert training["train_episode_count"] == 140
    assert training["fresh_confirmation_episode_count"] == 50
    assert training["confirmation_used_for_training"] is False
    assert training["old_final_test_read"] is False
    assert training["hyperparameter_sweep"] is False
    assert gate["all_six_seeds_complete"] is True
    assert gate["all_outputs_finite"] is True
    assert gate["old_final_test_read"] is False
    assert len(gate["seed_rows"]) == 6


def test_paired_checkpoint_architectures_match_and_no_extra_seed_exists() -> None:
    expected = {"seed_200820", "seed_201820", "seed_202820", "seed_203820", "seed_204820", "seed_205820"}
    root = W20 / "representation"
    assert {path.name for path in root.glob("seed_*")} == expected
    for seed_dir in root.glob("seed_*"):
        shapes = []
        for condition in CONDITIONS:
            payload = torch.load(seed_dir / condition / "checkpoint_ema.pt", map_location="cpu")
            shapes.append({name: tuple(value.shape) for name, value in payload["model_state_dict"].items()})
            assert payload["epoch"] == 40
            assert payload["ema_decay"] == 0.999
        assert shapes[0] == shapes[1] == shapes[2]


def test_gate_controls_downstream_authorization() -> None:
    gate = read_json(W20 / "wave20_representation_gate.json")
    frozen = W20 / "wave20_frozen_libero_representation_manifest.json"
    if gate["gate_pass"]:
        assert frozen.is_file()
        assert read_json(frozen)["old_final_test_read"] is False
    else:
        assert not frozen.exists()
        assert (W20 / "wave20_representation_gate_failure.md").is_file()
        assert not (W20 / "wave20_final_test_open_manifest.json").exists()


def test_offline_gate_uses_exact_f1_initializer_and_enforces_stop() -> None:
    offline = read_json(W20 / "wave20_offline_replication_gate.json")
    manifest = read_json(W20 / "wave20_frozen_model_manifest.json")
    f1 = torch.load(ROOT / manifest["F1"]["path"], map_location="cpu")["model_state_dict"]
    f2 = torch.load(ROOT / manifest["F2"]["path"], map_location="cpu")["model_state_dict"]
    for name, value in f1.items():
        torch.testing.assert_close(value, f2[f"initializer.{name}"], rtol=0.0, atol=0.0)
    assert manifest["F2_refinement_iterations"] == 4
    assert manifest["F2_exact_F1_initializer"] is True
    assert offline["cross_domain_offline_replication"] == "REJECTED"
    assert offline["test_split_read"] is False
    assert manifest["closed_loop_authorized"] is False
    assert not (W20 / "wave20_final_test_open_manifest.json").exists()

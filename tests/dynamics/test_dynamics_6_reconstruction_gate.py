"""Test the wave-18 reconstruction gate and mandatory-stop implementation.

Purpose
-------
Verify that wave 18 freezes the requested six-task protocol, recognizes that
rendered CALVIN observations are not complete simulator snapshots, preserves all
historical model files, and forbids embodied inference after a failed gate.

Parameters
----------
No command-line parameters. Pytest discovers this module.

Usage
-----
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python -m pytest \
  tests/dynamics/test_dynamics_6_reconstruction_gate.py -q

Outputs
-------
Pytest writes only its requested console/JUnit output. Tests do not modify model,
dataset, result, or checkpoint files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((ROOT / "configs/dynamics_6.yaml").read_text(encoding="utf-8"))
OUT = ROOT / CONFIG["experiment"]["output_root"]


def read_json(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_six_tasks_and_confirmatory_counts_are_exactly_frozen():
    assert CONFIG["source"]["primary_tasks"] == [
        "lift_blue_block_slider",
        "lift_red_block_table",
        "place_in_slider",
        "push_pink_block_right",
        "turn_off_lightbulb",
        "turn_on_lightbulb",
    ]
    assert CONFIG["closed_loop"]["minimum_episodes_per_task"] == 30
    assert CONFIG["closed_loop"]["minimum_total_episodes"] == 180
    assert CONFIG["closed_loop"]["maximum_task_fraction"] == 0.35


def test_branch_and_horizon_preregistration_matches_prompt():
    prereg = read_json("branch_point_preregistration.json")
    assert prereg["fractions"] == [0.25, 0.5, 0.75]
    assert prereg["horizons"] == [1, 2, 4, 8]
    assert prereg["frames_per_latent_step"] == 16
    assert prereg["cluster_unit"] == "source_episode"
    assert prereg["materialized_branch_points"] == 0


def test_rendered_source_schema_is_not_mislabeled_as_exact_state():
    sources = read_json("branch_source_manifest.json")
    validation = sources["official_debug"]["validation"]
    schema = validation["frame_schema"]
    assert "robot_obs" in schema and schema["robot_obs"]["shape"] == [15]
    assert "scene_obs" in schema and schema["scene_obs"]["shape"] == [24]
    assert "rel_actions" in schema and schema["rel_actions"]["shape"] == [7]
    assert "robot_joint_velocities" not in schema
    assert "movable_object_velocities" not in schema
    assert "controller_target" not in schema
    assert "bullet_state" not in schema
    assert not validation["contains_complete_simulator_snapshot"]
    assert sources["exactly_reconstructable_source_episodes"] == 0


def test_official_validation_segments_do_not_become_independent_episodes():
    sources = read_json("branch_source_manifest.json")
    validation = sources["official_debug"]["validation"]
    assert validation["authoritative_source_session_rows"] == 1
    assert validation["language_segments"] == 8
    assert not sources["source_adequacy_pass"]


def test_reconstruction_gate_failed_before_model_inference():
    gate = read_json("reconstruction_gate.json")
    assert gate["gate"] == "FAIL"
    assert gate["stop_condition_triggered"] == "required simulator state is unavailable"
    assert not gate["complete_source_snapshot"]
    assert not gate["embodied_model_comparison_authorized"]
    assert not gate["F1_F2_outputs_read"]
    assert not gate["primary_inference_run"]


def test_twin_contact_mismatch_and_missing_source_state_fail_gate():
    replay = read_json("zero_intervention_twin_replay_report.json")
    assert replay["diagnostic_segments"] == 6
    assert replay["terminal_task_predicate_agreement"] == 1.0
    assert replay["all_values_finite"]
    assert not replay["approximate_observation_reset_twins_pass"]
    assert not replay["exact_source_snapshot_available"]
    assert not replay["source_branch_reconstruction_pass"]
    assert not replay["overall_reconstruction_gate_passed"]
    for row in replay["segments"]:
        assert row["twin_component_max_abs_errors"]["contacts"] == 1.0
        assert all(
            value == 0.0
            for key, value in row["twin_component_max_abs_errors"].items()
            if key != "contacts"
        )


def test_diagnostic_ranges_stay_inside_official_session_boundary():
    sources = read_json("branch_source_manifest.json")
    lower, upper = sources["official_debug"]["validation"]["session_ranges"][0]
    replay = read_json("zero_intervention_twin_replay_report.json")
    for row in replay["segments"]:
        assert lower <= row["start_frame"] < row["end_frame"] <= upper


def test_all_simulator_gate_outputs_are_finite():
    replay = read_json("zero_intervention_twin_replay_report.json")
    numeric_fields = (
        "initial_source_robot_max_abs_error",
        "initial_source_scene_max_abs_error",
        "source_robot_error_median",
        "source_robot_error_p95",
        "source_robot_error_max",
        "source_scene_error_median",
        "source_scene_error_p95",
        "source_scene_error_max",
        "twin_error_median",
        "twin_error_p95",
        "twin_error_max",
    )
    for row in replay["segments"]:
        assert all(np.isfinite(row[key]) for key in numeric_fields)
        assert all(np.isfinite(value) for value in row["twin_component_max_abs_errors"].values())


def test_frozen_model_decoder_and_historical_del_hashes_unchanged():
    manifest = read_json("wave18_frozen_model_manifest.json")
    assert manifest["all_matched"]
    assert not manifest["models_loaded"]
    assert manifest["optimizer_or_training_calls"] == 0
    for row in manifest["files"]:
        assert row["matched"]
        assert sha256_file(ROOT / row["path"]) == row["expected_sha256"]
    assert {row["label"] for row in manifest["files"]} == {
        "representation", "semantic", "F1", "F2", "historical_DEL"
    }


def test_gate_runner_contains_no_training_or_model_execution_path():
    source = (ROOT / "scripts/dynamics/run_dynamics_6.py").read_text(encoding="utf-8")
    assert "import torch" not in source
    assert ".backward(" not in source
    assert "optimizer.step(" not in source
    assert "load_state_dict(" not in source
    assert "env_a.step(action.copy())" in source
    assert "env_b.step(action.copy())" in source


def test_action_preprocessing_matches_installed_calvin_interface():
    source = (
        ROOT / "third_party/calvin/calvin_env/calvin_env/robot/robot.py"
    ).read_text(encoding="utf-8")
    assert "max_rel_pos=0.02" in source
    assert "max_rel_orn=0.05" in source
    assert "use_target_pose=True" in source
    assert "self.target_pos += rel_pos" in source
    assert "self.target_orn += rel_orn" in source


def test_official_task_predicate_is_documented_and_installed():
    task_source = (
        ROOT / "third_party/calvin/calvin_env/calvin_env/envs/tasks.py"
    ).read_text(encoding="utf-8")
    audit = (OUT / "calvin_closed_loop_state_audit.md").read_text(encoding="utf-8")
    assert "def get_task_info_for_set" in task_source
    assert "Tasks.get_task_info_for_set" in audit


def test_all_closed_loop_outputs_are_explicitly_not_run_not_empty_results():
    manifest = read_json("closed_loop_not_run_manifest.json")
    assert manifest["status"] == "NOT_RUN_RECONSTRUCTION_GATE_FAILURE"
    assert not manifest["model_outputs_read"]
    assert len(manifest["artifacts"]) >= 20
    assert set(manifest["artifacts"].values()) == {"NOT_APPLICABLE_NOT_RUN"}


def test_claim_decision_does_not_overstate_gate_failure():
    claims = read_json("wave18_claim_decision.json")
    assert claims["reconstruction_gate"] == "FAIL"
    assert not claims["embodied_claim_authorized"]
    assert not claims["historical_claims_changed"]
    assert not claims["closed_loop_refinement_failed"]
    assert claims["closed_loop_refinement_outcome"] == "NOT_OBSERVED_STUDY_NOT_EXECUTED"
    assert claims["study_execution"] == "NOT_EXECUTED_EXACT_SOURCE_BRANCH_RECONSTRUCTION_UNAVAILABLE"
    for claim in ("C4_closed_loop_embodied_refinement", "C5_learned_correction_direction_causal_value", "C6_proposal_perturbation_recovery"):
        assert claims[claim] == "NOT_TESTED_RECONSTRUCTION_GATE_FAILURE"
    report = (ROOT / "reports/dynamics_6_results.md").read_text(encoding="utf-8")
    assert (
        "The planned closed-loop causal continuation study could not be executed because the retained public "
        "CALVIN artifacts do not permit exact reconstruction of source branch states."
    ) in report
    assert "This is not evidence that closed-loop refinement failed." in report

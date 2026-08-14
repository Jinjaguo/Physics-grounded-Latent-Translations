"""Validate Wave 23 train-only goal alignment and development stop protocol.

Purpose
-------
Check frozen Wave21 identities, exact train-only 75th-percentile goal cores,
K=20 neighborhoods, absence of held-out leakage and banned losses, six-way
language interventions, gradient isolation, registered clustered statistics,
and the no-held-out development-selection stop branch.

Parameters
----------
No command-line parameters; pytest discovers these tests.

Usage
-----
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python -m pytest \
  tests/dynamics/test_dynamics_11_goal_alignment.py -q

Outputs
-------
Pytest writes console/JUnit results. The final result is copied to
``results/dynamics/twenty_third_wave/.../tests_report.txt``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts.dynamics.run_dynamics_9 import load_representation, sha256
from scripts.dynamics.run_dynamics_10 import cycle_numpy


ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((ROOT / "configs/dynamics_11.yaml").read_text())
W21_CONFIG = yaml.safe_load((ROOT / CONFIG["wave21_config"]).read_text())
OUT = ROOT / CONFIG["experiment"]["output_root"]
W21 = ROOT / CONFIG["experiment"]["wave21_root"]


def load(name: str):
    return json.loads((OUT / name).read_text())


def finite_json(value):
    if isinstance(value, dict): return all(finite_json(item) for item in value.values())
    if isinstance(value, list): return all(finite_json(item) for item in value)
    if isinstance(value, float): return math.isfinite(value)
    return True


def test_frozen_wave21_hashes_split_and_inventory_unchanged():
    manifest = load("wave23_frozen_manifest.json")
    assert sha256(ROOT / manifest["representation_checkpoint"]) == manifest["representation_sha256"]
    assert sha256(W21 / "wave21_session_split_manifest.json") == manifest["session_split_sha256"]
    assert sha256(W21 / "wave21_transition_inventory.csv") == manifest["transition_inventory_sha256"]
    for seed, digest in manifest["Wave21_B1_hashes"].items():
        assert sha256(W21 / "checkpoints" / "B1_correct_language" / f"seed_{seed}.pt") == digest
    assert all(manifest[key] == 0 for key in ("representation_optimizer_steps", "encoder_optimizer_steps", "decoder_optimizer_steps", "text_encoder_optimizer_steps"))


def test_goal_cores_are_exact_train_only_75th_percentile_and_k20():
    manifest = load("wave23_goal_core_manifest.json")
    assert manifest["source_split"] == "train_only"
    assert manifest["test_samples_used"] == 0 and manifest["development_samples_used"] == 0
    assert manifest["primary_percentile"] == 75 and manifest["K"] == 20
    representation, _, _, _ = load_representation(W21_CONFIG, torch.device("cpu"))
    with np.load(W21 / "wave21_train_regions.npz") as regions, np.load(OUT / "wave23_goal_cores.npz") as cores:
        for task, row in manifest["goals"].items():
            values = regions[task].copy()
            _, _, correction = cycle_numpy(representation, values, torch.device("cpu"))
            residual = np.linalg.norm(correction, axis=1)
            threshold = float(np.percentile(residual, 75))
            assert np.isclose(threshold, row["thresholds"]["75"])
            assert len(cores[task]) == int(np.sum(residual <= threshold)) == row["primary_core_count"]
            assert len(cores[task]) >= 20


def test_model_is_single_factor_and_training_gradients_are_isolated():
    prereg = load("wave23_model_preregistration.json")
    assert prereg["K"] == 20 and prereg["lambda_candidates"] == [0.03, 0.1, 0.3]
    assert set(prereg["forbidden"].values()) == {0.0, False}
    records = load("wave23_training_records.json")
    assert len(records) == 18
    assert {row["seed"] for row in records} == set(CONFIG["model"]["seeds"])
    assert {row["lambda_align"] for row in records} == {0.03, 0.1, 0.3}
    assert all(row["gradient_audit"]["transition_gradients_nonzero"] for row in records)
    assert all(row["gradient_audit"]["representation_gradients_none"] for row in records)
    assert all(row["gradient_audit"]["classification_loss"] == row["gradient_audit"]["prototype_loss"] == row["gradient_audit"]["cycle_loss"] == 0.0 for row in records)


def test_lambda_selection_uses_development_only_and_stops_before_test():
    selection = load("wave23_alignment_weight_selection.json")
    assert selection["selection_split"] == "development_only"
    assert selection["candidate_set"] == [0.03, 0.1, 0.3]
    assert selection["status"] == "NO_CANDIDATE_PASSED" and selection["selected_lambda_align"] is None
    assert selection["held_out_test_opened"] is False and selection["no_new_lambda_allowed"] is True
    final = load("wave23_final_test_preregistration.json")
    assert final["status"] == "NOT_ACTIVATED_NO_DEVELOPMENT_CANDIDATE"
    assert final["held_out_test_opened"] is False
    assert final["bootstrap"] == {"cluster": "source_session", "replicates": 10000, "seed": 230823}


def test_same_state_sixway_changes_only_language_and_prototype_has_no_state_input():
    with np.load(OUT / "publication_figures_data" / "development_same_state_trajectories.npz") as archive:
        current = archive["z_current"]
        b1 = archive["Wave21_B1"]
        ga = archive["GA_lambda_0.03"]
        prototype = archive["language_prototype"]
    assert b1.shape == ga.shape == prototype.shape == (len(current), 6, 4, 32)
    assert current.shape[1] == 32
    assert np.any(np.linalg.norm(ga[:, 0] - ga[:, 1], axis=-1) > 0)
    assert np.allclose(np.var(prototype, axis=0), 0.0)


def test_phase_a_clustered_protocol_and_claim_statuses():
    phase = load("wave23_phaseA_results.json")
    assert phase["M1_goal_specific_executable_alignment"] == "SUPPORTED_FOR_INTERVENTION"
    assert phase["gates"] == {"A1": True, "A2": True, "A3": True, "A4": True, "A5": True}
    assert phase["optimizer_steps_before_decision"] == 0 and phase["test_opened"] is False
    for key in ("D1_margin_vs_correctness", "D2_distance_vs_decoded_error"):
        assert phase[key]["cluster"] == "source_session" and phase[key]["replicates"] == 10000
    claim = load("wave23_claim_decision.json")
    assert claim["C11_goal_specific_executable_alignment"] == "NOT_TESTED"
    assert claim["C12_language_as_goal_specific_executable_coordinate"] == "NOT_TESTED"
    assert claim["held_out_test_opened"] is False


def test_json_finite_and_all_stop_branch_deliverables_exist():
    for path in OUT.rglob("*.json"):
        assert finite_json(json.loads(path.read_text())), path
    required = [
        "twenty_third_wave_results.md", "twenty_third_wave_next_experiment.md", "wave23_frozen_manifest.json", "wave23_goal_core_manifest.json",
        "wave23_phaseA_goal_geometry.md", "wave23_goal_core_association_report.md", "wave23_mechanism_gate.json", "wave23_model_preregistration.json",
        "wave23_seed_preregistration.json", "wave23_alignment_weight_selection.json", "wave23_final_test_preregistration.json", "wave23_training_report.md",
        "wave23_statistical_report.md", "wave23_main_comparison.md", "wave23_same_state_language_swap.md", "wave23_decode_reencode_results.md",
        "wave23_continuity_results.md", "wave23_goal_geometry_analysis.md", "wave23_lift_to_place_case.md", "wave23_failure_taxonomy.md",
        "wave23_claim_decision.json", "exact_commands.sh", "environment_freeze.txt", "files_changed.txt", "updated_RESEARCH_LOG.md", "updated_NEXT_EXPERIMENT.md",
    ]
    assert all((OUT / name).exists() for name in required)
    assert len(list((OUT / "publication_figures").glob("figure_*.png"))) == 7
    assert len(list((OUT / "publication_tables").glob("table_*.csv"))) == 5

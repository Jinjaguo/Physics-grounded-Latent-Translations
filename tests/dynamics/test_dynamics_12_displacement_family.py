"""Validate Wave 24 paired displacement-family protocol and stop branch.

Purpose
-------
Check exact paired H1/H2/H4 reconstruction, frozen Wave21 identities and split,
train-only support/tau construction, development-only M2 statistics, the
registered no-training/no-test stop, same-state language diagnostics, and all
required publication artifacts.

Parameters
----------
No command-line parameters; pytest discovers these tests.

Usage
-----
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python -m pytest \
  tests/dynamics/test_dynamics_12_displacement_family.py -q

Outputs
-------
Pytest writes console output. The registered experiment copies it to
``results/dynamics/twenty_fourth_wave/2026-08-14_dynamics_12/tests_report.txt``.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((ROOT / "configs/dynamics_12.yaml").read_text())
OUT = ROOT / CONFIG["experiment"]["output_root"]
W21 = ROOT / CONFIG["experiment"]["wave21_root"]


def load(name: str):
    return json.loads((OUT / name).read_text())


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def finite_json(value):
    if isinstance(value, dict):
        return all(finite_json(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_json(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def test_frozen_wave21_identity_split_and_inputs_unchanged():
    manifest = load("wave24_frozen_manifest.json")
    assert digest(ROOT / manifest["representation_checkpoint"]) == manifest["representation_sha256"]
    assert digest(W21 / "wave21_session_split_manifest.json") == manifest["session_split_sha256"]
    assert digest(W21 / "wave21_transition_inventory.csv") == manifest["transition_inventory_sha256"]
    assert digest(W21 / "datasets" / "train.npz") == manifest["train_dataset_sha256"]
    assert digest(W21 / "datasets" / "development.npz") == manifest["development_dataset_sha256"]
    assert digest(W21 / "datasets" / "test.npz") == manifest["historical_test_dataset_sha256"]
    for seed, expected in manifest["Wave21_B1_hashes"].items():
        assert digest(W21 / "checkpoints" / "B1_correct_language" / f"seed_{seed}.pt") == expected
    assert all(manifest[key] == 0 for key in (
        "representation_optimizer_steps", "encoder_optimizer_steps",
        "decoder_optimizer_steps", "text_encoder_optimizer_steps",
        "Wave21_LCT_optimizer_steps_phaseA",
    ))


def test_paired_parquet_is_exact_and_test_arrays_are_masked():
    rows = pq.read_table(OUT / "wave24_paired_transition_inventory.parquet").to_pylist()
    assert len(rows) == 560
    assert {split: sum(row["split"] == split for row in rows) for split in ("train", "development", "test")} == {
        "train": 257, "development": 139, "test": 164,
    }
    assert all(row["source_frame_contiguous"] and not row["reset_or_discontinuity"] for row in rows)
    array_fields = ["z_previous", "z_current"] + [
        f"{name}_H{horizon}"
        for horizon in (1, 2, 4)
        for name in ("z_future", "delta", "future_actions")
    ]
    assert all(all(row[field] is None for field in array_fields) for row in rows if row["split"] == "test")
    assert all(all(row[field] is not None for field in array_fields) for row in rows if row["split"] != "test")

    for split in ("train", "development"):
        with np.load(W21 / "datasets" / f"{split}.npz") as data:
            lookup = {(int(data["session_row"][i]), int(data["boundary_frame"][i])): i for i in range(len(data["goal_id"]))}
            for row in (item for item in rows if item["split"] == split):
                index = lookup[(row["session_row"], row["boundary_frame"])]
                assert np.allclose(row["z_previous"], data["z_previous"][index])
                assert np.allclose(row["z_current"], data["z_current"][index])
                for horizon, hindex in ((1, 0), (2, 1), (4, 3)):
                    future = data["future_latents"][index, hindex]
                    assert np.allclose(row[f"z_future_H{horizon}"], future)
                    assert np.allclose(row[f"delta_H{horizon}"], future - data["z_current"][index])
                    assert np.allclose(row[f"future_actions_H{horizon}"], data["future_actions"][index, hindex].reshape(-1))


def test_supports_neighbors_and_tau_are_train_only_and_frozen():
    family = load("wave24_transition_family_manifest.json")
    horizon = load("wave24_horizon_core_manifest.json")
    static = load("wave24_static_core_manifest.json")
    assert family["source_split"] == horizon["source_split"] == "train_only"
    assert static["test_used"] is horizon["test_used"] is family["test_used"] is False
    assert family["development_used"] is False
    assert family["K"] == 20 and family["minimum_support"] == 8
    assert family["cell_count"] == len(family["eligible_cells"]) == 18
    assert family["adequate_goal_count"] == 6
    assert "query current latent" in family["neighbor_selection"] and "endpoint never used" in family["neighbor_selection"]

    with np.load(W21 / "datasets" / "train.npz") as train:
        for goal, task in enumerate(family["tau_by_goal"]):
            values = train["z_current"][train["goal_id"] == goal, 16:]
            distance = np.linalg.norm(values[:, None] - values[None], axis=-1)
            np.fill_diagonal(distance, np.inf)
            expected = float(np.median(np.partition(distance, 19, axis=1)[:, :20]))
            assert np.isclose(family["tau_by_goal"][task], expected)
            for h in (1, 2, 4):
                cell = family["statistics"][f"{task}__H{h}"]
                assert cell["train_transition_count"] == len(values)
                assert cell["adequate"] and cell["K_used"] == 20
                assert np.isclose(cell["tau_train_only"], expected)


def test_m2_uses_registered_cluster_bootstrap_and_stops_before_training_or_test():
    phase = load("wave24_phaseA_results.json")
    gate = load("wave24_mechanism_gate.json")
    assert phase["M2_state_horizon_conditioned_displacement_family"] == "REJECTED"
    assert phase["gates"] == gate["gates"] == {
        "A1": False, "A2": True, "A3": True,
        "A4": False, "A5": False, "A6": False,
    }
    assert phase["optimizer_steps_before_decision"] == 0 and phase["test_arrays_opened"] is False
    for result in (
        phase["A1_HorizonCoreGain"], phase["A2_D2_cosine"]["full"],
        phase["A2_D2_cosine"]["execution"],
        phase["A3_D2_minus_goal_mean"]["full_MSE_improvement"],
        phase["A3_D2_minus_goal_mean"]["execution_MSE_improvement"],
    ):
        assert result["cluster"] == "source_session" and result["replicates"] == 10000
    selection = load("wave24_transition_weight_selection.json")
    final = load("wave24_final_test_preregistration.json")
    claim = load("wave24_claim_decision.json")
    assert selection["status"] == "NOT_RUN_M2_REJECTED" and selection["selected_lambda_TM"] is None
    assert final["status"] == "NOT_ACTIVATED_M2_REJECTED" and final["bootstrap"] == {
        "cluster": "source_session", "replicates": 10000, "seed": 240824,
    }
    assert claim["C13_language_selects_state_conditioned_transition_family"] == "NOT_TESTED"
    assert claim["C14_language_as_state_horizon_conditioned_executable_transition_selector"] == "NOT_TESTED"
    assert not (OUT / "checkpoints").exists()


def test_only_registered_factor_and_same_state_language_diagnostic():
    prereg = load("wave24_model_preregistration.json")
    assert prereg["K"] == 20 and prereg["lambda_TM_candidates"] == [0.03, 0.1, 0.3]
    assert set(prereg["forbidden"].values()) == {0.0, False}
    with np.load(OUT / "publication_figures_data" / "development_same_state_displacements.npz") as archive:
        current = archive["z_current"]
        b1 = archive["Wave21_B1"]
        d2 = archive["D2_diagnostic"]
    assert current.shape == (139, 32)
    assert b1.shape == (139, 6, 4, 32)
    assert d2.shape == (139, 6, 3, 32)
    assert np.any(np.linalg.norm(b1[:, 0, -1] - b1[:, 1, -1], axis=-1) > 0)
    assert np.any(np.linalg.norm(d2[:, 0, -1] - d2[:, 1, -1], axis=-1) > 0)


def test_json_is_finite_and_all_required_stop_branch_artifacts_exist():
    for path in OUT.rglob("*.json"):
        assert finite_json(json.loads(path.read_text())), path
    required = [
        "twenty_fourth_wave_results.md", "twenty_fourth_wave_next_experiment.md",
        "wave24_frozen_manifest.json", "wave24_split_freeze.json",
        "wave24_paired_transition_inventory.parquet", "wave24_paired_transition_inventory_report.md",
        "wave24_static_core_manifest.json", "wave24_horizon_core_manifest.json",
        "wave24_transition_family_manifest.json", "wave24_phaseA_horizon_core_diagnosis.md",
        "wave24_phaseA_source_conditioned_displacement.md", "wave24_mechanism_gate.json",
        "wave24_model_preregistration.json", "wave24_seed_preregistration.json",
        "wave24_transition_weight_selection.json", "wave24_final_test_preregistration.json",
        "wave24_training_report.md", "wave24_statistical_report.md", "wave24_main_comparison.md",
        "wave24_same_state_language_swap.md", "wave24_transition_family_results.md",
        "wave24_decode_reencode_results.md", "wave24_continuity_results.md",
        "wave24_lift_to_place_case.md", "wave24_failure_taxonomy.md",
        "wave24_claim_decision.json", "exact_commands.sh", "environment_freeze.txt",
        "files_changed.txt", "updated_RESEARCH_LOG.md", "updated_NEXT_EXPERIMENT.md",
    ]
    assert all((OUT / name).exists() for name in required)
    assert len(list((OUT / "publication_figures").glob("figure_*.png"))) == 7
    assert len(list((OUT / "publication_tables").glob("table_*.csv"))) == 6
    assert len([line for line in (OUT / "twenty_fourth_wave_results.md").read_text().splitlines() if line[:1].isdigit()]) == 32

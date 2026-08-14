"""Validate the frozen Wave 22 cycle-consistency diagnosis and stop branch.

Purpose
-------
Check frozen Wave21 identities, exact E(D(.)) behavior, transition-only cycle
gradients, forbidden-loss absence, split/inventory reuse, preregistered K/lambda/
bootstrap/test discipline, valid finite outputs, and the no-training stop rule.

Parameters
----------
No command-line parameters; pytest discovers these tests.

Usage
-----
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python -m pytest \
  tests/dynamics/test_dynamics_10_cycle_consistency.py -q

Outputs
-------
Pytest writes only requested console/JUnit output. The completed experiment
copies the result to ``results/dynamics/twenty_second_wave/.../tests_report.txt``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts.dynamics.run_dynamics_9 import LCT, load_representation, sha256
from scripts.dynamics.run_dynamics_10 import decode_and_cycle_tensor


ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((ROOT / "configs/dynamics_10.yaml").read_text())
W21_CONFIG = yaml.safe_load((ROOT / CONFIG["wave21_config"]).read_text())
OUT = ROOT / CONFIG["experiment"]["output_root"]
W21 = ROOT / CONFIG["experiment"]["wave21_root"]


def load(name: str):
    return json.loads((OUT / name).read_text())


def finite_json(value):
    if isinstance(value, dict):
        return all(finite_json(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_json(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def test_wave21_hashes_and_protocol_inputs_are_unchanged():
    manifest = load("wave22_frozen_wave21_manifest.json")
    assert sha256(ROOT / manifest["representation_checkpoint"]) == manifest["representation_sha256"]
    for condition, key in (("B0_unconditional", "wave21_B0_hashes"), ("B1_correct_language", "wave21_B1_hashes"), ("B2_shuffled_language", "wave21_B2_hashes")):
        for seed, digest in manifest[key].items():
            assert sha256(W21 / "checkpoints" / condition / f"seed_{seed}.pt") == digest
    assert sha256(W21 / "wave21_session_split_manifest.json") == manifest["session_split_sha256"]
    assert sha256(W21 / "wave21_transition_inventory.csv") == manifest["boundary_inventory_sha256"]
    assert manifest["phase_a_uses_historical_wave21_heldout_outputs"] is True


def test_cycle_map_is_exact_frozen_decoder_then_encoder():
    representation, _, _, _ = load_representation(W21_CONFIG, torch.device("cpu"))
    latent = torch.randn(3, 32)
    decoded, actual = decode_and_cycle_tensor(representation, latent)
    manual_input = decoded.clone()
    manual_input[..., 6] = torch.where(decoded[..., 6] >= 0, 1.0, -1.0)
    expected = representation.encode(manual_input)
    assert torch.equal(actual, expected)
    assert all(not parameter.requires_grad for parameter in representation.parameters())


def test_cycle_loss_gradients_reach_only_transition_model():
    representation, _, _, _ = load_representation(W21_CONFIG, torch.device("cpu"))
    model = LCT(True, W21_CONFIG)
    previous = torch.randn(4, 32); current = torch.randn(4, 32); goal = torch.randn(4, 16)
    predicted = model.rollout(previous, current, goal, steps=2)
    _, cycled = decode_and_cycle_tensor(representation, predicted.flatten(0, 1))
    loss = (cycled - predicted.flatten(0, 1)).square().mean()
    loss.backward()
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0 for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in representation.parameters())
    assert all(not parameter.requires_grad for parameter in representation.parameters())


def test_single_factor_objective_contains_no_banned_losses():
    prereg = load("wave22_model_preregistration.json")
    assert prereg["architecture"] == "exact Wave21 B1 LCT architecture"
    assert set(prereg["forbidden_losses"].values()) == {0.0}
    assert prereg["same_wave21_split_inventory_chunking_and_gap_handling"] is True
    assert prereg["trainable"].startswith("transition model only")


def test_same_state_exact_k4_and_registered_selection_protocol():
    geometry = load("publication_figures_data/geometry_diagnostics.json")
    assert set(geometry["same_state"]) == {"Wave21_B1", "Wave21_B1_cycle4_diagnostic"}
    assert load("publication_figures_data/cycle_projection_diagnostic.json")["K_cycle"] == 4
    selection = load("wave22_cycle_weight_selection.json")
    assert selection["candidates"] == [0.1, 0.3, 1.0]
    assert selection["selection_split"] == "development only"
    assert selection["selected_lambda_cycle"] is None
    assert selection["held_out_LCTCC_test_used"] is False


def test_phase_a_stop_prevented_training_and_new_test_opening():
    phase = load("wave22_phaseA_results.json")
    assert phase["M0_decoder_consistency_mechanism"] == "REJECTED"
    assert phase["optimizer_steps_before_decision"] == 0
    assert phase["gates"] == {"A1": True, "A2": True, "A3": True, "A4": True, "A5": False}
    assert not (OUT / "checkpoints").exists()
    final = load("wave22_final_test_preregistration.json")
    assert final["LCT_CC_heldout_predictions_opened"] is False
    assert final["bootstrap"] == {"cluster": "source_session", "replicates": 10000, "seed": 220822}


def test_claim_statuses_json_validity_finiteness_and_deliverables():
    claim = load("wave22_claim_decision.json")
    assert claim["M0_decoder_consistency_mechanism"] == "REJECTED"
    assert claim["C9_executable_language_redirect"] == "NOT_TESTED"
    assert claim["C10_language_as_executable_target_coordinate"] == "NOT_TESTED"
    for path in OUT.rglob("*.json"):
        payload = json.loads(path.read_text())
        assert finite_json(payload), path
    required = [
        "twenty_second_wave_results.md", "twenty_second_wave_next_experiment.md", "wave22_frozen_wave21_manifest.json",
        "wave22_phaseA_cycle_diagnosis.md", "wave22_cycle_association_report.md", "wave22_cycle_projection_diagnostic.md",
        "wave22_cycle_weight_selection.json", "wave22_seed_preregistration.json", "wave22_model_preregistration.json",
        "wave22_final_test_preregistration.json", "wave22_training_report.md", "wave22_statistical_report.md",
        "wave22_main_comparison.md", "wave22_same_state_language_swap.md", "wave22_decode_reencode_results.md",
        "wave22_continuity_results.md", "wave22_geometry_analysis.md", "wave22_lift_to_place_case.md",
        "wave22_failure_taxonomy.md", "wave22_claim_decision.json", "exact_commands.sh", "environment_freeze.txt",
        "files_changed.txt", "updated_RESEARCH_LOG.md", "updated_NEXT_EXPERIMENT.md",
    ]
    assert all((OUT / name).exists() for name in required)
    assert len(list((OUT / "publication_figures").glob("figure_*.png"))) == 6
    assert len(list((OUT / "publication_tables").glob("table_*.csv"))) == 4

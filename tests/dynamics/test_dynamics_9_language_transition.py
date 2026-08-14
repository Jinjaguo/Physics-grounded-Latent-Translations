"""Validate the frozen Wave 21 language-conditioned transition protocol.

Purpose
-------
Check the completed Wave 21 artifacts for physical/session continuity, frozen
representation identity, causal input isolation, correct controls, clustered
statistics, finite outputs, and valid JSON.

Parameters
----------
No command-line parameters; pytest discovers these tests.

Usage
-----
PYTHONPATH=src python -m pytest \
  tests/dynamics/test_dynamics_9_language_transition.py -q

Outputs
-------
Pytest writes only its requested console/JUnit report.  The experiment copies
the final result to ``results/dynamics/twenty_first_wave/.../tests_report.txt``.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts.dynamics.run_dynamics_9 import LCT, sha256


ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((ROOT / "configs/dynamics_9.yaml").read_text())
OUT = ROOT / CONFIG["experiment"]["output_root"]


def load(name: str):
    return json.loads((OUT / name).read_text())


def test_frozen_representation_and_decoder_unchanged():
    manifest = load("wave21_frozen_representation_manifest.json")
    assert sha256(ROOT / manifest["checkpoint"]) == manifest["checkpoint_sha256"]
    assert manifest["representation_optimizer_steps"] == 0
    assert manifest["decoder_optimizer_steps"] == 0
    assert manifest["text_encoder_optimizer_steps"] == 0
    assert manifest["ema_updates"] == 0


def test_boundary_chunks_and_session_split_are_physical_and_disjoint():
    rows = list(csv.DictReader((OUT / "wave21_transition_inventory.csv").open()))
    assert rows and all(row["source_frame_contiguous"] == "True" for row in rows)
    assert all(row["reset_or_discontinuity"] == "False" for row in rows)
    assert all(row["h4_supported"] == "True" for row in rows)
    split = load("wave21_session_split_manifest.json")["sessions"]
    sets = [set(split[name]) for name in ("train", "development", "test")]
    assert sets[0].isdisjoint(sets[1]) and sets[0].isdisjoint(sets[2]) and sets[1].isdisjoint(sets[2])
    assert load("wave21_action_region_manifest.json")["training_only"] is True


def test_model_and_control_information_sets():
    b0 = LCT(False, CONFIG); b1 = LCT(True, CONFIG)
    zp = torch.randn(4, 32); zc = torch.randn(4, 32); g = torch.randn(4, 16)
    assert b0.uses_language is False and b1.uses_language is True
    assert torch.equal(b0.rollout(zp, zc, None), b0.rollout(zp, zc, None))
    assert not torch.equal(b1.rollout(zp, zc, g), b1.rollout(zp, zc, torch.zeros_like(g)))
    records = load("wave21_training_records.json")
    assert all(record["future_actions_as_input"] is False for record in records)
    assert all(record["target_region_loss"] is False for record in records)
    shuffled = [record for record in records if record["condition"] == "B2_shuffled_language"]
    assert shuffled and all(record["shuffle_audit"]["frequency_preserved"] for record in shuffled)


def test_same_state_interventions_share_state_and_only_language_changes():
    with np.load(OUT / "wave21_same_state_trajectories.npz") as archive:
        trajectories = archive["trajectories"]
        current = archive["z_current"]
    assert trajectories.shape[1:] == (6, 4, 32)
    assert current.shape == (len(trajectories), 32)
    assert load("final_integrity.json")["same_state_language_only_intervention"] is True


def test_clustered_statistics_and_registered_bootstrap():
    claim = load("wave21_claim_decision.json")
    for key in ("RedirectGain", "execution_RedirectGain", "B0_minus_B1_H2_execution", "B0_minus_B1_H4_decoded"):
        row = claim["primary_metrics"][key]
        assert row["cluster"] == "source_session"
        assert row["replicates"] == 10000
        assert np.isfinite([row["mean"], row["lower_95"], row["upper_95"]]).all()


def test_all_json_valid_and_numeric_outputs_finite():
    for path in OUT.rglob("*.json"):
        json.loads(path.read_text())
    assert load("final_integrity.json")["all_outputs_finite"] is True
    metrics = load("wave21_main_metrics.json")
    for values in metrics["model_table"].values():
        assert np.isfinite(list(values.values())).all()


def test_final_test_was_frozen_before_opening():
    prereg = load("wave21_final_test_preregistration.json")
    assert prereg["held_out_test_opened_before_freeze"] is False
    assert prereg["post_test_tuning_allowed"] is False
    assert load("final_integrity.json")["test_opened_after_preregistration"] is True

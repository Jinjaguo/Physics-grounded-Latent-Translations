"""Validate the registered Wave 26 phase-aware structured-flow experiment.

Purpose
-------
Check frozen representation/split identity, causal history reconstruction,
S0--S7 and D0--D3 availability decisions, Wave25 anchor reproduction,
development-family coverage, preregistration-before-heldout discipline,
source-session bootstrap settings, finite heldout results, and deliverables.

Parameters
----------
No command-line parameters; pytest discovers this module after the registered
Wave 26 run has completed.

Usage
-----
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python -m pytest \
  tests/dynamics/test_dynamics_14_phase_flow.py -q

Outputs
-------
Writes pytest console output; the experiment copies it to
``results/dynamics/twenty_sixth_wave/2026-08-14_dynamics_14/tests_report.txt``.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import yaml

from scripts.dynamics.run_dynamics_9 import sha256


ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((ROOT / "configs/dynamics_14.yaml").read_text())
OUT = ROOT / CONFIG["experiment"]["output_root"]
W21 = ROOT / CONFIG["experiment"]["wave21_root"]


def load(name: str):
    return json.loads((OUT / name).read_text())


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def test_frozen_assets_split_and_horizon_protocol_are_unchanged():
    manifest = load("wave26_frozen_manifest.json")
    assert sha256(ROOT / manifest["representation_checkpoint"]) == manifest["representation_sha256"]
    assert sha256(W21 / "wave21_session_split_manifest.json") == manifest["session_split_sha256"]
    assert sha256(W21 / "datasets/train.npz") == manifest["train_dataset_sha256"]
    assert sha256(W21 / "datasets/development.npz") == manifest["development_dataset_sha256"]
    assert manifest["corrected_horizon_indices"] == [0, 1, 3]
    assert all(manifest[key] == 0 for key in ("representation_optimizer_steps", "encoder_optimizer_steps", "decoder_optimizer_steps", "text_encoder_optimizer_steps"))


def test_causal_state_history_contact_and_data_scale_audits_are_explicit():
    features = load("wave26_state_feature_manifest.json")
    assert set(features) == {"S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"}
    assert features["S7"]["status"] == "UNAVAILABLE"
    for state in ("S0", "S1", "S2", "S3", "S4", "S5", "S6"):
        assert features[state]["latest_input_time"] == "query time"
        assert features[state]["future_inputs"] == []
    scale = load("wave26_data_scale_manifest.json")
    assert scale["nested"] and scale["D3"]["status"] == "UNAVAILABLE"
    sessions = [set(scale["conditions"][key]["sessions"]) for key in ("D0", "D1", "D2")]
    assert sessions[0] <= sessions[1] <= sessions[2]
    contact = (OUT / "wave26_contact_proxy_audit.md").read_text()
    assert "not ground-truth contact" in contact and "S7" in contact


def test_wave25_anchor_reproduces_and_development_covers_all_requested_axes():
    inventory = load("wave26_sweep_inventory.json")
    assert inventory["phase_reproduction_max_abs_metric_delta"] <= 1e-7
    assert len(inventory["selected_states"]) == 3
    assert len(inventory["state_models"]) == 21
    families = {name.rsplit("_", 1)[-1] for name in inventory["flow_models"] if name.startswith("Flow_") and not name.endswith("16step")}
    assert {"Phase-CFM", "History-CFM", "Prior-CFM", "R-CFM", "Streaming-CFM", "TC-CFM", "Hetero-CFM", "MP-CFM"} <= families
    assert len(inventory["objective_models"]) == 6
    metrics = load("wave26_development_metrics.json")
    assert all(finite(value) for value in metrics.values())
    assert len([name for name in metrics if name.startswith("Scale_")]) == 9


def test_final_selection_was_frozen_before_one_shot_heldout():
    selection = load("wave26_final_candidate_selection.json")
    prereg = load("wave26_final_test_preregistration.json")
    open_audit = load("wave26_heldout_open_audit.json")
    assert 1 <= len(selection["selected_models"]) <= 3
    assert selection["created_before_heldout_open"] and not selection["heldout_opened"]
    assert prereg["frozen_before_heldout_arrays_loaded"]
    assert prereg["bootstrap"] == {"cluster": "source_session", "replicates": 10000, "seed": 260826}
    assert open_audit["opened_after_preregistration"] and not open_audit["winner_tuning"]
    assert open_audit["selected_models_only"] == selection["selected_models"]
    held = load("wave26_heldout_metrics.json")
    assert set(held) == set(selection["selected_models"])
    assert all(finite(value) for value in held.values())


def test_claims_switch_return_efficiency_and_scorecard_are_complete():
    claims = load("wave26_claim_matrix.json")
    decisions = {"SUPPORTED", "NOT_SUPPORTED", "MIXED", "NOT_TESTED"}
    for key in ("C18_rich_causal_state_matters", "C19_continuous_flow_strongest_family", "C20_enriched_flow_reduces_identity_continuity_tradeoff", "C21_more_transition_data_helps", "C22_language_and_state_shape_transition_distribution"):
        assert claims[key] in decisions
    assert isinstance(claims["READY_FOR_RETARGETING_TEST"], bool)
    switches = load("wave26_offline_switch_metrics.json")
    returns = load("wave26_history_return_metrics.json")
    efficiency = load("wave26_efficiency_metrics.json")
    assert all(value["only_language_changed_at_switch"] for value in switches.values())
    assert all(not value["physical_time_reversal_tested"] for value in returns.values())
    assert all(value["inference_ms_per_query"] > 0 for value in efficiency.values())
    scorecard = list(csv.DictReader((OUT / "wave26_development_scorecard.csv").open()))
    assert scorecard and set(scorecard[0]) == {"model", "family", "PRED", "DECODE", "IDENTITY", "CYCLE-ID", "CONT", "LANG"}


def test_reports_tables_and_figure_data_are_complete():
    required = [
        "twenty_sixth_wave_results.md", "twenty_sixth_wave_next_experiment.md",
        "wave26_frozen_manifest.json", "wave26_dataset_audit.md", "wave26_data_scale_manifest.json",
        "wave26_phase_diagnostics.md", "wave26_contact_proxy_audit.md", "wave26_state_sweep_results.md",
        "wave26_flow_family_results.md", "wave26_objective_sweep_results.md", "wave26_nonflow_control_results.md",
        "wave26_data_scale_results.md", "wave26_development_scorecard.csv", "wave26_development_pareto.csv",
        "wave26_final_candidate_selection.json", "wave26_model_preregistration.json", "wave26_seed_preregistration.json",
        "wave26_final_test_preregistration.json", "wave26_heldout_results.md", "wave26_claim_matrix.json",
        "wave26_same_state_language_switch.md", "wave26_same_language_state_ablation.md",
        "wave26_retargeting_compatibility.md", "wave26_history_return_compatibility.md", "wave26_lift_to_place_case.md",
        "wave26_efficiency_report.md", "wave26_failure_taxonomy.md", "wave26_statistical_report.md",
        "exact_commands.sh", "environment_freeze.txt", "files_changed.txt", "updated_RESEARCH_LOG.md", "updated_NEXT_EXPERIMENT.md",
    ]
    assert all((OUT / name).exists() for name in required)
    assert len(list((OUT / "publication_tables").glob("table_*.csv"))) == 9
    assert len(list((OUT / "publication_figures_data").glob("figure_*.csv"))) == 8
    questions = [line for line in (OUT / "twenty_sixth_wave_results.md").read_text().splitlines() if line[:1].isdigit()]
    assert len(questions) == 40
    for path in OUT.rglob("*.json"):
        assert finite(json.loads(path.read_text())), path

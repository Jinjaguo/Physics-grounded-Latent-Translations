"""Validate Wave 25 broad transition sweep and no-heldout stop protocol.

Purpose
-------
Check frozen Wave21/24 identities, causal features, train-only clustering,
finite common-schema results across all requested families, oracle isolation,
fixed-seed generative reproducibility, development-only selection, held-out
masking, canonical case analysis, and complete reports.

Parameters
----------
No command-line parameters; pytest discovers these tests.

Usage
-----
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python -m pytest \
  tests/dynamics/test_dynamics_13_broad_sweep.py -q

Outputs
-------
Pytest writes console output. The registered run copies it to
``results/dynamics/twenty_fifth_wave/2026-08-14_dynamics_13/tests_report.txt``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts.dynamics.run_dynamics_9 import sha256
from scripts.dynamics.run_dynamics_13 import load_context, load_npz, load_predictor


ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((ROOT / "configs/dynamics_13.yaml").read_text())
OUT = ROOT / CONFIG["experiment"]["output_root"]
W21 = ROOT / CONFIG["experiment"]["wave21_root"]
W24 = ROOT / CONFIG["experiment"]["wave24_root"]


def load(name: str):
    return json.loads((OUT / name).read_text())


def finite(value):
    if isinstance(value, dict): return all(finite(item) for item in value.values())
    if isinstance(value, list): return all(finite(item) for item in value)
    if isinstance(value, float): return math.isfinite(value)
    return True


def test_frozen_assets_split_and_optimizer_status_are_unchanged():
    manifest = load("wave25_frozen_manifest.json")
    assert sha256(ROOT / manifest["representation_checkpoint"]) == manifest["representation_sha256"]
    assert sha256(W21 / "wave21_session_split_manifest.json") == manifest["session_split_sha256"]
    assert sha256(W21 / "wave21_transition_inventory.csv") == manifest["transition_inventory_sha256"]
    assert sha256(W21 / "datasets/train.npz") == manifest["train_dataset_sha256"]
    assert sha256(W21 / "datasets/development.npz") == manifest["development_dataset_sha256"]
    assert sha256(W24 / "wave24_paired_transition_inventory.parquet") == manifest["Wave24_paired_parquet_sha256"]
    assert all(manifest[key] == 0 for key in (
        "representation_optimizer_steps", "encoder_optimizer_steps",
        "decoder_optimizer_steps", "text_encoder_optimizer_steps",
    ))
    assert manifest["heldout_arrays_materialized"] is False


def test_dataset_and_features_are_causal_and_source_session_disjoint():
    audit_text = (OUT / "wave25_dataset_audit.md").read_text()
    assert "257" in audit_text and "139" in audit_text and "164" in audit_text
    features = load("wave25_feature_manifest.json")
    for variant in ("base", "phase"):
        assert features[variant]["future_inputs"] == []
        assert "z_previous" in features[variant]["inputs"]
        assert "z_current" in features[variant]["inputs"]
        assert "delta_previous" in features[variant]["inputs"]
    prereg = load("wave25_model_preregistration.json")
    assert set(prereg["forbidden_inputs"]) == {"future latent", "future action", "future task label", "future contact", "future simulator state"}
    split = json.loads((W21 / "wave21_session_split_manifest.json").read_text())
    groups = [set(split["sessions"][name]) for name in ("train", "development", "test")]
    assert groups[0].isdisjoint(groups[1]) and groups[0].isdisjoint(groups[2]) and groups[1].isdisjoint(groups[2])


def test_direction_modes_and_magnitudes_are_train_only_normalized_and_finite():
    modes = load("wave25_direction_modes.json")
    assert modes["fit_split"] == "train_only" and set(modes["candidate_counts"]) == {"1", "2", "3", "4"}
    for count, cells in modes["candidate_counts"].items():
        assert len(cells) == 18
        for cell in cells.values():
            centers = np.asarray(cell["centers"])
            assert centers.shape == (int(count), 32)
            assert np.allclose(np.linalg.norm(centers, axis=1), 1.0, atol=1e-5)
            assert np.isfinite(cell["log_magnitude"]).all()
    magnitudes = load("wave25_magnitude_modes.json")
    assert magnitudes["fit_split"] == "train_only" and len(magnitudes["cells"]) == 18


def test_broad_sweep_covers_all_families_and_results_are_finite():
    metrics = load("wave25_development_metrics.json")
    assert len(metrics) == 66
    families = {value["model_family"] for value in metrics.values()}
    assert {"historical", "deterministic_local", "direction_magnitude", "discrete_modes", "MDN", "MoE", "cVAE", "flow", "diffusion", "retrieval_augmented", "phase_augmented"} <= families
    assert all(finite(value) for value in metrics.values())
    assert all(value["parameter_count"] < 2_000_000 for value in metrics.values())
    assert metrics["D2_Wave24"]["dev_metrics"]["H2"]["full_mse"] == np.float32(1.2081164)
    assert np.isclose(metrics["D2_Wave24"]["dev_metrics"]["H4"]["decoded_mse"], 0.0541480221)


def test_oracles_are_isolated_and_distribution_sampling_is_reproducible():
    oracle = load("wave25_oracle_metrics.json")
    assert oracle["causal_performance"] is False
    assert all(value.get("causal_performance") is False for key, value in oracle.items() if key.startswith("O4_"))
    context = load_context(CONFIG, torch.device("cpu"))
    train = load_npz(W21 / "datasets/train.npz")
    dev = load_npz(W21 / "datasets/development.npz")
    predictor = load_predictor("Latent_CFM_8step_mean8", train, context, CONFIG, torch.device("cpu"))
    first = predictor({key: value[:4] for key, value in dev.items()}, dev["goal_id"][:4])
    second = predictor({key: value[:4] for key, value in dev.items()}, dev["goal_id"][:4])
    assert np.array_equal(first, second)


def test_selection_stops_before_heldout_and_claims_remain_not_tested():
    selection = load("wave25_final_candidate_selection.json")
    prereg = load("wave25_final_test_preregistration.json")
    claims = load("wave25_claim_decision.json")
    assert selection["eligible_models"] == selection["selected_models"] == []
    assert selection["heldout_opened"] is False and prereg["heldout_opened_before_freeze"] is False
    assert prereg["bootstrap"] == {"cluster": "source_session", "replicates": 10000, "seed": 250825}
    assert claims["C15_distributional_language_conditioned_transition"] == "NOT_TESTED_NO_DEVELOPMENT_CANDIDATE"
    assert claims["C16_executable_language_conditioned_transition_modes"] == "NOT_TESTED"
    assert claims["C17_language_and_state_shape_transition_distribution"] == "NOT_TESTED"
    assert claims["heldout_opened"] is False
    assert not (OUT / "wave25_heldout_metrics.json").exists()


def test_reports_canonical_case_and_required_artifacts_are_complete():
    required = [
        "twenty_fifth_wave_results.md", "twenty_fifth_wave_next_experiment.md",
        "wave25_frozen_manifest.json", "wave25_dataset_audit.md",
        "wave25_distribution_diagnostics.md", "wave25_direction_modes.json",
        "wave25_magnitude_modes.json", "wave25_cancellation_analysis.md", "wave25_oracle_suite.md",
        "wave25_deterministic_family_results.md", "wave25_factorized_direction_magnitude_results.md",
        "wave25_discrete_mode_results.md", "wave25_mdn_results.md", "wave25_moe_results.md",
        "wave25_cvae_results.md", "wave25_flow_results.md", "wave25_diffusion_results.md",
        "wave25_retrieval_augmented_results.md", "wave25_phase_augmented_results.md",
        "wave25_development_pareto.csv", "wave25_final_candidate_selection.json",
        "wave25_model_preregistration.json", "wave25_seed_preregistration.json",
        "wave25_final_test_preregistration.json", "wave25_heldout_results.md",
        "wave25_same_state_language_swap.md", "wave25_same_language_different_state.md",
        "wave25_retargeting_compatibility.md", "wave25_history_return_compatibility.md",
        "wave25_lift_to_place_case.md", "wave25_claim_decision.json",
        "wave25_failure_taxonomy.md", "wave25_statistical_report.md",
        "wave25_future_implementation_plan.md", "wave25_execution_log.md",
        "exact_commands.sh", "environment_freeze.txt", "files_changed.txt",
        "updated_RESEARCH_LOG.md", "updated_NEXT_EXPERIMENT.md",
    ]
    assert all((OUT / name).exists() for name in required)
    assert len(list((OUT / "publication_figures").glob("figure_*.png"))) == 7
    assert len(list((OUT / "publication_tables").glob("table_*.csv"))) == 6
    questions = [line for line in (OUT / "twenty_fifth_wave_results.md").read_text().splitlines() if line[:1].isdigit()]
    assert len(questions) == 37
    lift = np.genfromtxt(OUT / "publication_figures_data/development_lift_to_place.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    assert len(lift) > 0 and set(np.atleast_1d(lift["horizon"]).tolist()) == {1, 2, 4}
    for path in OUT.rglob("*.json"):
        assert finite(json.loads(path.read_text())), path

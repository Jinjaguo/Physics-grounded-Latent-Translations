"""Tests for the amended wave-16 public CALVIN H1/H2 replication.

Purpose
-------
Detect selection drift, overlap/padding, action incompatibility, H4/H8 access,
model mutation, window-level bootstrap, non-finite metrics, or missing explicit
H1/H2-only reporting in the post-audit external replication.

Parameters
----------
No custom parameters; pytest provides its standard command-line options.

Usage
-----
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  tests/dynamics/test_dynamics_4_external_h12.py -q

Outputs
-------
Pytest pass/fail output; the formal full-suite run writes JUnit XML into the
amended wave-16 external-H1/H2 result directory.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import yaml

from pglt.dynamics.dynamics_data import sha256_file
from pglt.dynamics.long_horizon import supported_rollout_offsets


ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((ROOT / "configs/dynamics_4_external_h12.yaml").read_text(encoding="utf-8"))
ACQUISITION = ROOT / CONFIG["experiment"]["acquisition_root"]
OUT = ROOT / CONFIG["experiment"]["output_root"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_finite(value: object) -> None:
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for nested in value.values():
            assert_finite(nested)
    elif isinstance(value, list):
        for nested in value:
            assert_finite(nested)


def test_post_audit_prompt_amendment_is_explicitly_h1_h2_only() -> None:
    prompt = (ROOT / "prompts/dynamics_4.md").read_text(encoding="utf-8")
    assert "Post-audit amendment" in prompt
    assert "evaluated rollout horizons = H1 and H2 only" in prompt
    assert "H4 and H8 = not run" in prompt


def test_acquisition_gate_has_exact_10_per_task_and_stops_after_000() -> None:
    manifest = load_json(ACQUISITION / "selected_segments_manifest.json")
    download = load_json(ACQUISITION / "download_manifest.json")
    assert manifest["gate_passed"] is True and manifest["total_segments"] == 60
    assert set(manifest["per_task_counts"].values()) == {10}
    assert [item["repo_path"] for item in download["files"]] == [
        "training/subset_training_023.zip", "training/subset_training_000.zip",
    ]
    staging = ROOT / CONFIG["data"]["staging_root"]
    assert not staging.exists() or not any(path.is_file() for path in staging.rglob("*"))


def test_every_selected_segment_is_contiguous_original_7d_and_hashed() -> None:
    manifest = load_json(ACQUISITION / "selected_segments_manifest.json")
    for segment in manifest["segments"]:
        path = ROOT / segment["path"]
        assert sha256_file(path) == segment["sha256"]
        with np.load(path, allow_pickle=False) as saved:
            actions = saved["rel_actions"]
            indices = saved["global_frame_indices"]
        assert actions.dtype == np.float64 and actions.shape in ((64, 7), (65, 7))
        assert np.array_equal(indices, np.arange(segment["start_frame"], segment["end_frame"] + 1))
        assert len(segment["four_window_ranges"]) == 4
        assert all(right - left + 1 == 16 for left, right in segment["four_window_ranges"])
        assert all(right[0] - left[0] == 16 for left, right in zip(segment["four_window_ranges"], segment["four_window_ranges"][1:]))
        assert segment["leftover_frames_not_windowed"] in (0, 1)


def test_preregistration_precedes_outputs_and_forbids_model_selection() -> None:
    acquisition = load_json(ACQUISITION / "external_h12_acquisition_preregistration.json")
    prospective = load_json(OUT / "external_h12_prospective_preregistration.json")
    assert acquisition["written_before_external_F1_F2_outputs"] is True
    assert acquisition["selection"]["model_dependent_filtering"] is False
    assert prospective["written_before_any_external_F1_F2_output"] is True
    assert prospective["evaluated_horizons"] == [1, 2]
    assert prospective["H4_H8_run"] is False
    assert prospective["bootstrap"]["replicates"] == 10000


def test_serialization_has_four_nonoverlapping_windows_and_exact_start_counts() -> None:
    rows = [json.loads(line) for line in (OUT / "external_sequences.jsonl").read_text(encoding="utf-8").splitlines()]
    with np.load(OUT / "external_frozen_latents.npz", allow_pickle=False) as saved:
        assert saved["latents"].shape == (240, 32)
        assert saved["raw_actions"].shape == (240, 16, 7)
    assert len(rows) == 60
    assert all(len(row["latent_indices"]) == 4 for row in rows)
    assert all(row["valid_H1_starts"] == [0, 1] for row in rows)
    assert all(row["valid_H2_starts"] == [0] for row in rows)
    assert supported_rollout_offsets(4, 1) == [0, 1]
    assert supported_rollout_offsets(4, 2) == [0]


def test_frozen_causal_audit_and_h4_h8_not_run() -> None:
    audit = load_json(OUT / "external_h12_freezing_and_causality_audit.json")
    counts = load_json(OUT / "external_h12_sample_counts.json")
    assert audit["all_parameters_unchanged"] is True
    assert audit["representation_optimizer_steps"] == audit["F1_optimizer_steps"] == audit["F2_optimizer_steps"] == 0
    assert audit["representation_backward_calls"] == audit["F1_backward_calls"] == audit["F2_backward_calls"] == 0
    assert audit["EMA_updates"] == 0 and audit["future_target_actions_used_as_model_input"] is False
    assert counts["H1_rollout_starts"] == 120 and counts["H2_rollout_starts"] == 60
    assert counts["H4_rollout_starts"] == counts["H8_rollout_starts"] == 0
    assert counts["H4_H8_run"] is False


def test_metrics_bootstrap_correction_and_iterations_are_complete_and_finite() -> None:
    metrics = load_json(OUT / "external_h12_rollout_metrics.json")
    bootstrap = load_json(OUT / "external_h12_paired_trajectory_bootstrap.json")
    correction = load_json(OUT / "external_h12_correction_alignment.json")
    iterations = load_json(OUT / "external_h12_refinement_intermediate_states.json")
    assert metrics["evaluated_horizons"] == [1, 2] and metrics["H4_H8_run"] is False
    assert set(metrics["models"]) == {"F1", "F2"}
    assert metrics["models"]["F1"]["1"]["sample_count"] == 120
    assert metrics["models"]["F2"]["2"]["sample_count"] == 60
    assert bootstrap["pooled"]["trajectory_count"] == 60
    assert bootstrap["pooled"]["bootstrap_replicates"] == 10000
    assert bootstrap["window_bootstrap"] is False
    assert set(bootstrap["per_task"]) == set(CONFIG["data"]["tasks"])
    assert correction["correction_target_cosine"]["pooled"]["count"] == 180
    assert iterations["iterations"] == [0, 1, 2, 3, 4]
    assert iterations["iteration_0_is_F1_initializer"] is True
    assert len(iterations["records"]) == 1200
    for payload in (metrics, bootstrap, correction, iterations):
        assert_finite(payload)


def test_final_report_and_research_log_explicitly_limit_scope() -> None:
    report = (OUT / "sixteenth_wave_results.md").read_text(encoding="utf-8")
    log = (ROOT / "RESEARCH_LOG.md").read_text(encoding="utf-8")
    assert "H1 and H2 only" in report and "H4 and H8 were not run" in report
    assert "H1 and H2 only" in log and "H4 and H8 were not run" in log

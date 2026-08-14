"""Integrity tests for the wave-17 continuous-play frozen evaluation.

These tests detect concrete protocol failures that would change the scientific
decision: reset/session crossings, frame overlap, invalid H16 support,
future-context leakage, checkpoint drift, incorrect session bootstrap units,
or non-finite reported metrics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.dynamics.acquire_dynamics_5 import valid_offsets


ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((ROOT / "configs/dynamics_5.yaml").read_text(encoding="utf-8"))
ACQUISITION = ROOT / CONFIG["experiment"]["acquisition_root"]
OUTPUT = ROOT / CONFIG["experiment"]["output_root"]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_context_offset_rules_are_causal_and_exogenous_only_when_labeled() -> None:
    contexts = [-1, 10, 10, 10, -1, 20, 20, -1, -1, -1]
    assert valid_offsets(contexts, 8, "A") == [0]
    assert valid_offsets(contexts, 8, "B") == []
    assert valid_offsets(contexts, 2, "A") == [0, 1, 2, 4, 5]
    assert valid_offsets(contexts, 2, "B") == [0, 1, 4]


def test_immutable_wave16_artifacts_match() -> None:
    audit = read_json(ACQUISITION / "wave16_immutable_artifact_audit.json")
    assert audit["all_matched"]
    assert all(row["actual_sha256"] == row["expected_sha256"] for row in audit["files"])


def test_blocks_are_contiguous_nonoverlapping_and_session_bounded() -> None:
    manifest = read_json(ROOT / CONFIG["data"]["continuous_block_manifest"])
    assert manifest["gate"]["passed"]
    assert manifest["no_raw_frame_overlap"]
    seen: set[int] = set()
    for block in manifest["blocks"]:
        assert block["authoritative_same_session"] and not block["reset_flags"]
        assert block["frame_count"] == 160 and block["number_H16_windows"] == 10
        frames = set(range(block["start_frame"], block["end_frame"] + 1))
        assert len(frames) == 160 and seen.isdisjoint(frames)
        seen.update(frames)
        with np.load(ROOT / block["path"], allow_pickle=False) as saved:
            assert saved["rel_actions"].shape == (160, 7)
            assert saved["robot_obs"].shape == (160, 15)
            assert np.array_equal(saved["global_frame_indices"], np.arange(block["start_frame"], block["end_frame"] + 1))


def test_primary_support_meets_exact_gate() -> None:
    support = read_json(OUTPUT / "H1_H2_H4_H8_support_table.json")
    assert support["primary_adequate"]
    for horizon, minimum in support["minimum_required_primary"].items():
        assert support["Protocol_A"][horizon] >= minimum


def test_annotation_boundaries_are_metadata_not_concatenation() -> None:
    manifest = read_json(ROOT / CONFIG["data"]["continuous_block_manifest"])
    assert any(block["annotation_boundaries"] for block in manifest["blocks"])
    for block in manifest["blocks"]:
        assert len(block["windows"]) == 10
        for window in block["windows"]:
            assert window["end_frame"] - window["start_frame"] + 1 == 16
            assert window["canonical_task"] != "" and window["language"] != ""


def test_protocol_a_never_uses_future_annotation_schedule() -> None:
    prereg = read_json(OUTPUT / "wave17_continuous_play_preregistration.json")
    audit = read_json(OUTPUT / "freezing_and_causality_audit.json")
    assert "held fixed" in prereg["protocol_A_context_rule"]
    assert audit["future_annotations_used_in_protocol_A"] is False
    assert audit["future_raw_actions_as_input"] is False
    assert audit["future_robot_states_as_input"] is False
    assert audit["teacher_forcing_after_start"] is False


def test_protocol_b_is_explicitly_exogenous() -> None:
    results = read_json(OUTPUT / "protocol_B_results.json")
    assert results["label"] == "EXOGENOUS_CONTEXT_SCHEDULE_DIAGNOSTIC"


def test_models_are_frozen_and_source_contains_no_training_call() -> None:
    audit = read_json(OUTPUT / "freezing_and_causality_audit.json")
    assert audit["all_parameters_unchanged"]
    assert audit["representation_optimizer_steps"] == audit["F1_optimizer_steps"] == audit["F2_optimizer_steps"] == 0
    assert audit["EMA_updates"] == audit["loss_backward_calls"] == 0
    source = (ROOT / "scripts/dynamics/run_dynamics_5.py").read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert ".backward(" not in source


def test_checkpoint_file_hashes_remain_frozen() -> None:
    manifest = read_json(ACQUISITION / "frozen_model_hash_manifest.json")
    for value in manifest.values():
        assert sha256(ROOT / value["path"]) == value["sha256"]


def test_bootstrap_uses_source_sessions_not_windows_or_blocks() -> None:
    bootstrap = read_json(OUTPUT / "source_session_clustered_paired_bootstrap.json")
    assert bootstrap["sampling_unit"] == "source play session"
    assert bootstrap["blocks_within_session_averaged_first"]
    assert bootstrap["window_bootstrap"] is False
    assert bootstrap["protocol_A"]["source_session_count"] >= 10


def test_manifold_reference_is_training_only() -> None:
    manifold = read_json(OUTPUT / "empirical_manifold_analysis.json")
    assert manifold["training_only"]
    assert "thirteenth_wave" in manifold["reference"]


def test_all_metric_numbers_are_finite() -> None:
    payloads = [
        read_json(OUTPUT / "horizon_wise_latent_metrics.json"),
        read_json(OUTPUT / "source_session_clustered_paired_bootstrap.json"),
        read_json(OUTPUT / "refinement_correction_alignment.json"),
        read_json(OUTPUT / "refinement_iteration_curves.json"),
        read_json(OUTPUT / "empirical_manifold_analysis.json"),
    ]

    def visit(value):
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, float):
            assert np.isfinite(value)

    for payload in payloads:
        visit(payload)

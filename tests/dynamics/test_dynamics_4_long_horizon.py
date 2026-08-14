"""Protocol tests for the sixteenth-wave long-horizon stop branch.

Purpose
-------
Detect window overlap/padding, unsupported H8 starts, non-trajectory bootstrap,
non-training PCA input, carried F1/F2 mismatch, unauthorized metric reads, and
project-size violations before accepting the wave-16 report.

Parameters
----------
No custom parameters; pytest supplies its standard CLI options.

Usage
-----
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  tests/dynamics/test_dynamics_4_long_horizon.py -q

Outputs
-------
Pytest pass/fail output; the formal command writes JUnit XML inside the
timestamped wave-16 result directory.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch
import yaml

from pglt.dynamics.dynamics_data import sha256_file
from pglt.dynamics.long_horizon import (
    decompose_tangent_normal,
    fit_training_neighbor_pca,
    non_overlapping_windows,
    paired_trajectory_bootstrap,
    supported_rollout_offsets,
)
from pglt.dynamics.open_data import SIX_TASKS, annotation_records, assert_disk_budget


ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((ROOT / "configs/dynamics_4.yaml").read_text(encoding="utf-8"))
OUT = ROOT / CONFIG["experiment"]["output_root"]
ACQUISITION = ROOT / CONFIG["experiment"]["acquisition_root"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_exact_nonoverlap_no_padding_and_h8_support() -> None:
    windows = non_overlapping_windows(100, 259, 16)
    assert len(windows) == 10
    assert windows[0] == (100, 115) and windows[-1] == (244, 259)
    assert all(left[1] < right[0] and right[0] - left[0] == 16 for left, right in zip(windows, windows[1:]))
    assert supported_rollout_offsets(10, 8) == [0]
    assert non_overlapping_windows(100, 258, 16)[-1] == (228, 243)


def test_bootstrap_samples_paired_trajectories() -> None:
    f1 = np.asarray([1.0, 2.0, 3.0, 4.0])
    f2 = f1 - 0.5
    result = paired_trajectory_bootstrap(f1, f2, replicates=10000, seed=1604)
    assert result["trajectory_count"] == 4
    assert result["bootstrap_replicates"] == 10000
    assert result["upper_95"] < 0


def test_training_neighbor_pca_finite_orthogonal_decomposition() -> None:
    rng = np.random.default_rng(1604)
    training_neighbors = rng.normal(size=(20, 16))
    pca = fit_training_neighbor_pca(training_neighbors, 0.9)
    tangent, normal = decompose_tangent_normal(rng.normal(size=16), pca)
    assert pca.tangent_dimension >= 1
    assert np.isfinite(tangent).all() and np.isfinite(normal).all()
    assert abs(float(tangent @ normal)) < 1e-9


def test_f2_frozen_initializer_equals_f1_and_iterations_are_four() -> None:
    f1 = torch.load(ROOT / CONFIG["models"]["f1_checkpoint"], map_location="cpu", weights_only=False)["model_state_dict"]
    f2 = torch.load(ROOT / CONFIG["models"]["f2_checkpoint"], map_location="cpu", weights_only=False)["model_state_dict"]
    for key, value in f1.items():
        assert torch.equal(value, f2[f"initializer.{key}"])
    assert CONFIG["models"]["refinement_iterations"] == 4


def test_failed_data_gate_blocks_all_metrics_and_del_primary_use() -> None:
    gate = json.loads((OUT / "data_adequacy_gate.json").read_text(encoding="utf-8"))
    prereg = json.loads((OUT / "prospective_evaluation_preregistration.json").read_text(encoding="utf-8"))
    assert gate["passed"] is False and gate["model_metrics_authorized"] is False
    assert prereg["historical_DEL_role"].startswith("frozen negative")
    for name in (
        "F1_F2_horizon_wise_latent_metrics.json", "trajectory_level_paired_bootstrap.json",
        "refinement_intermediate_state_table.json", "local_tangent_normal_manifold_audit.json",
    ):
        payload = json.loads((OUT / name).read_text(encoding="utf-8"))
        assert payload["status"] == "not_evaluated_due_data_adequacy_gate"
        assert payload["F1_F2_metrics_read"] is False


def test_project_remains_below_20_gib() -> None:
    observed = int(subprocess.run(["du", "-sb", str(ROOT)], check=True, capture_output=True, text=True).stdout.split()[0])
    assert observed < int(CONFIG["storage"]["maximum_project_gb"]) * 1024**3


def test_local_inventory_and_exact_task_mapping() -> None:
    inventory = load_json(ACQUISITION / "local_calvin_inventory.json")
    mapping = load_json(ACQUISITION / "canonical_task_mapping.json")
    assert inventory["episode_npz_count"] > 0
    assert inventory["direct_candidate_count_ge_160"] == 0
    assert set(mapping["tasks"]) == set(SIX_TASKS)
    assert mapping["mapping_method"].startswith("exact frozen CALVIN task ID")
    assert all(item["paraphrases"] for item in mapping["tasks"].values())


def test_huggingface_enumeration_and_download_hash_records() -> None:
    manifests = load_json(ACQUISITION / "per_source_download_manifest.json")
    assert [item["repo_id"] for item in manifests] == [
        "CollisionCode/calvin_d_d_lerobot_v2.1",
        "RoboVerseOrg/roboverse_data",
        "VyoJ/calvin-ABCD-D-subsets",
    ]
    assert manifests[1]["repository_file_count"] == 77768
    assert all("calvin" in item["repo_path"] for item in manifests[1]["files"])
    assert manifests[2]["selected_files"] == ["training/subset_training_023.zip"]
    for manifest in manifests:
        for item in manifest["files"]:
            assert len(item["sha256"]) == 64
    for item in manifests[0]["files"]:
        assert sha256_file(ROOT / item["path"]) == item["sha256"]


def test_disk_guard_and_staged_cleanup() -> None:
    minimum = CONFIG["storage"]["minimum_free_bytes"]
    assert assert_disk_budget(ROOT, minimum)["passed"] is True
    with pytest.raises(RuntimeError, match="Disk guard"):
        assert_disk_budget(ROOT, minimum, 10**15)
    cleanup = load_json(ACQUISITION / "staged_download_cleanup_log.json")
    assert any(event["source"] == "RoboVerseOrg/roboverse_data" for event in cleanup["events"])
    assert any(event["source"] == "VyoJ/calvin-ABCD-D-subsets" for event in cleanup["events"])
    assert not (ROOT / ".staging/sixteenth_wave/roboverse").exists()
    assert not (ROOT / ".staging/sixteenth_wave/abcd").exists()
    assert cleanup["final_free_space"]["free_bytes"] >= minimum


def test_roboverse_per_trajectory_schema_and_action_gate() -> None:
    audit = load_json(ACQUISITION / "tier1b_trajectory_audit.json")
    summaries = audit["trajectory_files"]
    assert len(summaries) == 358
    assert sum(item["trajectory_count"] for item in summaries) == 4836
    assert all(len(item["trajectory_audit"]) == item["trajectory_count"] for item in summaries)
    assert max(item["length_distribution"]["max"] for item in summaries) == 64
    assert sum(item["candidate_count_ge_160"] for item in summaries) == 0
    numeric_dims = {
        int(dimension)
        for item in summaries if "calvin_traj_ann" in item["repo_path"]
        for dimension in item["trajectory_action_dimension_distribution"]
    }
    assert numeric_dims == {9}


def test_exact_rel_action_convention_and_time_base_gate() -> None:
    convention = load_json(OUT / "rel_action_convention_audit.json")
    compatibility = load_json(OUT / "primary_compatibility_audit.json")
    assert convention["abcd_probe_passed"] is True
    assert convention["conversion_performed"] is False
    assert convention["calvin_source"]["output_order"] == [
        "relative_tcp_translation_xyz", "relative_tcp_euler_xyz", "gripper",
    ]
    by_repo = {item["repo_id"]: item for item in compatibility["sources"]}
    assert by_repo["CollisionCode/calvin_d_d_lerobot_v2.1"]["compatibility_status"] == "SCOUTING_ONLY"
    assert "10 Hz" in by_repo["CollisionCode/calvin_d_d_lerobot_v2.1"]["rejection_reason"]
    assert by_repo["RoboVerseOrg/roboverse_data"]["compatibility_status"] == "REJECTED"
    assert by_repo["VyoJ/calvin-ABCD-D-subsets"]["compatibility_status"] == "PRIMARY_COMPATIBLE"


def test_annotation_eligibility_is_direct_and_does_not_cross_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "auto_lang_ann.npy"
    np.save(path, {
        "info": {"indx": np.asarray([[0, 159], [160, 258], [259, 358]])},
        "language": {
            "task": np.asarray([SIX_TASKS[0], SIX_TASKS[0], SIX_TASKS[0]], dtype=object),
            "ann": np.asarray(["a", "a", "a"], dtype=object),
        },
    }, allow_pickle=True)
    audit = annotation_records(path, "test")
    assert audit["candidate_count_ge_160"] == 1
    assert audit["candidate_records"][0]["inclusive_frame_count"] == 160
    assert audit["candidate_records"][0]["contains_other_annotation_boundary"] is False


def test_selection_preregistration_freezing_and_no_future_actions() -> None:
    availability = load_json(OUT / "long_trajectory_availability_audit.json")
    prereg = load_json(OUT / "long_horizon_open_data_preregistration.json")
    frozen = load_json(OUT / "frozen_model_hash_manifest.json")
    gate = load_json(OUT / "data_adequacy_gate.json")
    assert availability["eligible_existing_segment_count"] == 0
    assert prereg["written_before_any_long_model_metric"] is True
    assert prereg["selected_segments"] == []
    assert prereg["future_target_actions"] is False
    assert prereg["configuration"]["data"]["chunk_length"] == 16
    assert prereg["configuration"]["data"]["stride"] == 16
    assert prereg["trajectory_bootstrap_replicates"] == 10000
    assert gate["model_metrics_authorized"] is False
    assert frozen["model_parameter_updates"] == frozen["representation_updates"] == frozen["ema_updates"] == 0
    for item in frozen["checkpoints"].values():
        assert sha256_file(ROOT / item["path"]) == item["sha256"]


def test_failed_gate_has_exact_missing_plan_and_finite_artifacts() -> None:
    failure = load_json(OUT / "data_adequacy_failure.json")
    plan = load_json(OUT / "targeted_missing_data_acquisition_plan.json")
    assert failure["primary_inference_permitted"] is False
    assert len(plan["missing"]) == 6 and plan["total_new_segments_required"] == 60
    assert all(item["missing_count_to_reach_10"] == 10 for item in plan["missing"])
    assert all(item["minimum_frames"] == 160 for item in plan["missing"])

    def assert_finite(value: object) -> None:
        if isinstance(value, float):
            assert math.isfinite(value)
        elif isinstance(value, dict):
            for nested in value.values():
                assert_finite(nested)
        elif isinstance(value, list):
            for nested in value:
                assert_finite(nested)

    for path in OUT.glob("*.json"):
        assert_finite(load_json(path))


def test_blocked_metrics_include_source_stratification_and_training_only_manifold() -> None:
    for name in ("per_source_per_task_metrics.json", "source_stratified_bootstrap.json"):
        payload = load_json(OUT / name)
        assert payload["F1_F2_metrics_read"] is False
    manifold = load_json(OUT / "local_tangent_normal_manifold_audit.json")
    assert manifold["F1_F2_metrics_read"] is False
    assert manifold["status"] == "not_evaluated_due_data_adequacy_gate"

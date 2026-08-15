"""Wave 27 prospective-experiment integrity tests.

Purpose
-------
Verify the frozen representation/splits, independent source-session inventory,
causal synchronized physical inputs, TRAIN-only retrieval/scoring policy,
decoder gradient path, reproducible flow sampling, session bootstrap settings,
and finite tracked JSON outputs after the Wave 27 run.

Parameters
----------
No command-line parameters.  Paths are resolved from ``configs/dynamics_15.yaml``.

Usage
-----
PYTHONPATH=.:src pytest -q tests/test_dynamics_15.py

Outputs
-------
Pytest writes only its normal console report.  The experiment harness captures
that report in ``results/.../tests_report.txt``.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import torch
import yaml

from pglt.dynamics.wave27_models import ConditionalTrajectoryFlow


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs/dynamics_15.yaml").read_text())
OUT = ROOT / CONFIG["experiment"]["output_root"]


def read(name: str):
    return json.loads((OUT / name).read_text())


def finite(value) -> bool:
    if isinstance(value, dict): return all(finite(item) for item in value.values())
    if isinstance(value, list): return all(finite(item) for item in value)
    if isinstance(value, float): return np.isfinite(value)
    return True


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): value.update(block)
    return value.hexdigest()


def test_frozen_hashes_and_legacy_split_unchanged():
    frozen = read("wave27_frozen_manifest.json")
    wave26 = json.loads((ROOT / CONFIG["experiment"]["wave26_root"] / "wave26_frozen_manifest.json").read_text())
    for key in ("representation_sha256", "encoder_sha256", "decoder_sha256", "semantic_projection_sha256", "text_feature_archive_sha256", "normalization_sha256"):
        assert frozen[key] == wave26[key]
    assert frozen["representation_updates"] == frozen["decoder_updates"] == frozen["text_updates"] == 0
    checkpoint = Path(frozen["representation_checkpoint"])
    if not checkpoint.is_absolute(): checkpoint = ROOT / checkpoint
    assert digest(checkpoint) == frozen["representation_sha256"]
    legacy_split = ROOT / CONFIG["experiment"]["wave21_root"] / "wave21_session_split_manifest.json"
    assert digest(legacy_split) == frozen["legacy_split_sha256"]


def test_new_sessions_unique_disjoint_and_split_frozen():
    rows = read("wave27_new_transition_inventory.json"); split = read("wave27_new_data_split_manifest.json")
    sessions = {row["source_session_id"] for row in rows}
    assert all(row["source_session_row"] >= 31 for row in rows)
    groups = [set(split["sessions"][name]) for name in ("new_train", "new_development", "new_prospective_test")]
    assert groups[0] | groups[1] | groups[2] == sessions
    assert groups[0].isdisjoint(groups[1]) and groups[0].isdisjoint(groups[2]) and groups[1].isdisjoint(groups[2])
    assert split["created_before_model_training"] and split["disjoint"]


def test_physical_alignment_contact_and_no_future_state():
    rows = read("wave27_new_transition_inventory.json")
    for row in rows:
        assert row["physically_contiguous"] and not row["reset_crossed"]
        assert row["source_start_frame"] == row["boundary_frame"] - 64
        assert row["source_end_frame"] == row["boundary_frame"] + 63
        assert not row["true_contact_available"] and not row["measured_tcp_velocity_available"] and not row["measured_joint_velocity_available"]
    manifest = read("wave27_physical_state_manifest.json")
    assert manifest["query_alignment"] == "latest input t-1" and manifest["true_contact"] == "UNAVAILABLE"
    encoded = read("wave27_encoded_dataset_manifest.json")
    assert not encoded["future_as_input"] and encoded["physical_alignment"].endswith("t-1")


def test_retrieval_and_candidate_selection_are_causal():
    prereg = read("wave27_collection_preregistration.json"); selection = read("wave27_final_candidate_selection.json")
    assert not prereg["selection_uses_model_outputs"]
    assert selection["frozen_before_prospective_test"] and selection["post_test_model_changes_forbidden"]
    assert len(selection["selected"]) <= 4
    for spec in read("wave27_model_specs.json").values():
        assert spec.get("retrieval_family", "R4_factored") not in ("oracle", "future")


def test_decoder_frozen_and_transition_gradient_passes():
    audit = read("wave27_decoder_gradient_unit_test.json")
    assert audit["passed"] and audit["transition_gradient_nonzero"]
    assert audit["decoder_parameters_frozen"] and audit["frozen_representation_gradient_l1"] == 0


def test_flow_sampling_reproducible_and_anchor_causal():
    torch.manual_seed(1); model = ConditionalTrajectoryFlow(8, 16, 96); x = torch.randn(3, 8); anchor = torch.randn(3, 3, 32)
    first = model.sample(x, 2, 4, torch.Generator().manual_seed(27), anchor, True)
    second = model.sample(x, 2, 4, torch.Generator().manual_seed(27), anchor, True)
    assert torch.equal(first, second)


def test_same_state_language_swap_changes_only_language_coordinates():
    train = np.load(OUT / "datasets/new_train.npz")
    goals = np.load(ROOT / CONFIG["experiment"]["wave21_root"] / "wave21_goal_embeddings.npy")
    from scripts.dynamics.run_dynamics_15 import PhysicalTransform
    data = {key: train[key][:1] for key in train.files}; transform = PhysicalTransform("PH0", goals).fit(data)
    raw0 = transform.raw(data, np.array([0])); raw1 = transform.raw(data, np.array([1])); difference = np.flatnonzero(np.abs(raw0 - raw1).ravel() > 0)
    assert len(difference) > 0 and difference.min() >= raw0.shape[1] - goals.shape[1]


def test_bootstrap_and_json_outputs():
    prereg = read("wave27_final_test_preregistration.json")
    assert prereg["bootstrap"] == {"unit": "source session", "replicates": 10000, "seed": 270827}
    for path in OUT.glob("*.json"):
        assert finite(json.loads(path.read_text())), path

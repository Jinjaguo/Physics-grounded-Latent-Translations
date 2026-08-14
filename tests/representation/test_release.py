"""Regression tests for the complete paper-release representation package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from pglt.data.calvin import Annotation, build_chunk_records
from pglt.representation.losses import reconstruction_loss, symmetric_contrastive_loss
from pglt.representation.model import ActionRepresentationModel
from pglt.representation.readiness import motor_sanity, stronger_control_delta


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/representation.yaml"


def load_config() -> dict:
    """Load the single released scientific configuration."""

    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    """Return a streaming digest for one frozen checkpoint."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def test_registered_episode_partitions_are_complete_and_disjoint() -> None:
    """Every official episode row belongs to exactly one registered role."""

    selection = load_config()["selection"]
    partitions = [
        selection["historical_training_episode_rows"],
        selection["development_episode_rows"],
        selection["confirmation_episode_rows"],
        selection["independent_replication_episode_rows"],
    ]
    flattened = [int(row) for partition in partitions for row in partition]
    assert sorted(flattened) == list(range(31))
    assert len(flattened) == len(set(flattened))


def test_compact_episode_schema_matches_official_bounds() -> None:
    """All 31 released episodes retain exact actions and global indices only."""

    config = load_config()
    metadata = ROOT / config["source"]["metadata_root"] / "training"
    compact = ROOT / config["source"]["compact_root"] / "training"
    bounds = np.load(metadata / "ep_start_end_ids.npy").reshape(-1, 2)
    assert len(bounds) == 31
    for row, (first, last) in enumerate(bounds):
        with np.load(compact / f"episode_row_{row:03d}.npz", allow_pickle=False) as archive:
            assert set(archive.files) == {"rel_actions", "global_frame_indices"}
            assert archive["rel_actions"].shape == (int(last - first + 1), 7)
            assert archive["rel_actions"].dtype == np.float64
            np.testing.assert_array_equal(
                archive["global_frame_indices"], np.arange(first, last + 1)
            )


def test_action_window_endpoint_convention() -> None:
    """A 16-step window consumes t..t+15 and records outcome index t+16."""

    records = build_chunk_records(
        [Annotation(0, 100, 133, "task", "instruction")],
        split="test",
        chunk_length=16,
        stride=4,
    )
    assert [record.start_index for record in records] == [100, 104, 108, 112, 116]
    assert records[0].action_end_index_inclusive == 115
    assert records[0].outcome_state_index == 116


def test_language_gradient_is_isolated_from_shared_action_trunk() -> None:
    """Contrastive gradients reach semantic head/text map but not the trunk."""

    torch.manual_seed(7)
    model = ActionRepresentationModel(
        input_mode="action_only",
        chunk_length=16,
        text_feature_dim=12,
    )
    actions = torch.randn(6, 16, 7)
    text_features = torch.randn(6, 12)
    semantic = model.isolated_clip_semantic_latent(actions)
    projected = model.project_text(text_features)
    loss, _ = symmetric_contrastive_loss(semantic, projected, 0.07)
    loss.backward()
    assert model.encoder[0].weight.grad is None
    final_head = model.encoder[-1].weight.grad
    assert final_head is not None
    assert torch.count_nonzero(final_head[:16]) > 0
    assert torch.count_nonzero(final_head[16:]) == 0
    assert model.text_projection.weight.grad is not None


def test_motor_loss_uses_binary_gripper_semantics() -> None:
    """The seventh action is BCE-logit supervised and rejects nonbinary targets."""

    target = torch.zeros(2, 16, 7)
    target[..., 6] = torch.tensor([-1.0, 1.0]).view(2, 1)
    losses = reconstruction_loss(torch.zeros_like(target), target)
    assert losses["gripper_bce"].item() == pytest.approx(np.log(2.0))
    target[0, 0, 6] = 0.0
    with pytest.raises(ValueError, match="exactly -1 or 1"):
        reconstruction_loss(torch.zeros_like(target), target)


def test_all_frozen_checkpoints_match_portable_manifest() -> None:
    """The 18 epoch-40 EMA checkpoints match hashes and load into final code."""

    config = load_config()
    checkpoint_root = ROOT / config["release"]["checkpoint_root"]
    manifest = json.loads((checkpoint_root / "manifest.json").read_text())
    assert len(manifest["checkpoints"]) == 18
    for item in manifest["checkpoints"]:
        assert not Path(item["path"]).is_absolute()
        path = (
            checkpoint_root
            / f"seed_{item['seed_base']}"
            / item["condition"]
            / "checkpoint_ema.pt"
        )
        assert file_sha256(path) == item["sha256"]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload["epoch"] == 40
        assert payload["ema_decay"] == pytest.approx(0.999)
    first = manifest["checkpoints"][0]
    first_path = (
        checkpoint_root
        / f"seed_{first['seed_base']}"
        / first["condition"]
        / "checkpoint_ema.pt"
    )
    model_config = config["model"]
    model = ActionRepresentationModel(
        input_mode="action_only",
        chunk_length=config["data"]["chunk_length"],
        action_dim=model_config["action_dim"],
        latent_dim=model_config["latent_dim"],
        hidden_dim=model_config["hidden_dim"],
        depth=model_config["depth"],
        text_feature_dim=config["text"]["frozen_feature_dim"],
        semantic_dim=model_config["semantic_dim"],
    )
    model.load_state_dict(torch.load(first_path, map_location="cpu", weights_only=False)["model_state_dict"])


def test_independent_replication_release_is_complete_and_read_only() -> None:
    """All 18 inference records use exact query exclusion and no optimization."""

    config = load_config()
    inference_root = ROOT / config["release"]["inference_root"]
    rows = [int(row) for row in config["selection"]["independent_replication_episode_rows"]]
    for seed in config["replication"]["seeds"]:
        for condition in config["replication"]["conditions"]:
            path = inference_root / f"seed_{seed}" / condition / "metrics.json"
            record = json.loads(path.read_text())
            assert record["parameter_tensors_unchanged"]
            assert record["optimizer_steps"] == record["ema_updates"] == record["backward_calls"] == 0
            assert record["independent_rows_excluded_from_normalization"]
            for query in rows:
                assert record["episode_metrics"][str(query)]["candidate_episode_rows"] == [
                    row for row in rows if row != query
                ]


def test_released_readiness_decision_preserves_history_and_passes() -> None:
    """The prospective gate passes without rewriting the historical failure."""

    decision_root = ROOT / "results/representation/independent_replication"
    decision = json.loads((decision_root / "readiness_gate_decision.json").read_text())
    status = json.loads((decision_root / "representation_status.json").read_text())
    assert decision["r_gate_passed"]
    assert decision["historical_all_cell_gate"] == "FAIL"
    assert not decision["historical_failure_erased"]
    assert len(decision["episode_summary"]) == 14
    assert len(decision["seed_summary"]) == 6
    assert decision["bootstrap"]["text_to_action"]["lower_95"] == pytest.approx(0.7992365126606198)
    assert decision["bootstrap"]["action_to_text"]["lower_95"] == pytest.approx(0.6606413721584853)
    assert decision["negative_cells"] == []
    assert status["representation_ready"] and status["representation_frozen_for_paper"]


def test_gate_thresholds_and_stronger_control_are_explicit() -> None:
    """Control selection and motor thresholds implement the frozen protocol."""

    delta, control = stronger_control_delta(0.8, reconstruction=0.2, shuffled=0.3)
    assert delta == pytest.approx(0.5)
    assert control == "shuffled_language"
    assert motor_sanity(1.2, 1.0, 0.95, 1.0)["passed"]
    assert not motor_sanity(1.2001, 1.0, 0.95, 1.0)["passed"]

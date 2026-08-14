"""Software-only tests for future frozen-latent dynamics input boundaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pglt.dynamics.frozen_latents import FrozenLatentTransitionDataset


def save_fixture(path: Path, starts: np.ndarray) -> None:
    """Write the exact current serialization using synthetic unit-test values."""

    count = len(starts)
    np.savez_compressed(
        path,
        latents=np.arange(count * 4, dtype=np.float32).reshape(count, 4),
        episode_id=np.full(count, 7, dtype=np.int64),
        window_start_index=starts,
        action_end_index_inclusive=starts + 15,
        outcome_state_index=starts + 16,
        task_labels=np.asarray(["task"] * count),
    )


def test_frozen_latent_dataset_returns_only_consecutive_triples(tmp_path: Path) -> None:
    """A valid episode retains its row and exact global start progression."""

    path = tmp_path / "episode.npz"
    save_fixture(path, np.arange(100, 196, 16, dtype=np.int64))
    dataset = FrozenLatentTransitionDataset(path)
    item = dataset[1]
    assert len(dataset) == 4
    assert item.episode_id == 7
    assert (item.previous_window_start_index, item.current_window_start_index, item.next_window_start_index) == (116, 132, 148)


def test_frozen_latent_dataset_rejects_nonconsecutive_windows(tmp_path: Path) -> None:
    """Missing windows cannot silently become one-step dynamics targets."""

    path = tmp_path / "episode.npz"
    save_fixture(path, np.asarray([100, 116, 133, 149], dtype=np.int64))
    with pytest.raises(ValueError, match="stride-16"):
        FrozenLatentTransitionDataset(path)

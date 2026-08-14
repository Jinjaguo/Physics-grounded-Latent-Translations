"""Read-only transition views over saved, frozen representation trajectories.

This infrastructure deliberately consumes the exact NPZ serialization written
by the released representation stage. It does not import representation
checkpoints, recompute latents, fit a transition, or assume that an action
latent is a physical generalized coordinate.  Consecutive tuples are exposed
only within one official episode and only when saved window starts differ by
the recorded stride.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch


SAVED_KEYS = (
    "latents",
    "episode_id",
    "window_start_index",
    "action_end_index_inclusive",
    "outcome_state_index",
    "task_labels",
)


@dataclass(frozen=True)
class FrozenLatentTransition:
    """Three consecutive saved coordinates with their exact source indices."""

    previous_latent: np.ndarray
    current_latent: np.ndarray
    next_latent: np.ndarray
    episode_id: int
    previous_window_start_index: int
    current_window_start_index: int
    next_window_start_index: int
    previous_task_label: str
    current_task_label: str
    next_task_label: str


class SameContextTransition(Protocol):
    """Interface for the future MLP transition baseline."""

    def predict_next(
        self,
        previous_latent: torch.Tensor,
        current_latent: torch.Tensor,
        context: torch.Tensor | None,
    ) -> torch.Tensor:
        """Predict one next frozen-latent coordinate."""


class CorrectedUnforcedDELTransition(Protocol):
    """Interface for the already diagnosed local unforced DEL ablation."""

    def predict_next_unforced(
        self,
        previous_latent: torch.Tensor,
        current_latent: torch.Tensor,
        context: torch.Tensor | None,
        delta_t: float,
    ) -> torch.Tensor:
        """Solve one corrected, explicitly unforced DEL transition."""


class FutureForcedDELTransition(Protocol):
    """Interface placeholder requiring an explicit audited generalized force."""

    def predict_next_forced(
        self,
        previous_latent: torch.Tensor,
        current_latent: torch.Tensor,
        context: torch.Tensor | None,
        generalized_force: torch.Tensor,
        delta_t: float,
    ) -> torch.Tensor:
        """Solve one forced DEL transition after force semantics are approved."""


class FrozenLatentTransitionDataset:
    """Validate a saved episode and expose stride-consistent coordinate triples."""

    def __init__(self, path: Path, expected_stride: int = 16) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as saved:
            if tuple(saved.files) != SAVED_KEYS:
                raise ValueError(f"Unexpected frozen-latent serialization keys: {saved.files}")
            self.arrays = {key: saved[key].copy() for key in saved.files}
        count = len(self.arrays["latents"])
        if any(len(self.arrays[key]) != count for key in SAVED_KEYS[1:]):
            raise ValueError("Frozen-latent serialization fields are not aligned")
        episode_ids = np.unique(self.arrays["episode_id"])
        if len(episode_ids) != 1:
            raise ValueError("One frozen-latent file must contain exactly one official episode")
        starts = self.arrays["window_start_index"]
        if expected_stride <= 0:
            raise ValueError("expected_stride must be positive")
        if len(starts) < 3 or not np.all(np.diff(starts) == expected_stride):
            raise ValueError(f"The dynamics interface requires saved stride-{expected_stride} windows")
        if not np.array_equal(self.arrays["action_end_index_inclusive"] + 1, self.arrays["outcome_state_index"]):
            raise ValueError("Saved action endpoint/outcome-state indexing is inconsistent")

    def __len__(self) -> int:
        """Return the number of within-episode consecutive latent triples."""

        return len(self.arrays["latents"]) - 2

    def __getitem__(self, index: int) -> FrozenLatentTransition:
        """Return one immutable consecutive triple and its original metadata."""

        if index < 0 or index >= len(self):
            raise IndexError(index)
        starts = self.arrays["window_start_index"]
        labels = self.arrays["task_labels"]
        return FrozenLatentTransition(
            previous_latent=self.arrays["latents"][index].copy(),
            current_latent=self.arrays["latents"][index + 1].copy(),
            next_latent=self.arrays["latents"][index + 2].copy(),
            episode_id=int(self.arrays["episode_id"][index]),
            previous_window_start_index=int(starts[index]),
            current_window_start_index=int(starts[index + 1]),
            next_window_start_index=int(starts[index + 2]),
            previous_task_label=str(labels[index]),
            current_task_label=str(labels[index + 1]),
            next_task_label=str(labels[index + 2]),
        )

"""Numerical utilities preregistered for prospective long-horizon evaluation.

These functions define exact non-overlapping windows, supported autonomous
rollout starts, paired whole-trajectory bootstrap, and training-neighbor PCA
decomposition.  They are independently testable before long data exist and do
not load or mutate representation or dynamics checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def non_overlapping_windows(start: int, end_inclusive: int, chunk_length: int = 16) -> list[tuple[int, int]]:
    """Return complete stride=H windows without padding beyond one segment."""

    if end_inclusive < start:
        raise ValueError("Segment end precedes start")
    count = (end_inclusive - start + 1) // chunk_length
    return [
        (start + index * chunk_length, start + (index + 1) * chunk_length - 1)
        for index in range(count)
    ]


def supported_rollout_offsets(window_count: int, horizon: int) -> list[int]:
    """Return offsets having previous/current plus the full future horizon."""

    if horizon < 1:
        raise ValueError("Horizon must be positive")
    return list(range(max(0, window_count - horizon - 1)))


def paired_trajectory_bootstrap(
    f1_auc: np.ndarray, f2_auc: np.ndarray, *, replicates: int = 10000, seed: int = 1604
) -> dict[str, float | int]:
    """Bootstrap paired whole-trajectory AUC differences, never windows."""

    f1 = np.asarray(f1_auc, dtype=np.float64)
    f2 = np.asarray(f2_auc, dtype=np.float64)
    if f1.ndim != 1 or f2.shape != f1.shape or len(f1) == 0:
        raise ValueError("Paired non-empty trajectory vectors are required")
    if not np.isfinite(f1).all() or not np.isfinite(f2).all():
        raise ValueError("Trajectory AUC inputs must be finite")
    delta = f2 - f1
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(replicates, len(delta)))
    sampled = delta[indices].mean(axis=1)
    return {
        "trajectory_count": len(delta), "bootstrap_replicates": replicates,
        "mean_delta_auc": float(delta.mean()),
        "lower_95": float(np.quantile(sampled, 0.025)),
        "upper_95": float(np.quantile(sampled, 0.975)),
    }


@dataclass(frozen=True)
class LocalPCA:
    """Training-neighbor empirical tangent basis and center."""

    center: np.ndarray
    tangent_basis: np.ndarray
    tangent_dimension: int


def fit_training_neighbor_pca(neighbors: np.ndarray, variance_fraction: float = 0.9) -> LocalPCA:
    """Fit local PCA only to caller-supplied frozen training neighbors."""

    values = np.asarray(neighbors, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("At least two training-neighbor vectors are required")
    if not 0 < variance_fraction <= 1:
        raise ValueError("variance_fraction must lie in (0,1]")
    center = values.mean(axis=0)
    _, singular, right = np.linalg.svd(values - center, full_matrices=False)
    variance = np.square(singular)
    cumulative = np.cumsum(variance) / max(float(variance.sum()), 1e-12)
    dimension = int(np.searchsorted(cumulative, variance_fraction) + 1)
    dimension = min(dimension, right.shape[0])
    return LocalPCA(center=center, tangent_basis=right[:dimension], tangent_dimension=dimension)


def decompose_tangent_normal(vector: np.ndarray, pca: LocalPCA) -> tuple[np.ndarray, np.ndarray]:
    """Decompose one displacement into empirical tangent and normal parts."""

    value = np.asarray(vector, dtype=np.float64)
    tangent = pca.tangent_basis.T @ (pca.tangent_basis @ value)
    return tangent, value - tangent

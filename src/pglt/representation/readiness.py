"""Frozen prospective-replication aggregation for representation readiness."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np


DIRECTIONS = ("text_to_action", "action_to_text")


def stronger_control_delta(
    correct: float, reconstruction: float, shuffled: float
) -> tuple[float, str]:
    """Subtract the direction-wise stronger frozen control."""

    controls = {
        "reconstruction_only": float(reconstruction),
        "shuffled_language": float(shuffled),
    }
    name = max(controls, key=controls.get)
    return float(correct) - controls[name], name


def motor_sanity(
    correct_mse: float,
    reconstruction_mse: float,
    correct_gripper: float,
    reconstruction_gripper: float,
) -> dict[str, float | bool]:
    """Apply the frozen 20% continuous-MSE and 0.05 gripper-drop limits."""

    baseline = float(reconstruction_mse)
    if baseline <= 0:
        raise ValueError("reconstruction_mse must be positive")
    relative = (float(correct_mse) - baseline) / baseline
    drop = float(reconstruction_gripper) - float(correct_gripper)
    tolerance = 1e-12
    return {
        "relative_mse_increase": relative,
        "gripper_drop": drop,
        "passed": relative <= 0.20 + tolerance and drop <= 0.05 + tolerance,
    }


def whole_episode_bootstrap(
    episode_values: Sequence[float], *, seed: int, replicates: int
) -> dict[str, float | int | str]:
    """Bootstrap the grand mean with whole episode means as resampling units."""

    values = np.asarray(episode_values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("episode_values must be a nonempty one-dimensional sequence")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(0, len(values), size=(int(replicates), len(values)))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
        "replicates": int(replicates),
        "unit": "whole_episode",
    }


def adjudicate_r_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int,
    integrity_passed: bool,
) -> dict[str, Any]:
    """Apply episode, seed, bootstrap, motor, and integrity requirements exactly."""

    if not rows:
        raise ValueError("At least one episode-seed cell is required")
    by_episode: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    by_seed: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_episode[int(row["episode_row"])].append(row)
        by_seed[int(row["seed_base"])].append(row)

    episode_summary = [
        {
            "episode_row": episode,
            **{
                direction: float(
                    np.mean([value["cross_episode_delta"][direction] for value in values])
                )
                for direction in DIRECTIONS
            },
        }
        for episode, values in sorted(by_episode.items())
    ]
    seed_summary = [
        {
            "seed_base": seed,
            **{
                direction: float(
                    np.mean([value["cross_episode_delta"][direction] for value in values])
                )
                for direction in DIRECTIONS
            },
        }
        for seed, values in sorted(by_seed.items())
    ]
    bootstrap = {
        direction: whole_episode_bootstrap(
            [row[direction] for row in episode_summary],
            seed=bootstrap_seed + direction_index,
            replicates=bootstrap_replicates,
        )
        for direction_index, direction in enumerate(DIRECTIONS)
    }
    checks = {
        "every_episode_positive": all(
            row[direction] > 0 for row in episode_summary for direction in DIRECTIONS
        ),
        "every_seed_positive": all(
            row[direction] > 0 for row in seed_summary for direction in DIRECTIONS
        ),
        "both_bootstrap_lower_bounds_positive": all(
            bootstrap[direction]["lower_95"] > 0 for direction in DIRECTIONS
        ),
        "all_motor_cells_passed": all(bool(row["motor_passed"]) for row in rows),
        "integrity_passed": bool(integrity_passed),
    }
    negative_cells = [
        dict(row)
        for row in rows
        if any(row["cross_episode_delta"][direction] < 0 for direction in DIRECTIONS)
    ]
    return {
        "episode_summary": episode_summary,
        "seed_summary": seed_summary,
        "bootstrap": bootstrap,
        "negative_cells": negative_cells,
        "checks": checks,
        "r_gate_passed": all(checks.values()),
    }

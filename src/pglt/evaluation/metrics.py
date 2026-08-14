"""Retrieval, motor-reconstruction, and latent-geometry metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import silhouette_score


def l2_normalize(array: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Normalize rows for cosine comparisons while guarding zero vectors."""

    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norms, epsilon)


def _direction_metrics(
    similarity: np.ndarray,
    query_tasks: Sequence[str],
    candidate_tasks: Sequence[str],
) -> dict[str, Any]:
    """Rank candidates using any same-task candidate as semantically relevant."""

    ranks: list[int] = []
    per_task: dict[str, list[int]] = defaultdict(list)
    order = np.argsort(-similarity, axis=1)
    for query_index, task in enumerate(query_tasks):
        relevant = [
            rank + 1
            for rank, candidate_index in enumerate(order[query_index])
            if candidate_tasks[candidate_index] == task
        ]
        if not relevant:
            raise ValueError(f"No relevant candidate exists for task {task!r}")
        best_rank = min(relevant)
        ranks.append(best_rank)
        per_task[task].append(best_rank)
    rank_array = np.asarray(ranks)
    return {
        "count": len(ranks),
        "recall_at_1": float(np.mean(rank_array <= 1)),
        "recall_at_5": float(np.mean(rank_array <= min(5, similarity.shape[1]))),
        "mean_rank": float(rank_array.mean()),
        "median_rank": float(np.median(rank_array)),
        "per_task_recall_at_1": {
            task: float(np.mean(np.asarray(task_ranks) <= 1))
            for task, task_ranks in sorted(per_task.items())
        },
    }


def symmetric_retrieval_metrics(
    action_latents: np.ndarray,
    text_latents: np.ndarray,
    tasks: Sequence[str],
) -> dict[str, Any]:
    """Evaluate semantic and exact-pair retrieval in both directions."""

    if action_latents.shape != text_latents.shape or action_latents.shape[0] != len(tasks):
        raise ValueError("Retrieval arrays and task metadata are not aligned")
    similarity = l2_normalize(action_latents) @ l2_normalize(text_latents).T
    text_to_action = _direction_metrics(similarity.T, tasks, tasks)
    action_to_text = _direction_metrics(similarity, tasks, tasks)
    action_order = np.argsort(-similarity, axis=1)
    text_order = np.argsort(-similarity.T, axis=1)
    action_ranks = np.asarray(
        [int(np.flatnonzero(action_order[index] == index)[0]) + 1 for index in range(len(tasks))]
    )
    text_ranks = np.asarray(
        [int(np.flatnonzero(text_order[index] == index)[0]) + 1 for index in range(len(tasks))]
    )
    exact = {
        "action_to_text_recall_at_1": float(np.mean(action_ranks == 1)),
        "action_to_text_recall_at_5": float(np.mean(action_ranks <= min(5, len(tasks)))),
        "action_to_text_median_rank": float(np.median(action_ranks)),
        "text_to_action_recall_at_1": float(np.mean(text_ranks == 1)),
        "text_to_action_recall_at_5": float(np.mean(text_ranks <= min(5, len(tasks)))),
        "text_to_action_median_rank": float(np.median(text_ranks)),
    }
    return {
        "semantic": {
            "text_to_action": text_to_action,
            "action_to_text": action_to_text,
        },
        "exact_pair": exact,
    }


def reconstruction_metrics(
    predictions: np.ndarray,
    normalized_targets: np.ndarray,
    raw_targets: np.ndarray,
    tasks: Sequence[str],
    action_mean: np.ndarray,
    action_std: np.ndarray,
) -> dict[str, Any]:
    """Report raw-unit continuous errors and gripper accuracy globally/by task."""

    predicted_raw = (
        predictions[..., :6] * action_std.reshape(1, 1, 6)
        + action_mean.reshape(1, 1, 6)
    )
    squared = (predicted_raw - raw_targets[..., :6]) ** 2
    absolute = np.abs(predicted_raw - raw_targets[..., :6])
    predicted_gripper = np.where(predictions[..., 6] >= 0.0, 1.0, -1.0)
    gripper_correct = predicted_gripper == raw_targets[..., 6]

    def aggregate(mask: np.ndarray) -> dict[str, float]:
        return {
            "continuous_mse_raw_rel_action": float(squared[mask].mean()),
            "continuous_mae_raw_rel_action": float(absolute[mask].mean()),
            "gripper_accuracy": float(gripper_correct[mask].mean()),
            "normalized_continuous_mse": float(
                ((predictions[mask, :, :6] - normalized_targets[mask, :, :6]) ** 2).mean()
            ),
        }

    labels = np.asarray(tasks)
    per_task = {task: aggregate(labels == task) for task in sorted(set(tasks))}
    return {"global": aggregate(np.ones(len(tasks), dtype=bool)), "per_task": per_task}


def latent_geometry_metrics(latents: np.ndarray, tasks: Sequence[str]) -> dict[str, Any]:
    """Compare within-task and between-task Euclidean/cosine distances."""

    if len(latents) != len(tasks):
        raise ValueError("Latents and tasks are not aligned")
    difference = latents[:, None, :] - latents[None, :, :]
    euclidean = np.linalg.norm(difference, axis=-1)
    normalized = l2_normalize(latents)
    cosine_distance = 1.0 - normalized @ normalized.T
    labels = np.asarray(tasks)
    same = labels[:, None] == labels[None, :]
    off_diagonal = ~np.eye(len(labels), dtype=bool)
    within = same & off_diagonal
    between = ~same
    if not within.any() or not between.any():
        raise ValueError("Geometry metrics require repeated examples and multiple tasks")
    within_euclidean = float(euclidean[within].mean())
    between_euclidean = float(euclidean[between].mean())
    within_cosine = float(cosine_distance[within].mean())
    between_cosine = float(cosine_distance[between].mean())
    result = {
        "within_euclidean_mean": within_euclidean,
        "between_euclidean_mean": between_euclidean,
        "between_over_within_euclidean": between_euclidean / max(within_euclidean, 1e-12),
        "within_cosine_distance_mean": within_cosine,
        "between_cosine_distance_mean": between_cosine,
        "between_over_within_cosine": between_cosine / max(within_cosine, 1e-12),
    }
    if len(set(tasks)) > 1 and min(Counter(tasks).values()) > 1:
        result["silhouette_euclidean"] = float(
            silhouette_score(latents, tasks, metric="euclidean")
        )
        result["silhouette_cosine"] = float(
            silhouette_score(latents, tasks, metric="cosine")
        )
    return result

"""Task-balanced retrieval and channel-wise reconstruction metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from pglt.evaluation.metrics import l2_normalize


DIRECTIONS = ("text_to_action", "action_to_text")
ACTION_DIMENSION_LABELS = (
    "relative_tcp_position_x_scaled_clipped",
    "relative_tcp_position_y_scaled_clipped",
    "relative_tcp_position_z_scaled_clipped",
    "relative_tcp_euler_x_scaled_clipped",
    "relative_tcp_euler_y_scaled_clipped",
    "relative_tcp_euler_z_scaled_clipped",
    "binary_gripper_close_minus1_open_plus1",
)


def annotation_representations(encoded: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Average L2-normalized chunks within exact episode/annotation identities."""

    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, metadata in enumerate(encoded["metadata"]):
        grouped[(int(metadata["episode_id"]), int(metadata["annotation_id"]))].append(index)
    items = []
    for (episode, annotation), indices in sorted(grouped.items()):
        tasks = {str(encoded["task"][index]) for index in indices}
        texts = {str(encoded["text"][index]) for index in indices}
        if len(tasks) != 1 or len(texts) != 1:
            raise ValueError("One annotation produced inconsistent task/text metadata")
        items.append(
            {
                "episode_id": episode,
                "annotation_id": annotation,
                "task": next(iter(tasks)),
                "text": next(iter(texts)),
                "action": l2_normalize(encoded["latents"][indices]).mean(axis=0),
                "text_latent": l2_normalize(encoded["text_latents"][indices]).mean(axis=0),
            }
        )
    return items


def add_macro_recall(metrics: dict[str, Any]) -> dict[str, Any]:
    """Add equal-task macro R@1 to both retrieval directions."""

    for direction in DIRECTIONS:
        per_task = metrics["semantic"][direction]["per_task_recall_at_1"]
        metrics["semantic"][direction]["macro_recall_at_1"] = float(
            np.mean(list(per_task.values()))
        )
    return metrics


def _cross_direction_metrics(
    similarity: np.ndarray,
    query_tasks: Sequence[str],
    candidate_tasks: Sequence[str],
) -> dict[str, Any]:
    """Rank candidates and report task-balanced semantic relevance."""

    ranks: list[int] = []
    per_task: dict[str, list[int]] = defaultdict(list)
    order = np.argsort(-similarity, axis=1)
    for query_index, task in enumerate(query_tasks):
        relevant = [
            rank + 1
            for rank, candidate in enumerate(order[query_index])
            if candidate_tasks[candidate] == task
        ]
        if not relevant:
            raise ValueError(f"Cross-episode candidates lack task {task}")
        rank = min(relevant)
        ranks.append(rank)
        per_task[task].append(rank)
    array = np.asarray(ranks)
    per_task_r1 = {
        task: float(np.mean(np.asarray(values) == 1))
        for task, values in sorted(per_task.items())
    }
    return {
        "count": len(ranks),
        "recall_at_1": float(np.mean(array == 1)),
        "recall_at_5": float(np.mean(array <= min(5, similarity.shape[1]))),
        "median_rank": float(np.median(array)),
        "per_task_recall_at_1": per_task_r1,
        "macro_recall_at_1": float(np.mean(list(per_task_r1.values()))),
    }


def cross_episode_retrieval_metrics(
    query_items: Sequence[Mapping[str, Any]],
    candidate_items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate both directions against a separately supplied candidate bank."""

    query_action = l2_normalize(np.stack([item["action"] for item in query_items]))
    query_text = l2_normalize(np.stack([item["text_latent"] for item in query_items]))
    candidate_action = l2_normalize(np.stack([item["action"] for item in candidate_items]))
    candidate_text = l2_normalize(np.stack([item["text_latent"] for item in candidate_items]))
    query_tasks = [str(item["task"]) for item in query_items]
    candidate_tasks = [str(item["task"]) for item in candidate_items]
    return {
        "text_to_action": _cross_direction_metrics(
            query_text @ candidate_action.T, query_tasks, candidate_tasks
        ),
        "action_to_text": _cross_direction_metrics(
            query_action @ candidate_text.T, query_tasks, candidate_tasks
        ),
    }


def per_action_dimension_reconstruction(
    encoded: Mapping[str, Any], action_mean: np.ndarray, action_std: np.ndarray
) -> dict[str, Any]:
    """Report six continuous-channel MSEs and binary-gripper error separately."""

    prediction = np.asarray(encoded["predictions"])
    normalized_target = np.asarray(encoded["normalized_targets"])
    raw_target = np.asarray(encoded["raw_targets"])
    predicted_raw = prediction[..., :6] * action_std.reshape(1, 1, 6) + action_mean.reshape(1, 1, 6)
    raw_mse = ((predicted_raw - raw_target[..., :6]) ** 2).mean(axis=(0, 1))
    normalized_mse = ((prediction[..., :6] - normalized_target[..., :6]) ** 2).mean(axis=(0, 1))
    predicted_gripper = np.where(prediction[..., 6] >= 0.0, 1.0, -1.0)
    result = [
        {
            "dimension": dimension,
            "label": ACTION_DIMENSION_LABELS[dimension],
            "raw_mse": float(raw_mse[dimension]),
            "normalized_mse": float(normalized_mse[dimension]),
            "metric_semantics": "continuous_scaled_clipped_rel_action_mse",
        }
        for dimension in range(6)
    ]
    result.append(
        {
            "dimension": 6,
            "label": ACTION_DIMENSION_LABELS[6],
            "raw_mse": float(((predicted_gripper - raw_target[..., 6]) ** 2).mean()),
            "normalized_mse": None,
            "metric_semantics": "thresholded_binary_command_mse_zero_or_four_per_element",
            "mismatch_rate": float(np.mean(predicted_gripper != raw_target[..., 6])),
        }
    )
    return {"dimensions": result}


def same_task_knn_purity(latents: np.ndarray, tasks: Sequence[str], k: int) -> float:
    """Measure same-task purity among nearest non-self chunk neighbors."""

    normalized = l2_normalize(latents)
    similarity = normalized @ normalized.T
    purities = []
    for index, task in enumerate(tasks):
        order = [candidate for candidate in np.argsort(-similarity[index]) if candidate != index]
        neighbors = order[: min(k, len(order))]
        purities.append(np.mean([tasks[candidate] == task for candidate in neighbors]))
    return float(np.mean(purities))


def cross_episode_action_nn_accuracy(
    query: Mapping[str, Any], candidates: Mapping[str, Any]
) -> float:
    """Classify each query action by its nearest candidate action latent."""

    similarity = l2_normalize(query["latents"]) @ l2_normalize(candidates["latents"]).T
    nearest = np.argmax(similarity, axis=1)
    return float(
        np.mean(
            [
                query["task"][index] == candidates["task"][candidate]
                for index, candidate in enumerate(nearest)
            ]
        )
    )

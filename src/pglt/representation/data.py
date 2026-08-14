"""Leakage-safe data and evaluation primitives for the released representation.

The module is intentionally action-only.  Each compact episode contains the
exact stored CALVIN ``rel_actions`` array and global indices, while language
annotations remain in the verified official metadata. Fold normalization is
fit only to permitted training episodes, and reserved rows are kept disjoint.

The factorized representation remains one 32-dimensional continuous vector:
the first 16 coordinates receive direct language contrastive gradients and
the final 16 form an unconstrained execution residual.  Reconstruction always
decodes the full concatenation, and no independence regularizer is present.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from pglt.data.calvin import Annotation, ChunkRecord, build_chunk_records, load_annotations
from pglt.evaluation.metrics import latent_geometry_metrics, reconstruction_metrics, symmetric_retrieval_metrics
from pglt.representation.model import ActionRepresentationModel
from pglt.representation.reproducibility import load_text_feature_archive
from pglt.representation.retrieval import (
    add_macro_recall,
    annotation_representations,
    cross_episode_action_nn_accuracy,
    same_task_knn_purity,
    cross_episode_retrieval_metrics,
    per_action_dimension_reconstruction,
)


DIRECTIONS = ("text_to_action", "action_to_text")
@dataclass(frozen=True)
class ActionOnlyEpisode:
    """One official episode with only the exact model-required action field."""

    episode_row: int
    first_index: int
    last_index: int
    rel_actions: np.ndarray
    global_frame_indices: np.ndarray
    source_path: str

    @property
    def length(self) -> int:
        """Return the exact inclusive official episode length."""

        return self.last_index - self.first_index + 1


@dataclass(frozen=True)
class ActionOnlyNormalization:
    """Training-fold-only moments for the six continuous rel-action channels."""

    action_mean: np.ndarray
    action_std: np.ndarray
    source_episode_rows: tuple[int, ...]
    source_frame_count: int

    def to_json(self) -> dict[str, Any]:
        """Serialize exact moments and their permitted source episodes."""

        return {
            "action_mean": self.action_mean.tolist(),
            "action_std": self.action_std.tolist(),
            "source_episode_rows": list(self.source_episode_rows),
            "source_frame_count": self.source_frame_count,
            "continuous_dimensions_only": True,
            "gripper_unchanged_minus1_plus1": True,
        }


@dataclass(frozen=True)
class DevelopmentFold:
    """One immutable training/new-development split with explicit candidates."""

    fold_index: int
    training_episode_rows: tuple[int, ...]
    development_episode_row: int
    train_episodes: dict[int, ActionOnlyEpisode]
    development_episodes: dict[int, ActionOnlyEpisode]
    train_annotations: tuple[Annotation, ...]
    development_annotations: tuple[Annotation, ...]
    train_records: tuple[ChunkRecord, ...]
    development_records: tuple[ChunkRecord, ...]
    candidate_records: tuple[ChunkRecord, ...]
    normalization: ActionOnlyNormalization
    text_features: dict[str, np.ndarray]
    leakage_checks: dict[str, bool]


@dataclass(frozen=True)
class ConfirmationData:
    """Frozen development-training source and four isolated confirmation rows."""

    training_episode_rows: tuple[int, ...]
    confirmation_episode_rows: tuple[int, ...]
    train_episodes: dict[int, ActionOnlyEpisode]
    confirmation_episodes: dict[int, ActionOnlyEpisode]
    train_annotations: tuple[Annotation, ...]
    confirmation_annotations: tuple[Annotation, ...]
    train_records: tuple[ChunkRecord, ...]
    confirmation_records: tuple[ChunkRecord, ...]
    candidate_records: tuple[ChunkRecord, ...]
    normalization: ActionOnlyNormalization
    text_features: dict[str, np.ndarray]
    leakage_checks: dict[str, bool]


class ActionChunkDataset:
    """Resolve normalized action chunks without loading any unused modality."""

    def __init__(
        self,
        episodes: Mapping[int, ActionOnlyEpisode],
        records: Sequence[ChunkRecord],
        normalization: ActionOnlyNormalization,
        text_features: Mapping[str, np.ndarray],
        annotation_text_override: Mapping[int, str] | None = None,
    ) -> None:
        self.episodes = dict(episodes)
        self.records = list(records)
        self.normalization = normalization
        self.text_features = dict(text_features)
        self.annotation_text_override = dict(annotation_text_override or {})

    def __len__(self) -> int:
        """Return the number of annotation-contained contiguous chunks."""

        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one Hx7 chunk with unchanged task/text/index metadata."""

        record = self.records[index]
        episode = self.episodes[record.episode_id]
        local_start = record.start_index - episode.first_index
        local_outcome = record.outcome_state_index - episode.first_index
        if local_start < 0 or local_outcome >= episode.length:
            raise IndexError(f"Chunk crosses official episode boundary: {record}")
        raw = episode.rel_actions[local_start:local_outcome]
        expected = record.outcome_state_index - record.start_index
        if raw.shape != (expected, 7):
            raise ValueError(f"Unexpected exact action chunk shape {raw.shape}")
        normalized = raw.astype(np.float32, copy=True)
        normalized[:, :6] = (
            normalized[:, :6] - self.normalization.action_mean.astype(np.float32)
        ) / self.normalization.action_std.astype(np.float32)
        text = self.annotation_text_override.get(record.annotation_id, record.text)
        if text not in self.text_features:
            raise KeyError(f"Frozen text feature absent for exact annotation {text!r}")
        return {
            "actions": torch.from_numpy(normalized),
            "raw_actions": torch.from_numpy(raw.astype(np.float32)),
            "text_feature": torch.from_numpy(np.asarray(self.text_features[text], dtype=np.float32)),
            "task": record.task,
            "text": text,
            "original_text": record.text,
            "metadata": asdict(record),
        }


def _load_one_episode(config: Mapping[str, Any], row: int, bounds: np.ndarray) -> ActionOnlyEpisode:
    """Resolve one compact episode from the released single episode root."""

    first, last = map(int, bounds[row])
    path = Path(config["source"]["compact_root"]) / "training" / f"episode_row_{row:03d}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Compact action episode is missing: {path}")
    sidecar_path = path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if [int(sidecar["first_index"]), int(sidecar["last_index"])] != [first, last]:
        raise ValueError(f"Sidecar bounds differ from official row {row}")
    with np.load(path, allow_pickle=False) as archive:
        if "rel_actions" not in archive.files or "global_frame_indices" not in archive.files:
            raise KeyError(f"Required compact fields absent from {path}")
        actions = archive["rel_actions"].copy()
        indices = archive["global_frame_indices"].copy()
    if actions.shape != (last - first + 1, 7) or actions.dtype != np.float64:
        raise ValueError(f"Exact rel_actions schema mismatch in {path}")
    if not np.array_equal(indices, np.arange(first, last + 1, dtype=np.int64)):
        raise ValueError(f"Global frame indices mismatch in {path}")
    return ActionOnlyEpisode(row, first, last, actions, indices, path.resolve().as_posix())


def load_action_only_episodes(
    config: Mapping[str, Any], rows: Sequence[int]
) -> dict[int, ActionOnlyEpisode]:
    """Load unique official training rows using authoritative metadata bounds."""

    unique = [int(row) for row in rows]
    if len(unique) != len(set(unique)):
        raise ValueError("Duplicate episode row requested")
    bounds = np.asarray(
        np.load(
            Path(config["source"]["metadata_root"]) / "training" / "ep_start_end_ids.npy",
            allow_pickle=False,
        )
    ).reshape(-1, 2)
    return {row: _load_one_episode(config, row, bounds) for row in unique}


def _group_annotations(
    config: Mapping[str, Any], episodes: Mapping[int, ActionOnlyEpisode]
) -> tuple[list[Annotation], dict[int, list[Annotation]]]:
    """Map exact selected task annotations to exactly one official episode."""

    all_annotations = load_annotations(Path(config["source"]["metadata_root"]) / "training")
    tasks = set(str(task) for task in config["data"]["tasks"])
    if tasks - {annotation.task for annotation in all_annotations}:
        raise KeyError("Configured exact task absent from official training annotations")
    grouped = {row: [] for row in episodes}
    selected = []
    for annotation in all_annotations:
        if annotation.task not in tasks:
            continue
        containing = [
            row
            for row, episode in episodes.items()
            if annotation.start_index >= episode.first_index and annotation.end_index <= episode.last_index
        ]
        if len(containing) == 1:
            grouped[containing[0]].append(annotation)
            selected.append(annotation)
        elif len(containing) > 1:
            raise ValueError(f"Annotation maps to multiple official episodes: {annotation}")
    return selected, grouped


def _records(
    grouped: Mapping[int, Sequence[Annotation]],
    episodes: Mapping[int, ActionOnlyEpisode],
    *,
    split: str,
    chunk_length: int,
    stride: int,
) -> list[ChunkRecord]:
    """Build chunks independently and assert each endpoint remains in its row."""

    result: list[ChunkRecord] = []
    for row in sorted(episodes):
        built = build_chunk_records(
            grouped[row], split=split, chunk_length=chunk_length, stride=stride, episode_id=row
        )
        for record in built:
            if record.start_index < episodes[row].first_index or record.outcome_state_index > episodes[row].last_index:
                raise ValueError(f"Chunk crossed episode row {row}: {record}")
        result.extend(built)
    return result


def fit_action_normalization(
    episodes: Mapping[int, ActionOnlyEpisode]
) -> ActionOnlyNormalization:
    """Fit six continuous moments exclusively on supplied fold-training rows."""

    rows = tuple(sorted(episodes))
    concatenated = np.concatenate([episodes[row].rel_actions for row in rows], axis=0)
    std = concatenated[:, :6].std(axis=0)
    if np.any(std <= 0.0):
        raise ValueError("A training-only continuous action standard deviation is nonpositive")
    return ActionOnlyNormalization(
        concatenated[:, :6].mean(axis=0), std, rows, len(concatenated)
    )


def prepare_development_fold(config: Mapping[str, Any], fold_index: int) -> DevelopmentFold:
    """Load one preregistered fold and prove confirmation/normalization isolation."""

    historical = [int(row) for row in config["selection"]["historical_training_episode_rows"]]
    development_rows = [int(row) for row in config["selection"]["development_episode_rows"]]
    confirmation_rows = {int(row) for row in config["selection"]["confirmation_episode_rows"]}
    if fold_index < 0 or fold_index >= len(development_rows):
        raise IndexError(fold_index)
    held_out = development_rows[fold_index]
    training_rows = historical + [row for row in development_rows if row != held_out]
    if confirmation_rows & set(training_rows + [held_out]):
        raise RuntimeError("A confirmation row entered development preparation")
    episodes = load_action_only_episodes(config, training_rows + [held_out])
    _, grouped = _group_annotations(config, episodes)
    train_episodes = {row: episodes[row] for row in training_rows}
    development_episodes = {held_out: episodes[held_out]}
    train_annotations = tuple(item for row in sorted(training_rows) for item in grouped[row])
    development_annotations = tuple(grouped[held_out])
    chunk_length = int(config["data"]["chunk_length"])
    train_records = tuple(
        _records(
            {row: grouped[row] for row in training_rows}, train_episodes,
            split="training", chunk_length=chunk_length, stride=int(config["data"]["train_stride"]),
        )
    )
    development_records = tuple(
        _records(
            {held_out: grouped[held_out]}, development_episodes,
            split="development", chunk_length=chunk_length,
            stride=int(config["data"]["evaluation_stride"]),
        )
    )
    candidate_records = tuple(
        _records(
            {row: grouped[row] for row in training_rows}, train_episodes,
            split="development_candidate", chunk_length=chunk_length,
            stride=int(config["data"]["cross_episode_candidate_stride"]),
        )
    )
    normalization = fit_action_normalization(train_episodes)
    text_features = load_text_feature_archive(Path(config["text"]["feature_archive"]))
    required_texts = {item.text for item in train_annotations + development_annotations}
    if required_texts - set(text_features):
        raise KeyError(f"Frozen text features missing {sorted(required_texts - set(text_features))}")
    train_frames = {int(index) for episode in train_episodes.values() for index in episode.global_frame_indices}
    development_frames = {
        int(index) for episode in development_episodes.values() for index in episode.global_frame_indices
    }
    train_annotation_ids = {item.annotation_id for item in train_annotations}
    development_annotation_ids = {item.annotation_id for item in development_annotations}
    checks = {
        "training_and_development_rows_disjoint": held_out not in training_rows,
        "confirmation_rows_absent": confirmation_rows.isdisjoint(training_rows + [held_out]),
        "training_and_development_frames_disjoint": train_frames.isdisjoint(development_frames),
        "training_and_development_annotations_disjoint": train_annotation_ids.isdisjoint(development_annotation_ids),
        "normalization_rows_equal_fold_training_rows": set(normalization.source_episode_rows) == set(training_rows),
        "normalization_frame_count_exact": normalization.source_frame_count == sum(item.length for item in train_episodes.values()),
        "all_training_chunks_reference_training_rows": all(item.episode_id in training_rows for item in train_records),
        "all_development_chunks_reference_held_out_row": all(item.episode_id == held_out for item in development_records),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Representation fold leakage check failed: {checks}")
    return DevelopmentFold(
        fold_index, tuple(training_rows), held_out, train_episodes, development_episodes,
        train_annotations, development_annotations, train_records, development_records,
        candidate_records, normalization, text_features, checks,
    )


def prepare_confirmation_data(config: Mapping[str, Any]) -> ConfirmationData:
    """Load the frozen final training rows and isolated confirmation episodes.

    Normalization and model fitting use only historical plus all eight
    development episodes; no confirmation value enters a fitted statistic.
    """

    training_rows = tuple(
        int(row)
        for row in config["selection"]["historical_training_episode_rows"]
        + config["selection"]["development_episode_rows"]
    )
    confirmation_rows = tuple(int(row) for row in config["selection"]["confirmation_episode_rows"])
    if not set(training_rows).isdisjoint(confirmation_rows):
        raise RuntimeError("Confirmation rows overlap frozen training rows")
    episodes = load_action_only_episodes(config, training_rows + confirmation_rows)
    _, grouped = _group_annotations(config, episodes)
    train_episodes = {row: episodes[row] for row in training_rows}
    confirmation_episodes = {row: episodes[row] for row in confirmation_rows}
    train_annotations = tuple(item for row in sorted(training_rows) for item in grouped[row])
    confirmation_annotations = tuple(item for row in sorted(confirmation_rows) for item in grouped[row])
    chunk_length = int(config["data"]["chunk_length"])
    train_records = tuple(
        _records(
            {row: grouped[row] for row in training_rows}, train_episodes,
            split="training", chunk_length=chunk_length, stride=int(config["data"]["train_stride"]),
        )
    )
    confirmation_records = tuple(
        _records(
            {row: grouped[row] for row in confirmation_rows}, confirmation_episodes,
            split="training_distribution_confirmation", chunk_length=chunk_length,
            stride=int(config["data"]["evaluation_stride"]),
        )
    )
    candidate_records = tuple(
        _records(
            {row: grouped[row] for row in training_rows}, train_episodes,
            split="confirmation_candidate", chunk_length=chunk_length,
            stride=int(config["data"]["cross_episode_candidate_stride"]),
        )
    )
    normalization = fit_action_normalization(train_episodes)
    text_features = load_text_feature_archive(Path(config["text"]["feature_archive"]))
    required = {item.text for item in train_annotations + confirmation_annotations}
    if required - set(text_features):
        raise KeyError(f"Frozen text features missing {sorted(required - set(text_features))}")
    train_frames = {int(index) for item in train_episodes.values() for index in item.global_frame_indices}
    confirmation_frames = {
        int(index) for item in confirmation_episodes.values() for index in item.global_frame_indices
    }
    checks = {
        "training_and_confirmation_rows_disjoint": set(training_rows).isdisjoint(confirmation_rows),
        "training_and_confirmation_frames_disjoint": train_frames.isdisjoint(confirmation_frames),
        "training_and_confirmation_annotations_disjoint": {
            item.annotation_id for item in train_annotations
        }.isdisjoint({item.annotation_id for item in confirmation_annotations}),
        "normalization_rows_equal_training_rows": set(normalization.source_episode_rows) == set(training_rows),
        "normalization_excludes_confirmation_rows": set(normalization.source_episode_rows).isdisjoint(confirmation_rows),
        "all_confirmation_chunks_reference_only_confirmation_rows": all(
            item.episode_id in confirmation_rows for item in confirmation_records
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Confirmation leakage check failed: {checks}")
    return ConfirmationData(
        training_rows, confirmation_rows, train_episodes, confirmation_episodes,
        train_annotations, confirmation_annotations, train_records, confirmation_records,
        candidate_records, normalization, text_features, checks,
    )


def create_confirmation_datasets(
    data: ConfirmationData,
    annotation_text_override: Mapping[int, str] | None = None,
) -> tuple[ActionChunkDataset, ActionChunkDataset, ActionChunkDataset]:
    """Create frozen training, confirmation, and training-candidate datasets."""

    common = {"normalization": data.normalization, "text_features": data.text_features}
    return (
        ActionChunkDataset(data.train_episodes, data.train_records, **common, annotation_text_override=annotation_text_override),
        ActionChunkDataset(data.confirmation_episodes, data.confirmation_records, **common),
        ActionChunkDataset(data.train_episodes, data.candidate_records, **common),
    )


def create_fold_datasets(
    fold: DevelopmentFold, annotation_text_override: Mapping[int, str] | None = None
) -> tuple[ActionChunkDataset, ActionChunkDataset, ActionChunkDataset]:
    """Create training, held-out query, and other-episode candidate datasets."""

    common = {"normalization": fold.normalization, "text_features": fold.text_features}
    return (
        ActionChunkDataset(fold.train_episodes, fold.train_records, **common, annotation_text_override=annotation_text_override),
        ActionChunkDataset(fold.development_episodes, fold.development_records, **common),
        ActionChunkDataset(fold.train_episodes, fold.candidate_records, **common),
    )


def stack_action_batch(
    dataset: ActionChunkDataset, indices: Sequence[int], device: torch.device
) -> dict[str, Any]:
    """Stack only action and frozen-text tensors plus immutable metadata."""

    samples = [dataset[index] for index in indices]
    batch = {
        field: torch.stack([sample[field] for sample in samples]).to(device)
        for field in ("actions", "raw_actions", "text_feature")
    }
    for field in ("task", "text", "original_text", "metadata"):
        batch[field] = [sample[field] for sample in samples]
    return batch


def build_representation_model(
    config: Mapping[str, Any], device: torch.device
) -> ActionRepresentationModel:
    """Build the released action-only 32-D model with a 16/16 latent split."""

    model = ActionRepresentationModel(
        input_mode="action_only",
        chunk_length=int(config["data"]["chunk_length"]),
        action_dim=int(config["model"]["action_dim"]),
        latent_dim=int(config["model"]["latent_dim"]),
        hidden_dim=int(config["model"]["hidden_dim"]),
        depth=int(config["model"]["depth"]),
        text_feature_dim=int(config["text"]["frozen_feature_dim"]),
        semantic_dim=int(config["model"]["semantic_dim"]),
    ).to(device)
    return model


@torch.no_grad()
def encode_action_dataset(
    model: ActionRepresentationModel,
    dataset: ActionChunkDataset,
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Encode full/subspace latents and reconstruct every deterministic record."""

    model.eval()
    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    metadata: dict[str, list[Any]] = defaultdict(list)
    for start in range(0, len(dataset), batch_size):
        indices = list(range(start, min(start + batch_size, len(dataset))))
        batch = stack_action_batch(dataset, indices, device)
        output = model(batch["actions"])
        arrays["latents"].append(output["latent"].cpu().numpy())
        arrays["semantic_latents"].append(output["semantic_latent"].cpu().numpy())
        arrays["execution_latents"].append(output["execution_latent"].cpu().numpy())
        arrays["text_latents"].append(model.project_text(batch["text_feature"]).cpu().numpy())
        arrays["predictions"].append(output["reconstruction"].cpu().numpy())
        arrays["normalized_targets"].append(batch["actions"].cpu().numpy())
        arrays["raw_targets"].append(batch["raw_actions"].cpu().numpy())
        for key in ("task", "text", "metadata"):
            metadata[key].extend(batch[key])
    result: dict[str, Any] = {key: np.concatenate(value, axis=0) for key, value in arrays.items()}
    result.update(metadata)
    return result


def evaluate_checkpoint(
    model: ActionRepresentationModel,
    development_dataset: ActionChunkDataset,
    candidate_dataset: ActionChunkDataset,
    *,
    device: torch.device,
    knn_k: int,
) -> dict[str, Any]:
    """Compute registered retrieval, motor, geometry, and per-dimension metrics."""

    development = encode_action_dataset(model, development_dataset, device)
    candidates = encode_action_dataset(model, candidate_dataset, device)
    # Retrieval uses only the directly aligned semantic prefix for factorized
    # models, while full-latent geometry remains separately observable.
    retrieval_view = {**development, "latents": development["semantic_latents"]}
    candidate_view = {**candidates, "latents": candidates["semantic_latents"]}
    development_items = annotation_representations(retrieval_view)
    candidate_items = annotation_representations(candidate_view)
    action = np.stack([item["action"] for item in development_items])
    text = np.stack([item["text_latent"] for item in development_items])
    tasks = [str(item["task"]) for item in development_items]
    retrieval = add_macro_recall(symmetric_retrieval_metrics(action, text, tasks))
    cross = cross_episode_retrieval_metrics(development_items, candidate_items)
    reconstruction = reconstruction_metrics(
        development["predictions"], development["normalized_targets"], development["raw_targets"],
        development["task"], development_dataset.normalization.action_mean,
        development_dataset.normalization.action_std,
    )
    full_geometry = latent_geometry_metrics(development["latents"], development["task"])
    full_geometry["same_task_knn_purity"] = same_task_knn_purity(
        development["latents"], development["task"], knn_k
    )
    semantic_geometry = latent_geometry_metrics(development["semantic_latents"], development["task"])
    semantic_geometry["same_task_knn_purity"] = same_task_knn_purity(
        development["semantic_latents"], development["task"], knn_k
    )
    semantic_geometry["cross_episode_nearest_neighbor_semantic_accuracy"] = (
        cross_episode_action_nn_accuracy(retrieval_view, candidate_view)
    )
    result = {
        "semantic_retrieval": retrieval,
        "cross_episode_retrieval": cross,
        "reconstruction": reconstruction,
        "per_action_dimension_reconstruction": per_action_dimension_reconstruction(
            development, development_dataset.normalization.action_mean,
            development_dataset.normalization.action_std,
        ),
        "full_latent_geometry": full_geometry,
        "semantic_comparison_geometry": semantic_geometry,
        "query_chunk_count": len(development_dataset),
        "candidate_chunk_count": len(candidate_dataset),
        "query_annotation_count": len(development_items),
        "candidate_annotation_count": len(candidate_items),
    }
    if development["execution_latents"].shape[1] > 0:
        execution_geometry = latent_geometry_metrics(development["execution_latents"], development["task"])
        execution_geometry["same_task_knn_purity"] = same_task_knn_purity(
            development["execution_latents"], development["task"], knn_k
        )
        execution_geometry["cross_episode_nearest_neighbor_semantic_accuracy"] = (
            cross_episode_action_nn_accuracy(
                {**development, "latents": development["execution_latents"]},
                {**candidates, "latents": candidates["execution_latents"]},
            )
        )
        result["execution_geometry"] = execution_geometry
    return result

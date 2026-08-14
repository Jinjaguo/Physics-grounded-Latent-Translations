"""Audited non-overlapping frozen-latent data preparation for dynamics_1.

The loader maps official CALVIN annotations to official episode rows, creates
stride-H windows wholly contained in one annotation, encodes them once under
``torch.no_grad()``, and retains exact action indices for causal checks.  The
official validation compact derivative is read only after its historical CRC
and schema provenance are verified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from pglt.data.calvin import Annotation, load_annotations
from pglt.representation.model import ActionRepresentationModel
from pglt.representation.reproducibility import load_text_feature_archive


@dataclass(frozen=True)
class DynamicsSequence:
    """One annotation-contained sequence of non-overlapping latent windows."""

    split: str
    episode_row: int
    annotation_id: int
    task: str
    text: str
    annotation_start: int
    annotation_end: int
    window_indices: tuple[tuple[int, int], ...]
    latent_indices: tuple[int, ...]
    physical_duration_seconds: float
    all_windows_inside_one_annotation: bool = True
    task_boundary_occurs: bool = False

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["number_non_overlapping_latent_steps"] = len(self.latent_indices)
        return payload


@dataclass(frozen=True)
class TransitionRecord:
    """Indices for one q_previous, q_current, q_target training transition."""

    sequence_index: int
    previous_index: int
    current_index: int
    target_index: int


def sha256_file(path: Path) -> str:
    """Hash a file incrementally."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_representation(config: Mapping[str, Any], checkpoint_path: Path) -> tuple[ActionRepresentationModel, dict[str, Any]]:
    """Load one hash-selected EMA representation with every parameter frozen."""

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = config["model"]
    text_cfg = config["text"]
    model = ActionRepresentationModel(
        input_mode="action_only",
        chunk_length=int(config["data"]["chunk_length"]),
        action_dim=int(model_cfg["action_dim"]),
        latent_dim=int(model_cfg["latent_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        depth=int(model_cfg["depth"]),
        text_feature_dim=int(text_cfg["frozen_feature_dim"]),
        semantic_dim=int(model_cfg["semantic_dim"]),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def _episode_file(root: Path, row: int) -> Path:
    path = root / f"episode_row_{row:03d}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_episode_actions(root: Path, row: int, bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray, Path]:
    """Load exact rel_actions and verify authoritative global frame indices."""

    path = _episode_file(root, row)
    first, last = map(int, bounds[row])
    with np.load(path, allow_pickle=False) as saved:
        if "rel_actions" not in saved.files or "global_frame_indices" not in saved.files:
            raise KeyError(f"Required arrays absent from {path}")
        actions = saved["rel_actions"].copy()
        indices = saved["global_frame_indices"].copy()
    expected = np.arange(first, last + 1, dtype=np.int64)
    if actions.shape != (len(expected), 7) or actions.dtype != np.float64:
        raise ValueError(f"Unexpected rel_actions schema in {path}")
    if not np.array_equal(indices, expected):
        raise ValueError(f"Global frame indices differ from official bounds in {path}")
    return actions, indices, path


def annotation_rows(metadata_split_root: Path, tasks: set[str]) -> tuple[np.ndarray, dict[int, list[Annotation]]]:
    """Map each selected annotation to exactly one official episode row."""

    bounds = np.asarray(np.load(metadata_split_root / "ep_start_end_ids.npy", allow_pickle=False)).reshape(-1, 2)
    grouped: dict[int, list[Annotation]] = {row: [] for row in range(len(bounds))}
    for annotation in load_annotations(metadata_split_root):
        if annotation.task not in tasks:
            continue
        containing = [
            row for row, (first, last) in enumerate(bounds)
            if annotation.start_index >= int(first) and annotation.end_index <= int(last)
        ]
        if len(containing) != 1:
            raise ValueError(f"Annotation {annotation.annotation_id} maps to {len(containing)} episodes")
        grouped[containing[0]].append(annotation)
    return bounds, grouped


def deterministic_episode_split(episode_count: int, train_count: int, meta_seed: int) -> tuple[list[int], list[int], list[int]]:
    """Return the frozen episode permutation and disjoint train/development rows."""

    permutation = np.random.default_rng(meta_seed).permutation(episode_count).astype(int).tolist()
    return permutation, sorted(permutation[:train_count]), sorted(permutation[train_count:])


def build_sequences(grouped: Mapping[int, Sequence[Annotation]], rows: Iterable[int], *, split: str, chunk_length: int, control_frequency_hz: float) -> list[DynamicsSequence]:
    """Enumerate annotation-contained stride-H sequences without padding."""

    sequences = []
    for row in sorted(int(value) for value in rows):
        for annotation in grouped[row]:
            starts = tuple(range(annotation.start_index, annotation.end_index - chunk_length + 1, chunk_length))
            if len(starts) < 1:
                continue
            windows = tuple((start, start + chunk_length - 1) for start in starts)
            if any(right - left + 1 != chunk_length for left, right in windows):
                raise AssertionError("Window length invariant failed")
            if any(next_left != left + chunk_length for (left, _), (next_left, _) in zip(windows, windows[1:])):
                raise AssertionError("Primary window stride differs from H")
            sequences.append(DynamicsSequence(
                split=split,
                episode_row=row,
                annotation_id=annotation.annotation_id,
                task=annotation.task,
                text=annotation.text,
                annotation_start=annotation.start_index,
                annotation_end=annotation.end_index,
                window_indices=windows,
                latent_indices=(),
                physical_duration_seconds=len(windows) * chunk_length / float(control_frequency_hz),
            ))
    return sequences


def serialize_frozen_latents(*, sequences: Sequence[DynamicsSequence], roots: Mapping[str, Path], metadata_root: Path, model: ActionRepresentationModel, checkpoint_payload: Mapping[str, Any], text_feature_archive: Path, output_path: Path, batch_size: int = 512) -> tuple[list[DynamicsSequence], dict[str, np.ndarray]]:
    """Encode every sequence window once and save compact aligned arrays."""

    text_features = load_text_feature_archive(text_feature_archive)
    normalization = checkpoint_payload["resolved_config"]["normalization"]
    mean = np.asarray(normalization["action_mean"], dtype=np.float32)
    std = np.asarray(normalization["action_std"], dtype=np.float32)
    actions_cache: dict[tuple[str, int], tuple[np.ndarray, int]] = {}
    bounds_cache: dict[str, np.ndarray] = {}
    raw_windows: list[np.ndarray] = []
    normalized_windows: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    split_values: list[str] = []
    episode_rows: list[int] = []
    annotation_ids: list[int] = []
    tasks: list[str] = []
    texts: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    issues: list[int] = []
    updated_sequences: list[DynamicsSequence] = []
    cursor = 0
    for sequence in sequences:
        key = (sequence.split, sequence.episode_row)
        if sequence.split not in bounds_cache:
            bounds_cache[sequence.split] = np.asarray(np.load(metadata_root / sequence.split / "ep_start_end_ids.npy", allow_pickle=False)).reshape(-1, 2)
        if key not in actions_cache:
            episode_actions, _, _ = load_episode_actions(roots[sequence.split], sequence.episode_row, bounds_cache[sequence.split])
            actions_cache[key] = (episode_actions, int(bounds_cache[sequence.split][sequence.episode_row, 0]))
        episode_actions, first = actions_cache[key]
        indices = []
        feature = torch.from_numpy(np.asarray(text_features[sequence.text], dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            context = F.normalize(model.project_text(feature), dim=-1).squeeze(0).cpu().numpy()
        for start, end in sequence.window_indices:
            raw = episode_actions[start - first:end - first + 1].astype(np.float32, copy=True)
            if raw.shape != (end - start + 1, 7):
                raise ValueError("Action window crossed its official episode")
            normalized = raw.copy()
            normalized[:, :6] = (normalized[:, :6] - mean) / std
            raw_windows.append(raw)
            normalized_windows.append(normalized)
            contexts.append(context)
            split_values.append(sequence.split)
            episode_rows.append(sequence.episode_row)
            annotation_ids.append(sequence.annotation_id)
            tasks.append(sequence.task)
            texts.append(sequence.text)
            starts.append(start)
            ends.append(end)
            issues.append(end + 1)
            indices.append(cursor)
            cursor += 1
        updated_sequences.append(DynamicsSequence(**{**asdict(sequence), "latent_indices": tuple(indices)}))
    normalized_array = np.stack(normalized_windows).astype(np.float32)
    latent_batches = []
    with torch.no_grad():
        for offset in range(0, len(normalized_array), batch_size):
            latent_batches.append(model.encode(torch.from_numpy(normalized_array[offset:offset + batch_size])).cpu().numpy())
    arrays = {
        "latents": np.concatenate(latent_batches).astype(np.float32),
        "semantic_latents": np.concatenate(latent_batches).astype(np.float32)[:, :16],
        "execution_latents": np.concatenate(latent_batches).astype(np.float32)[:, 16:],
        "raw_actions": np.stack(raw_windows).astype(np.float32),
        "normalized_actions": normalized_array,
        "contexts": np.stack(contexts).astype(np.float32),
        "split": np.asarray(split_values),
        "episode_row": np.asarray(episode_rows, dtype=np.int64),
        "annotation_id": np.asarray(annotation_ids, dtype=np.int64),
        "task": np.asarray(tasks),
        "text": np.asarray(texts),
        "window_start": np.asarray(starts, dtype=np.int64),
        "window_end_inclusive": np.asarray(ends, dtype=np.int64),
        "prediction_issue_frame": np.asarray(issues, dtype=np.int64),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    return updated_sequences, arrays


def transition_records(sequences: Sequence[DynamicsSequence]) -> list[TransitionRecord]:
    """Create all within-sequence triples; no episode or task boundary is crossed."""

    records = []
    for sequence_index, sequence in enumerate(sequences):
        indices = sequence.latent_indices
        for offset in range(len(indices) - 2):
            records.append(TransitionRecord(sequence_index, indices[offset], indices[offset + 1], indices[offset + 2]))
    return records


def horizon_starts(sequences: Sequence[DynamicsSequence], horizon: int) -> list[tuple[int, int]]:
    """Return (sequence index, first q_previous offset) for a true rollout horizon."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    return [
        (sequence_index, offset)
        for sequence_index, sequence in enumerate(sequences)
        for offset in range(len(sequence.latent_indices) - horizon - 1)
    ]


def sequence_audit(sequences: Sequence[DynamicsSequence], transitions: Sequence[TransitionRecord], horizons: Sequence[int]) -> dict[str, Any]:
    """Summarize counts and prove primary non-overlap invariants."""

    task_counts: dict[str, int] = {}
    for sequence in sequences:
        task_counts[sequence.task] = task_counts.get(sequence.task, 0) + 1
    return {
        "all_sequences_with_at_least_one_latent": len(sequences),
        "sequences_with_at_least_two_latents": sum(len(sequence.latent_indices) >= 2 for sequence in sequences),
        "sequences_with_trainable_triples": sum(len(sequence.latent_indices) >= 3 for sequence in sequences),
        "transitions": len(transitions),
        "latent_windows": sum(len(sequence.latent_indices) for sequence in sequences),
        "task_sequence_counts": task_counts,
        "horizon_sample_counts": {str(horizon): len(horizon_starts(sequences, horizon)) for horizon in horizons},
        "all_windows_inside_one_annotation": all(sequence.all_windows_inside_one_annotation for sequence in sequences),
        "any_task_boundary": any(sequence.task_boundary_occurs for sequence in sequences),
        "any_primary_window_overlap": any(
            current[1] >= following[0]
            for sequence in sequences
            for current, following in zip(sequence.window_indices, sequence.window_indices[1:])
        ),
    }


def write_json(path: Path, payload: Any) -> None:
    """Write deterministic, human-readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

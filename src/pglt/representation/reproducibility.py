"""Deterministic utilities used by the released representation pipeline."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from pglt.data.calvin import Annotation, ChunkRecord


def set_deterministic_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch and enable deterministic Torch kernels."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def load_text_feature_archive(path: Path) -> dict[str, np.ndarray]:
    """Load the frozen OpenCLIP feature archive with unique text keys."""

    with np.load(path, allow_pickle=True) as archive:
        texts = [str(value) for value in archive["texts"]]
        features = np.asarray(archive["features"], dtype=np.float32)
    if len(texts) != len(features) or len(texts) != len(set(texts)):
        raise ValueError("Frozen text-feature archive has inconsistent or duplicate keys")
    return dict(zip(texts, features))


def unique_task_batches(
    records: Sequence[ChunkRecord], seed: int, batch_size: int
) -> list[list[int]]:
    """Build deterministic batches containing at most one item per task."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    by_task: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_task[record.task].append(index)
    generator = np.random.default_rng(seed)
    for indices in by_task.values():
        generator.shuffle(indices)
    batches: list[list[int]] = []
    while any(by_task.values()):
        tasks = [task for task, indices in by_task.items() if indices]
        generator.shuffle(tasks)
        for start in range(0, len(tasks), batch_size):
            selected = tasks[start : start + batch_size]
            batches.append([by_task[task].pop() for task in selected])
    return batches


def shuffled_language_override(
    annotations: Sequence[Annotation], seed: int
) -> tuple[dict[int, str], list[dict[str, Any]]]:
    """Create a deterministic task-mismatched annotation-text control."""

    tasks = sorted({annotation.task for annotation in annotations})
    if len(tasks) < 2:
        raise ValueError("Shuffled-language control needs at least two tasks")
    texts_by_task: dict[str, list[str]] = defaultdict(list)
    for annotation in annotations:
        texts_by_task[annotation.task].append(annotation.text)
    generator = np.random.default_rng(seed)
    override: dict[int, str] = {}
    manifest: list[dict[str, Any]] = []
    for annotation in annotations:
        other_tasks = [task for task in tasks if task != annotation.task]
        source_task = other_tasks[int(generator.integers(len(other_tasks)))]
        candidates = sorted(set(texts_by_task[source_task]))
        text = candidates[int(generator.integers(len(candidates)))]
        override[annotation.annotation_id] = text
        manifest.append(
            {
                "annotation_id": annotation.annotation_id,
                "original_task": annotation.task,
                "replacement_task": source_task,
                "replacement_text": text,
            }
        )
    return override, manifest

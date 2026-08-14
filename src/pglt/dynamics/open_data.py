"""Open-data acquisition and compatibility helpers for the wave-16 protocol.

The helpers are deliberately model-free: candidate selection depends only on
source metadata, exact task labels, length, continuity, and action/time-base
compatibility.  No frozen F1/F2 checkpoint is imported here.
"""

from __future__ import annotations

from collections import Counter
import gzip
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
from typing import Any, Iterable, Mapping
import zipfile

import numpy as np

from pglt.dynamics.dynamics_data import sha256_file, write_json


SIX_TASKS = (
    "lift_blue_block_slider",
    "lift_red_block_table",
    "place_in_slider",
    "push_pink_block_right",
    "turn_off_lightbulb",
    "turn_on_lightbulb",
)


def disk_snapshot(path: Path) -> dict[str, int | str]:
    """Return byte-exact filesystem capacity for the filesystem containing path."""

    usage = shutil.disk_usage(path)
    return {
        "filesystem": os.statvfs(path).f_fsid,
        "mount_probe": str(path.resolve()),
        "total_capacity": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def directory_bytes(path: Path) -> int:
    """Measure regular-file bytes below one local data root."""

    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def assert_disk_budget(path: Path, minimum_free: int, planned_bytes: int = 0) -> dict[str, Any]:
    """Fail before an operation that would intentionally cross the 200-GB floor."""

    snapshot = disk_snapshot(path)
    projected = int(snapshot["free_bytes"]) - int(planned_bytes)
    if projected < minimum_free:
        raise RuntimeError(
            f"Disk guard: projected free bytes {projected} below required {minimum_free}"
        )
    return {**snapshot, "planned_bytes": planned_bytes, "projected_free_bytes": projected, "passed": True}


def annotation_records(path: Path, source_name: str, environment: str | None = None) -> dict[str, Any]:
    """Audit exact CALVIN annotation spans without merging adjacent records."""

    saved = np.load(path, allow_pickle=True).item()
    indices = np.asarray(saved.get("info", {}).get("indx", []), dtype=np.int64).reshape(-1, 2)
    language = saved.get("language", {})
    tasks = np.asarray(language.get("task", [""] * len(indices)), dtype=object)
    annotations = np.asarray(language.get("ann", [""] * len(indices)), dtype=object)
    records = []
    for position, (start, end) in enumerate(indices):
        task = str(tasks[position])
        raw = str(annotations[position])
        count = int(end - start + 1)
        records.append({
            "source_repo": source_name,
            "environment": environment,
            "canonical_task": task if task in SIX_TASKS else None,
            "source_task": task,
            "raw_language": raw,
            "annotation_position": position,
            "start_frame": int(start),
            "end_frame": int(end),
            "inclusive_frame_count": count,
            "contiguous": True,
            "contains_other_annotation_boundary": False,
            "direct_eligible_ge_160": task in SIX_TASKS and count >= 160,
        })
    lengths = [item["inclusive_frame_count"] for item in records]
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "annotation_count": len(records),
        "minimum_length": min(lengths) if lengths else None,
        "median_length": float(np.median(lengths)) if lengths else None,
        "maximum_length": max(lengths) if lengths else None,
        "candidate_count_ge_160": sum(item["direct_eligible_ge_160"] for item in records),
        "task_counts": dict(sorted(Counter(item["source_task"] for item in records).items())),
        "candidate_records": [item for item in records if item["direct_eligible_ge_160"]],
    }


def inspect_npz_action(path: Path) -> dict[str, Any]:
    """Inspect action keys and dimensions without changing the source arrays."""

    with np.load(path, allow_pickle=True) as saved:
        keys = list(saved.files)
        report: dict[str, Any] = {"path": str(path), "sha256": sha256_file(path), "keys": keys}
        for key in ("rel_actions", "actions", "action"):
            if key in saved:
                value = np.asarray(saved[key])
                report[key] = {
                    "shape": list(value.shape), "dtype": str(value.dtype),
                    "minimum_per_dimension": value.reshape(-1, value.shape[-1]).min(axis=0).astype(float).tolist(),
                    "maximum_per_dimension": value.reshape(-1, value.shape[-1]).max(axis=0).astype(float).tolist(),
                }
        return report


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one compact JSON-lines metadata file."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_lerobot_metadata(meta: Path) -> dict[str, Any]:
    """Summarize episode/task metadata while keeping the conversion scouting-only."""

    episodes = load_jsonl(meta / "episodes.jsonl")
    tasks = load_jsonl(meta / "tasks.jsonl")
    info = json.loads((meta / "info.json").read_text(encoding="utf-8"))
    lengths = [int(item.get("length", 0)) for item in episodes]
    task_by_index = {item.get("task_index"): item.get("task") for item in tasks}
    covered = set()
    rows = []
    for episode in episodes:
        task_indices = episode.get("tasks", [])
        descriptions = [task_by_index.get(value, str(value)) for value in task_indices]
        covered.update(text for text in descriptions if text)
        rows.append({
            "episode_index": episode.get("episode_index"),
            "length": int(episode.get("length", 0)),
            "task_ids": task_indices,
            "task_descriptions": descriptions,
        })
    fps = info.get("fps")
    return {
        "episodes": rows,
        "episode_count": len(rows),
        "length_distribution": {
            "min": min(lengths) if lengths else None,
            "median": float(np.median(lengths)) if lengths else None,
            "max": max(lengths) if lengths else None,
        },
        "fps": fps,
        "task_description_count": len(covered),
        "info": info,
        "compatibility_status": "SCOUTING_ONLY",
        "rejection_reason": (
            f"LeRobot conversion reports {fps} Hz rather than the required original 30 Hz; "
            "no interpolation or repeated actions are permitted"
        ),
    }


def pickle_summary(path: Path) -> dict[str, Any]:
    """Inspect one explicitly requested RoboVerse pickle trajectory bundle."""

    # RoboVerse pickles were emitted by NumPy 2, whose public arrays are
    # serialized through the renamed ``numpy._core`` package.  NumPy 1.x has
    # the same implementations under ``numpy.core``.
    if "numpy._core" not in sys.modules:
        sys.modules["numpy._core"] = np.core
        sys.modules["numpy._core.multiarray"] = np.core.multiarray
        sys.modules["numpy._core.numeric"] = np.core.numeric
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        payload = pickle.load(stream)

    report: dict[str, Any] = {
        "path": str(path), "sha256": sha256_file(path), "python_type": type(payload).__name__,
    }
    items: Iterable[Any]
    if isinstance(payload, dict):
        report["top_level_keys"] = sorted(map(str, payload.keys()))
        for key in ("trajs", "trajectories", "episodes", "data", "franka"):
            if key in payload and isinstance(payload[key], (list, tuple)):
                items = payload[key]
                break
            if key in payload and isinstance(payload[key], dict):
                items = list(payload[key].values())
                break
        else:
            items = [payload]
    elif isinstance(payload, (list, tuple)):
        items = payload
    else:
        items = [payload]
    items = list(items)
    report["trajectory_count"] = len(items)
    lengths: list[int] = []
    trajectory_audit: list[dict[str, Any]] = []
    for trajectory_index, item in enumerate(items):
        schema: dict[str, Any] = {
            "trajectory_index": trajectory_index,
            "type": type(item).__name__,
        }
        if isinstance(item, dict):
            schema["keys"] = sorted(map(str, item.keys()))
            action_key = next((key for key in ("rel_actions", "actions", "action") if key in item), None)
            action_value = item.get(action_key) if action_key else None
            if action_key is None:
                for nested_key, value in item.items():
                    if not isinstance(value, dict):
                        continue
                    nested_action_key = next(
                        (key for key in ("rel_actions", "actions", "action") if key in value), None
                    )
                    if nested_action_key is not None:
                        action_key = f"{nested_key}.{nested_action_key}"
                        action_value = value[nested_action_key]
                        break
            if action_key is not None:
                action_length = len(action_value)
                dimensions = []
                dtypes = set()
                for action in action_value:
                    array = np.asarray(action)
                    dimensions.append(int(array.size))
                    dtypes.add(str(array.dtype))
                dimension_counts = Counter(dimensions)
                schema.update({
                    "action_key": action_key,
                    "action_container_type": type(action_value).__name__,
                    "action_length": action_length,
                    "action_dimension_counts": {
                        str(dimension): count for dimension, count in sorted(dimension_counts.items())
                    },
                    "action_dtypes": sorted(dtypes),
                    "uniform_action_dim": dimensions[0] if dimensions and len(dimension_counts) == 1 else None,
                    "exact_7d_actions": bool(dimensions) and set(dimensions) == {7},
                })
                lengths.append(action_length)
            state_value = item.get("states")
            if state_value is not None:
                schema["state_length"] = len(state_value)
                if action_key is not None:
                    schema["state_action_length_relation"] = len(state_value) - len(action_value)
            schema["continuity_status"] = (
                "sequence_lengths_consistent_but_no_source_frame_ids"
                if action_key is not None
                and state_value is not None
                and len(state_value) - len(action_value) in (0, 1)
                else "not_proven"
            )
        trajectory_audit.append(schema)
    report["schemas"] = trajectory_audit[:5]
    report["trajectory_audit"] = trajectory_audit
    report["length_distribution"] = {
        "min": min(lengths) if lengths else None,
        "median": float(np.median(lengths)) if lengths else None,
        "max": max(lengths) if lengths else None,
    }
    report["candidate_count_ge_160"] = sum(length >= 160 for length in lengths)
    report["primary_candidate_count_ge_160_and_7d"] = sum(
        item.get("action_length", 0) >= 160 and item.get("exact_7d_actions", False)
        for item in trajectory_audit
    )
    report["trajectory_action_dimension_distribution"] = dict(sorted(Counter(
        item.get("uniform_action_dim") for item in trajectory_audit
        if item.get("uniform_action_dim") is not None
    ).items()))
    return report


def extract_zip_metadata_and_probe(zip_path: Path, destination: Path) -> dict[str, Any]:
    """Extract metadata plus one NPZ probe, never a full shard by default."""

    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        metadata = [
            name for name in names
            if name.endswith(("auto_lang_ann.npy", "scene_info.npy", "ep_lens.npy", "ep_start_end_ids.npy"))
        ]
        episode_names = [name for name in names if Path(name).name.startswith("episode_") and name.endswith(".npz")]
        selected = metadata + episode_names[:1]
        for name in selected:
            archive.extract(name, destination)
    annotation_paths = sorted(destination.rglob("auto_lang_ann.npy"))
    if len(annotation_paths) != 1:
        raise RuntimeError(f"Expected one annotation file in shard, found {len(annotation_paths)}")
    audit = annotation_records(annotation_paths[0], "VyoJ/calvin-ABCD-D-subsets")
    probes = [inspect_npz_action(path) for path in sorted(destination.rglob("episode_*.npz"))]
    return {
        "zip_member_count": len(names), "metadata_members": metadata,
        "episode_member_count": len(episode_names), "extracted_members": selected,
        "annotation_audit": audit, "action_probes": probes,
    }


def write_not_evaluated(path: Path, reason: str, **extra: Any) -> None:
    """Write an explicit prospective stop artifact after a failed data gate."""

    write_json(path, {
        "status": "NOT_TESTED_INSUFFICIENT_DATA", "reason": reason,
        "F1_F2_metrics_read": False, **extra,
    })

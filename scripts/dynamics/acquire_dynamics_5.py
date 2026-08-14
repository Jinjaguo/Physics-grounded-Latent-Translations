#!/usr/bin/env python3
"""Acquire wave-17 continuous CALVIN play blocks without model inspection.

Purpose
-------
Freeze the wave-17 source/checkpoint/continuity rules, then stage VyoJ CALVIN
ZIP shards one at a time and retain only non-overlapping 160-frame compact
blocks that stay inside one authoritative source-session row.

Parameters
----------
``--config`` selects the wave-17 YAML. ``--stage prepare`` verifies immutable
wave-16 inputs and freezes kinematic audit thresholds. ``--stage acquire``
downloads/audits shards until every prospective data-adequacy gate passes.

Usage
-----
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/acquire_dynamics_5.py --config configs/dynamics_5.yaml \
  --stage prepare
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/acquire_dynamics_5.py --config configs/dynamics_5.yaml \
  --stage acquire

Outputs
-------
Pre-registration inputs, source/hash/disk audits, and compact action/robot
state blocks are saved under ``artifacts/seventeenth_wave/data_acquisition``.
Temporary ZIPs live under ``.staging/seventeenth_wave`` and are deleted after
each shard has been compacted and audited.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
from io import BytesIO
import json
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Sequence
import zipfile

from huggingface_hub import HfApi, hf_hub_download
import numpy as np
import yaml

from pglt.dynamics.dynamics_data import sha256_file, write_json
from pglt.dynamics.open_data import assert_disk_budget, directory_bytes, disk_snapshot
from pglt.representation.reproducibility import load_text_feature_archive


ROOT = Path(__file__).resolve().parents[2]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("prepare", "acquire"))
    return parser.parse_args()


def roots(config: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    acquisition = ROOT / config["experiment"]["acquisition_root"]
    staging = ROOT / config["data"]["staging_root"]
    compact = ROOT / config["data"]["compact_root"]
    return acquisition, staging, compact


def append_command(acquisition: Path, command: str) -> None:
    path = acquisition / "executed_commands.txt"
    previous = path.read_text(encoding="utf-8") if path.exists() else ""
    path.write_text(previous + command.rstrip() + "\n", encoding="utf-8")


def historical_hash_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    records = []
    for label, value in config["historical_immutable"].items():
        path = ROOT / value["path"]
        actual = sha256_file(path)
        records.append({
            "label": label,
            "path": value["path"],
            "expected_sha256": value["sha256"],
            "actual_sha256": actual,
            "matched": actual == value["sha256"],
        })
    return {"all_matched": all(row["matched"] for row in records), "files": records}


def frozen_checkpoint_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    manifest = read_json(ROOT / config["representation"]["checkpoint_manifest"])
    representation = next(
        row for row in manifest["checkpoints"]
        if row["condition"] == config["representation"]["expected_condition"]
        and int(row["seed_base"]) == int(config["representation"]["expected_seed_base"])
    )
    result = {
        "representation": {"path": representation["path"], "sha256": representation["sha256"]},
    }
    for name, key in (("semantic", "semantic_checkpoint"), ("F1", "f1_checkpoint"), ("F2", "f2_checkpoint")):
        path = ROOT / config["models"][key]
        result[name] = {"path": config["models"][key], "sha256": sha256_file(path)}
    return result


def wrapped_orientation_jump(left: np.ndarray, right: np.ndarray) -> float:
    delta = (right - left + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.linalg.norm(delta))


def local_kinematic_reference(config: Mapping[str, Any]) -> dict[str, Any]:
    reference = ROOT / config["source"]["local_continuity_reference"]
    session_rows = np.asarray(np.load(reference / "ep_start_end_ids.npy"), dtype=np.int64)
    membership = {}
    for row_index, (start, end) in enumerate(session_rows):
        for frame in range(int(start), int(end) + 1):
            membership[frame] = row_index
    observations: dict[int, np.ndarray] = {}
    for path in sorted(reference.glob("episode_*.npz")):
        match = re.search(r"episode_(\d+)\.npz$", path.name)
        if match is None:
            continue
        with np.load(path, allow_pickle=False) as saved:
            observations[int(match.group(1))] = np.asarray(saved["robot_obs"], dtype=np.float64)
    values: dict[str, list[float]] = defaultdict(list)
    for frame in sorted(observations):
        if frame + 1 not in observations or membership.get(frame) != membership.get(frame + 1):
            continue
        left, right = observations[frame], observations[frame + 1]
        values["arm_joint_delta_norm"].append(float(np.linalg.norm(right[7:14] - left[7:14])))
        values["tcp_position_jump"].append(float(np.linalg.norm(right[:3] - left[:3])))
        values["tcp_orientation_jump"].append(wrapped_orientation_jump(left[3:6], right[3:6]))
        values["gripper_jump"].append(float(abs(right[6] - left[6])))
    summary = {}
    for metric, samples in values.items():
        array = np.asarray(samples, dtype=np.float64)
        summary[metric] = {
            "pairs": int(len(array)),
            "median": float(np.median(array)),
            "q99": float(np.quantile(array, 0.99)),
            "q999": float(np.quantile(array, 0.999)),
            "observed_max": float(array.max()),
            "diagnostic_flag_threshold": float(array.max()),
        }
    return {
        "source": reference.relative_to(ROOT).as_posix(),
        "robot_obs_schema": {
            "shape": [15], "tcp_position": [0, 3], "tcp_euler": [3, 6],
            "gripper_width": 6, "arm_joint_positions": [7, 14],
            "gripper_action_state": 14,
        },
        "same_authoritative_session_pairs": sum(len(v) for v in values.values()) // 4,
        "threshold_role": "diagnostic flags only; official ep_start_end_ids row is authoritative for reset exclusion",
        "metrics": summary,
    }


def prepare(config: Mapping[str, Any]) -> None:
    acquisition, _, compact = roots(config)
    acquisition.mkdir(parents=True, exist_ok=True)
    compact.mkdir(parents=True, exist_ok=True)
    minimum = int(config["storage"]["minimum_free_bytes"])
    planned = int(config["storage"]["planned_largest_subset_bytes"]) + int(config["storage"]["planned_compact_bytes"])
    disk = assert_disk_budget(ROOT, minimum, planned)
    historical = historical_hash_audit(config)
    write_json(acquisition / "wave16_immutable_artifact_audit.json", historical)
    if not historical["all_matched"]:
        raise RuntimeError("An immutable wave-16 artifact hash changed")
    checkpoints = frozen_checkpoint_manifest(config)
    write_json(acquisition / "frozen_model_hash_manifest.json", checkpoints)
    kinematics = local_kinematic_reference(config)
    write_json(acquisition / "frozen_kinematic_thresholds.json", kinematics)
    prereg = {
        "created_at": now(),
        "written_before_block_selection": True,
        "written_before_any_wave17_F1_F2_outputs": True,
        "source_repo": config["source"]["repo_id"],
        "source_order": config["source"]["subset_order"],
        "selection": {
            "ordering": "source shard order, then global unique annotation start/end/task/position; authoritative session-specific cap applied",
            "block_start": "annotation_start - 16 source frames",
            "block_frames": int(config["data"]["frames_per_block"]),
            "windows": int(config["data"]["windows_per_block"]),
            "window_frames": 16,
            "stride_frames": 16,
            "no_raw_frame_overlap": True,
            "maximum_blocks_per_session": int(config["data"]["maximum_blocks_per_session"]),
            "model_output_filtering": False,
            "session_boundary": "one exact authoritative ep_start_end_ids.npy row",
            "kinematic_threshold_role": "secondary diagnostic only",
        },
        "adequacy_gate": {
            "minimum_blocks": int(config["data"]["minimum_blocks"]),
            "minimum_distinct_sessions": int(config["data"]["minimum_sessions"]),
            "minimum_protocol_A_starts": config["evaluation"]["minimum_starts"],
        },
        "protocol_A": {
            "name": "CAUSAL_CONTEXT_HELD",
            "context": "annotation active at first frame of current H16 window, held for full rollout",
            "future_annotation_schedule_used": False,
            "unlabeled_start_eligible": False,
        },
        "protocol_B": {
            "name": "EXOGENOUS_CONTEXT_SCHEDULE_DIAGNOSTIC",
            "context": "annotation active at first frame of each current H16 window",
            "within_window_boundary_switch": "next H16 window at earliest",
            "unlabeled_window_eligible": False,
        },
        "horizons": config["evaluation"]["horizons"],
        "primary_endpoint": "session-clustered paired block normalized execution-error AUC; F2-F1",
        "bootstrap": {"replicates": int(config["evaluation"]["bootstrap_replicates"]), "upper_95_required_below_zero": True},
        "frozen_checkpoints": checkpoints,
        "updates": {"representation": 0, "semantic": 0, "F1": 0, "F2": 0, "EMA": 0},
        "future_raw_actions_as_model_input": False,
        "teacher_forcing_after_rollout_start": False,
    }
    write_json(acquisition / "wave17_acquisition_preregistration.json", prereg)
    write_json(acquisition / "disk_budget.json", {
        "created_at": now(), "initial": disk,
        "minimum_free_bytes": minimum,
        "planned_largest_subset_bytes": int(config["storage"]["planned_largest_subset_bytes"]),
        "planned_compact_bytes": int(config["storage"]["planned_compact_bytes"]),
        "checks": [{"phase": "prepare", **disk}],
    })
    write_json(acquisition / "staged_cleanup_log.json", {"events": []})
    append_command(acquisition, "df -h /home/jinjaguo/PGLT && df -B1 /home/jinjaguo/PGLT")
    append_command(acquisition, "PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/acquire_dynamics_5.py --config configs/dynamics_5.yaml --stage prepare")
    print(json.dumps({"stage": "prepare", "historical_hashes_match": True, "projected_free_bytes": disk["projected_free_bytes"]}, indent=2))


def one_member(archive: zipfile.ZipFile, suffix: str) -> str:
    names = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(names) != 1:
        raise RuntimeError(f"Expected one {suffix}, found {len(names)}")
    return names[0]


def load_pickle_npy(archive: zipfile.ZipFile, suffix: str) -> tuple[str, Any]:
    name = one_member(archive, suffix)
    with archive.open(name) as stream:
        value = np.load(BytesIO(stream.read()), allow_pickle=True)
        if isinstance(value, np.ndarray) and value.shape == ():
            value = value.item()
    return name, value


def frame_members(archive: zipfile.ZipFile) -> dict[int, str]:
    result = {}
    for name in archive.namelist():
        match = re.search(r"/episode_(\d+)\.npz$", name)
        if match:
            result[int(match.group(1))] = name
    return result


def unique_events(payload: Mapping[str, Any], text_keys: set[str]) -> list[dict[str, Any]]:
    ranges = np.asarray(payload["info"]["indx"], dtype=np.int64).reshape(-1, 2)
    language = payload["language"]
    tasks = np.asarray(language["task"], dtype=object)
    texts = np.asarray(language["ann"], dtype=object)
    grouped: dict[tuple[int, int, str], list[tuple[int, str]]] = defaultdict(list)
    for position, ((start, end), task, text) in enumerate(zip(ranges, tasks, texts)):
        grouped[(int(start), int(end), str(task))].append((position, str(text)))
    events = []
    for (start, end, task), variants in grouped.items():
        supported = sorted((position, text) for position, text in variants if text in text_keys)
        if not supported:
            continue
        position, text = supported[0]
        events.append({
            "annotation_position": position, "annotation_positions": [item[0] for item in sorted(variants)],
            "start_frame": start, "end_frame": end, "canonical_task": task,
            "language": text, "language_variants": [item[1] for item in sorted(variants)],
        })
    return sorted(events, key=lambda row: (row["start_frame"], row["end_frame"], row["canonical_task"], row["annotation_position"]))


def session_for_range(rows: np.ndarray, start: int, end: int) -> int | None:
    matches = np.flatnonzero((rows[:, 0] <= start) & (rows[:, 1] >= end))
    return int(matches[0]) if len(matches) == 1 else None


def overlaps_any(start: int, end: int, selected: Sequence[Mapping[str, Any]]) -> bool:
    return any(not (end < int(row["start_frame"]) or start > int(row["end_frame"])) for row in selected)


def active_event(events: Sequence[Mapping[str, Any]], frame: int) -> Mapping[str, Any] | None:
    active = [row for row in events if int(row["start_frame"]) <= frame <= int(row["end_frame"])]
    if not active:
        return None
    return sorted(active, key=lambda row: (int(row["start_frame"]), int(row["end_frame"]), str(row["canonical_task"]), int(row["annotation_position"])))[0]


def valid_offsets(window_context_positions: Sequence[int], horizon: int, protocol: str) -> list[int]:
    offsets = []
    for offset in range(len(window_context_positions) - horizon - 1):
        if protocol == "A":
            eligible = window_context_positions[offset + 1] >= 0
        else:
            eligible = all(window_context_positions[index] >= 0 for index in range(offset + 1, offset + horizon + 1))
        if eligible:
            offsets.append(offset)
    return offsets


def block_metadata(event: Mapping[str, Any], events: Sequence[Mapping[str, Any]], session_row: int, subset: str) -> dict[str, Any]:
    start = int(event["start_frame"]) - 16
    end = start + 159
    intersecting = [row for row in events if int(row["end_frame"]) >= start and int(row["start_frame"]) <= end]
    contexts = [active_event(events, start + 16 * index) for index in range(10)]
    positions = [int(row["annotation_position"]) if row is not None else -1 for row in contexts]
    tasks = [str(row["canonical_task"]) if row is not None else "NO_LANGUAGE_ANNOTATION" for row in contexts]
    texts = [str(row["language"]) if row is not None else "NO_LANGUAGE_ANNOTATION" for row in contexts]
    boundary_frames = sorted({
        frame for row in intersecting for frame in (int(row["start_frame"]), int(row["end_frame"]) + 1)
        if start < frame <= end
    })
    windows = []
    for index in range(10):
        window_start = start + 16 * index
        window_end = window_start + 15
        inside = [frame for frame in boundary_frames if window_start < frame <= window_end]
        windows.append({
            "window_index": index, "start_frame": window_start, "end_frame": window_end,
            "annotation_position": positions[index], "canonical_task": tasks[index], "language": texts[index],
            "boundary_inside_window": bool(inside),
            "boundary_before_next_window": any(window_end < frame <= window_end + 16 for frame in boundary_frames),
        })
    return {
        "source_subset": subset, "source_session_row": session_row,
        "source_session_id": f"training_ep_row_{session_row:05d}",
        "start_frame": start, "end_frame": end, "frame_count": 160, "number_H16_windows": 10,
        "anchor_annotation_position": int(event["annotation_position"]),
        "annotation_boundaries": boundary_frames,
        "annotation_sequence": intersecting,
        "window_context_annotation_positions": positions,
        "window_tasks": tasks, "window_languages": texts, "windows": windows,
        "valid_protocol_A_offsets": {str(h): valid_offsets(positions, h, "A") for h in (1, 2, 4, 8)},
        "valid_protocol_B_offsets": {str(h): valid_offsets(positions, h, "B") for h in (1, 2, 4, 8)},
        "reset_flags": [], "authoritative_same_session": True,
    }


def extract_block(archive: zipfile.ZipFile, members: Mapping[int, str], record: Mapping[str, Any], output: Path) -> dict[str, Any]:
    actions, robot, scene = [], [], []
    for frame in range(int(record["start_frame"]), int(record["end_frame"]) + 1):
        with archive.open(members[frame]) as stream:
            with np.load(BytesIO(stream.read()), allow_pickle=False) as saved:
                actions.append(np.asarray(saved["rel_actions"], dtype=np.float64))
                robot.append(np.asarray(saved["robot_obs"], dtype=np.float64))
                scene.append(np.asarray(saved["scene_obs"], dtype=np.float64))
    action_array, robot_array, scene_array = np.stack(actions), np.stack(robot), np.stack(scene)
    if action_array.shape != (160, 7) or robot_array.shape != (160, 15) or scene_array.shape != (160, 24):
        raise ValueError(f"Unexpected compact schema {action_array.shape}/{robot_array.shape}/{scene_array.shape}")
    if not all(np.isfinite(value).all() for value in (action_array, robot_array, scene_array)):
        raise ValueError("Non-finite compact source values")
    frame_ids = np.arange(int(record["start_frame"]), int(record["end_frame"]) + 1, dtype=np.int64)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, rel_actions=action_array, robot_obs=robot_array, scene_obs=scene_array, global_frame_indices=frame_ids)
    return {
        "path": output.relative_to(ROOT).as_posix(), "sha256": sha256_file(output),
        "rel_actions_shape": list(action_array.shape), "robot_obs_shape": list(robot_array.shape),
        "scene_obs_shape": list(scene_array.shape), "source_frames_contiguous": True,
    }


def kinematic_audit(record: Mapping[str, Any], compact: Mapping[str, Any], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    with np.load(ROOT / compact["path"], allow_pickle=False) as saved:
        robot = saved["robot_obs"].copy()
    metrics = {
        "arm_joint_delta_norm": np.linalg.norm(np.diff(robot[:, 7:14], axis=0), axis=1),
        "tcp_position_jump": np.linalg.norm(np.diff(robot[:, :3], axis=0), axis=1),
        "tcp_orientation_jump": np.linalg.norm((np.diff(robot[:, 3:6], axis=0) + np.pi) % (2.0 * np.pi) - np.pi, axis=1),
        "gripper_jump": np.abs(np.diff(robot[:, 6])),
    }
    result = {}
    for name, values in metrics.items():
        threshold = float(thresholds["metrics"][name]["diagnostic_flag_threshold"])
        result[name] = {"maximum": float(values.max()), "mean": float(values.mean()), "threshold": threshold, "flagged_pairs": int(np.sum(values > threshold))}
    result["any_diagnostic_flag"] = any(value["flagged_pairs"] > 0 for value in result.values() if isinstance(value, dict))
    result["official_session_metadata_reset_crossed"] = False
    result["block_id"] = record["block_id"]
    return result


def gate_status(selected: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    sessions = {row["source_session_id"] for row in selected}
    starts = {str(h): sum(len(row["valid_protocol_A_offsets"][str(h)]) for row in selected) for h in (1, 2, 4, 8)}
    minimums = {str(key): int(value) for key, value in config["evaluation"]["minimum_starts"].items()}
    passed = (
        len(selected) >= int(config["data"]["minimum_blocks"])
        and len(sessions) >= int(config["data"]["minimum_sessions"])
        and all(starts[h] >= minimums[h] for h in minimums)
    )
    return {"passed": passed, "blocks": len(selected), "distinct_sessions": len(sessions), "protocol_A_starts": starts, "minimum_protocol_A_starts": minimums}


def acquire(config: Mapping[str, Any]) -> None:
    acquisition, staging, compact_root = roots(config)
    if not (acquisition / "wave17_acquisition_preregistration.json").is_file():
        raise RuntimeError("Run prepare before acquisition")
    manifest_path = ROOT / config["data"]["continuous_block_manifest"]
    if manifest_path.exists():
        raise RuntimeError("Continuous block manifest already exists")
    minimum_free = int(config["storage"]["minimum_free_bytes"])
    thresholds = read_json(acquisition / "frozen_kinematic_thresholds.json")
    text_keys = set(load_text_feature_archive(ROOT / config["representation"]["text_feature_archive"]))
    api = HfApi()
    info = api.dataset_info(config["source"]["repo_id"], files_metadata=True)
    sizes = {row.rfilename: int(row.size or 0) for row in info.siblings}
    partial_blocks_path = acquisition / "continuous_blocks.partial.json"
    partial_audits_path = acquisition / "source_shard_audits.partial.json"
    if partial_blocks_path.exists() and partial_audits_path.exists():
        selected = read_json(partial_blocks_path)["blocks"]
        shard_audits = read_json(partial_audits_path)
    else:
        selected, shard_audits = [], []
    per_session = Counter(row["source_session_id"] for row in selected)
    downloads = [{
        "repo_path": row["source_subset"], "bytes": row["source_bytes"],
        "sha256": row["source_zip_sha256"], "revision": info.sha,
    } for row in shard_audits]
    integrity_rows = [kinematic_audit(row, row, thresholds) for row in selected]
    processed_subsets = {row["source_subset"] for row in shard_audits}
    cleanup = read_json(acquisition / "staged_cleanup_log.json")
    budget = read_json(acquisition / "disk_budget.json")
    for subset in config["source"]["subset_order"]:
        if subset in processed_subsets:
            continue
        if gate_status(selected, config)["passed"]:
            break
        before = assert_disk_budget(ROOT, minimum_free, sizes[subset] + int(config["storage"]["planned_compact_bytes"]))
        budget["checks"].append({"phase": "before_download", "subset": subset, **before})
        write_json(acquisition / "disk_budget.json", budget)
        subset_stage = staging / Path(subset).stem
        zip_path = Path(hf_hub_download(repo_id=config["source"]["repo_id"], filename=subset, repo_type="dataset", local_dir=subset_stage))
        zip_hash = sha256_file(zip_path)
        chosen = []
        with zipfile.ZipFile(zip_path) as archive:
            annotation_member, annotation_payload = load_pickle_npy(archive, "lang_annotations/auto_lang_ann.npy")
            episode_member, episode_rows = load_pickle_npy(archive, "ep_start_end_ids.npy")
            episode_rows = np.asarray(episode_rows, dtype=np.int64).reshape(-1, 2)
            members = frame_members(archive)
            events = unique_events(annotation_payload, text_keys)
            for event in events:
                block_start = int(event["start_frame"]) - 16
                block_end = block_start + 159
                session_row = session_for_range(episode_rows, block_start, block_end)
                if session_row is None:
                    continue
                session_id = f"training_ep_row_{session_row:05d}"
                if per_session[session_id] >= int(config["data"]["maximum_blocks_per_session"]):
                    continue
                if overlaps_any(block_start, block_end, selected):
                    continue
                if not all(frame in members for frame in range(block_start, block_end + 1)):
                    continue
                record = block_metadata(event, events, session_row, subset)
                block_id = f"{Path(subset).stem}_session_{session_row:05d}_frame_{block_start:07d}"
                record.update({
                    "block_id": block_id, "source_repo": config["source"]["repo_id"],
                    "source_revision": info.sha, "source_zip_sha256": zip_hash,
                    "selection_uses_model_outputs": False,
                })
                compact = extract_block(archive, members, record, compact_root / f"{block_id}.npz")
                record.update(compact)
                audit = kinematic_audit(record, compact, thresholds)
                record["kinematic_diagnostic_flags"] = audit["any_diagnostic_flag"]
                selected.append(record)
                chosen.append(record)
                integrity_rows.append(audit)
                per_session[session_id] += 1
                if gate_status(selected, config)["passed"]:
                    break
        shard_audits.append({
            "source_subset": subset, "source_zip_sha256": zip_hash,
            "source_bytes": zip_path.stat().st_size, "annotation_member": annotation_member,
            "episode_member": episode_member, "frame_members": len(members),
            "frame_min": min(members), "frame_max": max(members),
            "authoritative_session_rows": len(episode_rows), "unique_supported_events": len(events),
            "selected_here": len(chosen), "selected_session_counts": dict(sorted(Counter(row["source_session_id"] for row in chosen).items())),
        })
        downloads.append({"repo_path": subset, "bytes": zip_path.stat().st_size, "sha256": zip_hash, "revision": info.sha})
        temporary_bytes = directory_bytes(subset_stage)
        shutil.rmtree(subset_stage)
        after = assert_disk_budget(ROOT, minimum_free)
        budget["checks"].append({"phase": "after_cleanup", "subset": subset, **after})
        cleanup["events"].append({
            "created_at": now(), "subset": subset, "removed": subset_stage.relative_to(ROOT).as_posix(),
            "removed_temporary_bytes": temporary_bytes, "retained_compact_blocks": len(chosen),
            "free_bytes_after_cleanup": after["free_bytes"],
        })
        write_json(acquisition / "disk_budget.json", budget)
        write_json(acquisition / "staged_cleanup_log.json", cleanup)
        write_json(acquisition / "source_shard_audits.partial.json", shard_audits)
        write_json(acquisition / "continuous_blocks.partial.json", {"blocks": selected, "gate": gate_status(selected, config)})
    gate = gate_status(selected, config)
    manifest = {
        "created_at": now(), "written_before_wave17_F1_F2_outputs": True,
        "selection_uses_model_outputs": False, "source_repo": config["source"]["repo_id"],
        "source_revision": info.sha, "selection_order": "shard, then global unique annotation start/end/task/position, with a per-session cap",
        "blocks": selected, "gate": gate,
        "per_session_block_counts": dict(sorted(per_session.items())),
        "no_raw_frame_overlap": all(not overlaps_any(int(row["start_frame"]), int(row["end_frame"]), selected[:index]) for index, row in enumerate(selected)),
    }
    write_json(manifest_path, manifest)
    write_json(acquisition / "source_shard_audits.json", shard_audits)
    write_json(acquisition / "download_manifest.json", {"repo_id": config["source"]["repo_id"], "revision": info.sha, "files": downloads, "downloaded_bytes": sum(row["bytes"] for row in downloads)})
    write_json(acquisition / "continuous_play_integrity_audit.json", {
        "authoritative_session_boundary": "ep_start_end_ids.npy row", "any_reset_crossed": False,
        "all_source_frame_ranges_contiguous": True, "kinematic_thresholds": thresholds,
        "blocks": integrity_rows,
    })
    write_json(acquisition / "sha256_manifest.json", {"source_zips": downloads, "compact_blocks": [{"path": row["path"], "sha256": row["sha256"]} for row in selected]})
    budget["updated_at"] = now()
    budget["final"] = assert_disk_budget(ROOT, minimum_free)
    write_json(acquisition / "disk_budget.json", budget)
    append_command(acquisition, "PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/acquire_dynamics_5.py --config configs/dynamics_5.yaml --stage acquire")
    print(json.dumps({"stage": "acquire", "gate": gate, "subsets": [row["source_subset"] for row in shard_audits], "free_bytes": budget["final"]["free_bytes"]}, indent=2))
    if not gate["passed"]:
        write_json(acquisition / "missing_data_plan.json", {"reason": "prospective data adequacy gate failed", "observed": gate, "required": gate["minimum_protocol_A_starts"], "next_source_subset": config["source"]["subset_order"][len(shard_audits):len(shard_audits) + 1], "primary_inference_prohibited": True})
        raise RuntimeError("Wave-17 data adequacy gate failed; primary inference prohibited")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.stage == "prepare":
        prepare(config)
    else:
        acquire(config)


if __name__ == "__main__":
    main()

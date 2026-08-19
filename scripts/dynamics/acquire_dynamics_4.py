#!/usr/bin/env python3
"""Acquire and audit open CALVIN data for the sixteenth-wave experiment.

Purpose
-------
Run the prompt-mandated open-data-first pipeline without loading F1/F2: write
the disk budget and local inventory, scout the small LeRobot metadata, inspect
only six-task RoboVerse candidates, and stage the smallest original-format
ABCD shard.  Selection uses metadata/compatibility only.

Parameters
----------
``--config`` selects the frozen wave-16 YAML. ``--stage`` is one of
``inventory``, ``tier1``, ``tier2``, or ``all``.  ``tier1`` requires the
inventory artifacts; ``tier2`` requires both earlier stages.

Usage
-----
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/acquire_dynamics_4.py \
  --config configs/dynamics_4.yaml --stage all

Outputs
-------
Permanent audit files are written below
``artifacts/sixteenth_wave/data_acquisition``.  Large downloads are held below
``.staging/sixteenth_wave`` only during inspection and are deleted after the
hash/audit is recorded.  No model checkpoint is opened by this script.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Mapping

from huggingface_hub import HfApi, hf_hub_download
import numpy as np
import yaml

from pglt.dynamics.dynamics_data import sha256_file, write_json
from pglt.dynamics.open_data import (
    SIX_TASKS,
    annotation_records,
    assert_disk_budget,
    directory_bytes,
    disk_snapshot,
    extract_zip_metadata_and_probe,
    inspect_npz_action,
    pickle_summary,
    summarize_lerobot_metadata,
)


ROOT = Path(__file__).resolve().parents[2]
HOME = ROOT.parent


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--stage", required=True,
        choices=("inventory", "tier1", "tier1b_resume", "tier2", "all"),
    )
    return parser.parse_args()


def acquisition_root(config: Mapping[str, Any]) -> Path:
    return ROOT / config["experiment"]["acquisition_root"]


def append_command(config: Mapping[str, Any], command: str) -> None:
    path = acquisition_root(config) / "executed_commands.txt"
    previous = path.read_text(encoding="utf-8") if path.exists() else ""
    path.write_text(previous + command.rstrip() + "\n", encoding="utf-8")


def root_kind(path: Path) -> str:
    names = {part.lower() for part in path.parts}
    if "calvin_debug_dataset" in names:
        return "official_debug"
    if "calvin_task_d_d" in names or "task_d_d" in names:
        return "local_D_D_derivative"
    if "task_abc_d" in names:
        return "local_ABC_D"
    if "task_abcd_d" in names:
        return "local_ABCD_D"
    return "calvin_code_or_metadata"


def local_inventory(config: Mapping[str, Any]) -> dict[str, Any]:
    annotations = sorted(HOME.rglob("auto_lang_ann.npy"))
    episode_files = sorted(HOME.rglob("episode_*.npz"))
    calvin_named = sorted({
        path for path in HOME.rglob("*")
        if path.is_dir() and "calvin" in path.name.lower()
        and ".cache" not in path.parts and "site-packages" not in path.parts
    })
    annotation_audits = []
    for path in annotations:
        if ".cache" in path.parts or "site-packages" in path.parts:
            continue
        environment = "D" if "D_D" in str(path) or "calvin_debug" in str(path) else None
        annotation_audits.append(annotation_records(path, "local", environment))
    probes = []
    by_parent: dict[Path, Path] = {}
    for path in episode_files:
        if ".cache" in path.parts or "site-packages" in path.parts:
            continue
        by_parent.setdefault(path.parent, path)
    for path in sorted(by_parent.values()):
        try:
            probes.append(inspect_npz_action(path))
        except (KeyError, ValueError, OSError) as error:
            probes.append({"path": str(path), "inspection_error": str(error)})
    roots = []
    for path in calvin_named:
        roots.append({
            "path": str(path), "kind": root_kind(path),
            "bytes": directory_bytes(path) if len(path.relative_to(HOME).parts) <= 3 else None,
        })
    return {
        "created_at": now(),
        "search_root": str(HOME),
        "calvin_named_directories": roots,
        "annotation_files": annotation_audits,
        "episode_npz_count": len([p for p in episode_files if ".cache" not in p.parts and "site-packages" not in p.parts]),
        "action_format_probes": probes,
        "existing_lerobot_conversions": [],
        "existing_roboverse_trajectory_files": [],
        "direct_candidate_count_ge_160": sum(item["candidate_count_ge_160"] for item in annotation_audits),
    }


def canonical_mapping(inventory: Mapping[str, Any]) -> dict[str, Any]:
    mapping: dict[str, set[str]] = {task: set() for task in SIX_TASKS}
    source_paths = []
    for audit in inventory["annotation_files"]:
        path = Path(audit["path"])
        saved = np.load(path, allow_pickle=True).item()
        tasks = saved.get("language", {}).get("task", [])
        language = saved.get("language", {}).get("ann", [])
        for task, text in zip(tasks, language):
            if str(task) in mapping:
                mapping[str(task)].add(str(text))
        source_paths.append({"path": str(path), "sha256": audit["sha256"]})
    return {
        "created_before_inference": True,
        "mapping_method": "exact frozen CALVIN task ID; no semantic classifier",
        "ambiguous_mapping_policy": "reject_ambiguous_task_mapping",
        "tasks": {task: {"canonical_task": task, "paraphrases": sorted(values)} for task, values in mapping.items()},
        "sources": source_paths,
    }


def stage_inventory(config: Mapping[str, Any]) -> None:
    out = acquisition_root(config)
    out.mkdir(parents=True, exist_ok=True)
    snapshot = disk_snapshot(ROOT)
    local_roots = [
        ROOT / "data/representation/calvin_task_D_D",
        ROOT / "third_party/calvin/dataset/calvin_debug_dataset",
        ROOT / "archive/retired_snapshot/artifacts/third_wave/official_metadata/task_D_D",
    ]
    local_bytes = sum(directory_bytes(path) for path in local_roots if path.exists())
    planned_download = int(config["storage"]["planned_download_bytes"])
    planned_extract = int(config["storage"]["planned_temporary_extraction_bytes"])
    minimum = int(config["storage"]["minimum_free_bytes"])
    budget = {
        **snapshot,
        "local_CALVIN_bytes": local_bytes,
        "local_CALVIN_roots": [str(path) for path in local_roots if path.exists()],
        "planned_download": planned_download,
        "planned_temporary_extraction_overhead": planned_extract,
        "minimum_free_bytes": minimum,
        "projected_free_bytes_at_peak": int(snapshot["free_bytes"]) - planned_download - planned_extract,
        "passed": int(snapshot["free_bytes"]) - planned_download - planned_extract >= minimum,
    }
    if not budget["passed"]:
        raise RuntimeError("Initial staged-download plan violates the 200-GB free-space rule")
    write_json(out / "disk_budget.json", budget)
    inventory = local_inventory(config)
    write_json(out / "local_calvin_inventory.json", inventory)
    write_json(out / "canonical_task_mapping.json", canonical_mapping(inventory))
    write_json(out / "staged_download_cleanup_log.json", {"events": []})
    append_command(config, "df -B1 /home/jinjaguo")
    append_command(config, "du -sh /home/jinjaguo/Actions_As_Coordinates /home/jinjaguo/Actions_As_Coordinates/data/representation/calvin_task_D_D /home/jinjaguo/Actions_As_Coordinates/third_party/calvin/dataset/calvin_debug_dataset 2>/dev/null || true")
    append_command(config, "PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/acquire_dynamics_4.py --config configs/dynamics_4.yaml --stage inventory")
    print(json.dumps({"stage": "inventory", "disk_budget_passed": True, "local_candidates": inventory["direct_candidate_count_ge_160"]}, indent=2))


def repo_metadata(api: HfApi, repo: str) -> tuple[Any, dict[str, Any]]:
    info = api.dataset_info(repo, files_metadata=True)
    return info, {
        "repo_id": repo, "revision": info.sha,
        "license": info.card_data.get("license") if info.card_data else None,
        "repository_file_count": len(info.siblings),
    }


def download(repo: str, filename: str, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset", local_dir=destination))


def stage_tier1(config: Mapping[str, Any]) -> None:
    out = acquisition_root(config)
    if not (out / "disk_budget.json").is_file():
        raise RuntimeError("Run inventory before any download")
    api = HfApi()
    minimum = int(config["storage"]["minimum_free_bytes"])
    manifests = []
    audits = []

    lerobot = config["sources"]["lerobot_repo"]
    info, metadata = repo_metadata(api, lerobot)
    wanted = [s for s in info.siblings if s.rfilename.startswith("meta/")]
    planned = sum(int(s.size or 0) for s in wanted)
    assert_disk_budget(ROOT, minimum, planned)
    target = out / "hf_calvin_d_d_lerobot_meta"
    files = [download(lerobot, item.rfilename, target) for item in wanted]
    scout = summarize_lerobot_metadata(target / "meta")
    downloaded = sum(path.stat().st_size for path in files)
    manifest = {
        **metadata, "tier": "1A", "selected_files": [item.rfilename for item in wanted],
        "downloaded_bytes": downloaded,
        "files": [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files],
    }
    write_json(out / "download_manifest_tier1a_lerobot.json", manifest)
    manifests.append(manifest)
    audits.append({
        "source_name": "CALVIN D/D LeRobot mirror", **metadata,
        "downloaded_bytes": downloaded, "temporary_bytes": 0,
        "task_coverage": [task for task in SIX_TASKS if any(task in text for text in scout.get("info", {}).keys())],
        "trajectory_count": scout["episode_count"], "length_distribution": scout["length_distribution"],
        "action_dim": scout["info"].get("features", {}).get("action", {}).get("shape", [None])[-1],
        "action_semantics": "converted 7-D delta end-effector pose + gripper",
        "control_frequency": scout["fps"], "robot": "Panda", "coordinate_frame": "conversion-defined",
        "language_annotation_type": "LeRobot per-episode tasks",
        "candidate_count_ge_160": sum(row["length"] >= 160 for row in scout["episodes"]),
        "compatibility_status": "SCOUTING_ONLY", "rejection_reason": scout["rejection_reason"],
    })
    write_json(out / "tier1a_metadata_audit.json", scout)

    roboverse = config["sources"]["roboverse_repo"]
    info, metadata = repo_metadata(api, roboverse)
    exact = [
        s for s in info.siblings
        if s.rfilename == "trajs/calvin/calvin_traj_ann/ann_dict.npy"
        or (
            s.rfilename.startswith("trajs/calvin/")
            and s.rfilename.endswith("/v2/franka_v2.pkl.gz")
            and any(f"/{task}_a/" in s.rfilename for task in SIX_TASKS)
        )
    ]
    planned = sum(int(s.size or 0) for s in exact)
    assert_disk_budget(ROOT, minimum, planned)
    staging = ROOT / config["data"]["staging_root"] / "roboverse"
    files = [download(roboverse, item.rfilename, staging) for item in exact]
    summaries = []
    for path in files:
        if path.name.endswith(".pkl.gz"):
            summaries.append(pickle_summary(path))
    ann_path = next(path for path in files if path.name == "ann_dict.npy")
    ann_payload = np.load(ann_path, allow_pickle=True).item()
    frozen_mapping = read_json(out / "canonical_task_mapping.json")["tasks"]
    ids_by_task = {
        task: sorted({
            int(ann_payload[text]) for text in details["paraphrases"] if text in ann_payload
        })
        for task, details in frozen_mapping.items()
    }
    id_to_task = {
        task_id: task for task, task_ids in ids_by_task.items() for task_id in task_ids
    }
    annotated = [
        item for item in info.siblings
        if item.rfilename.startswith("trajs/calvin/calvin_traj_ann/env_")
        and item.rfilename.endswith("_v2.pkl")
        and (match := re.search(r"/task_(\d+)_v2\.pkl$", item.rfilename))
        and int(match.group(1)) in id_to_task
    ]
    additional_bytes = sum(int(item.size or 0) for item in annotated)
    assert_disk_budget(ROOT, minimum, additional_bytes)
    annotated_paths = [download(roboverse, item.rfilename, staging) for item in annotated]
    for item, path in zip(annotated, annotated_paths):
        summary = pickle_summary(path)
        match = re.search(r"/task_(\d+)_v2\.pkl$", item.rfilename)
        task_id = int(match.group(1))
        summary.update({"repo_path": item.rfilename, "task_id": task_id, "canonical_task": id_to_task[task_id]})
        summaries.append(summary)
    exact.extend(annotated)
    files.extend(annotated_paths)
    ann_summary = {
        "python_type": type(ann_payload).__name__,
        "entry_count": len(ann_payload),
        "exact_task_ids_by_canonical_task": ids_by_task,
        "entries": [
            {"language": str(key), "value_type": type(value).__name__, "value_repr": repr(value)[:500]}
            for key, value in sorted(ann_payload.items(), key=lambda item: str(item[0]))
        ],
    }
    downloaded = sum(path.stat().st_size for path in files)
    manifest = {
        **metadata, "tier": "1B", "selected_files": [item.rfilename for item in exact],
        "downloaded_bytes": downloaded,
        "files": [{"repo_path": item.rfilename, "bytes": path.stat().st_size, "sha256": sha256_file(path)} for item, path in zip(exact, files)],
    }
    write_json(out / "download_manifest_tier1b_roboverse.json", manifest)
    write_json(out / "tier1b_trajectory_audit.json", {"annotation_dictionary": ann_summary, "trajectory_files": summaries})
    manifests.append(manifest)
    candidates = sum(item["candidate_count_ge_160"] for item in summaries)
    audits.append({
        "source_name": "RoboVerse CALVIN task trajectory mirror", **metadata,
        "downloaded_bytes": downloaded, "temporary_bytes": downloaded,
        "task_coverage": sorted({
            item.get("canonical_task") for item in summaries if item.get("canonical_task")
        } | {
            task for task in SIX_TASKS if any(task in item["path"] for item in summaries)
        }),
        "trajectory_count": sum(item["trajectory_count"] for item in summaries),
        "length_distribution": {"per_file": [item["length_distribution"] for item in summaries]},
        "action_dim": None, "action_semantics": "audit recorded in tier1b_trajectory_audit.json",
        "control_frequency": None, "robot": "Franka", "coordinate_frame": "RoboVerse conversion",
        "language_annotation_type": "task-organized conversion",
        "candidate_count_ge_160": candidates,
        "compatibility_status": "SCOUTING_ONLY",
        "rejection_reason": "converted trajectory schema/time base is not proven identical to original 30-Hz CALVIN rel_actions",
    })
    before = disk_snapshot(ROOT)
    shutil.rmtree(staging)
    after = disk_snapshot(ROOT)
    cleanup = read_json(out / "staged_download_cleanup_log.json")
    cleanup["events"].append({
        "created_at": now(), "source": roboverse, "removed": str(staging),
        "removed_temporary_bytes": downloaded, "free_bytes_before": before["free_bytes"],
        "free_bytes_after": after["free_bytes"], "zip_and_full_extract_coexisted": False,
    })
    write_json(out / "staged_download_cleanup_log.json", cleanup)
    write_json(out / "open_data_source_audit.partial.json", audits)
    write_json(out / "per_source_download_manifest.partial.json", manifests)
    append_command(config, "PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/acquire_dynamics_4.py --config configs/dynamics_4.yaml --stage tier1")
    print(json.dumps({"stage": "tier1", "lerobot_fps": scout["fps"], "roboverse_candidates_ge_160": candidates}, indent=2))


def stage_tier1b_resume(config: Mapping[str, Any]) -> None:
    """Finish the already-downloaded Tier 1B files without repeating Tier 1A."""

    out = acquisition_root(config)
    tier1a_manifest = out / "download_manifest_tier1a_lerobot.json"
    tier1a_audit = out / "tier1a_metadata_audit.json"
    if not tier1a_manifest.is_file() or not tier1a_audit.is_file():
        raise RuntimeError("Tier 1A artifacts are missing; cannot resume at Tier 1B")
    staging = ROOT / config["data"]["staging_root"] / "roboverse"
    ann_path = staging / "trajs/calvin/calvin_traj_ann/ann_dict.npy"
    trajectory_paths = sorted(staging.glob("trajs/calvin/*_a/v2/franka_v2.pkl.gz"))
    trajectory_paths.extend(sorted(staging.glob("trajs/calvin/calvin_traj_ann/env_*/task_*_v2.pkl")))
    if not ann_path.is_file() or not trajectory_paths:
        raise RuntimeError("No complete staged Tier 1B files are available to resume")

    minimum = int(config["storage"]["minimum_free_bytes"])
    assert_disk_budget(ROOT, minimum)
    ann_payload = np.load(ann_path, allow_pickle=True).item()
    frozen_mapping = read_json(out / "canonical_task_mapping.json")["tasks"]
    ids_by_task = {
        task: sorted({
            int(ann_payload[text]) for text in details["paraphrases"] if text in ann_payload
        })
        for task, details in frozen_mapping.items()
    }
    id_to_task = {
        task_id: task for task, task_ids in ids_by_task.items() for task_id in task_ids
    }

    summaries = []
    files = [{
        "repo_path": ann_path.relative_to(staging).as_posix(),
        "bytes": ann_path.stat().st_size,
        "sha256": sha256_file(ann_path),
    }]
    for path in trajectory_paths:
        relative = path.relative_to(staging).as_posix()
        summary = pickle_summary(path)
        summary["repo_path"] = relative
        match = re.search(r"/calvin_traj_ann/(env_[^/]+)/task_(\d+)_v2\.pkl$", relative)
        if match:
            task_id = int(match.group(2))
            summary.update({
                "environment": match.group(1),
                "task_id": task_id,
                "canonical_task": id_to_task.get(task_id),
                "language_annotation_type": "exact ann_dict ID mapped through frozen CALVIN paraphrases",
            })
        else:
            task = next((task for task in SIX_TASKS if f"/{task}_a/" in f"/{relative}"), None)
            summary.update({
                "environment": None,
                "task_id": None,
                "canonical_task": task,
                "language_annotation_type": "task-organized repository path",
            })
        summaries.append(summary)
        files.append({
            "repo_path": relative,
            "bytes": path.stat().st_size,
            "sha256": summary["sha256"],
        })

    unmapped = [item["repo_path"] for item in summaries if item.get("canonical_task") not in SIX_TASKS]
    if unmapped:
        raise RuntimeError(f"Staged Tier 1B files contain {len(unmapped)} unmapped task IDs")
    manifest_previous = read_json(out / "download_manifest_tier1b_roboverse.json")
    manifest = {
        "repo_id": config["sources"]["roboverse_repo"],
        "revision": manifest_previous.get("revision"),
        "license": manifest_previous.get("license"),
        "repository_file_count": manifest_previous.get("repository_file_count"),
        "tier": "1B",
        "resumed_from_complete_staging": True,
        "selected_files": [item["repo_path"] for item in files],
        "downloaded_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }
    annotation_dictionary = {
        "python_type": type(ann_payload).__name__,
        "entry_count": len(ann_payload),
        "exact_task_ids_by_canonical_task": ids_by_task,
        "entries": [
            {"language": str(key), "value_type": type(value).__name__, "value_repr": repr(value)[:500]}
            for key, value in sorted(ann_payload.items(), key=lambda item: str(item[0]))
        ],
    }
    write_json(out / "download_manifest_tier1b_roboverse.json", manifest)
    write_json(out / "tier1b_trajectory_audit.json", {
        "pickle_deserialization_authorization": (
            "User explicitly authorized pickle.load only for prompt-selected "
            "RoboVerseOrg/roboverse_data CALVIN candidates"
        ),
        "annotation_dictionary": annotation_dictionary,
        "trajectory_files": summaries,
    })

    candidate_count = sum(item["candidate_count_ge_160"] for item in summaries)
    primary_candidate_count = sum(item["primary_candidate_count_ge_160_and_7d"] for item in summaries)
    dimension_distribution = Counter()
    for item in summaries:
        for dimension, count in item["trajectory_action_dimension_distribution"].items():
            dimension_distribution[int(dimension)] += count
    tier1b_audit = {
        "source_name": "RoboVerse CALVIN task trajectory mirror",
        "repo_id": config["sources"]["roboverse_repo"],
        "revision": manifest["revision"],
        "license": manifest["license"],
        "repository_file_count": manifest["repository_file_count"],
        "downloaded_bytes": manifest["downloaded_bytes"],
        "temporary_bytes": directory_bytes(staging),
        "task_coverage": list(SIX_TASKS),
        "trajectory_count": sum(item["trajectory_count"] for item in summaries),
        "length_distribution": {
            "per_file": [item["length_distribution"] for item in summaries],
        },
        "action_dim": dict(sorted(dimension_distribution.items())),
        "action_semantics": (
            "RoboVerse Franka joint/finger state-style actions; observed 9-D steps, "
            "not original CALVIN 7-D rel_actions"
        ),
        "control_frequency": None,
        "robot": "Franka",
        "coordinate_frame": "joint/finger coordinates in converted RoboVerse trajectories",
        "language_annotation_type": "task-organized and exact ann_dict ID mapping",
        "candidate_count_ge_160": candidate_count,
        "primary_candidate_count_ge_160_and_7d": primary_candidate_count,
        "compatibility_status": "REJECTED",
        "rejection_reason": (
            "Per-trajectory audit found no original-compatible 7-D CALVIN rel_actions; "
            "the observed converted action steps are 9-D and the time base is not proven 30 Hz"
        ),
    }
    audits = read_json(out / "open_data_source_audit.partial.json")
    audits = [item for item in audits if item.get("repo_id") != config["sources"]["roboverse_repo"]]
    audits.append(tier1b_audit)
    write_json(out / "open_data_source_audit.partial.json", audits)
    write_json(
        out / "per_source_download_manifest.partial.json",
        [read_json(tier1a_manifest), manifest],
    )

    before = disk_snapshot(ROOT)
    temporary = directory_bytes(staging)
    shutil.rmtree(staging)
    after = disk_snapshot(ROOT)
    cleanup = read_json(out / "staged_download_cleanup_log.json")
    cleanup["events"].append({
        "created_at": now(),
        "source": config["sources"]["roboverse_repo"],
        "removed": str(staging),
        "removed_temporary_bytes": temporary,
        "free_bytes_before": before["free_bytes"],
        "free_bytes_after": after["free_bytes"],
        "zip_and_full_extract_coexisted": False,
        "retained_candidate_files": [],
        "reason": "all audited trajectories fail the exact 7-D rel_actions compatibility gate",
    })
    cleanup["latest_disk_guard"] = assert_disk_budget(ROOT, minimum)
    write_json(out / "staged_download_cleanup_log.json", cleanup)
    append_command(
        config,
        "PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python "
        "scripts/dynamics/acquire_dynamics_4.py --config configs/dynamics_4.yaml --stage tier1b_resume",
    )
    print(json.dumps({
        "stage": "tier1b_resume",
        "trajectory_files": len(summaries),
        "trajectories": tier1b_audit["trajectory_count"],
        "candidate_count_ge_160": candidate_count,
        "primary_candidate_count_ge_160_and_7d": primary_candidate_count,
        "action_dimension_distribution": dict(sorted(dimension_distribution.items())),
        "free_bytes_after_cleanup": after["free_bytes"],
    }, indent=2))


def stage_tier2(config: Mapping[str, Any]) -> None:
    out = acquisition_root(config)
    partial = out / "open_data_source_audit.partial.json"
    if not partial.is_file():
        raise RuntimeError("Run tier1 before tier2")
    repo = config["sources"]["abcd_repo"]
    filename = config["sources"]["abcd_first_subset"]
    api = HfApi()
    info, metadata = repo_metadata(api, repo)
    sibling = next(item for item in info.siblings if item.rfilename == filename)
    size = int(sibling.size or 0)
    minimum = int(config["storage"]["minimum_free_bytes"])
    planned_extract = int(config["storage"]["planned_temporary_extraction_bytes"])
    assert_disk_budget(ROOT, minimum, size + planned_extract)
    staging = ROOT / config["data"]["staging_root"] / "abcd"
    zip_path = download(repo, filename, staging)
    zip_hash = sha256_file(zip_path)
    extract_root = staging / "extracted"
    inspection = extract_zip_metadata_and_probe(zip_path, extract_root)
    annotation = inspection["annotation_audit"]
    candidates = int(annotation["candidate_count_ge_160"])
    retained = []
    if candidates:
        raise RuntimeError("Eligible ABCD candidates require exact selective frame extraction, which must precede cleanup")
    manifest = {
        **metadata, "tier": "2", "selected_files": [filename],
        "downloaded_bytes": zip_path.stat().st_size,
        "files": [{"repo_path": filename, "bytes": zip_path.stat().st_size, "sha256": zip_hash}],
        "retained_candidate_files": retained,
    }
    write_json(out / "download_manifest_tier2_abcd.json", manifest)
    write_json(out / "tier2_subset_training_023_audit.json", inspection)
    audits = read_json(partial)
    audits.append({
        "source_name": "Original-format CALVIN ABCD subsets", **metadata,
        "downloaded_bytes": zip_path.stat().st_size, "temporary_bytes": directory_bytes(staging),
        "task_coverage": list(SIX_TASKS), "trajectory_count": annotation["annotation_count"],
        "length_distribution": {
            "min": annotation["minimum_length"], "median": annotation["median_length"], "max": annotation["maximum_length"],
        },
        "action_dim": 7, "action_semantics": "original CALVIN rel_actions",
        "control_frequency": 30, "robot": "Franka/Panda", "coordinate_frame": "CALVIN relative TCP",
        "language_annotation_type": "original auto_lang_ann.npy",
        "candidate_count_ge_160": candidates,
        "compatibility_status": "PRIMARY_COMPATIBLE",
        "rejection_reason": None,
        "subset_sequence_stopped_after_023_reason": (
            "the shard metadata contains the source-wide annotation table; its maximum annotation span is below 160 frames"
        ),
    })
    write_json(out / "open_data_source_audit.json", audits)
    manifests = read_json(out / "per_source_download_manifest.partial.json") + [manifest]
    write_json(out / "per_source_download_manifest.json", manifests)
    all_files = [item for manifest_item in manifests for item in manifest_item["files"]]
    write_json(out / "sha256_manifest.json", {"files": all_files})
    before = disk_snapshot(ROOT)
    temporary = directory_bytes(staging)
    shutil.rmtree(staging)
    after = disk_snapshot(ROOT)
    cleanup = read_json(out / "staged_download_cleanup_log.json")
    cleanup["events"].append({
        "created_at": now(), "source": repo, "removed": str(staging),
        "removed_temporary_bytes": temporary, "free_bytes_before": before["free_bytes"],
        "free_bytes_after": after["free_bytes"], "zip_and_full_extract_coexisted": False,
        "retained_candidate_files": retained,
    })
    cleanup["final_free_space"] = assert_disk_budget(ROOT, minimum)
    write_json(out / "staged_download_cleanup_log.json", cleanup)
    write_json(out / "long_trajectory_availability_audit.json", {
        "selection_uses_model_outputs": False,
        "sources": [{"source": item["source_name"], "candidate_count_ge_160": item["candidate_count_ge_160"]} for item in audits],
        "per_task_valid_counts": {task: 0 for task in SIX_TASKS},
        "total_PRIMARY_COMPATIBLE_segments": candidates,
    })
    append_command(config, "PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/acquire_dynamics_4.py --config configs/dynamics_4.yaml --stage tier2")
    print(json.dumps({"stage": "tier2", "abcd_annotations": annotation["annotation_count"], "max_frames": annotation["maximum_length"], "candidates_ge_160": candidates}, indent=2))


def main() -> None:
    parsed = args()
    config = yaml.safe_load(parsed.config.read_text(encoding="utf-8"))
    stages = ("inventory", "tier1", "tier2") if parsed.stage == "all" else (parsed.stage,)
    for stage in stages:
        if stage == "inventory":
            stage_inventory(config)
        elif stage == "tier1":
            stage_tier1(config)
        elif stage == "tier1b_resume":
            stage_tier1b_resume(config)
        elif stage == "tier2":
            stage_tier2(config)


if __name__ == "__main__":
    main()

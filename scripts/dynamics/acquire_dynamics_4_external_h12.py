#!/usr/bin/env python3
"""Acquire the amended wave-16 public CALVIN H1/H2 replication set.

Purpose
-------
Stage VyoJ CALVIN ABCD subset ZIPs in the preregistered order, select exact
six-task annotations with at least 64 contiguous frames, and retain only their
original 30-Hz 7-D ``rel_actions`` for a frozen H1/H2 external replication.

Parameters
----------
``--config`` selects the external-H1/H2 YAML. ``--stage`` is ``prepare`` or
``acquire``. Prepare freezes selection/checkpoint rules before any model output;
acquire stops as soon as 10 eligible segments exist for every frozen task.

Usage
-----
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/acquire_dynamics_4_external_h12.py \
  --config configs/dynamics_4_external_h12.yaml --stage prepare
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/acquire_dynamics_4_external_h12.py \
  --config configs/dynamics_4_external_h12.yaml --stage acquire

Outputs
-------
Audit JSON, hashes, and compact selected segment files are saved below
``artifacts/sixteenth_wave/external_h12``. Temporary ZIPs are downloaded below
``.staging/sixteenth_wave/external_h12`` and deleted after each subset audit.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from io import BytesIO
import json
from pathlib import Path
import re
import shutil
from typing import Any, Mapping
import zipfile

from huggingface_hub import HfApi, hf_hub_download
import numpy as np
import yaml

from pglt.dynamics.dynamics_data import sha256_file, write_json
from pglt.dynamics.open_data import assert_disk_budget, directory_bytes, disk_snapshot


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
    cache = ROOT / config["data"]["candidate_cache"]
    return acquisition, staging, cache


def append_command(acquisition: Path, command: str) -> None:
    path = acquisition / "executed_commands.txt"
    previous = path.read_text(encoding="utf-8") if path.exists() else ""
    path.write_text(previous + command.rstrip() + "\n", encoding="utf-8")


def prepare(config: Mapping[str, Any]) -> None:
    acquisition, _, cache = roots(config)
    acquisition.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    minimum = int(config["storage"]["minimum_free_bytes"])
    planned = (
        int(config["storage"]["planned_largest_subset_bytes"])
        + int(config["storage"]["planned_selective_extraction_bytes"])
    )
    disk = assert_disk_budget(ROOT, minimum, planned)
    representation_manifest = read_json(ROOT / config["representation"]["checkpoint_manifest"])
    representation = next(
        item for item in representation_manifest["checkpoints"]
        if item["condition"] == config["representation"]["expected_condition"]
        and int(item["seed_base"]) == int(config["representation"]["expected_seed_base"])
    )
    checkpoints = {
        "representation": {
            "path": representation["path"], "sha256": representation["sha256"],
        },
        "semantic": {
            "path": config["models"]["semantic_checkpoint"],
            "sha256": sha256_file(ROOT / config["models"]["semantic_checkpoint"]),
        },
        "F1": {
            "path": config["models"]["f1_checkpoint"],
            "sha256": sha256_file(ROOT / config["models"]["f1_checkpoint"]),
        },
        "F2": {
            "path": config["models"]["f2_checkpoint"],
            "sha256": sha256_file(ROOT / config["models"]["f2_checkpoint"]),
        },
    }
    preregistration = {
        "created_at": now(),
        "written_after_initial_open_data_audit": True,
        "written_before_external_F1_F2_outputs": True,
        "amendment_source": "prompts/dynamics_4.md post-audit amendment",
        "source_repo": config["source"]["repo_id"],
        "subset_order": config["source"]["subset_order"],
        "selection": {
            "canonical_tasks": config["data"]["tasks"],
            "minimum_contiguous_annotation_frames": 64,
            "windows": 4, "window_frames": 16, "stride_frames": 16,
            "segments_per_task": 10, "total_segments": 60,
            "ordering": "subset order, then source annotation position",
            "model_dependent_filtering": False,
            "leftover_65th_frame": "retained in source audit but excluded from four H16 windows",
        },
        "evaluation": {
            "horizons": [1, 2], "H4_H8_run": False,
            "primary_endpoint": "paired per-trajectory normalized execution-error AUC; F2-F1",
            "bootstrap_replicates": 10000,
            "success_gate": "upper_95_CI(Delta_AUC) < 0",
            "secondary": [
                "decoded action error", "off-manifold drift",
                "refinement correction-target direction", "refinement intermediate states",
            ],
        },
        "frozen_checkpoints": checkpoints,
        "updates": {
            "representation_optimizer_steps": 0, "representation_backward_calls": 0,
            "F1_optimizer_steps": 0, "F1_backward_calls": 0,
            "F2_optimizer_steps": 0, "F2_backward_calls": 0, "EMA_updates": 0,
        },
        "future_target_actions": False,
    }
    write_json(acquisition / "external_h12_acquisition_preregistration.json", preregistration)
    write_json(acquisition / "frozen_checkpoint_manifest.json", checkpoints)
    write_json(acquisition / "disk_budget.json", {
        **disk,
        "minimum_free_bytes": minimum,
        "planned_largest_subset_bytes": config["storage"]["planned_largest_subset_bytes"],
        "planned_selective_extraction_bytes": config["storage"]["planned_selective_extraction_bytes"],
    })
    write_json(acquisition / "staged_cleanup_log.json", {"events": []})
    append_command(acquisition, "df -h /home/jinjaguo/Actions_As_Coordinates && df -B1 /home/jinjaguo/Actions_As_Coordinates")
    append_command(
        acquisition,
        "PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python "
        "scripts/dynamics/acquire_dynamics_4_external_h12.py "
        "--config configs/dynamics_4_external_h12.yaml --stage prepare",
    )
    print(json.dumps({"stage": "prepare", "projected_free_bytes": disk["projected_free_bytes"]}, indent=2))


def load_annotation_payload(archive: zipfile.ZipFile) -> tuple[str, dict[str, Any]]:
    names = [name for name in archive.namelist() if name.endswith("lang_annotations/auto_lang_ann.npy")]
    if len(names) != 1:
        raise RuntimeError(f"Expected one auto_lang_ann.npy, found {len(names)}")
    with archive.open(names[0]) as stream:
        payload = np.load(BytesIO(stream.read()), allow_pickle=True).item()
    return names[0], payload


def frame_members(archive: zipfile.ZipFile) -> dict[int, str]:
    result = {}
    for name in archive.namelist():
        match = re.search(r"/episode_(\d+)\.npz$", name)
        if match:
            result[int(match.group(1))] = name
    return result


def subset_candidates(
    payload: Mapping[str, Any], members: Mapping[int, str], tasks: set[str]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    indices = np.asarray(payload["info"]["indx"], dtype=np.int64).reshape(-1, 2)
    language = payload["language"]
    task_values = np.asarray(language["task"], dtype=object)
    texts = np.asarray(language["ann"], dtype=object)
    eligible = []
    all_direct = Counter()
    for position, (start, end) in enumerate(indices):
        task = str(task_values[position])
        count = int(end - start + 1)
        if task not in tasks or count < 64:
            continue
        all_direct[task] += 1
        available = all(frame in members for frame in range(int(start), int(end) + 1))
        if available:
            eligible.append({
                "annotation_position": position,
                "canonical_task": task,
                "raw_language": str(texts[position]),
                "start_frame": int(start), "end_frame": int(end),
                "inclusive_frame_count": count,
                "contiguous": True, "contains_other_annotation_boundary": False,
                "source_frames_all_in_subset": True,
            })
    return eligible, dict(sorted(all_direct.items()))


def extract_segment(
    archive: zipfile.ZipFile, members: Mapping[int, str], candidate: Mapping[str, Any], output: Path
) -> dict[str, Any]:
    actions = []
    for frame in range(int(candidate["start_frame"]), int(candidate["end_frame"]) + 1):
        with archive.open(members[frame]) as stream:
            with np.load(BytesIO(stream.read()), allow_pickle=False) as saved:
                value = np.asarray(saved["rel_actions"])
        if value.shape != (7,) or value.dtype != np.float64 or not np.isfinite(value).all():
            raise ValueError(f"Invalid original rel_actions schema at frame {frame}: {value.shape}/{value.dtype}")
        actions.append(value)
    array = np.stack(actions)
    indices = np.arange(int(candidate["start_frame"]), int(candidate["end_frame"]) + 1, dtype=np.int64)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, rel_actions=array, global_frame_indices=indices)
    return {
        "path": output.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(output),
        "rel_actions_shape": list(array.shape), "rel_actions_dtype": str(array.dtype),
        "continuous_min": array[:, :6].min(axis=0).astype(float).tolist(),
        "continuous_max": array[:, :6].max(axis=0).astype(float).tolist(),
        "gripper_values": sorted(np.unique(array[:, 6]).astype(float).tolist()),
        "four_window_ranges": [
            [int(indices[offset]), int(indices[offset + 15])] for offset in range(0, 64, 16)
        ],
        "leftover_frames_not_windowed": int(len(indices) - 64),
    }


def acquire(config: Mapping[str, Any]) -> None:
    acquisition, staging, cache = roots(config)
    prereg_path = acquisition / "external_h12_acquisition_preregistration.json"
    if not prereg_path.is_file():
        raise RuntimeError("Prepare/preregister must run before acquisition")
    if (acquisition / "selected_segments_manifest.json").exists():
        raise RuntimeError("Selected segment manifest already exists; acquisition is already complete")
    minimum = int(config["storage"]["minimum_free_bytes"])
    api = HfApi()
    info = api.dataset_info(config["source"]["repo_id"], files_metadata=True)
    size_by_path = {item.rfilename: int(item.size or 0) for item in info.siblings}
    selected: list[dict[str, Any]] = []
    counts = Counter()
    subset_audits = []
    subset_manifests = []
    cleanup = read_json(acquisition / "staged_cleanup_log.json")
    tasks = set(config["data"]["tasks"])
    target = int(config["data"]["segments_per_task"])
    for subset_path in config["source"]["subset_order"]:
        if all(counts[task] >= target for task in tasks):
            break
        expected_bytes = size_by_path[subset_path]
        assert_disk_budget(ROOT, minimum, expected_bytes + int(config["storage"]["planned_selective_extraction_bytes"]))
        subset_stage = staging / Path(subset_path).stem
        zip_path = Path(hf_hub_download(
            repo_id=config["source"]["repo_id"], filename=subset_path,
            repo_type="dataset", local_dir=subset_stage,
        ))
        zip_hash = sha256_file(zip_path)
        before_extract = disk_snapshot(ROOT)
        chosen_here = []
        with zipfile.ZipFile(zip_path) as archive:
            annotation_member, payload = load_annotation_payload(archive)
            members = frame_members(archive)
            eligible, source_wide_counts = subset_candidates(payload, members, tasks)
            for candidate in eligible:
                task = candidate["canonical_task"]
                if counts[task] >= target:
                    continue
                segment_id = f"{Path(subset_path).stem}_ann_{candidate['annotation_position']:05d}"
                output = cache / task / f"{segment_id}.npz"
                compact = extract_segment(archive, members, candidate, output)
                record = {
                    **candidate, **compact,
                    "segment_id": segment_id,
                    "source_repo": config["source"]["repo_id"],
                    "source_revision": info.sha,
                    "source_subset": subset_path,
                    "source_zip_sha256": zip_hash,
                    "selection_uses_model_outputs": False,
                    "valid_H1_starts": 2, "valid_H2_starts": 1,
                }
                selected.append(record)
                chosen_here.append(record)
                counts[task] += 1
        subset_audits.append({
            "subset": subset_path, "source_zip_sha256": zip_hash,
            "downloaded_bytes": zip_path.stat().st_size,
            "annotation_member": annotation_member,
            "source_wide_direct_ge_64_counts": source_wide_counts,
            "subset_local_eligible_count": len(eligible),
            "subset_local_eligible_per_task": dict(sorted(Counter(
                item["canonical_task"] for item in eligible
            ).items())),
            "selected_here": len(chosen_here),
            "selected_here_per_task": dict(sorted(Counter(
                item["canonical_task"] for item in chosen_here
            ).items())),
        })
        subset_manifests.append({
            "repo_path": subset_path, "bytes": zip_path.stat().st_size,
            "sha256": zip_hash, "revision": info.sha,
        })
        temporary_bytes = directory_bytes(subset_stage)
        shutil.rmtree(subset_stage)
        after_cleanup = assert_disk_budget(ROOT, minimum)
        cleanup["events"].append({
            "created_at": now(), "subset": subset_path,
            "removed": str(subset_stage), "removed_temporary_bytes": temporary_bytes,
            "free_bytes_before_cleanup": before_extract["free_bytes"],
            "free_bytes_after_cleanup": after_cleanup["free_bytes"],
            "retained_compact_segments": len(chosen_here),
        })
        write_json(acquisition / "subset_audits.partial.json", subset_audits)
        write_json(acquisition / "download_manifest.partial.json", subset_manifests)
        write_json(acquisition / "staged_cleanup_log.json", cleanup)
    passed = len(selected) >= 60 and all(counts[task] >= target for task in tasks)
    manifest = {
        "created_at": now(), "written_before_external_F1_F2_outputs": True,
        "selection_uses_model_outputs": False,
        "source_repo": config["source"]["repo_id"], "source_revision": info.sha,
        "selection_rule": "subset order then annotation position; exact task; >=64 frames; all source frames local",
        "segments": selected,
        "per_task_counts": {task: counts[task] for task in config["data"]["tasks"]},
        "total_segments": len(selected), "gate_passed": passed,
        "H1_starts": sum(item["valid_H1_starts"] for item in selected),
        "H2_starts": sum(item["valid_H2_starts"] for item in selected),
        "H4_H8_run": False,
    }
    write_json(acquisition / "selected_segments_manifest.json", manifest)
    write_json(acquisition / "subset_audits.json", subset_audits)
    write_json(acquisition / "download_manifest.json", {
        "repo_id": config["source"]["repo_id"], "revision": info.sha,
        "files": subset_manifests,
        "downloaded_bytes": sum(item["bytes"] for item in subset_manifests),
    })
    write_json(acquisition / "sha256_manifest.json", {
        "source_zips": subset_manifests,
        "retained_segments": [{"path": item["path"], "sha256": item["sha256"]} for item in selected],
    })
    final_disk = assert_disk_budget(ROOT, minimum)
    budget = read_json(acquisition / "disk_budget.json")
    budget.update({"final": final_disk, "updated_at": now()})
    write_json(acquisition / "disk_budget.json", budget)
    append_command(
        acquisition,
        "PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python "
        "scripts/dynamics/acquire_dynamics_4_external_h12.py "
        "--config configs/dynamics_4_external_h12.yaml --stage acquire",
    )
    print(json.dumps({
        "stage": "acquire", "gate_passed": passed,
        "total_segments": len(selected), "per_task": manifest["per_task_counts"],
        "subsets_processed": [item["subset"] for item in subset_audits],
        "free_bytes": final_disk["free_bytes"],
    }, indent=2))
    if not passed:
        raise RuntimeError("Public H1/H2 acquisition did not reach 10 segments per task")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.stage == "prepare":
        prepare(config)
    else:
        acquire(config)


if __name__ == "__main__":
    main()

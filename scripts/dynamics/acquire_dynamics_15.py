#!/usr/bin/env python3
"""Acquire prospective CALVIN transitions with synchronized physical state.

Purpose
-------
Audit locally available collection routes, freeze a prospective acquisition
protocol, then stream unused official CALVIN human-play shards beginning after
the Wave21 session range.  Around each supported language onset, extract a
physically contiguous 128-frame record containing causal history, H1/H2/H4
future support, 7-D relative actions, 15-D robot state, and 24-D scene state.
Large source ZIPs are removed after compact extraction; interrupted collection
resumes from a partial manifest.

Parameters
----------
--config: Wave 27 YAML configuration.
--stage: ``prepare`` writes pre-collection manifests; ``collect`` downloads and
extracts until the prospective adequacy target is reached; ``all`` runs both.

Usage
-----
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/acquire_dynamics_15.py --config configs/dynamics_15.yaml \
  --stage all

Outputs
-------
Compact local-only NPZ records are saved below
``data/wave27_prospective/transitions``.  Tracked capability, preregistration,
inventory Parquet, split, completeness, collection report, download manifest,
and execution logs are written below
``results/dynamics/twenty_seventh_wave/2026-08-15_dynamics_15``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
GOALS = (
    "lift_blue_block_slider", "lift_red_block_table", "place_in_slider",
    "push_pink_block_right", "turn_off_lightbulb", "turn_on_lightbulb",
)


def now() -> str:
    return datetime.now().astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def out_path(config: dict) -> Path:
    return ROOT / config["experiment"]["output_root"]


def append_log(path: Path, line: str) -> None:
    previous = path.read_text() if path.exists() else ""
    path.write_text(previous + line.rstrip() + "\n")


def load_pickle_npy(archive: zipfile.ZipFile, suffix: str) -> Any:
    names = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(names) != 1:
        raise RuntimeError(f"expected one {suffix}, found {len(names)}")
    with archive.open(names[0]) as stream:
        value = np.load(BytesIO(stream.read()), allow_pickle=True)
        return value.item() if isinstance(value, np.ndarray) and value.shape == () else value


def frame_members(archive: zipfile.ZipFile) -> dict[int, str]:
    result = {}
    for name in archive.namelist():
        if "/episode_" not in name or not name.endswith(".npz"):
            continue
        stem = name.rsplit("/episode_", 1)[1][:-4]
        if stem.isdigit():
            result[int(stem)] = name
    return result


def unique_events(payload: dict) -> list[dict[str, Any]]:
    ranges = np.asarray(payload["info"]["indx"], np.int64).reshape(-1, 2)
    grouped: dict[tuple[int, int, str], list[str]] = defaultdict(list)
    for bounds, task, text in zip(ranges, payload["language"]["task"], payload["language"]["ann"]):
        if str(task) in GOALS:
            grouped[(int(bounds[0]), int(bounds[1]), str(task))].append(str(text))
    return [
        {"start_frame": start, "end_frame": end, "goal": goal, "language_variants": sorted(set(texts))}
        for (start, end, goal), texts in sorted(grouped.items())
    ]


def session_for_frame(bounds: np.ndarray, start: int, end: int) -> int | None:
    rows = np.flatnonzero((bounds[:, 0] <= start) & (bounds[:, 1] >= end))
    return int(rows[0]) if len(rows) == 1 else None


def prepare(config: dict) -> None:
    out = out_path(config); out.mkdir(parents=True, exist_ok=True)
    w21 = ROOT / config["experiment"]["wave21_root"]
    w26 = ROOT / config["experiment"]["wave26_root"]
    capability = {
        "created_at": now(),
        "routes": {
            "official_human_play_archive": {"available": True, "selected": True, "physical_fields": ["rel_actions", "robot_obs", "scene_obs"], "true_contact": False},
            "official_debug_archive": {"available": True, "selected": False, "reason": "only two source sessions and 17 annotations"},
            "trained_policy_collector": {"available": False, "selected": False, "reason": "no verified CALVIN policy checkpoint with the frozen six-goal interface"},
            "scripted_controller": {"available": False, "selected": False, "reason": "no verified six-goal primitive controller"},
            "manual_teleoperation": {"available": False, "selected": False, "reason": "VR/SpaceMouse hardware and operator loop unavailable in this run"},
        },
        "selected_route": "official human-play continuous source shards 005+",
        "collector_type": "official_human_play_archive",
        "collector_version": config["collection"]["revision"],
    }
    (out / "wave27_collection_capability_audit.md").write_text("# Wave 27 collection capability audit\n\n```json\n" + json.dumps(capability, indent=2) + "\n```\n")
    prereg = {
        "created_before_download_or_record_selection": True, "created_at": now(),
        "source_repo": config["collection"]["repo_id"], "revision": config["collection"]["revision"],
        "shard_order": config["collection"]["shard_order"],
        "excluded_authoritative_sessions": list(range(31)),
        "record": "language onset t with source frames [t-64,t+63]",
        "certification": ["one authoritative source session", "contiguous 128 frames", "no reset", "finite action/robot/scene", "H1/H2/H4 supported", "synchronized goal"],
        "counting": "transitions reported by count; inference/bootstrap clustered by source session; overlapping source ranges prohibited",
        "stopping": {key: config["collection"][key] for key in ("minimum_transitions", "preferred_transitions", "minimum_sessions", "minimum_per_goal")},
        "selection_uses_model_outputs": False, "prospective_test_opened": False,
    }
    write_json(out / "wave27_collection_preregistration.json", prereg)
    manifest26 = read_json(w26 / "wave26_frozen_manifest.json")
    frozen = {
        "created_before_new_data_collection": True, "created_at": now(),
        "representation_checkpoint": manifest26["representation_checkpoint"],
        "representation_sha256": manifest26["representation_sha256"],
        "encoder_sha256": manifest26["encoder_sha256"], "decoder_sha256": manifest26["decoder_sha256"],
        "semantic_projection_sha256": manifest26["semantic_projection_sha256"],
        "text_feature_archive_sha256": manifest26["text_feature_archive_sha256"],
        "normalization_sha256": manifest26["normalization_sha256"],
        "Wave21_B1_hashes": manifest26["Wave21_B1_hashes"], "Wave21_B0_hashes": manifest26["Wave21_B0_hashes"],
        "legacy_split_sha256": sha256(w21 / "wave21_session_split_manifest.json"),
        "Wave26_claim_matrix_sha256": sha256(w26 / "wave26_claim_matrix.json"),
        "Wave26_development_metrics_sha256": sha256(w26 / "wave26_development_metrics.json"),
        "representation_updates": 0, "decoder_updates": 0, "text_updates": 0,
    }
    write_json(out / "wave27_frozen_manifest.json", frozen)
    (out / "wave27_collection_execution_log.md").write_text("# Wave 27 collection execution log\n\n- Prepared before any new shard download or transition selection.\n")
    print(json.dumps({"stage": "prepare", "selected_route": capability["selected_route"], "prospective_test": "sealed"}), flush=True)


def adequate(rows: list[dict[str, Any]], config: dict) -> bool:
    counts = Counter(row["goal"] for row in rows)
    sessions = {row["source_session_id"] for row in rows}
    return len(rows) >= int(config["collection"]["minimum_transitions"]) and len(sessions) >= int(config["collection"]["minimum_sessions"]) and min((counts.get(goal, 0) for goal in GOALS), default=0) >= int(config["collection"]["minimum_per_goal"])


def preferred(rows: list[dict[str, Any]], config: dict) -> bool:
    counts = Counter(row["goal"] for row in rows)
    sessions = {row["source_session_id"] for row in rows}
    return len(rows) >= int(config["collection"]["preferred_transitions"]) and len(sessions) >= int(config["collection"]["minimum_sessions"]) and min((counts.get(goal, 0) for goal in GOALS), default=0) >= 2 * int(config["collection"]["minimum_per_goal"])


def split_sessions(rows: list[dict[str, Any]], seed: int) -> dict[str, list[str]]:
    sessions = sorted({row["source_session_id"] for row in rows})
    by_session = {session: Counter(row["goal"] for row in rows if row["source_session_id"] == session) for session in sessions}
    rng = np.random.default_rng(seed)
    ndev = max(1, round(.15 * len(sessions))); ntest = max(1, round(.15 * len(sessions)))
    best = None
    for _ in range(10000):
        order = list(rng.permutation(sessions))
        groups = {"new_development": order[:ndev], "new_prospective_test": order[ndev:ndev + ntest], "new_train": order[ndev + ntest:]}
        totals = {name: sum((by_session[s] for s in group), Counter()) for name, group in groups.items()}
        minimum = min((totals[name].get(goal, 0) for name in groups for goal in GOALS), default=0)
        imbalance = sum(abs(totals["new_development"].get(goal, 0) - totals["new_prospective_test"].get(goal, 0)) for goal in GOALS)
        score = (minimum, -imbalance)
        if best is None or score > best[0]: best = (score, groups, totals)
    assert best is not None
    return {name: sorted(group) for name, group in best[1].items()}


def collect(config: dict) -> None:
    out = out_path(config)
    if not (out / "wave27_collection_preregistration.json").exists():
        raise RuntimeError("prepare must run before collection")
    compact_root = ROOT / config["collection"]["compact_root"]; compact_root.mkdir(parents=True, exist_ok=True)
    partial = out / "wave27_collection_partial.json"
    rows = read_json(partial)["records"] if partial.exists() else []
    processed = set(read_json(partial).get("processed_shards", [])) if partial.exists() else set()
    downloads = read_json(partial).get("downloads", []) if partial.exists() else []
    api = HfApi(); filesystem = HfFileSystem(); info = api.dataset_info(config["collection"]["repo_id"], revision=config["collection"]["revision"], files_metadata=True)
    sizes = {row.rfilename: int(row.size or 0) for row in info.siblings}
    for shard_index in config["collection"]["shard_order"]:
        shard = f"training/subset_training_{int(shard_index):03d}.zip"
        if shard in processed: continue
        if preferred(rows, config): break
        remote_stream = int(shard_index) > int(config["collection"]["remote_stream_after_shard"])
        stage = ROOT / config["collection"]["staging_root"] / f"subset_{int(shard_index):03d}"
        remote_handle = None
        if remote_stream:
            remote_path = f"datasets/{config['collection']['repo_id']}@{config['collection']['revision']}/{shard}"
            remote_handle = filesystem.open(remote_path, "rb")
            archive_source = remote_handle
        else:
            zip_path = Path(hf_hub_download(repo_id=config["collection"]["repo_id"], filename=shard, repo_type="dataset", revision=config["collection"]["revision"], local_dir=stage))
            archive_source = zip_path
        chosen = []
        with zipfile.ZipFile(archive_source) as archive:
            annotations = load_pickle_npy(archive, "lang_annotations/auto_lang_ann.npy")
            bounds = np.asarray(load_pickle_npy(archive, "ep_start_end_ids.npy"), np.int64).reshape(-1, 2)
            members = frame_members(archive)
            occupied: dict[int, list[tuple[int, int]]] = defaultdict(list)
            for existing in rows:
                occupied[int(existing["source_session_row"])].append((int(existing["source_start_frame"]), int(existing["source_end_frame"])))
            for event in unique_events(annotations):
                start, end = event["start_frame"] - 64, event["start_frame"] + 63
                session = session_for_frame(bounds, start, end)
                if session is None or session < 31: continue
                if any(not (end < left or start > right) for left, right in occupied[session]): continue
                if not all(frame in members for frame in range(start, end + 1)): continue
                actions, robot, scene = [], [], []
                for frame in range(start, end + 1):
                    with archive.open(members[frame]) as stream:
                        with np.load(BytesIO(stream.read()), allow_pickle=False) as value:
                            actions.append(np.asarray(value["rel_actions"], np.float32))
                            robot.append(np.asarray(value["robot_obs"], np.float32))
                            scene.append(np.asarray(value["scene_obs"], np.float32))
                arrays = {"rel_actions": np.stack(actions), "robot_obs": np.stack(robot), "scene_obs": np.stack(scene), "global_frame_indices": np.arange(start, end + 1, dtype=np.int64)}
                if arrays["rel_actions"].shape != (128, 7) or arrays["robot_obs"].shape != (128, 15) or arrays["scene_obs"].shape != (128, 24) or not all(np.isfinite(value).all() for value in arrays.values()):
                    continue
                record_id = f"wave27_s{session:05d}_f{event['start_frame']:07d}"
                path = compact_root / f"{record_id}.npz"; np.savez_compressed(path, **arrays)
                record = {
                    "record_id": record_id, "goal": event["goal"], "language_variants": event["language_variants"],
                    "source_session_id": f"prospective_training_ep_row_{session:05d}", "source_session_row": session,
                    "source_shard": shard, "source_revision": info.sha,
                    "boundary_frame": event["start_frame"], "source_start_frame": start, "source_end_frame": end,
                    "collector_type": "official_human_play_archive", "collector_version": info.sha,
                    "environment_seed": None, "operator_seed": None, "policy_checkpoint": None,
                    "control_frequency_hz": int(config["collection"]["control_frequency_hz"]),
                    "timestamp_field": "global control step; seconds derived at 30 Hz", "compact_path": path.relative_to(ROOT).as_posix(),
                    "action_available": True, "robot_obs_available": True, "scene_obs_available": True,
                    "gripper_width_available": True, "tcp_pose_available": True, "joint_position_available": True,
                    "measured_joint_velocity_available": False, "measured_tcp_velocity_available": False,
                    "true_contact_available": False, "success_indicator_available": False, "annotation_task_predicate_available": True,
                    "h1_valid": True, "h2_valid": True, "h4_valid": True, "physically_contiguous": True, "reset_crossed": False,
                    "recoverability": "unknown", "physical_time_reversal_claim": False,
                }
                rows.append(record); chosen.append(record); occupied[session].append((start, end))
        if remote_handle is not None:
            remote_handle.close()
        downloads.append({"shard": shard, "source_bytes": sizes.get(shard), "transfer_mode": "remote_member_range_stream" if remote_stream else "resumable_full_zip", "selected": len(chosen), "sessions": sorted({row["source_session_id"] for row in chosen})})
        processed.add(shard)
        if stage.exists():
            shutil.rmtree(stage)
        write_json(partial, {"records": rows, "processed_shards": sorted(processed), "downloads": downloads, "adequate": adequate(rows, config)})
        append_log(out / "wave27_collection_execution_log.md", f"- Processed `{shard}`: selected {len(chosen)} non-overlapping transitions; cumulative={len(rows)}; staging removed.")
        print(json.dumps({"shard": shard, "selected": len(chosen), "cumulative": len(rows), "sessions": len({row['source_session_id'] for row in rows}), "per_goal": Counter(row["goal"] for row in rows)}), flush=True)
    split = split_sessions(rows, int(config["collection"]["split_seed"]))
    assignment = {session: name for name, sessions in split.items() for session in sessions}
    for row in rows: row["split"] = assignment[row["source_session_id"]]
    frame_ranges = [(row["source_start_frame"], row["source_end_frame"], row["source_session_id"]) for row in rows]
    overlap = sum(1 for i, (a, b, session) in enumerate(frame_ranges) for c, d, other in frame_ranges[:i] if session == other and not (b < c or a > d))
    if overlap: raise RuntimeError(f"found {overlap} overlapping transition ranges")
    inventory = pd.DataFrame(rows)
    inventory.to_parquet(out / "wave27_new_transition_inventory.parquet", index=False)
    write_json(out / "wave27_new_transition_inventory.json", rows)
    counts = Counter(row["goal"] for row in rows); session_counts = {name: len(sessions) for name, sessions in split.items()}
    split_manifest = {
        "created_before_model_training": True, "seed": int(config["collection"]["split_seed"]), "sampling_unit": "prospective source session",
        "sessions": split, "session_counts": session_counts, "transition_counts": {name: sum(row["split"] == name for row in rows) for name in split},
        "per_goal": {name: dict(Counter(row["goal"] for row in rows if row["split"] == name)) for name in split},
        "disjoint": all(set(split[a]).isdisjoint(split[b]) for i, a in enumerate(split) for b in list(split)[i + 1:]), "prospective_test_opened": False,
    }
    write_json(out / "wave27_new_data_split_manifest.json", split_manifest)
    completeness = {
        "records": len(rows), "sessions": len({row["source_session_id"] for row in rows}), "per_goal": dict(counts),
        "fields": {field: sum(bool(row[field]) for row in rows) for field in ("action_available", "robot_obs_available", "scene_obs_available", "gripper_width_available", "tcp_pose_available", "joint_position_available", "measured_joint_velocity_available", "measured_tcp_velocity_available", "true_contact_available")},
        "velocity_policy": "finite differences may be used only as explicitly derived causal features; they are not labeled measured velocity", "contact_policy": "unavailable, not replaced by proxy in true-contact claims",
    }
    (out / "wave27_physical_state_completeness.md").write_text("# Wave 27 physical-state completeness\n\n```json\n" + json.dumps(completeness, indent=2) + "\n```\n")
    report = {
        "collection_status": "PREFERRED_MET" if preferred(rows, config) else ("MINIMUM_MET" if adequate(rows, config) else "LIMITED_BELOW_TARGET"), "records": len(rows), "sessions": len({row["source_session_id"] for row in rows}),
        "per_goal": dict(counts), "non_overlapping_ranges": overlap == 0, "all_certified": all(row["h1_valid"] and row["h2_valid"] and row["h4_valid"] and row["physically_contiguous"] and not row["reset_crossed"] for row in rows),
        "downloads": downloads, "split": split_manifest, "true_contact": "UNAVAILABLE", "prospective_test_arrays_opened": False,
    }
    (out / "wave27_collection_report.md").write_text("# Wave 27 prospective collection report\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n")
    write_json(out / "wave27_collection_download_manifest.json", {"repo": config["collection"]["repo_id"], "revision": info.sha, "downloads": downloads})
    print(json.dumps({"stage": "collect", "status": report["collection_status"], "records": len(rows), "sessions": report["sessions"], "per_goal": counts, "prospective_test": "sealed"}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--stage", choices=("prepare", "collect", "all"), default="all"); args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text())
    for stage in (("prepare", "collect") if args.stage == "all" else (args.stage,)):
        print(json.dumps({"stage": stage, "started_at": now()}), flush=True)
        (prepare if stage == "prepare" else collect)(config)


if __name__ == "__main__": main()

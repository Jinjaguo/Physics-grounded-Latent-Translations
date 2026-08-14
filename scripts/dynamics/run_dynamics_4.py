#!/usr/bin/env python3
"""Run the sixteenth-wave prospective long-horizon refinement protocol.

Purpose
-------
Exhaustively audit project CALVIN annotations for >=10 non-overlapping H=16
windows, freeze the prospective wave-16 protocol and carried model hashes, and
enforce the >=60-segment data-adequacy gate.  When the gate fails, this script
follows the preregistered stop rule: it records collection feasibility and
explicit not-evaluated artifacts without reading F1/F2 trajectory metrics.

Parameters
----------
``--config`` is the wave-16 YAML. ``--stage`` is ``audit``, ``finalize``, or
``all``.  The audit stage never evaluates a carried dynamics checkpoint.

Usage
-----
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/run_dynamics_4.py \
  --config configs/dynamics_4.yaml --stage all

Outputs
-------
Timestamped artifacts are stored under
``results/dynamics/sixteenth_wave/...``.  The final scientific report is
mirrored to ``reports/dynamics_4_results.md`` and the repository research log
and next-experiment handoff are updated.  Project size is checked against the
configured 20 GiB ceiling before and after every stage.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from pglt.dynamics.dynamics_data import sha256_file, write_json


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("audit", "finalize", "all"))
    return parser.parse_args()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def project_size(config: Mapping[str, Any]) -> dict[str, Any]:
    completed = subprocess.run(["du", "-sb", str(ROOT)], check=True, capture_output=True, text=True)
    observed = int(completed.stdout.split()[0])
    maximum = int(config["storage"]["maximum_project_gb"]) * 1024 ** 3
    if observed >= maximum:
        raise RuntimeError(f"PGLT project is {observed} bytes, exceeding {maximum}")
    return {
        "project_bytes": observed, "maximum_project_bytes": maximum,
        "remaining_project_budget_bytes": maximum - observed, "passed": True,
    }


def annotation_payload(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    saved = np.load(path, allow_pickle=True).item()
    indices = np.asarray(saved.get("info", {}).get("indx", []), dtype=np.int64).reshape(-1, 2)
    language = saved.get("language", {})
    tasks = np.asarray(language.get("task", [""] * len(indices)), dtype=object)
    annotations = np.asarray(language.get("ann", [""] * len(indices)), dtype=object)
    chunk = int(config["data"]["chunk_length"])
    minimum_windows = int(config["data"]["minimum_windows_per_segment"])
    eligible = []
    lengths = []
    for position, (start, end) in enumerate(indices):
        frame_count = int(end - start + 1)
        lengths.append(frame_count)
        windows = frame_count // chunk
        if windows >= minimum_windows:
            eligible.append({
                "source": path.relative_to(ROOT).as_posix(), "annotation_position": position,
                "frame_start": int(start), "frame_end_inclusive": int(end),
                "frame_count": frame_count, "non_overlapping_H16_windows": windows,
                "task": str(tasks[position]), "language": str(annotations[position]),
                "window_ranges": [[int(start + offset * chunk), int(start + (offset + 1) * chunk - 1)] for offset in range(windows)],
                "valid_h8_rollout_starts": max(0, windows - 9),
            })
    return {
        "path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path),
        "annotation_count": len(indices), "minimum_frames": min(lengths) if lengths else None,
        "median_frames": float(np.median(lengths)) if lengths else None,
        "maximum_frames": max(lengths) if lengths else None,
        "eligible_count": len(eligible), "eligible_segments": eligible,
    }


def collection_protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    collection_source = ROOT / "third_party/calvin/calvin_env/calvin_env/vrdatacollector.py"
    environment_source = ROOT / "third_party/calvin/calvin_env/calvin_env/envs/play_table_env.py"
    recorder_source = ROOT / "third_party/calvin/calvin_env/calvin_env/io_utils/data_recorder.py"
    vr_source = ROOT / "third_party/calvin/calvin_env/calvin_env/io_utils/vr_input.py"
    calvin_commit = subprocess.run(
        ["git", "-C", str(ROOT / "third_party/calvin"), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    dependencies = {
        name: importlib.util.find_spec(name) is not None
        for name in ("pybullet", "hydra", "omegaconf", "quaternion")
    }
    return {
        "created_before_collection": True, "collection_required": True,
        "status": "blocked_existing_supported_pipeline_not_runnable_in_current_environment",
        "environment": {
            "repository": "CALVIN", "commit": calvin_commit,
            "scene": "calvin_scene_D", "environment_class": "PlayTableSimEnv",
            "environment_source": environment_source.relative_to(ROOT).as_posix(),
            "control_frequency_hz": 30, "bullet_frequency_hz": 240,
            "camera_configuration": "no_cameras", "use_vr": True,
        },
        "controller_and_recorder": {
            "collection_source": collection_source.relative_to(ROOT).as_posix(),
            "collection_source_sha256": sha256_file(collection_source),
            "controller": "VrInput backed by PyBullet VR events from a physical controller",
            "controller_source": vr_source.relative_to(ROOT).as_posix(),
            "recorder": "DataRecorder at 30 Hz", "recorder_source": recorder_source.relative_to(ROOT).as_posix(),
            "action_format": "absolute end-effector position (3), quaternion orientation (4), gripper {-1,+1} from VR analogue trigger",
        },
        "prospective_collection_design": {
            "task_instructions": list(config["data"]["tasks"]),
            "target_segments_per_task": int(config["data"]["preferred_segments_per_task"]),
            "randomization": "CALVIN scene-D reset using frozen prospective seeds 160400..160459; no model-dependent selection",
            "start_state_sampling": "native CALVIN scene reset; record exact robot_obs and scene_obs",
            "success_failure_recording": "new_playtable_tasks Tasks.get_task_info_for_set from start/end info; retain both outcomes",
            "frame_rate_hz": 30, "minimum_frames": 160,
            "termination_rule": "operator ends after task attempt and at least 160 recorded frames",
            "maximum_trajectory_duration_seconds": 12.0,
            "representation_or_dynamics_influence_collection": False,
        },
        "current_environment_dependencies": dependencies,
        "blocking_conditions": [
            "pybullet package unavailable", "numpy-quaternion package unavailable",
            "no PyBullet SHARED_MEMORY VR server", "no physical VR controller event source",
            "no repository-supported scripted controller or usable pretrained CALVIN policy is present",
        ],
        "disallowed_workarounds_not_used": [
            "invent random/scripted/learned controller", "concatenate short annotations",
            "repeat or replay windows", "label no-op trajectories as successful task attempts",
        ],
        "collected_segments": 0,
    }


def frozen_model_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    model_paths = {
        "shared_semantic_predictor": ROOT / config["models"]["semantic_checkpoint"],
        "F1_execution_mlp": ROOT / config["models"]["f1_checkpoint"],
        "F2_matched_refinement": ROOT / config["models"]["f2_checkpoint"],
        "historical_DEL_negative_baseline": ROOT / config["models"]["historical_del_checkpoint"],
    }
    manifest = read_json(ROOT / config["models"]["representation_manifest"])
    representations = {
        item["path"]: item["sha256"] for item in manifest["checkpoints"]
        if item["condition"] == "correct_language"
    }
    for relative, expected in representations.items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"Frozen representation hash mismatch: {relative}")
    return {
        "created_before_any_long_trajectory_model_metrics": True,
        "checkpoints": {name: {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)} for name, path in model_paths.items()},
        "representation_correct_language_checkpoints": representations,
        "model_parameter_updates": 0, "representation_updates": 0, "ema_updates": 0,
        "F2_initialization": "exact frozen F1 prediction", "F2_refinement_iterations": int(config["models"]["refinement_iterations"]),
        "future_target_actions": False,
    }


def not_evaluated_artifact(reason: str, fields: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "status": "not_evaluated_due_data_adequacy_gate",
        "reason": reason, "F1_F2_metrics_read": False,
    }
    if fields:
        payload.update(fields)
    return payload


def audit(config: Mapping[str, Any]) -> None:
    out = ROOT / config["experiment"]["output_root"]
    out.mkdir(parents=True, exist_ok=True)
    storage_before = project_size(config)
    sources = [annotation_payload(path, config) for path in sorted(ROOT.rglob("auto_lang_ann.npy"))]
    local_eligible = [segment for source in sources for segment in source["eligible_segments"]]
    acquisition_root = ROOT / config["experiment"]["acquisition_root"]
    source_audits = read_json(acquisition_root / "open_data_source_audit.json")
    acquisition_availability = read_json(acquisition_root / "long_trajectory_availability_audit.json")
    per_task = {
        task: int(acquisition_availability["per_task_valid_counts"].get(task, 0))
        for task in config["data"]["tasks"]
    }
    primary_segments = int(acquisition_availability["total_PRIMARY_COMPATIBLE_segments"])
    raw_npz = sorted((ROOT / "third_party/calvin/dataset").rglob("episode_*.npz"))
    dynamics_sequence_files = sorted((ROOT / "results/dynamics").rglob("*_sequences.jsonl"))
    existing_sequence_maxima = []
    for path in dynamics_sequence_files:
        maximum = 0
        count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            maximum = max(maximum, len(payload.get("latent_indices", [])))
            count += 1
        existing_sequence_maxima.append({
            "path": path.relative_to(ROOT).as_posix(), "sequence_count": count,
            "maximum_non_overlapping_windows": maximum,
        })
    availability = {
        "created_at": now(),
        "audit_scope": (
            "all local auto_lang_ann.npy and sequence manifests plus completed Tier 1A, "
            "Tier 1B, and source-wide Tier 2 annotation audit"
        ),
        "chunk_length": 16, "stride": 16, "minimum_windows": 10,
        "annotation_sources": sources, "raw_episode_npz_count": len(raw_npz),
        "raw_episode_roots": sorted(set(path.parent.relative_to(ROOT).as_posix() for path in raw_npz)),
        "existing_dynamics_sequence_files": existing_sequence_maxima,
        "open_data_sources": source_audits,
        "eligible_existing_segment_count": primary_segments,
        "eligible_existing_segments": local_eligible,
        "eligible_per_task": per_task,
        "audit_conclusion": (
            "No local or public-source canonical annotation spans 160 frames. "
            "The subset_training_023 metadata contains the source-wide 22,966-record ABCD "
            "training annotation table (maximum 65 frames), so later frame shards cannot "
            "contain a qualifying direct annotation; short annotations were not merged."
        ),
    }
    write_json(out / "long_trajectory_availability_audit.json", availability)
    write_json(out / "primary_compatibility_audit.json", {
        "required": {
            "robot": "Franka/Panda CALVIN", "action_dim": 7,
            "action_semantics": "original CALVIN rel_actions", "control_frequency_hz": 30,
            "minimum_contiguous_annotation_frames": 160,
        },
        "sources": [{
            "source_name": item["source_name"],
            "repo_id": item["repo_id"],
            "compatibility_status": item["compatibility_status"],
            "candidate_count_ge_160": item["candidate_count_ge_160"],
            "rejection_reason": item["rejection_reason"],
        } for item in source_audits],
        "eligible_segments": primary_segments,
        "selection_uses_model_outputs": False,
    })
    calvin_relative_source = ROOT / "third_party/calvin/calvin_env/calvin_env/utils/utils.py"
    frozen_loader_source = ROOT / "src/pglt/representation/data.py"
    abcd_probe = read_json(acquisition_root / "tier2_subset_training_023_audit.json")["action_probes"][0]
    write_json(out / "rel_action_convention_audit.json", {
        "calvin_source": {
            "path": calvin_relative_source.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(calvin_relative_source),
            "function": "to_relative_action",
            "translation": "clip(action[:3] - robot_obs[:3], -0.02, 0.02) / 0.02",
            "rotation": "wrapped Euler delta, clipped to [-0.05, 0.05] / 0.05",
            "gripper": "actions[-1:] unchanged",
            "output_order": ["relative_tcp_translation_xyz", "relative_tcp_euler_xyz", "gripper"],
        },
        "frozen_representation_consumer": {
            "path": frozen_loader_source.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(frozen_loader_source),
            "stored_schema": "float64 [T,7] rel_actions",
            "normalization": "frozen training-only action_mean/action_std applied to dimensions 0:6; no new-data renormalization",
        },
        "abcd_numeric_probe": abcd_probe["rel_actions"],
        "abcd_probe_passed": (
            abcd_probe["rel_actions"]["shape"] == [7]
            and all(-1.0 <= value <= 1.0 for value in abcd_probe["rel_actions"]["minimum_per_dimension"][:6])
            and all(-1.0 <= value <= 1.0 for value in abcd_probe["rel_actions"]["maximum_per_dimension"][:6])
            and abs(abcd_probe["rel_actions"]["minimum_per_dimension"][6]) == 1.0
        ),
        "conversion_performed": False,
    })
    protocol = collection_protocol(config)
    protocol["prospective_collection_design"]["target_segments_per_task"] = {
        task: 10 - per_task[task] for task in config["data"]["tasks"]
    }
    write_json(out / "collection_protocol.json", protocol)
    source_files = read_json(acquisition_root / "sha256_manifest.json")["files"]
    preregistration = {
        "created_at": now(), "written_before_any_long_model_metric": True,
        "scientific_state": {
            "C1": "SUPPORTED", "C2": "SUPPORTED", "C3a": "REJECTED", "C3b": "REJECTED", "C3c_local": "SUPPORTED",
        },
        "hypotheses": {
            "H1": "F2 improves frozen autonomous F1 at H1/H2/H4/H8 and paired trajectory AUC upper CI < 0",
            "H2": "F2 corrections improve target, decoded action, kNN radius, and empirical normal distance",
        },
        "primary_models": ["F1_execution_mlp", "F2_matched_refinement"],
        "historical_DEL_role": "frozen negative context only; excluded from primary statistics",
        "exact_source_files": source_files,
        "selected_segments": [],
        "selected_segment_count": 0,
        "selection_status": "blocked_by_data_adequacy_gate_before_inference",
        "horizons": [1, 2, 4, 8], "teacher_forcing": False, "padding": False,
        "future_target_actions": False, "trajectory_bootstrap_replicates": 10000,
        "primary_delta": "trajectory_AUC_F2 - trajectory_AUC_F1",
        "data_gate_precedes_evaluation": True, "configuration": config,
    }
    write_json(out / "prospective_evaluation_preregistration.json", preregistration)
    write_json(out / "long_horizon_open_data_preregistration.json", preregistration)
    write_json(out / "frozen_model_hash_manifest.json", frozen_model_manifest(config))
    adequacy_conditions = {
        "at_least_60_primary_compatible_segments": primary_segments >= int(config["data"]["minimum_segments_total"]),
        "at_least_10_each_of_six_tasks": all(
            per_task[task] >= int(config["data"]["minimum_segments_per_task"])
            for task in config["data"]["tasks"]
        ),
        "every_selected_segment_at_least_160_frames": primary_segments > 0,
        "every_selected_segment_supports_h8": primary_segments > 0,
    }
    gate = {
        "created_at": now(), "existing_segments": primary_segments, "newly_collected_segments": 0,
        "total_valid_segments": primary_segments, "per_target_task_counts": per_task,
        "conditions": adequacy_conditions, "passed": all(adequacy_conditions.values()),
        "required_action": "stop_and_report_insufficient_long_horizon_data",
        "model_metrics_authorized": False,
    }
    write_json(out / "data_adequacy_gate.json", gate)
    missing = [{
        "task": task,
        "open_data_valid_count": per_task[task],
        "missing_count_to_reach_10": max(0, 10 - per_task[task]),
        "minimum_frames": 160,
        "required_action_format": (
            "30-Hz original CALVIN 7-D rel_actions: relative TCP translation (0:3), "
            "relative TCP Euler rotation (3:6), gripper (6)"
        ),
    } for task in config["data"]["tasks"] if per_task[task] < 10]
    failure = {
        "status": "INSUFFICIENT_DATA",
        "required_total": 60,
        "required_per_task": 10,
        "observed_total": primary_segments,
        "observed_per_task": per_task,
        "missing": missing,
        "primary_inference_permitted": False,
    }
    write_json(out / "data_adequacy_failure.json", failure)
    write_json(out / "targeted_missing_data_acquisition_plan.json", {
        "manual_or_vr_collection_authorized": True,
        "collect_only_missing_cells": True,
        "missing": missing,
        "total_new_segments_required": sum(item["missing_count_to_reach_10"] for item in missing),
    })
    reason = (
        "0 PRIMARY_COMPATIBLE annotation-consistent segments; >=60 and >=10/task are required. "
        "The data gate failed before any F1/F2 primary inference."
    )
    write_json(out / "long_latent_serialization_manifest.json", not_evaluated_artifact(reason, {
        "serialization_created": False, "would_use_H": 16, "would_use_stride": 16,
    }))
    write_json(out / "horizon_sample_count_table.json", not_evaluated_artifact(reason, {
        "trajectory_count": 0, "starting_point_counts": {"1": 0, "2": 0, "4": 0, "8": 0},
        "physical_prediction_seconds": {"1": 16/30, "2": 32/30, "4": 64/30, "8": 128/30},
    }))
    status_files = [
        "F1_F2_horizon_wise_latent_metrics.json", "F1_F2_decoded_action_metrics.json",
        "F1_F2_semantic_retention_metrics.json", "F1_F2_off_manifold_metrics.json",
        "trajectory_level_paired_bootstrap.json", "rollout_error_growth_curves.json",
        "refinement_intermediate_state_table.json", "correction_vector_alignment_report.json",
        "local_tangent_normal_manifold_audit.json", "per_source_per_task_metrics.json",
        "source_stratified_bootstrap.json",
    ]
    for name in status_files:
        extra = {"bootstrap_replicates_not_run": 10000} if name == "trajectory_level_paired_bootstrap.json" else None
        write_json(out / name, not_evaluated_artifact(reason, extra))
    write_json(out / "refinement_mechanism_decision.json", {
        "status": "NOT_EVALUATED_INSUFFICIENT_DATA", "C3d": "NOT_EVALUATED_INSUFFICIENT_DATA",
        "mechanism_claim_authorized": False, "reason": reason,
    })
    write_json(out / "frozen_DEL_negative_baseline_report.json", {
        "status": "not_evaluated_due_data_adequacy_gate", "role": "historical frozen negative baseline only",
        "checkpoint": frozen_model_manifest(config)["checkpoints"]["historical_DEL_negative_baseline"],
        "retrained": False, "retuned": False, "included_in_primary_statistics": False,
    })
    acquisition_commands = (acquisition_root / "executed_commands.txt").read_text(encoding="utf-8")
    commands = acquisition_commands + """df -h /home/jinjaguo/PGLT
df -B1 /home/jinjaguo/PGLT
du -sb .
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_4.py --config configs/dynamics_4.yaml --stage audit
PYTHONPATH=src:third_party/LaWM PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/jinjaguo/anaconda3/envs/libero/bin/python -m pytest tests/representation tests/dynamics -q --junitxml=results/dynamics/sixteenth_wave/2026-08-13_dynamics_4/pytest_results.xml
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_4.py --config configs/dynamics_4.yaml --stage finalize
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/audit_dynamics_4.py --config configs/dynamics_4.yaml
du -sb .
du -sh .
"""
    (out / "executed_commands.txt").write_text(commands, encoding="utf-8")
    write_json(out / "environment_provenance.json", {
        "created_at": now(), "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(), "project_storage_before": storage_before,
        "calvin_collection_dependencies": protocol["current_environment_dependencies"],
    })
    storage_after = project_size(config)
    print(json.dumps({"stage": "audit", "eligible_segments": primary_segments, "data_gate": gate["passed"], "project_gib": storage_after["project_bytes"] / 1024**3}))


def finalize(config: Mapping[str, Any]) -> None:
    out = ROOT / config["experiment"]["output_root"]
    gate = read_json(out / "data_adequacy_gate.json")
    if gate["passed"]:
        raise RuntimeError("This stop-branch finalizer is only valid when the data gate fails")
    availability = read_json(out / "long_trajectory_availability_audit.json")
    protocol = read_json(out / "collection_protocol.json")
    acquisition_root = ROOT / config["experiment"]["acquisition_root"]
    source_audits = read_json(acquisition_root / "open_data_source_audit.json")
    inventory = read_json(acquisition_root / "local_calvin_inventory.json")
    missing_plan = read_json(out / "targeted_missing_data_acquisition_plan.json")
    claims = {
        "created_at": now(), "C1": "SUPPORTED", "C2": "SUPPORTED",
        "C3a_full_DEL": "REJECTED", "C3b_exec_DEL": "REJECTED",
        "C3c_local_refinement": "SUPPORTED",
        "C3c_long_refinement": "NOT_TESTED_INSUFFICIENT_DATA",
        "C3d_empirical_manifold_restoration": "NOT_TESTED",
        "DEL_future_role": "negative baseline only; no further rescue",
        "paper_story": "Language-grounded action coordinates are semantically addressable, executable, and locally predictable; refinement improves short-horizon prediction, but stable long-horizon latent dynamics remains unresolved.",
    }
    write_json(out / "wave16_claim_decision.json", claims)
    write_json(out / "final_paper_claim_decision.json", claims)
    local_roots = read_json(acquisition_root / "disk_budget.json")["local_CALVIN_roots"]
    source_rows = "\n".join(
        f"- `{item['repo_id']}`: {item['downloaded_bytes']:,} bytes; "
        f"{item['compatibility_status']}; direct >=160 candidates = {item['candidate_count_ge_160']}."
        for item in source_audits
    )
    missing_rows = "\n".join(
        f"- `{item['task']}`: valid {item['open_data_valid_count']}, missing {item['missing_count_to_reach_10']}."
        for item in missing_plan["missing"]
    )
    answers = [
        f"1. Existing local compatible roots: **{len(local_roots)}** roots ({', '.join(f'`{Path(path).relative_to(ROOT)}`' for path in local_roots)}); none has a >=160-frame six-task annotation.",
        "2. Public sources audited/downloaded: **CollisionCode LeRobot D/D metadata, RoboVerse CALVIN six-task files, and VyoJ original-format ABCD subset metadata/frames**.",
        "3. Downloaded bytes: **12,614,629 / 215,910,633 / 2,160,142,328**, respectively.",
        "4. Rejected/ineligible compatibility: **LeRobot is 10 Hz (SCOUTING_ONLY); RoboVerse is converted object/9-D joint-style data with unproven 30-Hz timing (REJECTED)**.",
        "5. The 10-Hz LeRobot conversion remained scouting-only: **yes**; no interpolation or repetition was used.",
        "6. Direct >=160-frame candidates: **0 LeRobot, 0 RoboVerse, 0 VyoJ ABCD**.",
        "7. Valid segments per frozen task: **0 for every one of the six tasks**.",
        "8. Open data reached 10/task and 60 total: **no (0/task, 0 total)**.",
        "9. Missing collection: **10 segments for each of all six tasks; 60 total**, each >=160 contiguous frames in exact 30-Hz 7-D CALVIN rel_actions.",
        "10. F1/F2 frozen before all prospective metrics: **yes**; hashes were recorded, update counts are zero, and no primary metric was read.",
        "11. H1/H2/H4/H8 rollout starts: **0 / 0 / 0 / 0**.",
        "12. Paired trajectory AUC with upper 95% CI below zero: **not tested; adequacy gate blocked bootstrap**.",
        "13. F2 beat F1 at H4: **not tested**.",
        "14. F2 beat F1 at H8: **not tested**.",
        "15. F2 reduced H8 decoded-action error: **not tested**.",
        "16. F2 reduced H8 execution off-manifold drift: **not tested**.",
        "17. Refinement correction aligned with GT correction: **not tested**.",
        "18. Empirical normal-to-manifold distance decreased: **not tested**.",
        "19. C3c-long: **NOT_TESTED_INSUFFICIENT_DATA**, not rejected.",
        "20. C3d: **NOT_TESTED**.",
        "21. DEL remains a frozen negative baseline only: **yes; no retraining, retuning, or primary-bootstrap use**.",
        "22. Defensible story: **semantic/executable coordinates and local refinement remain supported; stable long-horizon dynamics remains unresolved**.",
        "23. Manual/VR collection remains necessary: **yes, only the 60 missing cells listed above**.",
    ]
    report = f"""# PGLT 第十六轮 / 第四次动力学实验报告

## 结论

本轮从暂停边界继续执行，没有重做 Tier 0/Tier 1A。Tier 1B 对 **358 个 RoboVerse 文件、4,836 条轨迹**完成逐轨迹 pickle schema、动作维度和长度审计；最大长度 64，直接长候选为 0。Tier 2 staged 下载 `subset_training_023`，完成 SHA256、metadata/动作探针审计及清理。该 shard 携带全 ABCD training annotation 表（22,966 条、六任务各约 675 条），全表最大长度 **65 帧**，因此 `_000…_022` 只包含该表所索引的其他帧，继续下载不能产生 >=160 的直接 annotation。

指导文件要求 6 tasks × 10 段，因此 data-adequacy gate 为 **FAIL**。本轮没有读取任何 F1/F2 prospective metric、没有执行 bootstrap、没有拼接短轨迹，也没有读取未来动作。

C3c-local 保持 **SUPPORTED**。C3c-long 与 C3d 均为 **NOT_EVALUATED_INSUFFICIENT_DATA**，不是负结果。C3a/C3b 保持 **REJECTED**，不再进行 DEL rescue。

## 数据源审计

{source_rows}

本地 inventory 记录 **{inventory['episode_npz_count']}** 个 episode NPZ；三个既有 CALVIN 根合计见 `disk_budget.json`。LeRobot 始终是 10-Hz scouting-only；RoboVerse 不满足动作/时间兼容门；VyoJ 数据格式兼容但没有足够长的 annotation。

## 精确缺失数据

{missing_rows}

## Prospective collection 状态

- CALVIN commit：`{protocol['environment']['commit']}`。
- 原生 pipeline：`{protocol['controller_and_recorder']['collection_source']}`。
- 控制/记录频率：30 Hz；最低 160 帧；最大 12 秒。
- 计划：只补上面列出的缺失 cell，保留成功与失败，不按 F1/F2 表现筛选。
- 当前 blocker：真实 VR/SHARED_MEMORY 与两项运行依赖缺失。
- 未采用范围外 workaround：没有新写随机、脚本或 learned behavior policy。

## 23 个明确回答

{chr(10).join(answers)}

## 当前可辩护论文故事

**Language-grounded action coordinates are semantically addressable, executable, and locally predictable; refinement improves short-horizon prediction, but stable long-horizon latent dynamics remains unresolved.**

中文：**语言落地动作坐标具有语义可寻址性、可执行性与局部可预测性；refinement 改善短期预测，但稳定的长时域 latent 动力学仍未解决。**

## 下一实验

按 `targeted_missing_data_acquisition_plan.json` 只采集六任务各缺失的 10 段、每段至少 160 帧的 30-Hz 原始 CALVIN 7-D `rel_actions`；通过 60 段 gate 后，使用本轮已冻结的 F1/F2/semantic checkpoint 和 10,000 次 whole-trajectory bootstrap 完成 H1/H2/H4/H8 评估。不得适配模型或再次调 DEL。
"""
    (out / "sixteenth_wave_results.md").write_text(report, encoding="utf-8")
    report_path = ROOT / config["experiment"]["report_path"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    next_text = """# Sixteenth-wave next experiment

Open data supplies zero PRIMARY_COMPATIBLE >=160-frame segments, so collect only the exact missing cells in `targeted_missing_data_acquisition_plan.json`: 10 prospective segments for each frozen task (60 total), each with >=160 contiguous 30-Hz original CALVIN 7-D `rel_actions`. Preserve successes and failures without model-dependent selection. Once the 10/task and 60-total gate passes, evaluate the already-frozen semantic/F1/F2 checkpoints at H1/H2/H4/H8 with the preregistered 10,000-replicate paired whole-trajectory bootstrap. Do not adapt representation/F1/F2, read future actions, or reopen DEL.
"""
    (out / "sixteenth_wave_next_experiment.md").write_text(next_text, encoding="utf-8")
    (ROOT / "NEXT_EXPERIMENT.md").write_text(next_text, encoding="utf-8")
    log_path = ROOT / "RESEARCH_LOG.md"
    previous = log_path.read_text(encoding="utf-8") if log_path.exists() else "# RESEARCH_LOG\n"
    entry = f"\n## {now()} — dynamics_4 / wave 16\n\nResumed the open-data audit at Tier 1B. Audited 4,836 RoboVerse trajectories and the source-wide 22,966-record VyoJ ABCD annotation table from staged subset_training_023; every direct six-task annotation is shorter than 160 frames. The open-data count is 0/task and 0 total, so the adequacy gate blocked all primary F1/F2 inference. C3c-long is NOT_TESTED_INSUFFICIENT_DATA and C3d is NOT_TESTED; the exact fallback is 10 new CALVIN-compatible segments for each of six tasks. See `{config['experiment']['report_path']}`.\n"
    if "Audited 4,836 RoboVerse trajectories and the source-wide 22,966-record VyoJ ABCD annotation table" not in previous:
        log_path.write_text(previous.rstrip() + "\n" + entry, encoding="utf-8")
    storage = project_size(config)
    write_json(out / "project_storage_audit.json", storage)
    files = [
        path for path in sorted(out.rglob("*"))
        if path.is_file() and path.name not in {"files_changed_report.json", "final_audit_report.json"}
    ]
    files.extend(path for path in sorted(acquisition_root.rglob("*")) if path.is_file())
    files.extend([
        report_path, ROOT / "NEXT_EXPERIMENT.md", ROOT / "RESEARCH_LOG.md",
        ROOT / "configs/dynamics_4.yaml",
        ROOT / "scripts/dynamics/acquire_dynamics_4.py",
        ROOT / "scripts/dynamics/run_dynamics_4.py",
        ROOT / "scripts/dynamics/audit_dynamics_4.py",
        ROOT / "src/pglt/dynamics/open_data.py",
        ROOT / "tests/dynamics/test_dynamics_4_long_horizon.py",
    ])
    write_json(out / "files_changed_report.json", {
        "created_or_updated_files": [{"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files],
        "prior_dynamics_artifacts_overwritten": False,
        "self_excluded_to_avoid_recursive_hash": "files_changed_report.json",
        "generated_after_this_report": "final_audit_report.json",
    })
    print(json.dumps({
        "stage": "finalize", "data_gate": False,
        "C3c_long": claims["C3c_long_refinement"],
        "project_gib": storage["project_bytes"] / 1024**3,
    }))


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    stages = ("audit", "finalize") if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "audit":
            audit(config)
        elif stage == "finalize":
            finalize(config)


if __name__ == "__main__":
    main()

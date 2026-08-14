#!/usr/bin/env python3
"""Freeze and audit the prospective Wave-20 LIBERO motor-margin experiment.

Purpose
-------
Verify the immutable Wave-19 split without opening any test episode payload,
freeze six new representation seeds, the sole 2x reconstruction intervention,
the fresh π0.5 collection schedule, exact snapshot tolerances, and the
checkpoint-selection rule before collection or confirmation inference.

Parameters
----------
``--config`` selects the Wave-20 YAML. ``--stage preregister`` performs the
manifest-only audit and writes every prospective Wave-20 preregistration.

Usage
-----
PYTHONPATH=src:/home/jinjaguo/LIBERO \
  /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/run_dynamics_8.py \
  --config configs/dynamics_8.yaml --stage preregister

Outputs
-------
Preregistrations, the exact command log, and the environment freeze are saved
under ``results/dynamics/twentieth_wave/2026-08-14_dynamics_8``. A provenance
copy of the environment freeze is saved below ``data/wave20_libero_confirmation``.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from run_dynamics_7 import write_environment_freeze


ROOT = Path(__file__).resolve().parents[2]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("preregister",), required=True)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git_commit(path: Path) -> str:
    import subprocess

    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config: Mapping[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out = ROOT / config["experiment"]["output_root"]
    data = ROOT / config["experiment"]["data_root"]
    out.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    sources = config["sources"]
    wave19 = ROOT / sources["wave19_output_root"]
    split_path = wave19 / "wave19_dataset_split_manifest.json"
    dataset_path = wave19 / "wave19_dataset_manifest.json"
    training_path = wave19 / "wave19_representation_training_manifest.json"
    source_policy_path = wave19 / "wave19_pi05_source_policy_manifest.json"
    split = read_json(split_path)
    if split.get("counts") != {"train": 140, "development": 50, "test": 50}:
        raise RuntimeError(f"Wave-19 split counts changed: {split.get('counts')}")
    assignments = split["assignments"]
    sets = {name: set(values) for name, values in assignments.items()}
    if any(sets[a] & sets[b] for a, b in (("train", "development"), ("train", "test"), ("development", "test"))):
        raise RuntimeError("Wave-19 source-episode splits overlap")
    if split.get("final_test_unread_for_model_selection") is not True:
        raise RuntimeError("Wave-19 final-test unread invariant is absent")
    wave19_seeds = set(read_json(wave19 / "wave19_representation_preregistration.json")["seeds"])
    seeds = [int(value) for value in config["representation"]["seeds"]]
    if len(seeds) != 6 or len(set(seeds)) != 6 or wave19_seeds.intersection(seeds):
        raise RuntimeError("Wave-20 seeds are not six unique seeds disjoint from Wave 19")
    if float(config["representation"]["reconstruction_weight"]) != 2.0:
        raise RuntimeError("Wave-20 reconstruction weight must equal exactly 2.0")
    source_policy = read_json(source_policy_path)
    current_libero = git_commit(Path(sources["libero_root"]))
    current_openpi = git_commit(Path(sources["openpi_root"]))
    if current_libero != source_policy["libero"]["commit"] or current_openpi != source_policy["openpi"]["commit"]:
        raise RuntimeError("LIBERO or OpenPI commit changed after Wave-19 freeze")
    freeze = {
        "frozen_at": now(),
        "wave19_dataset_manifest": str(dataset_path.relative_to(ROOT)),
        "wave19_dataset_manifest_sha256": sha256_file(dataset_path),
        "wave19_split_manifest": str(split_path.relative_to(ROOT)),
        "wave19_split_manifest_sha256": sha256_file(split_path),
        "wave19_representation_training_manifest_sha256": sha256_file(training_path),
        "assignment_sha256": {name: sha256_json(values) for name, values in assignments.items()},
        "counts": split["counts"],
        "episode_disjoint": True,
        "source_episode_membership_unchanged": True,
        "wave20_test_payload_files_read": False,
        "wave19_final_test_remains_unopened": True,
    }
    write_json(out / "wave20_existing_split_freeze.json", freeze)
    write_json(
        out / "wave20_seed_preregistration.json",
        {
            "frozen_at": now(),
            "written_before_fresh_confirmation_evaluation": True,
            "seeds": seeds,
            "wave19_seeds": sorted(wave19_seeds),
            "disjoint_from_wave19": True,
            "required_complete": 6,
            "no_seed_drop_or_replacement": True,
            "additional_seeds_forbidden": True,
        },
    )
    write_json(
        out / "wave20_checkpoint_selection_rule.json",
        {
            "frozen_at": now(),
            "written_before_fresh_confirmation_evaluation": True,
            "rule": (
                "among registered seeds with finite outputs and positive per-seed bidirectional semantic delta, "
                "select the seed with median EMA continuous-MSE ratio; for an even count select the lower of the "
                "two central registered ratios; tie by lower registered seed"
            ),
            "checkpoint": "EMA epoch 40",
            "confirmation_based_epoch_selection": False,
        },
    )
    snapshot = read_json(wave19 / "wave19_snapshot_certification_preregistration.json")
    write_json(
        out / "wave20_snapshot_certification_preregistration.json",
        {
            **snapshot,
            "frozen_at": now(),
            "inherited_exactly_from_wave19": True,
            "restore_order": "mj_setState -> mj_forward -> mj_setState",
            "action_boundary": "env.step(action.copy())",
            "frozen_before_fresh_collection": True,
        },
    )
    tasks = read_json(wave19 / "wave19_libero_suite_tasks.json")
    collection = config["collection"]
    write_json(
        out / "wave20_collection_preregistration.json",
        {
            "frozen_at": now(),
            "written_before_fresh_collection": True,
            "suite": "libero_10",
            "task_list": tasks,
            "required_certified_successes_per_task": 5,
            "attempt_cap_per_task": int(collection["attempts_per_task_maximum"]),
            "environment_seed_schedule": "200820 + task_id*10000 + attempt",
            "task_seed_schedule": "1200820 + task_id*10000 + attempt",
            "policy_seed_schedule": "single global π0.5 sampling stream seeded 200820 before ordered batched collection",
            "checkpoint_path": source_policy["local_checkpoint_path"],
            "checkpoint_sha256": source_policy["model_file_sha256"],
            "openpi_commit": current_openpi,
            "libero_commit": current_libero,
            "snapshot_format": "complete mjSTATE_INTEGRATION plus Python/controller/observable payload",
            "restore_order": "mj_setState -> mj_forward -> mj_setState",
            "branch_fractions": collection["branch_fractions"],
            "minimum_future_steps": int(collection["minimum_future_steps"]),
            "certification_tolerances": {"median": 0.0, "p95": 0.0, "controller_max": 0.0},
            "action_mutation_safety": "copy on save and env.step(action.copy())",
            "jit_mode": collection["jit_mode"],
            "numba_disable_jit_environment_variable": "unset",
            "wave19_episode_ids_forbidden": sorted(set().union(*sets.values())),
        },
    )
    write_json(
        out / "wave20_representation_preregistration.json",
        {
            "frozen_at": now(),
            "written_before_fresh_confirmation_evaluation": True,
            "architecture": {
                "input": "action_only",
                "action_chunk_horizon": 16,
                "latent_dim": 32,
                "semantic_dim": 16,
                "execution_dim": 16,
                "hidden_dim": 128,
                "depth": 3,
            },
            "primary_pair": {
                "R0": "L_reconstruction",
                "R1": "2.0 * L_reconstruction + 1.0 * L_semantic",
            },
            "shuffled_language_control": "same R1 objective; used only in the frozen semantic-delta control",
            "seeds": seeds,
            "epochs": 40,
            "checkpoint": "EMA epoch 40",
            "ema_decay": 0.999,
            "training_split": "unchanged Wave-19 140-episode train split",
            "evaluation_split": "fresh Wave-20 50-episode confirmation-development set",
            "old_development_use": "descriptive diagnostics only",
            "old_final_test_read": False,
            "bootstrap": {"replicates": 10000, "seed": 200820, "cluster": "source_episode", "task_stratified": True},
            "gate": {
                "bidirectional_mean_delta_strictly_positive": True,
                "bidirectional_lower95_strictly_positive": True,
                "motor_ratio_maximum": 1.15,
                "gripper_drop_maximum": 0.02,
                "six_of_six_complete": True,
                "all_outputs_finite": True,
            },
            "forbidden": ["hyperparameter_sweep", "extra_seed", "early_stopping", "confirmation_training", "CALVIN_weights"],
        },
    )
    write_json(
        out / "wave20_dynamics_preregistration.json",
        {
            "frozen_at": now(),
            "authorization": "only if wave20_representation_gate.gate_pass is true",
            "design": "exact Wave-19 F1/free and F2/four-step matched iterative refinement",
            "offline_gate": "O1-O8 exactly as prompts/dynamics_8.md",
            "final_test_open": "only after representation and O1-O8 pass",
            "DEL_forbidden": True,
        },
    )
    write_environment_freeze(config)
    command = (
        "PYTHONPATH=src:/home/jinjaguo/LIBERO /home/jinjaguo/anaconda3/envs/libero/bin/python "
        "scripts/dynamics/run_dynamics_8.py --config configs/dynamics_8.yaml --stage preregister"
    )
    (out / "exact_commands.sh").write_text(f"# {now()} phase=preregister\n{command}\n", encoding="utf-8")
    print(json.dumps({"status": "PREREGISTERED", "output_root": str(out), "seeds": seeds}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Materialize every-step MuJoCo contacts from immutable Wave-19 raw states.

Purpose
-------
After source collection freezes, reconstruct contacts deterministically from
each saved ``mjSTATE_INTEGRATION`` boundary state and the episode's official
LIBERO BDDL model. Save compact ragged arrays for contact geometry pairs,
distance, position, frame, and 6-D force for every successful and failed raw
episode. Raw source directories are read-only and are never modified.

Parameters
----------
``--config`` selects the frozen Wave-19 YAML. The script has no task, outcome,
or tolerance override: it processes every finalized raw episode in the data
root and requires the dataset split to have been frozen first.

Usage
-----
PYTHONPATH=src:/home/jinjaguo/LIBERO MUJOCO_GL=egl \
  /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/derive_wave19_contacts.py --config configs/dynamics_7.yaml

Outputs
-------
Per-episode compressed ragged arrays are written below
``data/wave19_libero_branchable/derived/contacts``. A manifest and Markdown
report are written below the Wave-19 result root, and the exact command and
disk records are appended to the existing experiment logs.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any

import mujoco
import numpy as np
import yaml

from libero.libero.envs import OffScreenRenderEnv

from pglt.libero.snapshot import restore_integration_state


ROOT = Path(__file__).resolve().parents[2]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def disk_record(data: Path, config: dict[str, Any], phase: str) -> None:
    usage = shutil.disk_usage(ROOT)
    row = {
        "recorded_at": now(),
        "phase": phase,
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "minimum_free_bytes": int(config["runtime"]["minimum_free_disk_bytes"]),
        "minimum_pass": usage.free >= int(config["runtime"]["minimum_free_disk_bytes"]),
    }
    if not row["minimum_pass"]:
        raise RuntimeError("Free disk is below the frozen Wave-19 minimum")
    with (data / "audits/disk_usage_log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def contact_archive(env: Any, states: np.ndarray) -> dict[str, np.ndarray]:
    offsets = [0]
    pairs = []
    distances = []
    positions = []
    frames = []
    forces = []
    for state in states:
        restore_integration_state(env.sim, state)
        env.sim.forward()
        data = env.sim.data
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(env.sim.model._model, data._data, index, force)
            pairs.append((int(contact.geom1), int(contact.geom2)))
            distances.append(float(contact.dist))
            positions.append(np.asarray(contact.pos).copy())
            frames.append(np.asarray(contact.frame).copy())
            forces.append(force)
        offsets.append(len(pairs))
    return {
        "step_offsets": np.asarray(offsets, dtype=np.int64),
        "geom_pairs": np.asarray(pairs, dtype=np.int32).reshape(-1, 2),
        "distance": np.asarray(distances, dtype=np.float64),
        "position": np.asarray(positions, dtype=np.float64).reshape(-1, 3),
        "frame": np.asarray(frames, dtype=np.float64).reshape(-1, 9),
        "force": np.asarray(forces, dtype=np.float64).reshape(-1, 6),
    }


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out = ROOT / config["experiment"]["output_root"]
    data = ROOT / config["experiment"]["data_root"]
    if not (out / "wave19_dataset_split_manifest.json").is_file():
        raise RuntimeError("Dataset must be frozen before contact materialization")
    disk_record(data, config, "contact-materialization-start")
    metadata_paths = sorted((data / "raw_collection").glob("task_*/successes/*/episode_metadata.json"))
    metadata_paths += sorted((data / "raw_collection").glob("task_*/failures/*/episode_metadata.json"))
    rows = []
    env = None
    active_bddl = None
    try:
        for index, metadata_path in enumerate(metadata_paths, start=1):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            episode = metadata_path.parent
            bddl = Path(metadata["bddl_path"])
            if bddl != active_bddl:
                if env is not None:
                    env.close()
                env = OffScreenRenderEnv(
                    bddl_file_name=bddl,
                    camera_heights=32,
                    camera_widths=32,
                    use_camera_obs=False,
                )
                env.seed(int(metadata["environment_seed"]))
                env.reset()
                active_bddl = bddl
            states = np.load(episode / "integration_states.npy")
            archive = contact_archive(env, states)
            relative = Path(f"task_{int(metadata['task_id']):02d}") / f"{metadata['episode_id']}.npz"
            destination = data / "derived/contacts" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, **archive)
            rows.append(
                {
                    "episode_id": metadata["episode_id"],
                    "task_id": int(metadata["task_id"]),
                    "outcome": "success" if metadata["terminal_official_success"] else "failure",
                    "boundary_states": len(states),
                    "contacts": len(archive["distance"]),
                    "archive": str(destination.relative_to(ROOT)),
                    "raw_unchanged": True,
                }
            )
            print(f"contacts {index}/{len(metadata_paths)} {metadata['episode_id']} n={len(archive['distance'])}", flush=True)
    finally:
        if env is not None:
            env.close()
    manifest = {
        "created_at": now(),
        "source": "immutable per-boundary mjSTATE_INTEGRATION plus official episode BDDL",
        "derivation": "mj_setState then mj_forward; active MuJoCo contacts serialized as ragged arrays",
        "raw_collection_modified": False,
        "episodes": len(rows),
        "boundary_states": sum(row["boundary_states"] for row in rows),
        "contacts": sum(row["contacts"] for row in rows),
        "rows": rows,
    }
    write_json(out / "wave19_contact_derivation_manifest.json", manifest)
    (out / "wave19_contact_derivation_report.md").write_text(
        "# Wave-19 every-step contact materialization\n\n"
        f"Reconstructed `{manifest['contacts']}` active contacts across `{manifest['boundary_states']}` control "
        f"boundaries in `{manifest['episodes']}` finalized raw episodes. Geometry pairs, distance, position, "
        "frame, and 6-D force were saved as compact ragged arrays. Immutable raw directories were not changed.\n",
        encoding="utf-8",
    )
    with (out / "exact_commands.sh").open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n# {now()} phase=contact-materialization\n"
            "PYTHONPATH=src:/home/jinjaguo/LIBERO MUJOCO_GL=egl "
            "/home/jinjaguo/anaconda3/envs/libero/bin/python "
            "scripts/dynamics/derive_wave19_contacts.py --config configs/dynamics_7.yaml\n"
        )
    disk_record(data, config, "contact-materialization-complete")


if __name__ == "__main__":
    main()

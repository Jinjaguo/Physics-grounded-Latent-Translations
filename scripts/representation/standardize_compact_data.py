#!/usr/bin/env python3
"""Standardize released compact episodes to the final two-field schema.

Purpose
-------
Mechanically remove legacy unused modalities/arrays from copied compact NPZ
files while preserving exact ``rel_actions`` and ``global_frame_indices``.
The script hashes retained arrays before and after each atomic replacement.

Parameters
----------
--config: released representation YAML; --archive-sidecars: optional directory
for pre-standardization JSON sidecars.

Usage
-----
python scripts/representation/standardize_compact_data.py \
  --config configs/representation.yaml \
  --archive-sidecars archive/representation_development/manifests/legacy_sidecars

Outputs
-------
Rewrites only NPZ files containing extra keys, updates their sidecars, and
writes ``standardization_manifest.json`` beside the compact training data.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import yaml


def array_digest(actions: np.ndarray, indices: np.ndarray) -> str:
    """Hash retained array identities and bytes."""

    digest = hashlib.sha256()
    for array in (actions, indices):
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(list(array.shape)).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--archive-sidecars", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    root = Path(config["source"]["compact_root"]) / "training"
    metadata = Path(config["source"]["metadata_root"]) / "training"
    bounds = np.load(metadata / "ep_start_end_ids.npy").reshape(-1, 2)
    records = []
    if args.archive_sidecars:
        args.archive_sidecars.mkdir(parents=True, exist_ok=True)
    for row, (first, last) in enumerate(bounds):
        path = root / f"episode_row_{row:03d}.npz"
        sidecar = path.with_suffix(".json")
        with np.load(path, allow_pickle=False) as archive:
            original_keys = sorted(archive.files)
            actions = archive["rel_actions"].copy()
            indices = archive["global_frame_indices"].copy()
        before = array_digest(actions, indices)
        changed = set(original_keys) != {"rel_actions", "global_frame_indices"}
        if changed:
            if args.archive_sidecars and sidecar.exists():
                shutil.copy2(sidecar, args.archive_sidecars / sidecar.name)
            temporary = path.with_suffix(".standardized.npz")
            np.savez_compressed(temporary, rel_actions=actions, global_frame_indices=indices)
            temporary.replace(path)
            payload = {
                "created_at": datetime.now().astimezone().isoformat(),
                "split": "training",
                "episode_row": row,
                "first_index": int(first),
                "last_index": int(last),
                "frame_count": int(last - first + 1),
                "retained_fields": ["rel_actions", "global_frame_indices"],
                "removed_legacy_fields": [key for key in original_keys if key not in {"rel_actions", "global_frame_indices"}],
                "retained_array_sha256": before,
            }
            sidecar.write_text(json.dumps(payload, indent=2) + "\n")
        with np.load(path, allow_pickle=False) as standardized:
            after = array_digest(standardized["rel_actions"], standardized["global_frame_indices"])
            keys_after = sorted(standardized.files)
        if before != after:
            raise RuntimeError(f"Retained array changed during standardization: {path}")
        records.append({"official_row": row, "changed": changed, "keys_before": original_keys, "keys_after": keys_after, "retained_array_sha256": after})
    manifest = {"created_at": datetime.now().astimezone().isoformat(), "episodes": records, "changed_count": sum(item["changed"] for item in records), "retained_arrays_unchanged": True}
    (root / "standardization_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"episodes": len(records), "changed": manifest["changed_count"], "retained_arrays_unchanged": True}))


if __name__ == "__main__":
    main()

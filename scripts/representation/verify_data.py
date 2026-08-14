#!/usr/bin/env python3
"""Verify the released action-only CALVIN representation dataset.

Purpose
-------
Check all 31 official task_D_D training episode rows against authoritative
metadata while enforcing the compact two-field schema: exact ``rel_actions``
and ``global_frame_indices`` only.

Parameters
----------
--config: released representation YAML; --output: optional JSON report.

Usage
-----
python scripts/representation/verify_data.py \
  --config configs/representation.yaml \
  --output results/representation/data_integrity.json

Outputs
-------
Prints a compact summary and optionally writes the complete integrity report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


def sha256(path: Path) -> str:
    """Hash one compact episode."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    root = Path(config["source"]["compact_root"]) / "training"
    metadata = Path(config["source"]["metadata_root"]) / "training"
    bounds = np.load(metadata / "ep_start_end_ids.npy").reshape(-1, 2)
    episodes = []
    for row, (first, last) in enumerate(bounds):
        path = root / f"episode_row_{row:03d}.npz"
        sidecar = path.with_suffix(".json")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"rel_actions", "global_frame_indices"}:
                raise ValueError(f"Unexpected fields: {path}")
            actions = archive["rel_actions"]
            indices = archive["global_frame_indices"]
        expected = int(last - first + 1)
        if actions.shape != (expected, 7) or actions.dtype != np.float64:
            raise ValueError(f"Unexpected action schema: {path}")
        if not np.array_equal(indices, np.arange(first, last + 1, dtype=np.int64)):
            raise ValueError(f"Unexpected frame indices: {path}")
        if not sidecar.exists():
            raise FileNotFoundError(sidecar)
        episodes.append({"official_row": row, "frame_range_inclusive": [int(first), int(last)], "frame_count": expected, "sha256": sha256(path)})
    report = {"dataset": "CALVIN task_D_D training", "retained_fields": ["rel_actions", "global_frame_indices"], "episode_count": len(episodes), "total_frames": sum(item["frame_count"] for item in episodes), "episodes": episodes, "complete": len(episodes) == 31}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"episodes": len(episodes), "frames": report["total_frames"], "complete": report["complete"]}))


if __name__ == "__main__":
    main()

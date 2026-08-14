#!/usr/bin/env python3
"""Build the released frozen representation checkpoint manifest.

Purpose
-------
Hash all six seeds x three conditions after checkpoint migration or training,
and verify epoch, EMA decay, seed, condition, and normalization provenance.

Parameters
----------
--config: released representation YAML; --checkpoint-root: optional override.

Usage
-----
python scripts/representation/build_checkpoint_manifest.py \
  --config configs/representation.yaml

Outputs
-------
Writes ``checkpoints/representation/manifest.json``.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import torch
import yaml


PAYLOAD_CONDITION_ALIASES = {
    "correct_language": {"correct_language", "ema_isolated_correct_language"},
    "shuffled_language": {"shuffled_language", "ema_isolated_shuffled_language"},
    "reconstruction_only": {"reconstruction_only", "ema_reconstruction_only"},
}


def sha256(path: Path) -> str:
    """Hash one checkpoint."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    root = args.checkpoint_root or Path(config["release"]["checkpoint_root"])
    checkpoints = []
    expected_epoch = int(config["optimization"]["epochs"])
    expected_decay = float(config["optimization"]["ema_decay"])
    for seed in config["replication"]["seeds"]:
        for condition in config["replication"]["conditions"]:
            path = root / f"seed_{seed}" / condition / "checkpoint_ema.pt"
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if int(payload["epoch"]) != expected_epoch or int(payload["seed_base"]) != int(seed):
                raise ValueError(f"Checkpoint identity mismatch: {path}")
            if float(payload["ema_decay"]) != expected_decay:
                raise ValueError(f"EMA decay mismatch: {path}")
            payload_condition = str(payload["condition"])
            resolved_condition = str(payload["resolved_config"]["condition"])
            if payload_condition not in PAYLOAD_CONDITION_ALIASES[condition]:
                raise ValueError(f"Checkpoint condition mismatch: {path}")
            if resolved_condition not in PAYLOAD_CONDITION_ALIASES[condition]:
                raise ValueError(f"Resolved checkpoint condition mismatch: {path}")
            checkpoints.append({"seed_base": int(seed), "condition": condition, "source_payload_condition": payload_condition, "path": path.as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size, "epoch": expected_epoch, "ema_decay": expected_decay, "normalization_source_rows": payload["resolved_config"]["normalization"]["source_episode_rows"]})
    manifest = {"created_at": datetime.now().astimezone().isoformat(), "model": "action-only 32-D latent (16 semantic + 16 execution)", "checkpoints": checkpoints}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"checkpoints": len(checkpoints), "epoch": expected_epoch, "ema_decay": expected_decay}))


if __name__ == "__main__":
    main()

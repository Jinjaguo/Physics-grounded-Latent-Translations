#!/usr/bin/env python3
"""Freeze OpenCLIP text features for the 10 official Wave-19 instructions.

Purpose
-------
Load the preregistered DataComp XL ViT-L/14 OpenCLIP checkpoint from the local
Hugging Face cache, encode the exact audited official ``libero_10`` language
instructions, L2-normalize the frozen 768-D text vectors, and record model and
feature provenance for representation training.

Parameters
----------
``--config`` selects the Wave-19 YAML. ``--weights`` selects the local
``open_clip_pytorch_model.bin`` file; no network download is performed.

Usage
-----
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/prepare_wave19_text_features.py \
  --config configs/dynamics_7.yaml \
  --weights /home/jinjaguo/.cache/huggingface/hub/models--laion--CLIP-ViT-L-14-DataComp.XL-s13B-b90K/snapshots/84c9828e63dc9a9351d1fe637c346d4c1c4db341/open_clip_pytorch_model.bin

Outputs
-------
Writes ``text_features.npz`` and ``text_feature_manifest.json`` below
``data/wave19_libero_branchable/derived/representation`` and appends the exact
command to the Wave-19 command log.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import open_clip
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = Path(
    "/home/jinjaguo/.cache/huggingface/hub/"
    "models--laion--CLIP-ViT-L-14-DataComp.XL-s13B-b90K/snapshots/"
    "84c9828e63dc9a9351d1fe637c346d4c1c4db341/open_clip_pytorch_model.bin"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out = ROOT / config["experiment"]["output_root"]
    data = ROOT / config["experiment"]["data_root"]
    tasks_path = out / "wave19_libero_suite_tasks.json"
    if not tasks_path.is_file() or not args.weights.is_file():
        raise FileNotFoundError("Audited task list or local OpenCLIP checkpoint is missing")
    target = data / "derived/representation"
    feature_path = target / "text_features.npz"
    manifest_path = target / "text_feature_manifest.json"
    if feature_path.exists() or manifest_path.exists():
        raise RuntimeError("Wave-19 text features are already frozen; refusing to overwrite")
    task_rows = json.loads(tasks_path.read_text(encoding="utf-8"))
    texts = [str(row["language_instruction"]) for row in task_rows]
    if len(texts) != 10 or len(set(texts)) != 10:
        raise RuntimeError("Expected exactly 10 distinct audited official instructions")
    torch.set_num_threads(8)
    model = open_clip.create_model("ViT-L-14", pretrained=str(args.weights), device="cpu")
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    with torch.inference_mode():
        features = model.encode_text(tokenizer(texts)).float()
        features = torch.nn.functional.normalize(features, dim=-1).cpu().numpy()
    if features.shape != (10, 768) or not np.isfinite(features).all():
        raise RuntimeError(f"Unexpected OpenCLIP text feature matrix {features.shape}")
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        feature_path,
        texts=np.asarray(texts),
        task_ids=np.arange(10, dtype=np.int16),
        features=features.astype(np.float32),
    )
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "architecture": "open_clip ViT-L-14",
        "pretrained": "DataComp.XL-s13B-b90K",
        "weights_path": str(args.weights),
        "weights_sha256": sha256_file(args.weights),
        "feature_path": str(feature_path.relative_to(ROOT)),
        "feature_sha256": sha256_file(feature_path),
        "feature_shape": list(features.shape),
        "l2_normalized": True,
        "text_encoder_frozen": True,
        "instructions": texts,
        "uses_episode_or_split_data": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (out / "exact_commands.sh").open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n# {manifest['created_at']} phase=text-feature-freeze\n"
            "PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python "
            "scripts/dynamics/prepare_wave19_text_features.py --config configs/dynamics_7.yaml "
            f"--weights {args.weights}\n"
        )
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

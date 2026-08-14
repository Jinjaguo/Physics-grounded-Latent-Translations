#!/usr/bin/env python3
"""Evaluate frozen representation checkpoints on independent CALVIN episodes.

Purpose
-------
Run read-only inference for six seeds and three conditions on all 14 official
training episodes that were unused by training, development, and confirmation.
For query episode E, candidates are every other independent episode.

Parameters
----------
--config: released representation YAML; --seed: optional registered seed;
--device: CUDA device; --checkpoint-root/--inference-root: optional overrides.

Usage
-----
PYTHONPATH=src python scripts/representation/evaluate.py \
  --config configs/representation.yaml --device cuda:0 [--seed 810]

Outputs
-------
Writes 18 read-only ``metrics.json`` records beneath
``results/representation/independent_replication/inference``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from pglt.data.calvin import build_chunk_records, load_annotations
from pglt.evaluation.metrics import reconstruction_metrics, symmetric_retrieval_metrics
from pglt.representation.data import (
    ActionChunkDataset,
    ActionOnlyEpisode,
    ActionOnlyNormalization,
    encode_action_dataset,
)
from pglt.representation.objectives import CONDITIONS, build_model
from pglt.representation.retrieval import (
    DIRECTIONS,
    add_macro_recall,
    annotation_representations,
    cross_episode_retrieval_metrics,
    per_action_dimension_reconstruction,
)


def sha256(path: Path) -> str:
    """Hash one checkpoint without loading it."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def concatenate(values: list[dict]) -> dict:
    """Concatenate independently encoded candidate episodes."""

    arrays = ("latents", "semantic_latents", "execution_latents", "text_latents", "predictions", "normalized_targets", "raw_targets")
    return {
        **{key: np.concatenate([item[key] for item in values], axis=0) for key in arrays},
        **{key: sum((list(item[key]) for item in values), []) for key in ("task", "text", "metadata")},
    }


def episode_metrics(query: dict, candidates: dict, normalization: ActionOnlyNormalization) -> dict:
    """Compute semantic and motor metrics for one query episode."""

    query_items = annotation_representations({**query, "latents": query["semantic_latents"]})
    candidate_items = annotation_representations({**candidates, "latents": candidates["semantic_latents"]})
    action = np.stack([item["action"] for item in query_items])
    text = np.stack([item["text_latent"] for item in query_items])
    tasks = [item["task"] for item in query_items]
    retrieval = add_macro_recall(symmetric_retrieval_metrics(action, text, tasks))
    reconstruction = reconstruction_metrics(
        query["predictions"],
        query["normalized_targets"],
        query["raw_targets"],
        query["task"],
        normalization.action_mean,
        normalization.action_std,
    )
    return {
        "semantic_retrieval": retrieval,
        "cross_episode_retrieval": cross_episode_retrieval_metrics(query_items, candidate_items),
        "reconstruction": reconstruction,
        "per_action_dimension_reconstruction": per_action_dimension_reconstruction(query, normalization.action_mean, normalization.action_std),
        "query_chunk_count": len(query["task"]),
        "candidate_chunk_count": len(candidates["task"]),
        "query_annotation_count": len(query_items),
        "candidate_annotation_count": len(candidate_items),
        "candidate_episode_rows": sorted({int(item["episode_id"]) for item in candidates["metadata"]}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--inference-root", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    rows = [int(row) for row in config["selection"]["independent_replication_episode_rows"]]
    registered = [int(seed) for seed in config["replication"]["seeds"]]
    seeds = registered if args.seed is None else [int(args.seed)]
    if any(seed not in registered for seed in seeds):
        raise ValueError("Unregistered seed")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Frozen evaluation requires CUDA")
    torch.set_num_threads(int(config["runtime"]["torch_cpu_threads_per_process"]))
    metadata_root = Path(config["source"]["metadata_root"])
    bounds = np.load(metadata_root / "training" / "ep_start_end_ids.npy").reshape(-1, 2)
    annotations_all = load_annotations(metadata_root / "training")
    tasks = set(config["data"]["tasks"])
    feature_archive = np.load(config["text"]["feature_archive"], allow_pickle=True)
    text_features = {str(key): value for key, value in zip(feature_archive["texts"], feature_archive["features"])}
    episodes = {}
    records = {}
    for row in rows:
        first, last = map(int, bounds[row])
        path = Path(config["source"]["compact_root"]) / "training" / f"episode_row_{row:03d}.npz"
        with np.load(path) as archive:
            episodes[row] = ActionOnlyEpisode(row, first, last, archive["rel_actions"].copy(), archive["global_frame_indices"].copy(), str(path.resolve()))
        annotations = [item for item in annotations_all if item.task in tasks and item.start_index >= first and item.end_index <= last]
        records[row] = build_chunk_records(annotations, split="independent_replication", chunk_length=int(config["data"]["chunk_length"]), stride=int(config["data"]["evaluation_stride"]), episode_id=row)
    checkpoint_root = args.checkpoint_root or Path(config["release"]["checkpoint_root"])
    manifest = json.loads((checkpoint_root / "manifest.json").read_text())
    expected_hashes = {(int(item["seed_base"]), item["condition"]): item["sha256"] for item in manifest["checkpoints"]}
    output_root = args.inference_root or Path(config["release"]["inference_root"])
    for seed in seeds:
        for condition in CONDITIONS:
            output = output_root / f"seed_{seed}" / condition
            metrics_path = output / "metrics.json"
            checkpoint = checkpoint_root / f"seed_{seed}" / condition / "checkpoint_ema.pt"
            checkpoint_hash = sha256(checkpoint)
            if checkpoint_hash != expected_hashes[(seed, condition)]:
                raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint}")
            if metrics_path.exists() and json.loads(metrics_path.read_text()).get("checkpoint_sha256") == checkpoint_hash:
                continue
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            normalization_json = payload["resolved_config"]["normalization"]
            normalization = ActionOnlyNormalization(
                np.asarray(normalization_json["action_mean"]),
                np.asarray(normalization_json["action_std"]),
                tuple(normalization_json["source_episode_rows"]),
                int(normalization_json["source_frame_count"]),
            )
            if not set(normalization.source_episode_rows).isdisjoint(rows):
                raise RuntimeError("Independent episode entered checkpoint normalization")
            model = build_model(config, device)
            model.load_state_dict(payload["model_state_dict"])
            model.eval()
            before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            encoded = {}
            for row in rows:
                dataset = ActionChunkDataset({row: episodes[row]}, records[row], normalization, text_features)
                encoded[row] = encode_action_dataset(model, dataset, device)
            metrics = {}
            for row in rows:
                candidates = concatenate([encoded[other] for other in rows if other != row])
                metrics[str(row)] = episode_metrics(encoded[row], candidates, normalization)
                if metrics[str(row)]["candidate_episode_rows"] != [other for other in rows if other != row]:
                    raise RuntimeError("Candidate query exclusion changed")
            unchanged = all(torch.equal(before[key], value.detach().cpu()) for key, value in model.state_dict().items())
            if not unchanged:
                raise RuntimeError("Frozen inference changed a model tensor")
            output.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps({"seed_base": seed, "condition": condition, "checkpoint_sha256": checkpoint_hash, "checkpoint_epoch": int(config["optimization"]["epochs"]), "ema_decay": float(config["optimization"]["ema_decay"]), "normalization_source_rows": list(normalization.source_episode_rows), "independent_rows_excluded_from_normalization": True, "candidate_pool_rule": config["evaluation"]["query_candidate_rule"], "parameter_tensors_unchanged": True, "optimizer_steps": 0, "ema_updates": 0, "backward_calls": 0, "episode_metrics": metrics}, indent=2))
            print(json.dumps({"seed": seed, "condition": condition, "episodes": len(rows)}), flush=True)


if __name__ == "__main__":
    main()

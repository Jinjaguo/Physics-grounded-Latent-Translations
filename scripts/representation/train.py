#!/usr/bin/env python3
"""Train the released PGLT action representation and frozen controls.

Purpose
-------
Train six deterministic seeds for correct-language, shuffled-language, and
reconstruction-only conditions using the final action-only 16/16 model and
EMA=0.999. Training and normalization use only the 13 registered source
episodes; the four confirmation episodes are evaluated without adaptation.

Parameters
----------
--config: released representation YAML; --seed: optional registered seed;
--device: CUDA device; --output-root: destination for reproduced checkpoints;
--force: replace an incomplete reproduced unit only.

Usage
-----
PYTHONPATH=src python scripts/representation/train.py \
  --config configs/representation.yaml --device cuda:0 [--seed 810]

Outputs
-------
Writes EMA/raw checkpoints, resolved settings, train logs, and confirmation
metrics under ``results/representation/reproduction/checkpoints`` by default.
An explicit ``--output-root`` selects a different reproduction destination.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path

import torch
import yaml

from pglt.representation.data import (
    ActionChunkDataset,
    create_confirmation_datasets,
    evaluate_checkpoint,
    prepare_confirmation_data,
    stack_action_batch,
)
from pglt.representation.ema import ParameterEMA
from pglt.representation.objectives import CONDITIONS, batch_objective, build_model
from pglt.representation.reproducibility import (
    set_deterministic_seed,
    shuffled_language_override,
    unique_task_batches,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=Path("results/representation/reproduction/checkpoints"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    registered = [int(seed) for seed in config["replication"]["seeds"]]
    seeds = registered if args.seed is None else [int(args.seed)]
    if any(seed not in registered for seed in seeds):
        raise ValueError("Unregistered seed")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Released training requires an available CUDA device")
    torch.set_num_threads(int(config["runtime"]["torch_cpu_threads_per_process"]))
    data = prepare_confirmation_data(config)
    common = {"normalization": data.normalization, "text_features": data.text_features}
    epochs = int(config["optimization"]["epochs"])
    decay = float(config["optimization"]["ema_decay"])
    root = args.output_root
    for seed_base in seeds:
        for condition in CONDITIONS:
            output = root / f"seed_{seed_base}" / condition
            ema_path = output / "checkpoint_ema.pt"
            metrics_path = output / "confirmation_metrics.json"
            if ema_path.exists() and metrics_path.exists() and not args.force:
                print(json.dumps({"seed": seed_base, "condition": condition, "status": "reused"}))
                continue
            if output.exists() and not args.force:
                raise RuntimeError(f"Incomplete unit requires --force: {output}")
            output.mkdir(parents=True, exist_ok=True)
            effective_seed = seed_base + int(
                config["replication"]["condition_seed_offsets"][condition]
            )
            set_deterministic_seed(effective_seed)
            model = build_model(config, device)
            ema = ParameterEMA(model, decay)
            override = None
            shuffle_manifest = []
            if condition == "shuffled_language":
                override, shuffle_manifest = shuffled_language_override(
                    data.train_annotations, effective_seed
                )
            train, confirmation, candidates = create_confirmation_datasets(data, override)
            parameters = list(model.parameters())
            optimizer = torch.optim.AdamW(
                parameters,
                lr=float(config["optimization"]["learning_rate"]),
                weight_decay=float(config["optimization"]["weight_decay"]),
            )
            resolved = {
                "created_at": datetime.now().astimezone().isoformat(),
                "seed_base": seed_base,
                "effective_seed": effective_seed,
                "condition": condition,
                "epochs": epochs,
                "ema_decay": decay,
                "training_episode_rows": list(data.training_episode_rows),
                "confirmation_episode_rows": list(data.confirmation_episode_rows),
                "normalization": data.normalization.to_json(),
                "leakage_checks": data.leakage_checks,
                "shuffled_language_mapping": shuffle_manifest,
                "device": str(device),
            }
            (output / "resolved_config.json").write_text(json.dumps(resolved, indent=2))
            log = output / "train_log.jsonl"
            log.write_text("")
            for epoch in range(1, epochs + 1):
                model.train()
                totals: dict[str, float] = defaultdict(float)
                count = 0
                for indices in unique_task_batches(
                    train.records,
                    seed=effective_seed + epoch,
                    batch_size=int(config["optimization"]["batch_size"]),
                ):
                    batch = stack_action_batch(train, indices, device)
                    objective = batch_objective(
                        model, batch, condition, config["optimization"]
                    )
                    optimizer.zero_grad(set_to_none=True)
                    objective["total"].backward()
                    torch.nn.utils.clip_grad_norm_(
                        parameters, float(config["optimization"]["gradient_clip_norm"])
                    )
                    optimizer.step()
                    ema.update(model)
                    count += len(indices)
                    for key in ("total", "clip", "reconstruction", "continuous_mse", "gripper_bce"):
                        totals[key] += float(objective[key].detach().cpu()) * len(indices)
                with log.open("a") as handle:
                    handle.write(json.dumps({"epoch": epoch, **{key: value / count for key, value in totals.items()}}) + "\n")

            def evaluate_all() -> dict:
                overall = evaluate_checkpoint(
                    model,
                    confirmation,
                    candidates,
                    device=device,
                    knn_k=int(config["evaluation"]["knn_k"]),
                )
                per_episode = {}
                for row in data.confirmation_episode_rows:
                    records = [item for item in data.confirmation_records if item.episode_id == row]
                    dataset = ActionChunkDataset(
                        {row: data.confirmation_episodes[row]}, records, **common
                    )
                    per_episode[str(row)] = evaluate_checkpoint(
                        model,
                        dataset,
                        candidates,
                        device=device,
                        knn_k=int(config["evaluation"]["knn_k"]),
                    )
                return {"overall": overall, "per_episode": per_episode}

            raw_metrics = evaluate_all()
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epochs, "condition": condition, "seed_base": seed_base},
                output / "checkpoint_raw.pt",
            )
            ema.store(model)
            ema.copy_to(model)
            ema_metrics = evaluate_all()
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epochs, "condition": condition, "seed_base": seed_base, "ema_updates": ema.updates, "ema_decay": decay, "resolved_config": resolved},
                ema_path,
            )
            ema.restore(model)
            metrics_path.write_text(
                json.dumps({"seed_base": seed_base, "condition": condition, "epoch": epochs, "ema_updates": ema.updates, "raw": raw_metrics, "ema": ema_metrics}, indent=2)
            )
            print(json.dumps({"seed": seed_base, "condition": condition, "status": "completed", "ema_updates": ema.updates}), flush=True)


if __name__ == "__main__":
    main()

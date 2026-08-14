#!/usr/bin/env python3
"""Run leave-one-development-episode-out representation validation.

Purpose
-------
Reproduce the 8 folds x 6 seeds x 3 conditions used to validate epoch 40 and
EMA=0.999 before confirmation. Confirmation and independent-replication rows
are never loaded. Only final per-cell metrics are retained.

Parameters
----------
--config: released representation YAML; --seed: optional registered seed;
--device: CUDA device.

Usage
-----
PYTHONPATH=src python scripts/representation/validate.py \
  --config configs/representation.yaml --device cuda:0 [--seed 810]

Outputs
-------
Writes 144 ``metrics.json`` records beneath the configured development root.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from pglt.representation.data import (
    create_fold_datasets,
    evaluate_checkpoint,
    prepare_development_fold,
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
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    registered = [int(seed) for seed in config["replication"]["seeds"]]
    seeds = registered if args.seed is None else [int(args.seed)]
    if any(seed not in registered for seed in seeds):
        raise ValueError("Unregistered seed")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Validation requires CUDA")
    torch.set_num_threads(int(config["runtime"]["torch_cpu_threads_per_process"]))
    epochs = int(config["optimization"]["epochs"])
    output_root = Path(config["release"]["development_root"])
    for seed_base in seeds:
        for fold_index in range(len(config["selection"]["development_episode_rows"])):
            fold = prepare_development_fold(config, fold_index)
            forbidden = set(config["selection"]["confirmation_episode_rows"]) | set(
                config["selection"]["independent_replication_episode_rows"]
            )
            if forbidden & set(fold.training_episode_rows + (fold.development_episode_row,)):
                raise RuntimeError("A reserved row entered development validation")
            for condition in CONDITIONS:
                output = output_root / f"fold_{fold_index}" / f"seed_{seed_base}" / condition
                metrics_path = output / "metrics.json"
                if metrics_path.exists():
                    continue
                output.mkdir(parents=True, exist_ok=True)
                effective_seed = seed_base + int(config["replication"]["condition_seed_offsets"][condition])
                set_deterministic_seed(effective_seed)
                model = build_model(config, device)
                ema = ParameterEMA(model, float(config["optimization"]["ema_decay"]))
                override = None
                if condition == "shuffled_language":
                    override, _ = shuffled_language_override(fold.train_annotations, effective_seed)
                train, development, candidates = create_fold_datasets(fold, override)
                parameters = list(model.parameters())
                optimizer = torch.optim.AdamW(
                    parameters,
                    lr=float(config["optimization"]["learning_rate"]),
                    weight_decay=float(config["optimization"]["weight_decay"]),
                )
                for epoch in range(1, epochs + 1):
                    for indices in unique_task_batches(
                        train.records,
                        seed=effective_seed + epoch,
                        batch_size=int(config["optimization"]["batch_size"]),
                    ):
                        batch = stack_action_batch(train, indices, device)
                        objective = batch_objective(model, batch, condition, config["optimization"])
                        optimizer.zero_grad(set_to_none=True)
                        objective["total"].backward()
                        torch.nn.utils.clip_grad_norm_(parameters, float(config["optimization"]["gradient_clip_norm"]))
                        optimizer.step()
                        ema.update(model)
                raw = evaluate_checkpoint(model, development, candidates, device=device, knn_k=int(config["evaluation"]["knn_k"]))
                ema.store(model)
                ema.copy_to(model)
                averaged = evaluate_checkpoint(model, development, candidates, device=device, knn_k=int(config["evaluation"]["knn_k"]))
                ema.restore(model)
                metrics_path.write_text(json.dumps({"fold": fold_index, "development_episode_row": fold.development_episode_row, "seed_base": seed_base, "condition": condition, "epoch": epochs, "ema_decay": config["optimization"]["ema_decay"], "raw": raw, "ema": averaged, "ema_updates": ema.updates}, indent=2))
                print(json.dumps({"fold": fold_index, "seed": seed_base, "condition": condition}), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train and evaluate the diagnosed stable LaWM adapter on official toy data.

Purpose
-------
This is a software-validation harness, not a CALVIN dynamics experiment.  It
uses the upstream LaWM objective, model components, optimizer, toy tensor, and
physical evaluator while replacing only the numerically unstable DEL solver
iteration with PGLT's diagnosed sign/time-scaled correction.  Every batch is
checked before backward, after backward, and after the optimizer step.

Parameters
----------
--toy-data
    Official tensor produced by upstream ``examples/toy_parabolic.py``.
--output-dir
    Separate directory for the stable-adapter checkpoint, JSONL training log,
    resolved configuration, and final software-validation metrics.
--epochs
    Number of smoke-training epochs.
--batch-size
    Number of official toy trajectories per optimizer batch.
--max-samples
    Optional leading subset used to keep CPU validation practical.
--dt
    Exact toy time step used by the discrete Lagrangian.
--seed
    Deterministic model, optimizer, and data-order seed.

Usage
-----
PYTHONPATH=src:third_party/LaWM python scripts/dynamics/train_corrected_solver_toy.py \
    --toy-data results/dynamics/software_validation/upstream_reference/toy_parabolic.pt \
    --output-dir results/dynamics/software_validation/corrected_solver/stable_smoke \
    --epochs 5 --batch-size 16 --max-samples 16 --dt 0.02 --seed 0

Outputs
-------
The requested output directory receives ``config.json``, ``train_log.jsonl``,
``stable_lawm_final.pt``, and ``metrics.json``.  Learned-model energy values
are reported only as upstream toy software diagnostics, never as physical
energy for a PGLT representation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from lawm.metrics import compute_metrics
from lawm.train import batch_objective
from lawm.utils import load_state_tensor, make_state_weights, make_time_grid, weighted_state_loss
from pglt.dynamics.lawm_adapter import StableToyLeastActionWorldModel, require_finite


def parse_args() -> argparse.Namespace:
    """Parse the bounded CPU smoke-training settings."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toy-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def assert_finite_gradients(model: torch.nn.Module) -> None:
    """Reject the first named nonfinite optimizer gradient."""

    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            require_finite(f"gradient {name}", parameter.grad)


def assert_finite_parameters(model: torch.nn.Module, stage: str) -> None:
    """Reject the first named nonfinite parameter at an optimizer boundary."""

    for name, parameter in model.named_parameters():
        require_finite(f"parameter {name} {stage}", parameter)


def main() -> None:
    """Run finite-checked toy training and upstream metric computation."""

    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.max_samples <= 0:
        raise ValueError("epochs, batch-size, and max-samples must be positive")
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    states = load_state_tensor(args.toy_data, state_dim=9, max_samples=args.max_samples)
    ts = make_time_grid(states.shape[1], args.dt, "cpu")
    weights = make_state_weights(9, "cpu")
    model = StableToyLeastActionWorldModel(
        state_dim=9,
        latent_dim=9,
        context_dim=16,
        hidden_dim=128,
        depth=3,
        solver_iters=8,
        solver_step_size=0.25,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(states),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resolved = {
        **vars(args),
        "toy_data": str(args.toy_data),
        "output_dir": str(args.output_dir),
        "state_dim": 9,
        "context_dim": 16,
        "hidden_dim": 128,
        "depth": 3,
        "solver_iters": 8,
        "solver_coefficient": 0.25,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "lambda_del": 1e-2,
        "lambda_reg": 1e-4,
        "adapter_change": "q_next += alpha * h_next * DEL_residual / detached_positive_mass",
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(resolved, indent=2, default=str), encoding="utf-8"
    )
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        totals = {key: 0.0 for key in ("loss", "traj", "del", "reg")}
        count = 0
        model.train()
        for (batch,) in loader:
            output = batch_objective(
                model, batch, ts, weights, lambda_del=1e-2, lambda_reg=1e-4
            )
            for name in totals:
                require_finite(f"epoch {epoch} forward {name}", output[name])
            optimizer.zero_grad(set_to_none=True)
            output["loss"].backward()
            assert_finite_gradients(model)
            assert_finite_parameters(model, "before optimizer step")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            assert_finite_gradients(model)
            optimizer.step()
            assert_finite_parameters(model, "after optimizer step")
            count += len(batch)
            for name in totals:
                totals[name] += float(output[name].detach()) * len(batch)
        record = {"epoch": epoch, **{name: value / count for name, value in totals.items()}}
        history.append(record)
        with (args.output_dir / "train_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    model.eval()
    with torch.enable_grad():
        prediction = model(states[:, 0], ts, state1=states[:, 1])
        rollout_mse = weighted_state_loss(prediction, states, weights)
        require_finite("final learned rollout", prediction)
        require_finite("final learned rollout MSE", rollout_mse)
        upstream_metrics = compute_metrics(
            model, states, ts, true_energy_mode="translational_gravity", gravity=9.8
        )
    if not all(
        isinstance(value, str) or torch.isfinite(torch.tensor(float(value)))
        for value in upstream_metrics.values()
    ):
        raise FloatingPointError(f"Final upstream metrics are nonfinite: {upstream_metrics}")
    metrics: dict[str, Any] = {
        "status": "passed_finite_stable_adapter_smoke_test",
        "history": history,
        "final_weighted_rollout_mse": float(rollout_mse.detach()),
        "upstream_physical_evaluator": upstream_metrics,
        "scope": "official toy software validation only; not a CALVIN or PGLT physics result",
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": resolved,
            "metrics": metrics,
        },
        args.output_dir / "stable_lawm_final.pt",
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Capture PyTorch's first anomaly for the deterministic LaWM horizon-8 failure.

Purpose
-------
This diagnostic complements the tensor-level numerical harness by enabling
PyTorch anomaly detection only around the known minimal failing backward pass.
It records the autograd operation named by the exception.  Anomaly detection
is not used during the corrected solver's training and is not itself a fix.

Parameters
----------
--toy-data
    Official upstream LaWM parabolic toy tensor.
--output-json
    Destination for exact horizon, solver parameters, finite forward losses,
    and the caught anomaly exception.
--seed
    Deterministic upstream model initialization seed.

Usage
-----
PYTHONPATH=third_party/LaWM python scripts/dynamics/trace_upstream_autograd_anomaly.py \
    --toy-data results/dynamics/software_validation/upstream_reference/toy_parabolic.pt \
    --output-json results/dynamics/software_validation/corrected_solver/anomaly_trace.json \
    --seed 0

Outputs
-------
The JSON artifact is saved at the requested location.  The script exits
successfully only when a nonfinite-backward RuntimeError is detected; an
unexpected finite upstream backward is treated as a regression mismatch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lawm.model import LeastActionWorldModel
from lawm.train import batch_objective
from lawm.utils import load_state_tensor, make_state_weights, make_time_grid


def parse_args() -> argparse.Namespace:
    """Parse official toy input, diagnostic output, and deterministic seed."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toy-data", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Run the first known failing horizon and capture its anomaly operation."""

    args = parse_args()
    torch.manual_seed(args.seed)
    states = load_state_tensor(args.toy_data, state_dim=9, max_samples=1)[:, :8]
    ts = make_time_grid(8, 0.02, "cpu")
    model = LeastActionWorldModel(
        state_dim=9,
        latent_dim=9,
        context_dim=16,
        hidden_dim=128,
        depth=3,
        solver_iters=8,
        solver_step_size=0.25,
    )
    output = batch_objective(
        model,
        states,
        ts,
        make_state_weights(9, "cpu"),
        lambda_del=1e-2,
        lambda_reg=1e-4,
    )
    record = {
        "seed": args.seed,
        "batch_size": 1,
        "horizon": 8,
        "dt": 0.02,
        "solver_iters": 8,
        "solver_step_size": 0.25,
        "forward_losses": {
            name: float(output[name].detach()) for name in ("loss", "traj", "del", "reg")
        },
        "all_forward_losses_finite": all(
            bool(torch.isfinite(output[name])) for name in ("loss", "traj", "del", "reg")
        ),
    }
    try:
        with torch.autograd.detect_anomaly(check_nan=True):
            output["loss"].backward()
    except RuntimeError as error:
        record["anomaly_detected"] = True
        record["exception"] = str(error)
    else:
        record["anomaly_detected"] = False
        record["exception"] = None
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    if not record["anomaly_detected"]:
        raise RuntimeError("Expected upstream horizon-8 anomaly was not reproduced")


if __name__ == "__main__":
    main()

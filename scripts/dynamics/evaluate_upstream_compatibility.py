#!/usr/bin/env python3
"""Run the upstream LaWM physical evaluator with a PyTorch 2.6+ compatibility shim.

Purpose
-------
The upstream LaWM evaluator loads a checkpoint containing ``pathlib.Path``
objects.  PyTorch 2.6 changed ``torch.load`` to use ``weights_only=True`` by
default, so the unmodified evaluator rejects that trusted, locally generated
checkpoint.  This wrapper allow-lists only the concrete pathlib classes used
by the checkpoint and then invokes the upstream evaluator without editing any
LaWM source file.

Parameters
----------
All command-line parameters are defined by and forwarded unchanged to
``third_party/LaWM/lawm/metrics.py``.  The required arguments are
``--checkpoint`` and ``--trajectory-pt``; useful optional arguments include
``--dt``, ``--device``, ``--max-samples``, and ``--true-energy-mode``.

Usage
-----
PYTHONPATH=third_party/LaWM python scripts/dynamics/evaluate_upstream_compatibility.py \
    --checkpoint results/dynamics/software_validation/upstream_reference/run/lawm_final.pth \
    --trajectory-pt results/dynamics/software_validation/upstream_reference/toy_parabolic.pt \
    --dt 0.02 --device cpu --max-samples 16

Outputs
-------
The wrapper prints the upstream JSON metrics to standard output.  It does not
create files itself; checkpoint and trajectory paths are read-only inputs.
Experiment orchestration should capture the printed JSON in the corresponding
software-validation manifest or log directory.
"""

from __future__ import annotations

import pathlib

import torch


def main() -> None:
    """Allow-list trusted pathlib checkpoint metadata and run LaWM evaluation."""

    torch.serialization.add_safe_globals([pathlib.Path, pathlib.PosixPath])
    from lawm.metrics import main as upstream_main

    upstream_main()


if __name__ == "__main__":
    main()

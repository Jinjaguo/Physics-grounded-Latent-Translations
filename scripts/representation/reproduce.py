#!/usr/bin/env python3
"""Orchestrate the released representation reproduction pipeline.

Purpose
-------
Provide one functional entry point for data verification, optional development
validation, final training, frozen independent evaluation, result aggregation,
tests, and release audit. Stages are explicit so checkpoint-only reproduction
does not retrain models.

Parameters
----------
--config: representation YAML; --stage: one or more ordered stages;
--device: CUDA device; --python: interpreter path.

Usage
-----
python scripts/representation/reproduce.py --config configs/representation.yaml \
  --stage evaluate summarize test audit --device cuda:0

python scripts/representation/reproduce.py --config configs/representation.yaml \
  --stage validate train evaluate summarize test audit --device cuda:0

Outputs
-------
Checkpoint-only stages verify the released files in their configured paths.
Runs containing ``train`` write new checkpoints, inference, and decisions
under ``results/representation/reproduction`` without replacing the release.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


STAGES = ("verify", "validate", "train", "evaluate", "summarize", "test", "audit")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", nargs="+", choices=STAGES, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "src" + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    full_reproduction = "train" in args.stage
    checkpoint_root = Path("results/representation/reproduction/checkpoints") if full_reproduction else Path("checkpoints/representation")
    inference_root = Path("results/representation/reproduction/inference") if full_reproduction else Path("results/representation/independent_replication/inference")
    decision_root = Path("results/representation/reproduction") if full_reproduction else Path("results/representation/independent_replication")
    commands = {
        "verify": [args.python, "scripts/representation/verify_data.py", "--config", str(args.config), "--output", "results/representation/data_integrity.json"],
        "validate": [args.python, "scripts/representation/validate.py", "--config", str(args.config), "--device", args.device],
        "train": [args.python, "scripts/representation/train.py", "--config", str(args.config), "--device", args.device, "--output-root", str(checkpoint_root)],
        "evaluate": [args.python, "scripts/representation/evaluate.py", "--config", str(args.config), "--device", args.device, "--checkpoint-root", str(checkpoint_root), "--inference-root", str(inference_root)],
        "summarize": [args.python, "scripts/representation/summarize.py", "--config", str(args.config), "--inference-root", str(inference_root), "--output-root", str(decision_root)],
        "test": [args.python, "-m", "pytest", "tests/representation", "-q", "--junitxml=results/representation/representation_tests.xml"],
        "audit": [args.python, "scripts/representation/audit.py", "--config", str(args.config), "--pytest-xml", "results/representation/representation_tests.xml", "--checkpoint-root", str(checkpoint_root), "--inference-root", str(inference_root), "--decision-root", str(decision_root)],
    }
    for stage in args.stage:
        print(f"[representation] stage={stage}", flush=True)
        subprocess.run(commands[stage], check=True, env=environment)
        if stage == "train":
            subprocess.run(
                [args.python, "scripts/representation/build_checkpoint_manifest.py", "--config", str(args.config), "--checkpoint-root", str(checkpoint_root)],
                check=True,
                env=environment,
            )


if __name__ == "__main__":
    main()

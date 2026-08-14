#!/usr/bin/env python3
"""Aggregate independent replication metrics and decide representation readiness.

Purpose
-------
Compute correct-language minus direction-wise stronger-control deltas, episode
and seed summaries, whole-episode bootstrap intervals, every-cell motor sanity,
and the final prospective Representation Readiness Gate.

Parameters
----------
--config: released representation YAML; --inference-root/--output-root:
optional reproduced-run overrides.

Usage
-----
PYTHONPATH=src python scripts/representation/summarize.py \
  --config configs/representation.yaml

Outputs
-------
Writes functional JSON tables and ``summary.md`` under
``results/representation/independent_replication``.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import yaml

from pglt.representation.readiness import (
    DIRECTIONS,
    adjudicate_r_gate,
    motor_sanity,
    stronger_control_delta,
)


CONDITIONS = ("correct_language", "shuffled_language", "reconstruction_only")


def write_json(path: Path, payload: object) -> None:
    """Write finite, stable JSON with a trailing newline."""

    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def macro(metrics: dict, direction: str) -> float:
    """Read one cross-episode task-balanced R@1."""

    return float(metrics["cross_episode_retrieval"][direction]["macro_recall_at_1"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inference-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    rows = [int(row) for row in config["selection"]["independent_replication_episode_rows"]]
    seeds = [int(seed) for seed in config["replication"]["seeds"]]
    inference_root = args.inference_root or Path(config["release"]["inference_root"])
    output = args.output_root or (Path(config["release"]["results_root"]) / "independent_replication")
    output.mkdir(parents=True, exist_ok=True)
    records = {}
    missing = []
    for seed in seeds:
        for condition in CONDITIONS:
            path = inference_root / f"seed_{seed}" / condition / "metrics.json"
            if path.exists():
                records[(seed, condition)] = json.loads(path.read_text())
            else:
                missing.append(str(path))
    read_only = all(
        item["parameter_tensors_unchanged"]
        and item["optimizer_steps"] == item["ema_updates"] == item["backward_calls"] == 0
        and item["independent_rows_excluded_from_normalization"]
        for item in records.values()
    )
    candidate_exact = all(
        item["episode_metrics"][str(query)]["candidate_episode_rows"]
        == [row for row in rows if row != query]
        for item in records.values()
        for query in rows
    )
    integrity = not missing and len(records) == 18 and read_only and candidate_exact
    semantic_table = []
    motor_table = []
    gate_rows = []
    negative_cells = []
    for episode in rows:
        for seed in seeds:
            correct = records[(seed, "correct_language")]["episode_metrics"][str(episode)]
            shuffled = records[(seed, "shuffled_language")]["episode_metrics"][str(episode)]
            reconstruction = records[(seed, "reconstruction_only")]["episode_metrics"][str(episode)]
            deltas = {}
            scores = {}
            controls = {}
            for direction in DIRECTIONS:
                correct_score = macro(correct, direction)
                shuffled_score = macro(shuffled, direction)
                reconstruction_score = macro(reconstruction, direction)
                delta, control = stronger_control_delta(
                    correct_score, reconstruction_score, shuffled_score
                )
                deltas[direction] = delta
                controls[direction] = control
                scores[direction] = {
                    "correct_language": correct_score,
                    "shuffled_language": shuffled_score,
                    "reconstruction_only": reconstruction_score,
                }
            correct_motor = correct["reconstruction"]["global"]
            reconstruction_motor = reconstruction["reconstruction"]["global"]
            motor = motor_sanity(
                correct_motor["continuous_mse_raw_rel_action"],
                reconstruction_motor["continuous_mse_raw_rel_action"],
                correct_motor["gripper_accuracy"],
                reconstruction_motor["gripper_accuracy"],
            )
            semantic_table.append(
                {
                    "episode_row": episode,
                    "seed_base": seed,
                    "scores": scores,
                    "semantic_delta": deltas,
                    "stronger_control": controls,
                    "candidate_annotation_count": correct["candidate_annotation_count"],
                    "candidate_chunk_count": correct["candidate_chunk_count"],
                }
            )
            motor_table.append(
                {
                    "episode_row": episode,
                    "seed_base": seed,
                    "correct_mse": correct_motor["continuous_mse_raw_rel_action"],
                    "reconstruction_mse": reconstruction_motor["continuous_mse_raw_rel_action"],
                    "correct_gripper_accuracy": correct_motor["gripper_accuracy"],
                    "reconstruction_gripper_accuracy": reconstruction_motor["gripper_accuracy"],
                    **motor,
                }
            )
            gate_rows.append(
                {
                    "episode_row": episode,
                    "seed_base": seed,
                    "cross_episode_delta": deltas,
                    "motor_passed": motor["passed"],
                }
            )
            for direction in DIRECTIONS:
                if deltas[direction] < 0:
                    negative_cells.append(
                        {
                            "episode_row": episode,
                            "seed_base": seed,
                            "direction": direction,
                            "delta": deltas[direction],
                            "stronger_control": controls[direction],
                            "candidate_annotation_count": correct["candidate_annotation_count"],
                        }
                    )
    decision = adjudicate_r_gate(
        gate_rows,
        bootstrap_seed=int(config["evaluation"]["bootstrap_seed"]),
        bootstrap_replicates=int(config["evaluation"]["bootstrap_replicates"]),
        integrity_passed=integrity,
    )
    episode_means = {item["episode_row"]: item for item in decision["episode_summary"]}
    seed_means = {item["seed_base"]: item for item in decision["seed_summary"]}
    for item in negative_cells:
        item["episode_mean"] = episode_means[item["episode_row"]][item["direction"]]
        item["seed_mean"] = seed_means[item["seed_base"]][item["direction"]]
    write_json(output / "episode_seed_semantic_metrics.json", semantic_table)
    write_json(output / "episode_seed_motor_metrics.json", motor_table)
    write_json(output / "episode_mean_deltas.json", decision["episode_summary"])
    write_json(output / "seed_mean_deltas.json", decision["seed_summary"])
    write_json(output / "whole_episode_bootstrap.json", decision["bootstrap"])
    write_json(output / "negative_semantic_cells.json", negative_cells)
    gate = {
        "created_at": datetime.now().astimezone().isoformat(),
        "historical_all_cell_gate": "FAIL",
        "prospective_representation_readiness_gate": "PASS" if decision["r_gate_passed"] else "FAIL",
        "historical_failure_erased": False,
        "integrity": {
            "records": len(records),
            "expected": 18,
            "missing": missing,
            "read_only_inference": read_only,
            "candidate_query_exclusion_exact": candidate_exact,
            "passed": integrity,
        },
        **decision,
    }
    write_json(output / "readiness_gate_decision.json", gate)
    status = {
        "historical_all_cell_gate": "FAIL",
        "independent_replication": "PASS" if decision["r_gate_passed"] else "FAIL",
        "representation_ready": bool(decision["r_gate_passed"]),
        "representation_frozen_for_paper": bool(decision["r_gate_passed"]),
        "latent_dynamics_authorized": bool(decision["r_gate_passed"]),
        "historical_failure_erased": False,
    }
    write_json(output / "representation_status.json", status)
    summary = f"""# Representation independent replication

The historical all-cell gate remains **FAIL**. The separate prospective
Representation Readiness Gate is **{'PASS' if decision['r_gate_passed'] else 'FAIL'}**.

- Independent official episodes: {len(rows)}
- Seeds / conditions: {len(seeds)} / {len(CONDITIONS)}
- Episode x seed cells: {len(gate_rows)}
- Every episode mean positive in both directions: {decision['checks']['every_episode_positive']}
- Every seed mean positive in both directions: {decision['checks']['every_seed_positive']}
- Both whole-episode bootstrap lower bounds positive: {decision['checks']['both_bootstrap_lower_bounds_positive']}
- Motor failures: {sum(not item['passed'] for item in motor_table)}
- Negative semantic cells: {len(negative_cells)}
"""
    (output / "summary.md").write_text(summary)
    print(json.dumps({"readiness_gate_passed": decision["r_gate_passed"], "negative_cells": len(negative_cells), "motor_failures": sum(not item["passed"] for item in motor_table)}))


if __name__ == "__main__":
    main()

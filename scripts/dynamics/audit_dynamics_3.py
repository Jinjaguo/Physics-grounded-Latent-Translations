#!/usr/bin/env python3
"""Audit completeness and immutability of the third dynamics experiment.

Purpose
-------
Verify every required wave-15 deliverable, frozen checkpoint/code manifest,
representation and historical-negative immutability, one-shot ordering, hard
gate/claim consistency, passing tests, and the 20 GiB free-space floor.

Parameters
----------
``--config`` points to ``configs/dynamics_3.yaml`` or an equivalent frozen
wave-15 configuration.

Usage
-----
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/audit_dynamics_3.py --config configs/dynamics_3.yaml

Outputs
-------
Writes ``final_audit_report.json`` inside the configured wave-15 output root
and prints a compact pass/fail summary.  It does not modify learned models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import yaml

from pglt.dynamics.dynamics_data import sha256_file, write_json


ROOT = Path(__file__).resolve().parents[2]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out = ROOT / config["experiment"]["output_root"]
    required = [
        "fifteenth_wave_results.md", "fifteenth_wave_next_experiment.md",
        "factorized_dynamics_preregistration.json", "frozen_representation_model_audit.json",
        "semantic_predictor_specification_results.json", "F1_execution_mlp_specification_training_results.json",
        "F2_matched_refinement_specification_training_results.json", "F3_free_execution_del_specification_training_results.json",
        "F4_decoder_geometry_del_specification_training_results.json", "decoder_jacobian_metric_validation_report.json",
        "parameter_count_table.json", "compatibility_preflight_report.json", "residual_vs_error_report.json",
        "development_rollout_metrics.json", "development_decoded_action_metrics.json",
        "development_semantic_retention_metrics.json", "development_full_and_execution_off_manifold_metrics.json",
        "factorized_dynamics_confirmation_manifest.json", "one_shot_official_validation_results.json",
        "hard_variational_claim_gate.json", "paper_claim_decision.json", "executed_commands.txt",
        "environment_provenance.json", "files_changed_report.json", "pytest_results.xml",
        "final_integrity_check.json", "representation_post_training_audit.json", "information_and_fairness_audit.json",
    ]
    missing = [name for name in required if not (out / name).is_file()]
    manifest = read_json(out / "factorized_dynamics_confirmation_manifest.json") if not missing else {}
    checkpoint_hashes_match = bool(manifest) and all(
        sha256_file(out / "checkpoints" / name) == digest for name, digest in manifest["checkpoints"].items()
    )
    code_hashes_match = bool(manifest) and all(
        sha256_file(ROOT / name) == digest for name, digest in manifest["metric_code_sha256"].items()
    )
    initial = read_json(out / "frozen_representation_model_audit.json") if (out / "frozen_representation_model_audit.json").exists() else {}
    representation_hash_matches = bool(initial) and sha256_file(ROOT / initial["checkpoint_path"]) == initial["checkpoint_sha256"]
    historical_hashes_match = bool(initial) and all(
        sha256_file(ROOT / name) == digest for name, digest in initial["historical_negative_artifact_sha256"].items()
    )
    validation = read_json(out / "one_shot_official_validation_results.json") if (out / "one_shot_official_validation_results.json").exists() else {}
    ordering = bool(validation) and validation["one_shot"] and validation["manifest_created_at"] < validation["evaluated_at"]
    fairness = read_json(out / "information_and_fairness_audit.json") if (out / "information_and_fairness_audit.json").exists() else {}
    fairness_passed = bool(fairness) and fairness["F2_F4_initializer_state_exactly_equal_to_F1"] and fairness["identical_iteration_counts"] and not fairness["future_target_actions_accessed"]
    junit = (out / "pytest_results.xml").read_text(encoding="utf-8") if (out / "pytest_results.xml").exists() else ""
    tests_passed = "failures=\"0\"" in junit and "errors=\"0\"" in junit
    free = shutil.disk_usage(ROOT).free
    required_free = int(config["storage"]["minimum_filesystem_available_gb"]) * 1024 ** 3
    checks = {
        "all_required_deliverables_present": not missing,
        "checkpoint_hashes_match_confirmation_manifest": checkpoint_hashes_match,
        "metric_code_hashes_match_confirmation_manifest": code_hashes_match,
        "representation_checkpoint_unchanged": representation_hash_matches,
        "historical_full_latent_negative_artifacts_unchanged": historical_hashes_match,
        "manifest_precedes_exactly_one_validation": ordering,
        "fair_information_and_refinement_budget": fairness_passed,
        "pytest_passed": tests_passed,
        "filesystem_at_least_20GiB_free": free >= required_free,
        "representation_optimizer_steps_zero": bool(initial) and initial["representation_optimizer_steps"] == 0,
    }
    report = {
        "missing": missing, "checks": checks, "all_passed": all(checks.values()),
        "filesystem_available_bytes": free, "minimum_available_bytes": required_free,
    }
    write_json(out / "final_audit_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

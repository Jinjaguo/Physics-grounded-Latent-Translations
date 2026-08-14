#!/usr/bin/env python3
"""Audit completeness and integrity of the wave-16 long-horizon stop branch.

Purpose
-------
Verify required deliverables, the failed data gate, zero unauthorized F1/F2
metric reads, carried checkpoint hashes, no overwritten prior artifacts,
passing tests, and the strict 20 GiB PGLT project-size ceiling.

Parameters
----------
``--config`` selects the frozen wave-16 YAML.

Usage
-----
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/audit_dynamics_4.py --config configs/dynamics_4.yaml

Outputs
-------
Writes ``final_audit_report.json`` in the wave-16 directory and exits nonzero
if any integrity condition fails.  It never loads a model for inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

import yaml

from pglt.dynamics.dynamics_data import sha256_file, write_json


ROOT = Path(__file__).resolve().parents[2]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out = ROOT / config["experiment"]["output_root"]
    required = [
        "sixteenth_wave_results.md", "sixteenth_wave_next_experiment.md",
        "long_trajectory_availability_audit.json", "collection_protocol.json",
        "data_adequacy_gate.json", "prospective_evaluation_preregistration.json",
        "frozen_model_hash_manifest.json", "long_latent_serialization_manifest.json",
        "horizon_sample_count_table.json", "F1_F2_horizon_wise_latent_metrics.json",
        "F1_F2_decoded_action_metrics.json", "F1_F2_semantic_retention_metrics.json",
        "F1_F2_off_manifold_metrics.json", "trajectory_level_paired_bootstrap.json",
        "rollout_error_growth_curves.json", "refinement_intermediate_state_table.json",
        "correction_vector_alignment_report.json", "local_tangent_normal_manifold_audit.json",
        "refinement_mechanism_decision.json", "frozen_DEL_negative_baseline_report.json",
        "final_paper_claim_decision.json", "executed_commands.txt", "environment_provenance.json",
        "files_changed_report.json", "pytest_results.xml", "project_storage_audit.json",
    ]
    missing = [name for name in required if not (out / name).is_file()]
    gate = read_json(out / "data_adequacy_gate.json") if (out / "data_adequacy_gate.json").exists() else {}
    availability = read_json(out / "long_trajectory_availability_audit.json") if (out / "long_trajectory_availability_audit.json").exists() else {}
    manifest = read_json(out / "frozen_model_hash_manifest.json") if (out / "frozen_model_hash_manifest.json").exists() else {}
    model_hashes = bool(manifest) and all(
        sha256_file(ROOT / item["path"]) == item["sha256"] for item in manifest["checkpoints"].values()
    )
    status_names = [
        "F1_F2_horizon_wise_latent_metrics.json", "F1_F2_decoded_action_metrics.json",
        "F1_F2_semantic_retention_metrics.json", "F1_F2_off_manifold_metrics.json",
        "trajectory_level_paired_bootstrap.json", "rollout_error_growth_curves.json",
        "refinement_intermediate_state_table.json", "correction_vector_alignment_report.json",
        "local_tangent_normal_manifold_audit.json",
    ]
    metrics_unread = all(
        read_json(out / name).get("F1_F2_metrics_read") is False for name in status_names if (out / name).exists()
    ) and all((out / name).exists() for name in status_names)
    xml = (out / "pytest_results.xml").read_text(encoding="utf-8") if (out / "pytest_results.xml").exists() else ""
    observed = int(subprocess.run(["du", "-sb", str(ROOT)], check=True, capture_output=True, text=True).stdout.split()[0])
    maximum = int(config["storage"]["maximum_project_gb"]) * 1024**3
    checks = {
        "required_deliverables_present": not missing,
        "availability_audit_found_zero_eligible": bool(availability) and availability["eligible_existing_segment_count"] == 0,
        "data_gate_failed_before_metrics": bool(gate) and gate["passed"] is False and gate["model_metrics_authorized"] is False,
        "all_F1_F2_metrics_unread": metrics_unread,
        "carried_checkpoint_hashes_unchanged": model_hashes,
        "model_representation_ema_updates_zero": bool(manifest) and manifest["model_parameter_updates"] == manifest["representation_updates"] == manifest["ema_updates"] == 0,
        "pytest_passed": "failures=\"0\"" in xml and "errors=\"0\"" in xml,
        "project_below_20GiB": observed < maximum,
        "prior_artifacts_not_overwritten": read_json(out / "files_changed_report.json").get("prior_dynamics_artifacts_overwritten") is False if (out / "files_changed_report.json").exists() else False,
    }
    report = {
        "missing": missing, "checks": checks, "project_bytes": observed,
        "maximum_project_bytes": maximum, "all_passed": all(checks.values()),
    }
    write_json(out / "final_audit_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

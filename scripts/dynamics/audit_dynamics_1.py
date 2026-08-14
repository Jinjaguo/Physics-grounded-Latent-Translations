#!/usr/bin/env python3
"""Audit the completed dynamics_1 experiment without recomputing metrics.

Purpose
-------
Verify required deliverables, frozen checkpoint/metric/representation hashes,
JUnit status, official-evaluation provenance, report identity, and the 20 GB
workspace limit after the complete thirteenth-wave experiment.

Parameters
----------
--config: the frozen dynamics_1 YAML containing output/report/storage paths.

Usage
-----
PYTHONPATH=src:third_party/LaWM \
  /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/audit_dynamics_1.py --config configs/dynamics_1.yaml

Outputs
-------
Writes ``final_integrity_check.json`` and a comprehensive
``files_changed_report.json`` inside the configured timestamped result root.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import yaml


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out = ROOT / config["experiment"]["output_root"]
    report = ROOT / config["experiment"]["report_path"]
    required = [
        "thirteenth_wave_results.md", "thirteenth_wave_next_experiment.md",
        "dynamics_dataset_audit.jsonl", "dynamics_dataset_audit_summary.json",
        "dynamics_split_preregistration.json", "frozen_latents.npz",
        "frozen_latent_serialization_manifest.json", "representation_checkpoint_hash_audit.json",
        "model_specs_and_parameter_counts.json", "solver_numerical_validation_report.json",
        "training_and_development_selection.json", "development_evaluation.json",
        "dynamics_confirmation_manifest.json", "held_out_dynamics_evaluation.json",
        "rollout_error_vs_horizon_tables.json", "semantic_retention_tables.json",
        "decoded_action_metrics.json", "off_manifold_metrics.json",
        "del_residual_stability_report.json", "causal_information_set_audit.json",
        "task_boundary_diagnostic.json", "oracle_future_action_diagnostic.json",
        "executed_commands.txt", "environment_provenance.json", "pytest_results.xml",
    ]
    missing = [name for name in required if not (out / name).is_file()]
    confirmation = read_json(out / "dynamics_confirmation_manifest.json")
    checkpoint_hashes_valid = all(
        sha256(out / "checkpoints" / name) == expected
        for name, expected in confirmation["checkpoint_sha256"].items()
    )
    metric_hashes_valid = all(
        sha256(ROOT / name) == expected
        for name, expected in confirmation["metric_code_sha256"].items()
    )
    representation = read_json(out / "representation_checkpoint_hash_audit.json")
    representation_hash_valid = sha256(ROOT / representation["selected_checkpoint"]["path"]) == representation["observed_sha256"]
    junit = ET.parse(out / "pytest_results.xml").getroot()
    suites = [junit] if junit.tag == "testsuite" else list(junit.findall("testsuite"))
    tests = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
    failures = sum(int(suite.attrib.get("failures", 0)) for suite in suites)
    errors = sum(int(suite.attrib.get("errors", 0)) for suite in suites)
    workspace_bytes = int(subprocess.run(["du", "-sb", str(ROOT)], check=True, capture_output=True, text=True).stdout.split()[0])
    maximum_bytes = int(config["storage"]["maximum_workspace_gb"]) * 1024 ** 3
    report_identity_valid = report.is_file() and "# dynamics_1 实验结果" in report.read_text(encoding="utf-8")
    files = [
        ROOT / "configs/dynamics_1.yaml",
        ROOT / "src/pglt/dynamics/dynamics_data.py",
        ROOT / "src/pglt/dynamics/variational.py",
        ROOT / "src/pglt/dynamics/runner.py",
        ROOT / "src/pglt/dynamics/frozen_latents.py",
        ROOT / "scripts/dynamics/run_dynamics_1.py",
        ROOT / "scripts/dynamics/audit_dynamics_1.py",
        ROOT / "tests/dynamics/test_dynamics_1_protocol.py",
        ROOT / "tests/dynamics/test_frozen_latent_interface.py",
        ROOT / "pyproject.toml",
        ROOT / "RESEARCH_LOG.md",
        ROOT / "NEXT_EXPERIMENT.md",
        report,
    ] + sorted(path for path in out.rglob("*") if path.is_file() and path.name not in {"files_changed_report.json", "final_integrity_check.json"})
    unique_files = sorted(set(files))
    write_json(out / "files_changed_report.json", {
        "created_or_modified_files": [
            {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in unique_files
        ]
    })
    checks = {
        "all_required_deliverables_present": not missing,
        "checkpoint_hashes_match_confirmation_manifest": checkpoint_hashes_valid,
        "metric_code_hashes_match_confirmation_manifest": metric_hashes_valid,
        "representation_checkpoint_hash_unchanged": representation_hash_valid,
        "representation_optimizer_steps_zero": representation["representation_optimizer_steps"] == 0,
        "representation_backward_calls_zero": representation["representation_backward_calls"] == 0,
        "representation_ema_updates_zero": representation["ema_updates"] == 0,
        "confirmation_frozen_before_official_metrics": confirmation["frozen_before_official_validation_metrics"],
        "oracle_excluded_from_primary": confirmation["oracle_excluded_from_primary"],
        "tests_passed": tests == 19 and failures == 0 and errors == 0,
        "report_identity_valid": report_identity_valid,
        "workspace_below_20gb": workspace_bytes < maximum_bytes,
    }
    payload = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "missing_deliverables": missing,
        "tests": {"tests": tests, "failures": failures, "errors": errors},
        "storage": {"workspace_bytes": workspace_bytes, "maximum_bytes": maximum_bytes, "remaining_bytes": maximum_bytes - workspace_bytes},
        "checks": checks,
        "complete": all(checks.values()),
    }
    write_json(out / "final_integrity_check.json", payload)
    print(json.dumps({"complete": payload["complete"], "missing": missing, "workspace_bytes": workspace_bytes}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit the completed dynamics_2 experiment without recomputing diagnostics.

Purpose
-------
Verify every fourteenth-wave deliverable, wave-13 frozen checkpoint hashes,
diagnostic settings/code/result hashes, exact residual reproduction, learned
parameter immutability, JUnit results, report identity, and at least 20 GiB of
remaining filesystem space.

Parameters
----------
--config: frozen dynamics_2 YAML containing source, result, report, and storage
requirements.

Usage
-----
PYTHONPATH=src:third_party/LaWM \
  /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/audit_dynamics_2.py --config configs/dynamics_2.yaml

Outputs
-------
Writes ``environment_provenance.json``, ``files_changed_report.json``, and
``final_integrity_check.json`` in the configured fourteenth-wave directory.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch
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
    wave13 = ROOT / config["experiment"]["wave13_root"]
    report = ROOT / config["experiment"]["report_path"]
    required = [
        "fourteenth_wave_results.md", "fourteenth_wave_next_experiment.md",
        "frozen_model_checkpoint_audit.json", "exact_del_residual_regression.json",
        "ground_truth_residual_compatibility_table.json", "residual_vs_error_analysis.json",
        "solver_budget_report.json", "robust_root_solver_report.json",
        "initialization_basin_report.json", "root_proximity_report.json",
        "jacobian_conditioning_report.json", "root_multiplicity_diagnostic.json",
        "unforced_vs_forced_failure_comparison.json", "matched_refinement_interpretation.json",
        "final_del_failure_mechanism.json", "longer_trajectory_next_experiment_decision.json",
        "diagnostic_settings_preregistration.json", "diagnostic_confirmation_manifest.json",
        "development_diagnostic_results.json", "validation_diagnostic_results.json",
        "validation_descriptive_replication.json", "development_diagnostic_raw_arrays.npz",
        "validation_diagnostic_raw_arrays.npz", "executed_commands.txt", "pytest_results.xml",
    ]
    missing = [name for name in required if not (out / name).is_file()]
    confirmation = read_json(out / "diagnostic_confirmation_manifest.json")
    wave13_confirmation = read_json(wave13 / "dynamics_confirmation_manifest.json")
    checkpoint_hashes_valid = all(
        sha256(wave13 / "checkpoints" / name) == expected
        for name, expected in wave13_confirmation["checkpoint_sha256"].items()
    )
    code_hashes_valid = all(
        sha256(ROOT / name) == expected
        for name, expected in confirmation["diagnostic_code_sha256"].items()
    )
    development_hash_valid = sha256(out / "development_diagnostic_results.json") == confirmation["development_results_sha256"]
    settings_hash_valid = sha256(out / "diagnostic_settings_preregistration.json") == confirmation["settings_sha256"]
    regression = read_json(out / "exact_del_residual_regression.json")
    exact_residual = all(value["exact_within_1e-7"] for value in regression.values())
    development = read_json(out / "development_diagnostic_results.json")
    validation = read_json(out / "validation_diagnostic_results.json")
    frozen = all(
        result["frozen_parameter_audit"]["all_model_hashes_unchanged"]
        and result["frozen_parameter_audit"]["representation_hash_unchanged"]
        and result["frozen_parameter_audit"]["learned_parameter_optimizer_steps"] == 0
        and result["frozen_parameter_audit"]["root_solver_optimized_only_q_next"]
        for result in (development, validation)
    )
    junit = ET.parse(out / "pytest_results.xml").getroot()
    suites = [junit] if junit.tag == "testsuite" else list(junit.findall("testsuite"))
    tests = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
    failures = sum(int(suite.attrib.get("failures", 0)) for suite in suites)
    errors = sum(int(suite.attrib.get("errors", 0)) for suite in suites)
    usage = shutil.disk_usage(ROOT)
    minimum = int(config["storage"]["minimum_filesystem_available_gb"]) * 1024 ** 3
    report_identity = report.is_file() and "# dynamics_2 实验结果" in report.read_text(encoding="utf-8")
    provenance = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "diagnostic_device": "cpu",
        "development_is_only_decision_source": True,
        "validation_role": confirmation["validation_role"],
        "filesystem": {"total_bytes": usage.total, "used_bytes": usage.used, "available_bytes": usage.free, "minimum_available_bytes": minimum},
    }
    write_json(out / "environment_provenance.json", provenance)
    changed = [
        ROOT / "configs/dynamics_2.yaml",
        ROOT / "src/pglt/dynamics/del_diagnostics.py",
        ROOT / "scripts/dynamics/run_dynamics_2.py",
        ROOT / "scripts/dynamics/audit_dynamics_2.py",
        ROOT / "tests/dynamics/test_dynamics_2_diagnostics.py",
        ROOT / "RESEARCH_LOG.md", ROOT / "NEXT_EXPERIMENT.md", report,
    ] + sorted(path for path in out.rglob("*") if path.is_file() and path.name not in {"files_changed_report.json", "final_integrity_check.json"})
    write_json(out / "files_changed_report.json", {
        "created_or_modified_files": [
            {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(set(changed))
        ]
    })
    checks = {
        "all_required_deliverables_present": not missing,
        "wave13_checkpoint_hashes_unchanged": checkpoint_hashes_valid,
        "diagnostic_code_hashes_match_frozen_manifest": code_hashes_valid,
        "development_result_hash_matches_manifest": development_hash_valid,
        "settings_hash_matches_manifest": settings_hash_valid,
        "settings_frozen_before_development": confirmation["settings_frozen_before_development"],
        "validation_is_descriptive_only": confirmation["validation_role"] == "descriptive replication only; not held-out and not used for settings",
        "exact_residual_reproduction_passed": exact_residual,
        "all_learned_models_and_representation_unchanged": frozen,
        "all_tests_passed": tests == 24 and failures == 0 and errors == 0,
        "report_identity_valid": report_identity,
        "available_space_at_least_20gb": usage.free >= minimum,
    }
    payload = {
        "created_at": provenance["created_at"],
        "missing_deliverables": missing,
        "tests": {"tests": tests, "failures": failures, "errors": errors},
        "storage": provenance["filesystem"],
        "checks": checks,
        "complete": all(checks.values()),
    }
    write_json(out / "final_integrity_check.json", payload)
    print(json.dumps({"complete": payload["complete"], "tests": tests, "available_bytes": usage.free, "missing": missing}))


if __name__ == "__main__":
    main()

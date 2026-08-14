#!/usr/bin/env python3
"""Audit the amended wave-16 public CALVIN H1/H2 replication.

Purpose
-------
Verify acquisition/selection counts, source and checkpoint hashes, frozen and
causal inference, H1/H2-only scope, whole-trajectory bootstrap, finite outputs,
tests, reports, research-log wording, and the 200-GB disk floor.

Parameters
----------
``--config`` selects the amended external-H1/H2 experiment YAML.

Usage
-----
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/audit_dynamics_4_external_h12.py \
  --config configs/dynamics_4_external_h12.yaml

Outputs
-------
Writes ``final_audit_report.json`` below the amended wave-16 result directory
and exits nonzero if any required integrity condition fails.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
from typing import Any

import yaml

from pglt.dynamics.dynamics_data import sha256_file, write_json


ROOT = Path(__file__).resolve().parents[2]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(nested) for nested in value.values())
    if isinstance(value, list):
        return all(finite(nested) for nested in value)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out = ROOT / config["experiment"]["output_root"]
    acquisition = ROOT / config["experiment"]["acquisition_root"]
    required = [
        "external_h12_prospective_preregistration.json", "external_frozen_latents.npz",
        "external_sequences.jsonl", "frozen_serialization_audit.json",
        "external_h12_rollout_metrics.json", "external_h12_decoded_action_metrics.json",
        "external_h12_off_manifold_metrics.json", "external_h12_per_task_metrics.json",
        "external_h12_trajectory_auc.json", "external_h12_paired_trajectory_bootstrap.json",
        "external_h12_correction_alignment.json", "external_h12_refinement_intermediate_states.json",
        "external_h12_sample_counts.json", "external_h12_replication_decision.json",
        "external_h12_freezing_and_causality_audit.json", "wave16_external_h12_claim_decision.json",
        "sixteenth_wave_results.md", "sixteenth_wave_next_experiment.md", "pytest_results.xml",
        "executed_commands.txt", "environment_provenance.json", "files_changed_report.json",
    ]
    missing = [name for name in required if not (out / name).is_file()]
    selected = read_json(acquisition / "selected_segments_manifest.json")
    prereg = read_json(out / "external_h12_prospective_preregistration.json")
    freezing = read_json(out / "external_h12_freezing_and_causality_audit.json")
    counts = read_json(out / "external_h12_sample_counts.json")
    bootstrap = read_json(out / "external_h12_paired_trajectory_bootstrap.json")
    metrics = read_json(out / "external_h12_rollout_metrics.json")
    checkpoints = read_json(acquisition / "frozen_checkpoint_manifest.json")
    report = (out / "sixteenth_wave_results.md").read_text(encoding="utf-8")
    log = (ROOT / "RESEARCH_LOG.md").read_text(encoding="utf-8")
    xml = (out / "pytest_results.xml").read_text(encoding="utf-8")
    checkpoint_hashes = all(
        sha256_file(ROOT / item["path"]) == item["sha256"] for item in checkpoints.values()
    )
    checks = {
        "required_deliverables_present": not missing,
        "exact_60_and_10_per_task": selected["total_segments"] == 60 and set(selected["per_task_counts"].values()) == {10},
        "selection_model_independent": selected["selection_uses_model_outputs"] is False,
        "preregistered_before_outputs": prereg["written_before_any_external_F1_F2_output"] is True,
        "checkpoint_hashes_unchanged": checkpoint_hashes,
        "all_models_frozen_and_causal": freezing["all_parameters_unchanged"] is True
        and freezing["future_target_actions_used_as_model_input"] is False
        and all(freezing[key] == 0 for key in (
            "representation_optimizer_steps", "representation_backward_calls",
            "F1_optimizer_steps", "F1_backward_calls", "F2_optimizer_steps",
            "F2_backward_calls", "EMA_updates",
        )),
        "H1_H2_only": counts["H1_rollout_starts"] == 120 and counts["H2_rollout_starts"] == 60
        and counts["H4_rollout_starts"] == counts["H8_rollout_starts"] == 0
        and counts["H4_H8_run"] is False,
        "whole_trajectory_bootstrap_10000": bootstrap["pooled"]["trajectory_count"] == 60
        and bootstrap["pooled"]["bootstrap_replicates"] == 10000
        and bootstrap["window_bootstrap"] is False,
        "all_metrics_finite": finite(metrics) and finite(bootstrap),
        "tests_passed": "failures=\"0\"" in xml and "errors=\"0\"" in xml,
        "report_and_log_scope_explicit": "H1 and H2 only" in report
        and "H4 and H8 were not run" in report
        and "H1 and H2 only" in log
        and "H4 and H8 were not run" in log,
        "disk_floor_passed": shutil.disk_usage(ROOT).free >= int(config["storage"]["minimum_free_bytes"]),
    }
    payload = {
        "missing": missing, "checks": checks, "all_passed": all(checks.values()),
        "free_bytes": shutil.disk_usage(ROOT).free,
        "minimum_free_bytes": config["storage"]["minimum_free_bytes"],
    }
    write_json(out / "final_audit_report.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

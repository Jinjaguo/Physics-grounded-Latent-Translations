#!/usr/bin/env python3
"""Audit the paper-release representation package.

Purpose
-------
Verify the single functional config, all 31 compact official training episodes,
18 frozen checkpoint hashes, 18 independent-inference records, read-only
invariants, final readiness decision, paper source, and test report.

Parameters
----------
--config: released representation YAML; --pytest-xml: optional JUnit XML;
--checkpoint-root/--inference-root/--decision-root: optional overrides.

Usage
-----
PYTHONPATH=src python scripts/representation/audit.py \
  --config configs/representation.yaml \
  --pytest-xml results/representation/representation_tests.xml

Outputs
-------
Writes ``release_integrity.json`` under the selected decision root; by default
this is ``results/representation/independent_replication``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml


def sha256(path: Path) -> str:
    """Compute a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: object) -> bool:
    """Return whether a nested JSON-compatible value contains no NaN/Inf."""

    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pytest-xml", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--inference-root", type=Path)
    parser.add_argument("--decision-root", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    compact_root = Path(config["source"]["compact_root"]) / "training"
    metadata_root = Path(config["source"]["metadata_root"]) / "training"
    bounds = np.load(metadata_root / "ep_start_end_ids.npy").reshape(-1, 2)
    bad_episodes = []
    for row, (first, last) in enumerate(bounds):
        path = compact_root / f"episode_row_{row:03d}.npz"
        sidecar = path.with_suffix(".json")
        if not path.exists() or not sidecar.exists():
            bad_episodes.append({"row": row, "reason": "missing"})
            continue
        with np.load(path) as archive:
            valid = (
                set(archive.files) == {"rel_actions", "global_frame_indices"}
                and archive["rel_actions"].shape == (int(last - first + 1), 7)
                and archive["rel_actions"].dtype == np.float64
                and np.array_equal(archive["global_frame_indices"], np.arange(first, last + 1))
            )
        if not valid:
            bad_episodes.append({"row": row, "reason": "schema"})
    checkpoint_root = args.checkpoint_root or Path(config["release"]["checkpoint_root"])
    manifest = json.loads((checkpoint_root / "manifest.json").read_text())
    checkpoint_hashes = all(
        sha256(checkpoint_root / f"seed_{item['seed_base']}" / item["condition"] / "checkpoint_ema.pt")
        == item["sha256"]
        for item in manifest["checkpoints"]
    )
    inference_root = args.inference_root or Path(config["release"]["inference_root"])
    records = []
    missing_records = []
    rows = [int(row) for row in config["selection"]["independent_replication_episode_rows"]]
    for seed in config["replication"]["seeds"]:
        for condition in config["replication"]["conditions"]:
            path = inference_root / f"seed_{seed}" / condition / "metrics.json"
            if path.exists():
                records.append(json.loads(path.read_text()))
            else:
                missing_records.append(str(path))
    read_only = all(
        record["parameter_tensors_unchanged"]
        and record["optimizer_steps"] == record["ema_updates"] == record["backward_calls"] == 0
        and record["independent_rows_excluded_from_normalization"]
        for record in records
    )
    candidate_exact = all(
        record["episode_metrics"][str(query)]["candidate_episode_rows"]
        == [row for row in rows if row != query]
        for record in records
        for query in rows
    )
    results_root = Path(config["release"]["results_root"])
    decision_root = args.decision_root or (results_root / "independent_replication")
    decision = json.loads((decision_root / "readiness_gate_decision.json").read_text())
    status = json.loads((decision_root / "representation_status.json").read_text())
    tests = None
    if args.pytest_xml:
        xml = ET.parse(args.pytest_xml).getroot()
        suite = xml if xml.tag == "testsuite" else xml.find("testsuite")
        tests = {key: int(suite.attrib[key]) for key in ("tests", "failures", "errors")}
    complete = (
        len(bounds) == 31
        and not bad_episodes
        and len(manifest["checkpoints"]) == 18
        and checkpoint_hashes
        and len(records) == 18
        and not missing_records
        and read_only
        and candidate_exact
        and all(finite(record) for record in records)
        and decision["r_gate_passed"]
        and status["representation_ready"]
        and Path("paper/representation_iclr2026.tex").exists()
        and (tests is None or tests["failures"] == tests["errors"] == 0)
    )
    payload = {
        "config": str(args.config),
        "config_sha256": sha256(args.config),
        "compact_official_episodes": len(bounds) - len(bad_episodes),
        "bad_episodes": bad_episodes,
        "checkpoint_count": len(manifest["checkpoints"]),
        "checkpoint_hashes_valid": checkpoint_hashes,
        "inference_record_count": len(records),
        "missing_inference_records": missing_records,
        "read_only_inference": read_only,
        "candidate_query_exclusion_exact": candidate_exact,
        "readiness_gate_passed": decision["r_gate_passed"],
        "representation_ready": status["representation_ready"],
        "tests": tests,
        "complete": complete,
    }
    audit_output = decision_root / "release_integrity.json"
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()

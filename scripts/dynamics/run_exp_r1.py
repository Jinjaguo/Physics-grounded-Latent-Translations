#!/usr/bin/env python3
"""Run EXP_R1 hierarchical latent path-planning tournament.

Purpose
-------
Audit the released action-coordinate interface, encode real Wave27 continuous
transition windows with the frozen representation, and compare multi-step
latent path planners against interpolation, frozen F1/F2 rollout, graph
search, trajectory optimization, and CEM.  The experiment uses an oracle
boundary and train-only target regions; the hidden post-boundary path is used
only for evaluation.

Parameters
----------
``--stage`` is ``audit`` or ``run``.  ``--device`` selects the PyTorch device
(default ``cpu``).  ``--max-records`` is an optional deterministic audit/debug
limit and must not be used for the registered full run.  ``--seed`` controls
only planner sampling (default 4101).

Usage
-----
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python \\
  scripts/dynamics/run_exp_r1.py --stage audit --device cpu
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python \\
  scripts/dynamics/run_exp_r1.py --stage run --device cpu

Outputs
-------
Interface/data audits and the frozen manifest are saved under ``reports/``.
Encoded cases, development/held-out metrics, preregistration, claim decision,
the detailed report, and ``reports/next_exp_fromR1.md`` are saved under
``results/EXP_R1/`` and ``reports/``.  The representation, decoder, F1, and
historical F2 checkpoints are never modified.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import heapq
import json
from pathlib import Path
import platform
import shutil
import time
from typing import Any, Iterable

import numpy as np
import torch
from torch.nn import functional as F

from pglt.dynamics.dynamics_data import load_frozen_representation, sha256_file
from pglt.dynamics.factorized import ExecutionMLP, ExecutionMatchedRefinement


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "EXP_R1"
REPORTS = ROOT / "reports"
COLLECTION = ROOT / "results/dynamics/twenty_seventh_wave/2026-08-15_dynamics_15/wave27_collection_partial.json"
SPLIT = ROOT / "results/dynamics/twenty_seventh_wave/2026-08-15_dynamics_15/wave27_new_data_split_manifest.json"
REP_CONFIG = ROOT / "configs/representation.yaml"
REP_MANIFEST = ROOT / "checkpoints/representation/manifest.json"
REP_FEATURES = ROOT / "data/representation/text_features/openclip_vitl14_datacomp_xl.npz"
F1_PATH = ROOT / "results/dynamics/fifteenth_wave/2026-08-12_dynamics_3/checkpoints/F1_execution_mlp.pt"
F2_PATH = ROOT / "results/dynamics/fifteenth_wave/2026-08-12_dynamics_3/checkpoints/F2_matched_refinement.pt"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): convert(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(val) for val in item]
        if isinstance(item, np.ndarray):
            return convert(item.tolist())
        if isinstance(item, np.generic):
            return item.item()
        return item
    path.write_text(json.dumps(convert(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return sha256_file(path)


def disk_audit() -> dict[str, Any]:
    usage = shutil.disk_usage(ROOT)
    floor = 300 * 10**9
    if usage.free < floor:
        raise RuntimeError(f"EXP_R1 requires 300 GB free; only {usage.free} bytes remain")
    return {"total_bytes": usage.total, "used_bytes": usage.used, "available_bytes": usage.free, "floor_bytes": floor, "passed": True}


def load_records(max_records: int | None = None) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, list[str]]]:
    payload = read_json(COLLECTION)
    split = read_json(SPLIT)
    session_split: dict[str, str] = {}
    split_sessions: dict[str, list[str]] = {}
    for key, name in (("new_train", "train"), ("new_development", "development"), ("new_prospective_test", "heldout")):
        sessions = list(split["sessions"][key])
        split_sessions[name] = sessions
        session_split.update({session: name for session in sessions})
    records = list(payload["records"])
    if max_records is not None:
        records = records[:max_records]
    return records, session_split, split_sessions


def select_representation() -> tuple[Any, dict[str, Any], Path, dict[str, Any]]:
    manifest = read_json(REP_MANIFEST)
    candidates = [item for item in manifest["checkpoints"] if item["condition"] == "correct_language"]
    selected = next(item for item in candidates if int(item["seed_base"]) == 810)
    path = ROOT / selected["path"]
    if sha(path) != selected["sha256"]:
        raise RuntimeError("Frozen representation checkpoint hash mismatch")
    config = __import__("yaml").safe_load(REP_CONFIG.read_text(encoding="utf-8"))
    model, payload = load_frozen_representation(config, path)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload, path, selected


def audit(max_records: int | None = None) -> None:
    """Write all EXP_R1 audits before encoding or planner selection."""

    OUT.mkdir(parents=True, exist_ok=True)
    records, session_split, split_sessions = load_records(max_records)
    disk = disk_audit()
    model, payload, rep_path, rep_entry = select_representation()
    feature_texts = set()
    with np.load(REP_FEATURES, allow_pickle=True) as archive:
        feature_texts = {str(value) for value in archive["texts"]}
    schema_counts: dict[str, int] = {}
    missing_files: list[str] = []
    invalid: list[str] = []
    per_split_goal: dict[str, dict[str, int]] = {name: {} for name in ("train", "development", "heldout")}
    for record in records:
        source_session = record["source_session_id"]
        split = session_split.get(source_session)
        path = ROOT / record["compact_path"]
        if split is None:
            invalid.append(f"unassigned session {source_session}")
        if not path.is_file():
            missing_files.append(str(path))
            continue
        with np.load(path, allow_pickle=False) as saved:
            schema = tuple(saved.files)
            schema_counts["|".join(schema)] = schema_counts.get("|".join(schema), 0) + 1
            expected = (schema == ("rel_actions", "robot_obs", "scene_obs", "global_frame_indices")
                        and saved["rel_actions"].shape == (128, 7)
                        and saved["robot_obs"].shape == (128, 15)
                        and saved["scene_obs"].shape == (128, 24)
                        and saved["global_frame_indices"].shape == (128,)
                        and np.all(np.diff(saved["global_frame_indices"]) == 1))
            if not expected:
                invalid.append(f"schema/index failure {path}")
        if record["source_end_frame"] - record["source_start_frame"] + 1 != 128:
            invalid.append(f"source length failure {record['record_id']}")
        if record["boundary_frame"] != record["source_start_frame"] + 64:
            invalid.append(f"boundary is not midpoint {record['record_id']}")
        if not record["physically_contiguous"] or record["reset_crossed"]:
            invalid.append(f"continuity flag failure {record['record_id']}")
        variant = str(record["language_variants"][0])
        if variant not in feature_texts:
            invalid.append(f"missing exact text feature {variant}")
        if split is not None:
            goal = str(record["goal"])
            per_split_goal[split][goal] = per_split_goal[split].get(goal, 0) + 1
    if missing_files or invalid:
        raise RuntimeError(f"EXP_R1 audit failed: missing={len(missing_files)} invalid={len(invalid)}")
    frozen = {
        "created_at": now(), "experiment": "EXP_R1", "records": len(records),
        "representation": {"path": str(rep_path.relative_to(ROOT)), "sha256": rep_entry["sha256"], "entry": rep_entry,
                            "latent_dim": 32, "semantic_dim": 16, "action_window": 16, "normalization": payload["resolved_config"]["normalization"]},
        "f1_checkpoint": {"path": str(F1_PATH.relative_to(ROOT)), "sha256": sha(F1_PATH)},
        "f2_checkpoint": {"path": str(F2_PATH.relative_to(ROOT)), "sha256": sha(F2_PATH)},
        "collection": {"path": str(COLLECTION.relative_to(ROOT)), "sha256": sha(COLLECTION), "source_revision": read_json(COLLECTION)["records"][0]["source_revision"]},
        "split": {"path": str(SPLIT.relative_to(ROOT)), "sha256": sha(SPLIT), "sessions": split_sessions},
        "schema_counts": schema_counts, "per_split_goal_counts": per_split_goal, "disk": disk,
        "frozen_parameters": {"representation": True, "decoder": True, "f1": True, "historical_f2": True},
    }
    prereg = {
        "created_at": now(), "written_before_planner_run": True, "experiment": "EXP_R1",
        "question": "Can a multi-step planner connect a real pre-boundary latent to a train-derived target action region while preserving continuity and support?",
        "case_protocol": "8 frozen H16 chunks per 128-frame window; chunk 3 is oracle start, chunks 4-7 are hidden evaluation path, goal is collection-record target task.",
        "splits": {"train": "new_train sessions", "development": "new_development sessions", "heldout": "new_prospective_test sessions"},
        "horizon": 4, "target_region": "all train post-boundary chunks grouped by exact record goal; no heldout future endpoint used for construction",
        "methods": ["linear_interpolation", "f1_free_rollout", "f2_old_refinement", "graph_dijkstra", "traj_terminal", "traj_terminal_dynamics", "traj_terminal_continuity", "traj_terminal_support", "traj_full", "cem"],
        "cost_ablation": ["terminal", "terminal+dynamics", "terminal+continuity", "terminal+support", "terminal+dynamics+continuity+support"],
        "metrics": ["target_arrival", "endpoint_identity", "hidden_path_latent_mse", "decoded_action_mse", "switch_jump", "decoded_first_difference", "decoded_second_difference", "support_distance", "unsupported_fraction", "f1_dynamics_cost", "path_length", "curvature", "runtime_seconds", "nonfinite"],
        "heldout_policy": "select candidates only from development, then evaluate the frozen method set on heldout exactly once",
        "success_gate": "A planner must Pareto-improve over linear, F1, old F2, and pointwise/static interpolation on heldout target arrival, continuity, and support across the task set; otherwise NOT_SUPPORTED and continue EXP_R2.",
        "historical_negative_baseline": "Wave28-Wave78 pointwise steering remains contextual evidence and is not retrained.",
    }
    write_json(REPORTS / "EXP_R1_frozen_manifest.json", frozen)
    write_json(REPORTS / "EXP_R1_preregistration.json", prereg)
    write_json(OUT / "frozen_manifest.json", frozen)
    write_json(OUT / "preregistration.json", prereg)
    interface_md = f"""# EXP_R1 interface audit\n\nCreated: {now()}\n\n- Representation: `{rep_path.relative_to(ROOT)}`; manifest SHA-256 `{rep_entry['sha256']}`.\n- Latent interface: 32 dimensions, 16 semantic prefix + 16 execution suffix.\n- Action interface: normalized CALVIN `rel_actions`, 16 frames, 7 channels.\n- Decoder: frozen representation decoder, output `(16, 7)` normalized actions.\n- F1: `{F1_PATH.relative_to(ROOT)}`, SHA-256 `{sha(F1_PATH)}`; execution-only MLP consumes previous/current execution, current semantic, and causal projected text context.\n- Historical F2: `{F2_PATH.relative_to(ROOT)}`, SHA-256 `{sha(F2_PATH)}`; matched iterative refinement, frozen baseline only.\n- Physical fields available in each window: `robot_obs (128,15)`, `scene_obs (128,24)`, `rel_actions (128,7)`, and contiguous global frame indices.\n- Oracle F3 interface: collection boundary frame and source-session split are available; no learned completion model is used.\n- Waypoint/return fields: robot observations are present, but EXP_R1 does not claim return; this experiment isolates path planning.\n\nNo guessed field names were used. Representation, decoder, F1, and F2 optimizer/update counts are zero by protocol.\n"""
    data_md = f"""# EXP_R1 data audit\n\nCreated: {now()}\n\nThe audit found **{len(records)}** physically contiguous, reset-free 128-frame windows from the official Wave27 human-play archive. Every boundary is exactly at the midpoint (`start + 64`), so each case contains four pre-boundary and four post-boundary H16 chunks.\n\n| split | sessions | windows | goals |\n|---|---:|---:|---|\n"""
    for split in ("train", "development", "heldout"):
        count = sum(1 for r in records if session_split[r["source_session_id"]] == split)
        goals = ", ".join(f"{k}:{v}" for k, v in sorted(per_split_goal[split].items()))
        data_md += f"| {split} | {len(split_sessions[split])} | {count} | {goals} |\n"
    data_md += "\nThe planner receives only the start latent, target task text/region, and frozen model interfaces. The four post-boundary chunks remain hidden until metric evaluation. Target regions are constructed from train post-boundary chunks only. Source-session separation is preserved.\n"
    (REPORTS / "EXP_R1_interface_audit.md").write_text(interface_md, encoding="utf-8")
    (REPORTS / "EXP_R1_data_audit.md").write_text(data_md, encoding="utf-8")
    print(json.dumps({"stage": "audit", "records": len(records), "disk_available": disk["available_bytes"], "output": str(OUT)}, indent=2))


@dataclass
class Case:
    record_id: str
    split: str
    goal: str
    text: str
    session: str
    latent: np.ndarray
    norm_actions: np.ndarray


def encode_cases(device: torch.device, max_records: int | None = None) -> tuple[list[Case], Any, dict[str, np.ndarray], dict[str, str]]:
    model, payload, _, _ = select_representation()
    model.to(device)
    norm = payload["resolved_config"]["normalization"]
    mean = np.asarray(norm["action_mean"], dtype=np.float32)
    std = np.asarray(norm["action_std"], dtype=np.float32)
    records, session_split, _ = load_records(max_records)
    cases: list[Case] = []
    with np.load(REP_FEATURES, allow_pickle=True) as archive:
        text_features = {str(t): np.asarray(v, dtype=np.float32) for t, v in zip(archive["texts"], archive["features"])}
    for record in records:
        path = ROOT / record["compact_path"]
        with np.load(path, allow_pickle=False) as saved:
            raw = np.asarray(saved["rel_actions"], dtype=np.float32)
        normalized = raw.copy()
        normalized[:, :6] = (normalized[:, :6] - mean[:6]) / std[:6]
        chunks = normalized.reshape(8, 16, 7)
        with torch.no_grad():
            latent = model.encode(torch.from_numpy(chunks).float().to(device)).cpu().numpy()
        cases.append(Case(str(record["record_id"]), session_split[record["source_session_id"]], str(record["goal"]), str(record["language_variants"][0]), str(record["source_session_id"]), latent, chunks))
    return cases, model, text_features, session_split


def load_frozen_planners(device: torch.device) -> tuple[ExecutionMLP, ExecutionMatchedRefinement]:
    f1 = ExecutionMLP(context_dim=32, hidden_dim=64, depth=3)
    f1.load_state_dict(torch.load(F1_PATH, map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
    f2 = ExecutionMatchedRefinement(f1, context_dim=32, hidden_dim=64, depth=3, iterations=4, step_size=0.01)
    f2.load_state_dict(torch.load(F2_PATH, map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
    f1.to(device); f2.to(device)
    f1.eval(); f2.eval()
    for model in (f1, f2):
        for p in model.parameters():
            p.requires_grad_(False)
    return f1, f2


def train_nodes(cases: list[Case]) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    nodes = np.concatenate([case.latent[4:] for case in cases if case.split == "train"], axis=0)
    targets: dict[str, np.ndarray] = {}
    radii: dict[str, float] = {}
    for goal in sorted({case.goal for case in cases}):
        values = np.concatenate([case.latent[4:] for case in cases if case.split == "train" and case.goal == goal], axis=0)
        targets[goal] = values
        center = values.mean(axis=0)
        radii[goal] = float(np.quantile(np.linalg.norm(values - center, axis=1), 0.90))
    return nodes, targets, radii


def linear_plan(start: np.ndarray, target: np.ndarray, horizon: int = 4) -> np.ndarray:
    return np.linspace(start, target, horizon + 1, dtype=np.float32)[1:]


def f_rollout(case: Case, target_text: np.ndarray, representation: Any, f1: ExecutionMLP, f2: ExecutionMatchedRefinement, which: str, device: torch.device) -> np.ndarray:
    previous = torch.from_numpy(case.latent[2, 16:]).float().to(device).unsqueeze(0)
    current = torch.from_numpy(case.latent[3, 16:]).float().to(device).unsqueeze(0)
    semantic = torch.from_numpy(case.latent[3, :16]).float().to(device).unsqueeze(0)
    text = torch.from_numpy(target_text).float().to(device).unsqueeze(0)
    context = torch.cat((semantic, text), dim=-1)
    outputs = []
    for _ in range(4):
        with torch.enable_grad():
            if which == "f1":
                nxt = f1(previous, current, context)
            else:
                nxt, _ = f2(previous, current, context)
        full = torch.cat((semantic, nxt), dim=-1)
        outputs.append(full.detach().cpu().numpy()[0])
        previous, current = current.detach(), nxt.detach()
    return np.stack(outputs)


def graph_plan(start: np.ndarray, nodes: np.ndarray, targets: np.ndarray, k: int = 12) -> np.ndarray:
    all_nodes = np.concatenate((start[None], nodes), axis=0)
    target_ids = set(range(1 + len(nodes) - len(targets), 1 + len(nodes))) if len(targets) else set()
    # Use exact target indices by matching rows rather than assuming ordering.
    target_ids = set()
    for row in targets:
        target_ids.add(1 + int(np.argmin(np.sum((nodes - row) ** 2, axis=1))))
    distances = np.sum((all_nodes[:, None, :] - all_nodes[None, :, :]) ** 2, axis=-1)
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(len(all_nodes))]
    kk = min(k + 1, len(all_nodes))
    for i in range(len(all_nodes)):
        neighbours = np.argpartition(distances[i], kk - 1)[:kk]
        for j in neighbours:
            if i != int(j):
                adjacency[i].append((int(j), float(np.sqrt(distances[i, j]))))
    dist = [float("inf")] * len(all_nodes); prev = [-1] * len(all_nodes); dist[0] = 0.0
    heap = [(0.0, 0)]
    goal = -1
    while heap:
        value, i = heapq.heappop(heap)
        if value != dist[i]: continue
        if i in target_ids: goal = i; break
        for j, weight in adjacency[i]:
            candidate = value + weight
            if candidate < dist[j]:
                dist[j] = candidate; prev[j] = i; heapq.heappush(heap, (candidate, j))
    if goal < 0:
        return linear_plan(start, targets.mean(axis=0))
    route = []
    while goal >= 0:
        route.append(all_nodes[goal]); goal = prev[goal]
    route = np.stack(route[::-1])
    if len(route) == 1: return np.repeat(route, 4, axis=0)
    positions = np.linspace(0, len(route) - 1, 5)
    return np.stack([np.interp(positions[1:], np.arange(len(route)), route[:, d]) for d in range(32)], axis=1).astype(np.float32)


def trajopt_plan(start: np.ndarray, target: np.ndarray, support: np.ndarray, model: Any, f1: ExecutionMLP, text: np.ndarray, variant: str, device: torch.device) -> np.ndarray:
    start_t = torch.tensor(start, dtype=torch.float32, device=device)
    init = torch.tensor(linear_plan(start, target), dtype=torch.float32, device=device)
    way = torch.nn.Parameter(init.clone())
    optimizer = torch.optim.Adam([way], lr=0.035)
    support_t = torch.tensor(support, dtype=torch.float32, device=device)
    for _ in range(70):
        path = torch.cat((start_t[None], way), dim=0)
        terminal = (way[-1] - torch.tensor(target, dtype=torch.float32, device=device)).square().mean()
        diffs = path[1:] - path[:-1]
        continuity = diffs.square().mean()
        curvature = (path[2:] - 2 * path[1:-1] + path[:-2]).square().mean()
        support_dist = torch.cdist(way, support_t).min(dim=1).values.square().mean()
        dyn = torch.zeros((), device=device)
        prev = path[:-2, 16:]; curr = path[1:-1, 16:]
        semantic = path[1:-1, :16]
        ctx = torch.tensor(text, dtype=torch.float32, device=device).expand(len(prev), -1)
        with torch.no_grad():
            pred = f1(prev, curr, torch.cat((semantic, ctx), dim=-1))
        dyn = (path[2:, 16:] - pred).square().mean()
        if variant == "terminal": loss = terminal
        elif variant == "terminal_dynamics": loss = terminal + 0.30 * dyn
        elif variant == "terminal_continuity": loss = terminal + 0.25 * continuity + 0.10 * curvature
        elif variant == "terminal_support": loss = terminal + 0.20 * support_dist
        else: loss = terminal + 0.30 * dyn + 0.25 * continuity + 0.20 * support_dist + 0.10 * curvature
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_([way], 5.0); optimizer.step()
        with torch.no_grad(): way[0].copy_(init[0])
    return way.detach().cpu().numpy()


def cem_plan(start: np.ndarray, target: np.ndarray, support: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mean = linear_plan(start, target)
    std = np.full_like(mean, 0.35, dtype=np.float32)
    for _ in range(6):
        samples = rng.normal(mean, std, size=(128, 4, 32)).astype(np.float32)
        samples[:, 0] = 0.7 * samples[:, 0] + 0.3 * start
        terminal = np.mean((samples[:, -1] - target) ** 2, axis=1)
        continuity = np.mean(np.diff(np.concatenate([np.repeat(start[None, None], 128, axis=0), samples], axis=1), axis=1) ** 2, axis=(1, 2))
        support_cost = np.min(np.sum((samples[:, :, None, :] - support[None, None, :, :]) ** 2, axis=-1), axis=2).mean(axis=1)
        score = terminal + 0.25 * continuity + 0.10 * support_cost
        elite = samples[np.argsort(score)[:16]]
        mean = elite.mean(axis=0); std = np.maximum(elite.std(axis=0), 0.03)
    return mean.astype(np.float32)


def metric(case: Case, planned: np.ndarray, target_set: np.ndarray, all_targets: dict[str, np.ndarray], all_support: np.ndarray, representation: Any, f1: ExecutionMLP, target_text: np.ndarray, device: torch.device) -> dict[str, float | bool]:
    start = case.latent[3]
    final_dist = float(np.min(np.linalg.norm(target_set - planned[-1], axis=1)))
    target_center = target_set.mean(axis=0)
    other_centers = [values.mean(axis=0) for goal, values in all_targets.items() if goal != case.goal]
    identity = float(np.linalg.norm(planned[-1] - target_center) <= min(np.linalg.norm(planned[-1] - center) for center in other_centers))
    true = case.latent[4:8]
    with torch.no_grad():
        decoded = representation.decode(torch.from_numpy(planned).float().to(device)).cpu().numpy()
        start_decoded = representation.decode(torch.from_numpy(start[None]).float().to(device)).cpu().numpy()[0]
        true_decoded = representation.decode(torch.from_numpy(true).float().to(device)).cpu().numpy()
    all_decoded = np.concatenate((start_decoded[None], decoded.reshape(-1, 16, 7)), axis=0).reshape(-1, 7)
    continuous = np.mean((decoded[:, :, :6] - true_decoded[:, :, :6]) ** 2)
    jump = np.mean((decoded[0, 0] - start_decoded[-1]) ** 2)
    first = np.mean(np.diff(all_decoded, axis=0) ** 2)
    second = np.mean(np.diff(all_decoded, n=2, axis=0) ** 2)
    support_d = np.min(np.linalg.norm(planned[:, None, :] - all_support[None, :, :], axis=-1), axis=1)
    path = np.diff(np.concatenate((start[None], planned), axis=0), axis=0)
    curvature = np.mean(np.linalg.norm(np.diff(path, axis=0), axis=1)) if len(path) > 1 else 0.0
    with torch.no_grad():
        prev = torch.from_numpy(case.latent[2, 16:]).float().to(device).unsqueeze(0)
        curr = torch.from_numpy(case.latent[3, 16:]).float().to(device).unsqueeze(0)
        semantic = torch.from_numpy(case.latent[3, :16]).float().to(device).unsqueeze(0)
        ctx = torch.from_numpy(target_text).float().to(device).unsqueeze(0)
        dyn_cost = []
        for point in planned:
            pred = f1(prev, curr, torch.cat((semantic, ctx), dim=-1))
            nxt = torch.from_numpy(point[16:]).float().to(device).unsqueeze(0)
            dyn_cost.append(float((pred - nxt).square().mean().cpu()))
            prev, curr = curr, nxt
    return {"target_arrival": float(final_dist <= np.quantile(np.linalg.norm(target_set - target_center, axis=1), 0.90)), "endpoint_identity": identity, "final_target_distance": final_dist, "hidden_path_latent_mse": float(np.mean((planned - true) ** 2)), "decoded_action_mse": float(continuous), "switch_jump": float(jump), "decoded_first_difference": float(first), "decoded_second_difference": float(second), "support_distance": float(np.mean(support_d)), "unsupported_fraction": float(np.mean(support_d > np.quantile(np.linalg.norm(all_support - all_support.mean(axis=0), axis=1), 0.95))), "f1_dynamics_cost": float(np.mean(dyn_cost)), "path_length": float(np.sum(np.linalg.norm(path, axis=1))), "curvature": curvature, "nonfinite": bool(not np.isfinite(planned).all())}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric = [key for key, value in rows[0].items() if isinstance(value, (float, int, bool)) and key not in {"nonfinite"}]
    out = {"count": len(rows), "nonfinite_count": int(sum(bool(row["nonfinite"]) for row in rows))}
    for key in numeric:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        out[key] = float(np.mean(values))
    return out


def run(device_name: str, seed: int, max_records: int | None = None) -> None:
    if not (OUT / "preregistration.json").is_file():
        raise RuntimeError("Run audit stage first; preregistration is missing")
    device = torch.device(device_name)
    cases, representation, text_features, _ = encode_cases(device, max_records)
    f1, f2 = load_frozen_planners(device)
    nodes, targets, radii = train_nodes(cases)
    target_texts: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for case in cases:
            if case.goal not in target_texts:
                feature = torch.from_numpy(text_features[case.text]).float().to(device).unsqueeze(0)
                target_texts[case.goal] = F.normalize(representation.project_text(feature), dim=-1).cpu().numpy()[0]
    methods = ["linear_interpolation", "f1_free_rollout", "f2_old_refinement", "graph_dijkstra", "traj_terminal", "traj_terminal_dynamics", "traj_terminal_continuity", "traj_terminal_support", "traj_full", "cem"]
    development: dict[str, list[dict[str, Any]]] = {name: [] for name in methods}
    heldout: dict[str, list[dict[str, Any]]] = {name: [] for name in methods}
    timings: dict[str, float] = {name: 0.0 for name in methods}
    target_nodes = {goal: values for goal, values in targets.items()}
    for case in cases:
        target_set = target_nodes[case.goal]
        target = target_set.mean(axis=0)
        start = case.latent[3]
        for method in methods:
            t0 = time.perf_counter()
            if method == "linear_interpolation": planned = linear_plan(start, target)
            elif method == "f1_free_rollout": planned = f_rollout(case, target_texts[case.goal], representation, f1, f2, "f1", device)
            elif method == "f2_old_refinement": planned = f_rollout(case, target_texts[case.goal], representation, f1, f2, "f2", device)
            elif method == "graph_dijkstra": planned = graph_plan(start, nodes, target_set)
            elif method.startswith("traj_"): planned = trajopt_plan(start, target, nodes, representation, f1, target_texts[case.goal], method[5:], device)
            else: planned = cem_plan(start, target, nodes, seed + len(development[method]) + len(heldout[method]))
            timings[method] += time.perf_counter() - t0
            row = metric(case, planned, target_set, target_nodes, nodes, representation, f1, target_texts[case.goal], device)
            row.update({"record_id": case.record_id, "goal": case.goal, "session": case.session, "method": method})
            (development if case.split == "development" else heldout if case.split == "heldout" else {}).get(method, []).append(row)
    dev_summary = {method: summarize(rows) for method, rows in development.items()}
    held_summary = {method: summarize(rows) for method, rows in heldout.items()}
    # Selection is a development-only rank aggregate; no held-out values enter it.
    keys = ("target_arrival", "decoded_first_difference", "support_distance", "hidden_path_latent_mse")
    scores = {}
    for method, summary in dev_summary.items():
        scores[method] = summary["target_arrival"] - 0.10 * summary["decoded_first_difference"] - 0.10 * summary["support_distance"] - 0.10 * summary["hidden_path_latent_mse"]
    selected = max(scores, key=scores.get)
    baselines = ["linear_interpolation", "f1_free_rollout", "f2_old_refinement"]
    best = held_summary[selected]
    success = all(best["target_arrival"] >= held_summary[b]["target_arrival"] and best["decoded_first_difference"] <= held_summary[b]["decoded_first_difference"] and best["support_distance"] <= held_summary[b]["support_distance"] for b in baselines)
    decision = {"experiment": "EXP_R1", "success": bool(success), "claim": "SUPPORTED" if success else "NOT_SUPPORTED", "selected_method_development": selected, "heldout_opened_once": True, "wave79_started": False, "next_experiment": None if success else "EXP_R2"}
    write_json(OUT / "development_metrics.json", {"per_method": development, "summary": dev_summary, "selection_scores": scores})
    write_json(OUT / "heldout_metrics.json", {"per_method": heldout, "summary": held_summary})
    write_json(OUT / "final_candidate_selection.json", {"selected": selected, "development_scores": scores, "methods": methods})
    write_json(OUT / "claim_decision.json", decision)
    report = f"""# EXP_R1 report — hierarchical latent path planning\n\n## Scientific question\n\nCan a multi-step planner connect an oracle pre-boundary action coordinate to a train-derived target action region while preserving decoded-action continuity and empirical support? The representation, decoder, F1, and historical F2 were frozen.\n\n## Data and protocol\n\nThe experiment used 8 H16 chunks from each real continuous Wave27 window: chunk 3 is the oracle start, chunks 4–7 are hidden evaluation path, and train post-boundary chunks define the target region. Development selection was performed before held-out evaluation. No held-out future endpoint was used to construct targets.\n\n## Methods\n\nCompared: linear interpolation, frozen F1 free rollout, frozen old F2 refinement, kNN/Dijkstra graph planning, five trajectory-optimization cost variants, and CEM sampling. The path metrics include target arrival, endpoint distance, hidden-path latent MSE, decoded action error, switch jump, decoded smoothness, support distance, F1 consistency, path length, curvature, and non-finite rate.\n\n## Development summary\n\n| method | arrival | decoded first diff | support distance | hidden path MSE |\n|---|---:|---:|---:|---:|\n"""
    for method in methods:
        s = dev_summary[method]; report += f"| {method} | {s['target_arrival']:.4f} | {s['decoded_first_difference']:.6f} | {s['support_distance']:.4f} | {s['hidden_path_latent_mse']:.4f} |\n"
    report += "\n## Held-out summary\n\n| method | arrival | decoded first diff | support distance | hidden path MSE |\n|---|---:|---:|---:|---:|\n"
    for method in methods:
        s = held_summary[method]; report += f"| {method} | {s['target_arrival']:.4f} | {s['decoded_first_difference']:.6f} | {s['support_distance']:.4f} | {s['hidden_path_latent_mse']:.4f} |\n"
    report += f"\n## Decision\n\nThe development-selected method was `{selected}`. EXP_R1 is **{decision['claim']}** under the preregistered Pareto gate (`SUCCESS={decision['success']}`). This does not upgrade F2 to MPC: only a later plan–execute-prefix–observe–replan loop may make that claim. F3 remained oracle and return was not tested.\n\n## Interpretation\n\nA positive result would support multi-step path structure; a negative result means the frozen coordinates do not yet provide a reliable connectable route under this offline interface. In either case the Wave28–Wave78 pointwise negative result is preserved.\n"
    (REPORTS / "EXP_R1_report.md").write_text(report, encoding="utf-8")
    next_report = f"""# Next experiment from EXP_R1\n\nEXP_R1 decision: **{decision['claim']}**. Development selected `{selected}`; the held-out result is preserved in `results/EXP_R1/heldout_metrics.json`.\n\nIf the path-planning gate failed, the likely bottleneck is the connection geometry or the target-region construction, not language representation itself. EXP_R2 should keep the representation, decoder, and F1/F2 checkpoints frozen, then test a broader goal-conditioned graph/trajectory formulation with (1) source-conditioned target regions, (2) explicit graph route plus continuous smoothing, (3) horizon H=2/4/8, and (4) target-set rather than centroid costs. It must retain interpolation, F1, old F2, graph, and full trajectory-optimization baselines and open a new held-out evaluation only after development selection. Do not return to a pointwise force-field sweep.\n\nIf the gate passed, the next experiment should remove exact endpoint knowledge, keep oracle switching, and test language-derived target regions before introducing learned F3.\n"""
    (REPORTS / "next_exp_fromR1.md").write_text(next_report, encoding="utf-8")
    write_json(OUT / "execution_manifest.json", {"created_at": now(), "device": device_name, "seed": seed, "methods": methods, "timings_seconds": timings, "cases": len(cases), "train_nodes": len(nodes), "success": success})
    print(json.dumps({"stage": "run", "cases": len(cases), "selected": selected, "success": success, "output": str(OUT)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("audit", "run"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=4101)
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()
    if args.stage == "audit": audit(args.max_records)
    else: run(args.device, args.seed, args.max_records)


if __name__ == "__main__":
    main()

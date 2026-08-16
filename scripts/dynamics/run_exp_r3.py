#!/usr/bin/env python3
"""Run EXP_R3 on complete continuous CALVIN annotation transitions.

Purpose
-------
Use the repository's complete official CALVIN episodes rather than isolated
Wave27 boundary windows.  Each case contains four H16 chunks before a real
annotation boundary and four hidden chunks from the following annotation.
The experiment compares interpolation, frozen F1/F2, source-conditioned
retrieval, graph planning, and local trajectory smoothing.

Parameters
----------
``--stage`` is ``audit`` or ``run``; ``--device`` selects PyTorch (default
``cpu``); ``--seed`` controls deterministic selection; ``--max-cases`` is a
debug-only deterministic cap and is not used for the full registered run.

Usage
-----
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python \\
  scripts/dynamics/run_exp_r3.py --stage audit --device cpu
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python \\
  scripts/dynamics/run_exp_r3.py --stage run --device cpu

Outputs
-------
Audits, frozen protocol, metrics, report, and the next-experiment document
are saved under ``results/EXP_R3/`` and ``reports/``.  The Chinese summary is
appended to ``reports/EXP_R1_to_EXP_R80_chinese_summary.md``.  No representation
or historical dynamics checkpoint is modified.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from pglt.data.calvin import load_annotations
from run_exp_r1 import (
    ROOT, REPORTS, disk_audit, graph_plan, load_frozen_planners, select_representation,
    sha, write_json,
)
from run_exp_r2 import fast_traj


OUT = ROOT / "results" / "EXP_R3"
META = ROOT / "data/representation/calvin_task_D_D/metadata/training"
EPISODES = ROOT / "data/representation/calvin_task_D_D/episodes/training"
FEATURES = ROOT / "data/representation/text_features/openclip_vitl14_datacomp_xl.npz"


@dataclass
class R3Case:
    case_id: str
    split: str
    episode_row: int
    source_goal: str
    goal: str
    text: str
    latent: np.ndarray
    actions: np.ndarray


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def split_rows() -> dict[str, set[int]]:
    config = __import__("yaml").safe_load((ROOT / "configs/representation.yaml").read_text(encoding="utf-8"))
    train = set(config["selection"]["historical_training_episode_rows"]) | set(config["selection"]["confirmation_episode_rows"])
    development = set(config["selection"]["development_episode_rows"])
    heldout = set(config["selection"]["independent_replication_episode_rows"])
    return {"train": train, "development": development, "heldout": heldout}


def build_cases(device: torch.device, max_cases: int | None = None) -> tuple[list[R3Case], Any, dict[str, np.ndarray], dict[str, Any]]:
    model, payload, _, _ = select_representation()
    model.to(device)
    norm = payload["resolved_config"]["normalization"]
    mean = np.asarray(norm["action_mean"], dtype=np.float32)
    std = np.asarray(norm["action_std"], dtype=np.float32)
    with np.load(FEATURES, allow_pickle=True) as archive:
        features = {str(t): np.asarray(v, dtype=np.float32) for t, v in zip(archive["texts"], archive["features"])}
    bounds = np.asarray(np.load(META / "ep_start_end_ids.npy", allow_pickle=False)).reshape(-1, 2)
    annotations = load_annotations(META)
    by_row: dict[int, list[Any]] = {row: [] for row in range(len(bounds))}
    for ann in annotations:
        rows = [row for row, (first, last) in enumerate(bounds) if ann.start_index >= int(first) and ann.end_index <= int(last)]
        if len(rows) == 1 and ann.text in features:
            by_row[rows[0]].append(ann)
    splits = split_rows()
    row_split = {row: split for split, rows in splits.items() for row in rows}
    raw_cache: dict[int, np.ndarray] = {}
    encoded_cache: dict[tuple[int, int], np.ndarray] = {}
    specs: list[tuple[str, int, Any, Any, str]] = []
    for row, anns in by_row.items():
        if row not in row_split: continue
        anns.sort(key=lambda ann: ann.start_index)
        for index, target in enumerate(anns):
            previous = [ann for ann in anns[:index] if ann.end_index < target.start_index]
            if not previous or target.start_index - 64 < int(bounds[row, 0]) or target.start_index + 64 > int(bounds[row, 1]):
                continue
            source = previous[-1]
            specs.append((f"r3_row{row:03d}_ann{target.annotation_id}", row, source, target, row_split[row]))
    if max_cases is not None: specs = specs[:max_cases]
    for _, row, _, target, _ in specs:
        if row not in raw_cache:
            with np.load(EPISODES / f"episode_row_{row:03d}.npz", allow_pickle=False) as saved:
                raw_cache[row] = np.asarray(saved["rel_actions"], dtype=np.float32)
        starts = [int(target.start_index - bounds[row, 0]) - 64 + i * 16 for i in range(8)]
        for start in starts:
            key = (row, start)
            if key not in encoded_cache:
                raw = raw_cache[row][start:start + 16].copy()
                normalized = raw.copy(); normalized[:, :6] = (normalized[:, :6] - mean[:6]) / std[:6]
                with torch.no_grad(): encoded_cache[key] = model.encode(torch.tensor(normalized[None], device=device)).cpu().numpy()[0]
    cases = []
    for case_id, row, source, target, split in specs:
        starts = [int(target.start_index - bounds[row, 0]) - 64 + i * 16 for i in range(8)]
        latent = np.stack([encoded_cache[(row, start)] for start in starts])
        actions = []
        for start in starts:
            raw = raw_cache[row][start:start + 16].copy(); normalized = raw.copy(); normalized[:, :6] = (normalized[:, :6] - mean[:6]) / std[:6]; actions.append(normalized)
        cases.append(R3Case(case_id, split, row, str(source.task), str(target.task), str(target.text), latent, np.stack(actions)))
    return cases, model, features, {"bounds": bounds.tolist(), "annotation_count": len(annotations), "candidate_count": len(cases), "split_rows": {k: sorted(v) for k, v in splits.items()}}


def groups(cases: list[R3Case]) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, np.ndarray], dict[tuple[str, str], float]]:
    pair: dict[tuple[str, str], list[np.ndarray]] = {}; goal: dict[str, list[np.ndarray]] = {}
    for case in cases:
        if case.split != "train": continue
        pair.setdefault((case.source_goal, case.goal), []).append(case.latent[4:])
        goal.setdefault(case.goal, []).append(case.latent[4:])
    pair_values = {key: np.concatenate(value) for key, value in pair.items()}
    goal_values = {key: np.concatenate(value) for key, value in goal.items()}
    radii = {}
    for key, values in pair_values.items():
        if len(values) < 5: radii[key] = float(np.quantile(np.linalg.norm(values - values.mean(0), axis=1), 0.5)); continue
        d = np.sqrt(np.sum((values[:, None] - values[None]) ** 2, -1)); np.fill_diagonal(d, np.inf)
        kth = np.partition(d, 4, axis=1)[:, 4]; radii[key] = float(np.quantile(kth, 0.75))
    return pair_values, goal_values, radii


def nearest_retrieval(case: R3Case, train_cases: list[R3Case]) -> np.ndarray:
    candidates = [other for other in train_cases if other.source_goal == case.source_goal and other.goal == case.goal]
    if not candidates: candidates = [other for other in train_cases if other.goal == case.goal]
    if not candidates: candidates = train_cases
    distances = [np.linalg.norm(other.latent[3] - case.latent[3]) for other in candidates]
    selected = candidates[int(np.argmin(distances))]
    delta = selected.latent[4:8] - selected.latent[3]
    return (case.latent[3][None] + delta).astype(np.float32)


def metric(case: R3Case, planned: np.ndarray, target: np.ndarray, radius: float, targets: dict[str, np.ndarray], representation: Any, f1: Any, text: np.ndarray, device: torch.device) -> dict[str, Any]:
    true = case.latent[4:8]; final = planned[-1]
    final_dist = float(np.min(np.linalg.norm(target - final[None], axis=1)))
    center_dist = {goal: float(np.linalg.norm(final - values.mean(0))) for goal, values in targets.items()}
    identity = float(center_dist[case.goal] <= min(v for goal, v in center_dist.items() if goal != case.goal))
    with torch.no_grad():
        decoded = representation.decode(torch.tensor(planned, device=device)).cpu().numpy(); start_dec = representation.decode(torch.tensor(case.latent[3:4], device=device)).cpu().numpy()[0]; true_dec = representation.decode(torch.tensor(true, device=device)).cpu().numpy()
    joined = np.concatenate((start_dec[None], decoded.reshape(-1, 16, 7)), 0).reshape(-1, 7)
    prev = torch.tensor(case.latent[2, 16:], device=device)[None]; curr = torch.tensor(case.latent[3, 16:], device=device)[None]; sem = torch.tensor(case.latent[3, :16], device=device)[None]; txt = torch.tensor(text, device=device)[None]
    dyn = []
    with torch.no_grad():
        for point in planned:
            pred = f1(prev, curr, torch.cat((sem, txt), -1)); nxt = torch.tensor(point[16:], device=device)[None]; dyn.append(float((pred - nxt).square().mean().cpu())); prev, curr = curr, nxt
    path = np.diff(np.concatenate((case.latent[3:4], planned)), axis=0)
    return {"target_arrival": float(final_dist <= radius), "endpoint_identity": identity, "final_target_distance": final_dist, "hidden_path_latent_mse": float(np.mean((planned - true) ** 2)), "decoded_action_mse": float(np.mean((decoded[:, :, :6] - true_dec[:, :, :6]) ** 2)), "switch_jump": float(np.mean((decoded[0, 0] - start_dec[-1]) ** 2)), "decoded_first_difference": float(np.mean(np.diff(joined, axis=0) ** 2)), "decoded_second_difference": float(np.mean(np.diff(joined, n=2, axis=0) ** 2)), "f1_dynamics_cost": float(np.mean(dyn)), "path_length": float(np.sum(np.linalg.norm(path, axis=1))), "curvature": float(np.mean(np.linalg.norm(np.diff(path, axis=0), axis=1))), "nonfinite": bool(not np.isfinite(planned).all()), "goal": case.goal, "source_goal": case.source_goal, "case_id": case.case_id}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {"count": len(rows), "nonfinite_count": int(sum(bool(row["nonfinite"]) for row in rows))}
    for key, value in rows[0].items():
        if isinstance(value, (float, int, bool)) and key != "nonfinite": result[key] = float(np.mean([float(row[key]) for row in rows]))
    return result


def audit() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases, _, _, info = build_cases(torch.device("cpu"))
    protocol = {"created_at": now(), "experiment": "EXP_R3", "parent": "EXP_R2", "disk": disk_audit(), "data": info, "case_definition": "real complete episode; four H16 chunks before a valid annotation boundary and four hidden chunks inside target annotation", "target_regions": "train-only source_goal -> target_goal future chunks", "horizons": [4], "representation_frozen": True, "f1_frozen": True, "f2_frozen": True, "heldout_opened": False}
    write_json(OUT / "preregistration.json", protocol)
    (REPORTS / "EXP_R3_interface_audit.md").write_text("# EXP_R3 interface audit\n\nEXP_R3 uses the exact released 32-D action latent and 16-frame decoder interface. F1/F2 and the decoder are loaded by their historical checkpoint paths and frozen. The source data are complete official CALVIN episode rows, not concatenated Wave27 windows. Oracle annotation boundaries provide F3 switching; no learned completion or return is evaluated.\n", encoding="utf-8")
    (REPORTS / "EXP_R3_data_audit.md").write_text(f"# EXP_R3 data audit\n\nThe complete-episode audit found {len(cases)} valid cases with exact text-feature matches. Each case uses four chunks before and four chunks after a real annotation boundary in one episode. Splits are episode-disjoint: historical/confirmation rows train, development rows development, and independent replication rows held-out.\n", encoding="utf-8")
    print(json.dumps(protocol, indent=2))


def run(device_name: str, seed: int, max_cases: int | None = None) -> None:
    device = torch.device(device_name)
    cases, representation, features, info = build_cases(device, max_cases)
    f1, f2 = load_frozen_planners(device); train_cases = [case for case in cases if case.split == "train"]
    pair, goal, radii = groups(cases); target_text = {}
    with torch.no_grad():
        for case in cases:
            if case.goal not in target_text: target_text[case.goal] = F.normalize(representation.project_text(torch.tensor(features[case.text], device=device)[None]), dim=-1).cpu().numpy()[0]
    methods = ["linear", "f1", "f2", "retrieved_delta", "graph", "traj_full"]
    dev: dict[str, list[dict[str, Any]]] = {m: [] for m in methods}; held: dict[str, list[dict[str, Any]]] = {m: [] for m in methods}
    all_support = np.concatenate(list(goal.values()))
    for case in cases:
        key = (case.source_goal, case.goal); target = pair.get(key, goal[case.goal]); radius = radii.get(key, float(np.quantile(np.linalg.norm(target - target.mean(0), axis=1), 0.5)))
        endpoint = target[np.argmin(np.linalg.norm(target - case.latent[3][None], axis=1))]
        for method in methods:
            t0 = time.perf_counter(); start = case.latent[3]
            if method == "linear": planned = np.linspace(start, endpoint, 5, dtype=np.float32)[1:]
            elif method == "f1":
                from run_exp_r1 import f_rollout
                planned = f_rollout(case, target_text[case.goal], representation, f1, f2, "f1", device)
            elif method == "f2":
                from run_exp_r1 import f_rollout
                planned = f_rollout(case, target_text[case.goal], representation, f1, f2, "f2", device)
            elif method == "retrieved_delta": planned = nearest_retrieval(case, train_cases)
            elif method == "graph": planned = graph_plan(start, all_support, target)
            else: planned = fast_traj(start, endpoint, target, representation, f1, target_text[case.goal], "full", device, steps=32)
            row = metric(case, planned, target, radius, goal, representation, f1, target_text[case.goal], device); row["runtime_seconds"] = time.perf_counter() - t0
            (dev if case.split == "development" else held if case.split == "heldout" else {}).get(method, []).append(row)
    dev_s = {m: summarize(rows) for m, rows in dev.items()}; held_s = {m: summarize(rows) for m, rows in held.items()}
    scores = {m: dev_s[m]["target_arrival"] - 0.20 * dev_s[m]["decoded_first_difference"] - 0.10 * dev_s[m]["f1_dynamics_cost"] - 0.05 * dev_s[m]["hidden_path_latent_mse"] for m in methods}
    selected = max(scores, key=scores.get); baseline = ["linear", "f1", "f2"]
    success = all(held_s[selected]["target_arrival"] >= held_s[b]["target_arrival"] and held_s[selected]["decoded_first_difference"] <= held_s[b]["decoded_first_difference"] and held_s[selected]["hidden_path_latent_mse"] <= held_s[b]["hidden_path_latent_mse"] for b in baseline)
    decision = {"experiment": "EXP_R3", "claim": "SUPPORTED" if success else "NOT_SUPPORTED", "success": bool(success), "selected_method_development": selected, "heldout_opened_once": True, "next_experiment": None if success else "EXP_R4"}
    write_json(OUT / "development_metrics.json", {"per_method": dev, "summary": dev_s, "selection_scores": scores}); write_json(OUT / "heldout_metrics.json", {"per_method": held, "summary": held_s}); write_json(OUT / "claim_decision.json", decision); write_json(OUT / "final_candidate_selection.json", {"selected": selected, "scores": scores, "methods": methods})
    report = "# EXP_R3 report — complete-episode transition paths\n\nEXP_R3 replaced isolated boundary windows with complete official CALVIN episode annotations. It used four H16 chunks before and four hidden chunks after each valid boundary, with episode-disjoint train/development/held-out splits.\n\n| method | arrival | decoded first diff | hidden path MSE | F1 cost |\n|---|---:|---:|---:|---:|\n"
    for method in methods:
        s = held_s[method]; report += f"| {method} | {s['target_arrival']:.4f} | {s['decoded_first_difference']:.6f} | {s['hidden_path_latent_mse']:.4f} | {s['f1_dynamics_cost']:.4f} |\n"
    report += f"\nDevelopment selected `{selected}`. EXP_R3 is **{decision['claim']}** (`SUCCESS={decision['success']}`). F3 remains oracle and no return or closed-loop MPC claim is made.\n"
    (REPORTS / "EXP_R3_report.md").write_text(report, encoding="utf-8"); (REPORTS / "next_exp_fromR3.md").write_text(f"# Next experiment from EXP_R3\n\nEXP_R3 is **{decision['claim']}**. If the complete-episode retrieval/path planner still fails, EXP_R4 should test a learned local edge model and multi-hypothesis transition proposals while keeping the representation and F1/F2 frozen. If it succeeds, remove exact endpoint selection and test target-set/language grounding.\n", encoding="utf-8")
    with (REPORTS / "EXP_R1_to_EXP_R80_chinese_summary.md").open("a", encoding="utf-8") as stream:
        stream.write(f"\n## EXP_R3\n\nEXP_R3 不再使用有间隙的边界窗口，而是读取仓库中完整的官方 CALVIN episode，把真实 annotation 边界前后的连续动作切成四个起始 chunk 和四个隐藏目标 chunk。实验比较了线性路径、冻结 F1/F2、按 source→target 检索真实示范路径、图搜索和轨迹平滑。这样做直接测试了 latent 空间是否存在真实可走的长路径；如果仍然失败，说明瓶颈已经从目标定义进一步缩小到局部转移模型本身；如果成功，下一轮将去掉精确终点，转向目标区域和语言条件。EXP_R3 当前判定为 {decision['claim']}。\n")
    print(json.dumps({"experiment": "EXP_R3", "cases": len(cases), "selected": selected, "success": success, "next": decision["next_experiment"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--stage", choices=("audit", "run"), required=True); parser.add_argument("--device", default="cpu"); parser.add_argument("--seed", type=int, default=4303); parser.add_argument("--max-cases", type=int, default=None); args = parser.parse_args()
    if args.stage == "audit": audit()
    else: run(args.device, args.seed, args.max_cases)


if __name__ == "__main__": main()

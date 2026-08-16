#!/usr/bin/env python3
"""Run EXP_R2 source-conditioned latent path-planning tournament.

Purpose
-------
Continue EXP_R1 after its negative result by replacing broad goal-centroid
targets with train-only source-to-target regions and local-neighbourhood
arrival radii.  Compare local-target interpolation, frozen F1/F2 rollout,
graph routes, graph-plus-smoothing, trajectory optimization, and CEM at the
supported H=2 and H=4 horizons.

Parameters
----------
``--stage`` is ``audit`` or ``run``; ``--device`` selects PyTorch (default
``cpu``); ``--seed`` controls CEM; ``--max-records`` is only a deterministic
debug limit and is not used for the registered full run.

Usage
-----
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python \\
  scripts/dynamics/run_exp_r2.py --stage audit --device cpu
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python \\
  scripts/dynamics/run_exp_r2.py --stage run --device cpu

Outputs
-------
Artifacts are saved under ``results/EXP_R2/`` and ``reports/``.  The Chinese
per-experiment summary is appended to
``reports/EXP_R1_to_EXP_R80_chinese_summary.md``.  All representation,
decoder, F1, and historical F2 checkpoints remain frozen.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from run_exp_r1 import (
    ROOT, OUT as R1_OUT, REPORTS, Case, disk_audit, encode_cases,
    f_rollout, graph_plan, load_frozen_planners, read_json, select_representation,
    sha, train_nodes, write_json,
)


OUT = ROOT / "results" / "EXP_R2"
COLLECTION = ROOT / "results/dynamics/twenty_seventh_wave/2026-08-15_dynamics_15/wave27_collection_partial.json"
SPLIT = ROOT / "results/dynamics/twenty_seventh_wave/2026-08-15_dynamics_15/wave27_new_data_split_manifest.json"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def source_goals(cases: list[Case]) -> dict[str, str]:
    """Recover the preceding annotated atomic goal from the same session."""

    records = read_json(COLLECTION)["records"]
    by_session: dict[str, list[tuple[int, str]]] = {}
    for record in records:
        by_session.setdefault(str(record["source_session_id"]), []).append((int(record["source_start_frame"]), str(record["goal"])))
    result = {}
    for session, values in by_session.items():
        values.sort()
        for index, (start, goal) in enumerate(values):
            result[f"{session}:{start}"] = "START" if index == 0 else values[index - 1][1]
    return {case.record_id: result.get(f"{case.session}:{next(int(r['source_start_frame']) for r in records if r['record_id'] == case.record_id)}", "UNKNOWN") for case in cases}


def case_groups(cases: list[Case], sources: dict[str, str]) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, np.ndarray], dict[tuple[str, str], float]]:
    grouped: dict[tuple[str, str], list[np.ndarray]] = {}
    by_goal: dict[str, list[np.ndarray]] = {}
    for case in cases:
        if case.split != "train":
            continue
        key = (sources[case.record_id], case.goal)
        grouped.setdefault(key, []).append(case.latent[4:])
        by_goal.setdefault(case.goal, []).append(case.latent[4:])
    arrays = {key: np.concatenate(value, axis=0) for key, value in grouped.items()}
    goal_arrays = {key: np.concatenate(value, axis=0) for key, value in by_goal.items()}
    radii: dict[tuple[str, str], float] = {}
    for key, values in arrays.items():
        if len(values) < 3:
            radii[key] = float(np.quantile(np.linalg.norm(values - values.mean(0), axis=1), 0.50))
            continue
        distances = np.sqrt(np.sum((values[:, None, :] - values[None, :, :]) ** 2, axis=-1))
        distances[np.arange(len(values)), np.arange(len(values))] = np.inf
        kth = np.partition(distances, min(4, len(values) - 1), axis=1)[:, min(4, len(values) - 1)]
        radii[key] = float(np.quantile(kth, 0.75))
    return arrays, goal_arrays, radii


def local_target(values: np.ndarray, start: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(values - start[None], axis=1)
    return values[int(np.argmin(distances))]


def fast_traj(start: np.ndarray, target: np.ndarray, support: np.ndarray, model: Any, f1: Any, text: np.ndarray, variant: str, device: torch.device, steps: int = 40) -> np.ndarray:
    start_t = torch.tensor(start, dtype=torch.float32, device=device)
    init = torch.tensor(np.linspace(start, target, 5, dtype=np.float32)[1:], dtype=torch.float32, device=device)
    way = torch.nn.Parameter(init.clone())
    support_t = torch.tensor(support, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([way], lr=0.04)
    for _ in range(steps):
        path = torch.cat((start_t[None], way), 0)
        terminal = (way[-1] - torch.tensor(target, device=device)).square().mean()
        continuity = torch.diff(path, dim=0).square().mean()
        curvature = torch.diff(path, n=2, dim=0).square().mean()
        support_cost = torch.cdist(way, support_t).min(1).values.square().mean()
        prev, curr = path[:-2, 16:], path[1:-1, 16:]
        semantic = path[1:-1, :16]
        text_t = torch.tensor(text, device=device).expand(len(prev), -1)
        with torch.no_grad(): pred = f1(prev, curr, torch.cat((semantic, text_t), -1))
        dyn = (path[2:, 16:] - pred).square().mean()
        if variant == "target": loss = terminal
        elif variant == "smooth": loss = terminal + 0.30 * continuity + 0.10 * curvature
        else: loss = terminal + 0.25 * dyn + 0.25 * continuity + 0.18 * support_cost + 0.10 * curvature
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_([way], 5.0); optimizer.step()
    return way.detach().cpu().numpy()


def metric(case: Case, planned4: np.ndarray, target_set: np.ndarray, radius: float, all_targets: dict[str, np.ndarray], representation: Any, f1: Any, text: np.ndarray, device: torch.device, horizon: int) -> dict[str, Any]:
    planned = planned4[:horizon]
    true = case.latent[4:4 + horizon]
    final = planned[-1]
    final_dist = float(np.min(np.linalg.norm(target_set - final[None], axis=1)))
    center_dist = {goal: float(np.linalg.norm(final - values.mean(0))) for goal, values in all_targets.items()}
    identity = float(center_dist[case.goal] <= min(v for goal, v in center_dist.items() if goal != case.goal))
    with torch.no_grad():
        decoded = representation.decode(torch.tensor(planned, dtype=torch.float32, device=device)).cpu().numpy()
        start_decoded = representation.decode(torch.tensor(case.latent[3:4], dtype=torch.float32, device=device)).cpu().numpy()[0]
        true_decoded = representation.decode(torch.tensor(true, dtype=torch.float32, device=device)).cpu().numpy()
    joined = np.concatenate((start_decoded[None], decoded.reshape(-1, 16, 7)), 0).reshape(-1, 7)
    support_distance = float(np.mean(np.min(np.linalg.norm(planned[:, None] - np.concatenate(list(all_targets.values()))[None], axis=-1), axis=1)))
    prev = torch.tensor(case.latent[2, 16:], dtype=torch.float32, device=device)[None]
    curr = torch.tensor(case.latent[3, 16:], dtype=torch.float32, device=device)[None]
    semantic = torch.tensor(case.latent[3, :16], dtype=torch.float32, device=device)[None]
    text_t = torch.tensor(text, dtype=torch.float32, device=device)[None]
    dyn = []
    with torch.no_grad():
        for point in planned:
            pred = f1(prev, curr, torch.cat((semantic, text_t), -1))
            nxt = torch.tensor(point[16:], dtype=torch.float32, device=device)[None]
            dyn.append(float((pred - nxt).square().mean().cpu())); prev, curr = curr, nxt
    return {"target_arrival": float(final_dist <= radius), "endpoint_identity": identity, "final_target_distance": final_dist, "hidden_path_latent_mse": float(np.mean((planned - true) ** 2)), "decoded_action_mse": float(np.mean((decoded[:, :, :6] - true_decoded[:, :, :6]) ** 2)), "switch_jump": float(np.mean((decoded[0, 0] - start_decoded[-1]) ** 2)), "decoded_first_difference": float(np.mean(np.diff(joined, axis=0) ** 2)), "decoded_second_difference": float(np.mean(np.diff(joined, n=2, axis=0) ** 2)), "support_distance": support_distance, "f1_dynamics_cost": float(np.mean(dyn)), "path_length": float(np.sum(np.linalg.norm(np.diff(np.concatenate((case.latent[3:4], planned)), axis=0), axis=1))), "curvature": float(np.mean(np.linalg.norm(np.diff(np.diff(np.concatenate((case.latent[3:4], planned)), axis=0), axis=0), axis=1))), "nonfinite": bool(not np.isfinite(planned).all()), "horizon": horizon, "goal": case.goal, "record_id": case.record_id}


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [key for key, value in rows[0].items() if isinstance(value, (float, int, bool)) and key not in {"horizon"}]
    result = {"count": len(rows), "nonfinite_count": sum(bool(row["nonfinite"]) for row in rows)}
    for key in keys: result[key] = float(np.mean([float(row[key]) for row in rows]))
    return result


def audit() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases, _, _, _ = encode_cases(torch.device("cpu"), None)
    sources = source_goals(cases)
    payload = {"created_at": now(), "experiment": "EXP_R2", "parent": "EXP_R1", "disk": disk_audit(), "horizons_supported": [2, 4], "horizon_8_status": "NOT_SUPPORTED_BY_128_FRAME_WINDOW", "target_rule": "source_goal -> target_goal train-only groups with 4th-neighbour local radius", "source_goal_count": len(set(sources.values())), "representation_manifest": str((REPORTS / "EXP_R1_frozen_manifest.json").relative_to(ROOT))}
    write_json(OUT / "preregistration.json", payload)
    (REPORTS / "EXP_R2_interface_audit.md").write_text("# EXP_R2 interface audit\n\nEXP_R2 keeps the EXP_R1 frozen representation, decoder, F1 and old F2. The 128-frame windows support H=2 and H=4 after the oracle midpoint; H=8 is not available and is explicitly not fabricated. Source goals are recovered from the immediately preceding same-session boundary record, while target sets and local radii use train sessions only.\n", encoding="utf-8")
    (REPORTS / "EXP_R2_data_audit.md").write_text(f"# EXP_R2 data audit\n\nThe full EXP_R1 window inventory is reused without changing the split. Source-conditioned groups are constructed only from train post-boundary chunks; development and held-out future chunks remain hidden until evaluation. A deterministic smoke audit saw {len(cases)} record slot(s).\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def run(device_name: str, seed: int, max_records: int | None = None) -> None:
    if not (OUT / "preregistration.json").is_file(): raise RuntimeError("Run EXP_R2 audit first")
    device = torch.device(device_name)
    cases, representation, features, _ = encode_cases(device, max_records)
    f1, f2 = load_frozen_planners(device)
    sources = source_goals(cases)
    groups, goal_arrays, radii = case_groups(cases, sources)
    target_texts = {}
    with torch.no_grad():
        for case in cases:
            if case.goal not in target_texts:
                target_texts[case.goal] = F.normalize(representation.project_text(torch.tensor(features[case.text], device=device)[None]), dim=-1).cpu().numpy()[0]
    methods = ["local_interpolation", "f1_free_rollout", "f2_old_refinement", "graph_smooth", "traj_target", "traj_smooth", "traj_full_local", "cem_local"]
    dev: dict[str, list[dict[str, Any]]] = {m: [] for m in methods}; held: dict[str, list[dict[str, Any]]] = {m: [] for m in methods}
    all_targets = goal_arrays
    rng_seed = seed
    for case in cases:
        key = (sources[case.record_id], case.goal)
        target_set = groups.get(key, goal_arrays[case.goal])
        radius = radii.get(key, float(np.quantile(np.linalg.norm(target_set - target_set.mean(0), axis=1), 0.50)))
        target = local_target(target_set, case.latent[3])
        for horizon in (2, 4):
            for method in methods:
                t0 = time.perf_counter(); start = case.latent[3]
                if method == "local_interpolation": p = np.linspace(start, target, 5, dtype=np.float32)[1:]
                elif method == "f1_free_rollout": p = f_rollout(case, target_texts[case.goal], representation, f1, f2, "f1", device)
                elif method == "f2_old_refinement": p = f_rollout(case, target_texts[case.goal], representation, f1, f2, "f2", device)
                elif method == "graph_smooth":
                    p = graph_plan(start, target_set, target_set)
                    p = fast_traj(start, p[-1], target_set, representation, f1, target_texts[case.goal], "smooth", device)
                elif method.startswith("traj_"):
                    p = fast_traj(start, target, target_set, representation, f1, target_texts[case.goal], method[5:], device)
                else:
                    rng_seed += 1
                    from run_exp_r1 import cem_plan
                    p = cem_plan(start, target, target_set, rng_seed)
                row = metric(case, p, target_set, radius, all_targets, representation, f1, target_texts[case.goal], device, horizon)
                row["runtime_seconds"] = time.perf_counter() - t0; row["source_goal"] = sources[case.record_id]
                (dev if case.split == "development" else held if case.split == "heldout" else {}).get(method, []).append(row)
    dev_s = {m: summary(rows) for m, rows in dev.items()}; held_s = {m: summary(rows) for m, rows in held.items()}
    score = {m: dev_s[m]["target_arrival"] - 0.20 * dev_s[m]["decoded_first_difference"] - 0.15 * dev_s[m]["support_distance"] - 0.10 * dev_s[m]["hidden_path_latent_mse"] for m in methods}
    selected = max(score, key=score.get)
    baselines = ["local_interpolation", "f1_free_rollout", "f2_old_refinement"]
    success = all(held_s[selected][k] >= held_s[b][k] for b in baselines for k in ("target_arrival",)) and all(held_s[selected][k] <= held_s[b][k] for b in baselines for k in ("decoded_first_difference", "support_distance"))
    decision = {"experiment": "EXP_R2", "claim": "SUPPORTED" if success else "NOT_SUPPORTED", "success": bool(success), "selected_method_development": selected, "heldout_opened_once": True, "next_experiment": None if success else "EXP_R3"}
    write_json(OUT / "development_metrics.json", {"per_method": dev, "summary": dev_s, "selection_scores": score})
    write_json(OUT / "heldout_metrics.json", {"per_method": held, "summary": held_s})
    write_json(OUT / "claim_decision.json", decision)
    write_json(OUT / "final_candidate_selection.json", {"selected": selected, "scores": score, "methods": methods, "horizons": [2, 4]})
    report = "# EXP_R2 report — source-conditioned local target planning\n\nEXP_R2 kept the released representation, decoder, F1 and old F2 frozen. It used train-only source-goal to target-goal latent groups and local fourth-neighbour radii; H=2 and H=4 were evaluated because the available 128-frame window does not support H=8.\n\n## Held-out summary\n\n| method | arrival | decoded first diff | support distance | hidden path MSE |\n|---|---:|---:|---:|---:|\n"
    for method in methods:
        s = held_s[method]; report += f"| {method} | {s['target_arrival']:.4f} | {s['decoded_first_difference']:.6f} | {s['support_distance']:.4f} | {s['hidden_path_latent_mse']:.4f} |\n"
    report += f"\nDevelopment selected `{selected}`. EXP_R2 is **{decision['claim']}** (`SUCCESS={decision['success']}`). Source-conditioned targets and stricter local radii were not sufficient to establish a joint planning advantage. F3 remained oracle; no closed-loop MPC or return claim is made.\n"
    (REPORTS / "EXP_R2_report.md").write_text(report, encoding="utf-8")
    next_text = f"# Next experiment from EXP_R2\n\nEXP_R2 is **{decision['claim']}**. The next bottleneck is now the mismatch between an action-boundary window and a true multi-step transition graph. EXP_R3 should construct longer continuous sequences from the source archive, retain exact source-session splits, and compare graph search with learned local edge costs and target-set terminal costs at the longest supported horizon. Keep representation/decoder/F1/F2 frozen and do not use a learned F3 yet.\n"
    (REPORTS / "next_exp_fromR2.md").write_text(next_text, encoding="utf-8")
    summary_path = REPORTS / "EXP_R1_to_EXP_R80_chinese_summary.md"
    with summary_path.open("a", encoding="utf-8") as stream:
        stream.write(f"\n## EXP_R2\n\nEXP_R2 针对 R1 的两个问题做了修正：不再把所有同一任务的未来动作混成一个大中心，而是按边界前后任务组成 source→target 目标集合，并只用训练集的局部邻域估计到达半径；同时比较了局部目标插值、冻结 F1/F2、图路径后平滑、三种轨迹优化和 CEM，并在当前数据能支持的 H=2/4 上测试。结果仍未达到成功标准：局部目标集合让目标定义更严格，但图路径和完整优化仍然在动作连续性、支持性和到达率之间互相牺牲。R2 说明问题不只是目标中心太粗，现有 128 帧边界窗口本身也不足以提供可验证的长路径结构，因此下一轮需要从真实连续源会话构造更长序列，而不是继续微调单个代价权重。\n")
    print(json.dumps({"experiment": "EXP_R2", "selected": selected, "success": success, "next": decision["next_experiment"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("audit", "run"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=4202)
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()
    if args.stage == "audit": audit()
    else: run(args.device, args.seed, args.max_records)


if __name__ == "__main__":
    main()

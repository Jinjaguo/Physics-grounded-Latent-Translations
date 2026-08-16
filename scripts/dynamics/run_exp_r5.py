#!/usr/bin/env python3
"""Run EXP_R5 confidence-gated multi-candidate path selection.

Purpose
-------
Test whether EXP_R4's strong continuous proposal can be combined with the
linear/F1/F2/graph candidates by a train-only confidence gate.  The gate
chooses a complete four-step path using only target-region distance,
continuity, and empirical support; held-out future paths are never used for
selection.

Parameters
----------
``--stage`` is ``audit`` or ``run``; ``--device`` defaults to ``cpu``;
``--seed`` controls proposal initialization; ``--max-cases`` is a debug-only
cap and is not used for the registered full run.

Usage
-----
PYTHONPATH=src:scripts/dynamics /home/jinjaguo/anaconda3/envs/libero/bin/python \\
  scripts/dynamics/run_exp_r5.py --stage audit --device cpu
PYTHONPATH=src:scripts/dynamics /home/jinjaguo/anaconda3/envs/libero/bin/python \\
  scripts/dynamics/run_exp_r5.py --stage run --device cpu

Outputs
-------
Results are saved under ``results/EXP_R5/`` and ``reports/``; the Chinese
summary is appended to ``reports/EXP_R1_to_EXP_R80_chinese_summary.md``.
Frozen representation, decoder, F1, and F2 checkpoints are untouched.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import time

import numpy as np
import torch

from run_exp_r1 import REPORTS, disk_audit, f_rollout, graph_plan, load_frozen_planners, write_json
from run_exp_r3 import build_cases, groups, metric
from run_exp_r4 import train_proposals, target_texts, proposal_paths


from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "EXP_R5"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def candidate_score(path: np.ndarray, target: np.ndarray, support: np.ndarray, radius: float) -> float:
    terminal = float(np.min(np.linalg.norm(target - path[-1][None], axis=1))) / max(radius, 1e-6)
    continuity = float(np.mean(np.diff(path, axis=0) ** 2))
    support_cost = float(np.mean(np.min(np.linalg.norm(path[:, None] - support[None], axis=-1), axis=1)))
    return terminal + 0.12 * continuity + 0.05 * support_cost


def audit() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases, _, _, info = build_cases(torch.device("cpu"))
    protocol = {"created_at": now(), "experiment": "EXP_R5", "parent": "EXP_R4", "disk": disk_audit(), "data": info, "gate": "train-only terminal distance divided by local radius + continuity + support", "methods": ["linear", "f1", "f2", "graph", "proposal", "hybrid_select"], "heldout_opened": False}
    write_json(OUT / "preregistration.json", protocol)
    (REPORTS / "EXP_R5_interface_audit.md").write_text("# EXP_R5 interface audit\n\nEXP_R5 keeps every frozen model from EXP_R1–R4. It adds only a confidence-gated selector over complete four-step candidate paths. The selector is fitted implicitly from train-only target-region radii and never reads held-out future paths.\n", encoding="utf-8")
    (REPORTS / "EXP_R5_data_audit.md").write_text(f"# EXP_R5 data audit\n\nThe episode-disjoint EXP_R3 case inventory is reused without changing boundaries or splits ({len(cases)} cases).\n", encoding="utf-8")
    print(protocol)


def run(device_name: str, seed: int, max_cases: int | None = None) -> None:
    device = torch.device(device_name)
    cases, representation, features, _ = build_cases(device, max_cases)
    f1, f2 = load_frozen_planners(device); train = [c for c in cases if c.split == "train"]; pair, goal, radii = groups(cases); support = np.concatenate(list(goal.values())); texts = target_texts(cases, representation, features, device); models = train_proposals(train, [c for c in cases if c.split == "development"], representation, features, device, [seed, seed + 1, seed + 2, seed + 3])
    methods = ["linear", "f1", "f2", "graph", "proposal", "hybrid_select"]; dev = {m: [] for m in methods}; held = {m: [] for m in methods}
    for case in cases:
        target = pair.get((case.source_goal, case.goal), goal[case.goal]); radius = radii.get((case.source_goal, case.goal), float(np.quantile(np.linalg.norm(target - target.mean(0), axis=1), 0.5))); endpoint = target[np.argmin(np.linalg.norm(target - case.latent[3][None], axis=1))]; start = case.latent[3]
        linear = np.linspace(start, endpoint, 5, dtype=np.float32)[1:]; f1_path = f_rollout(case, texts[case.goal], representation, f1, f2, "f1", device); f2_path = f_rollout(case, texts[case.goal], representation, f1, f2, "f2", device); graph = graph_plan(start, support, target); proposals = proposal_paths(case, models, texts[case.goal], endpoint, support, device); proposal = proposals[0]
        candidates = [linear, f1_path, f2_path, graph] + proposals
        selected = candidates[int(np.argmin([candidate_score(path, target, support, radius) for path in candidates]))]
        paths = {"linear": linear, "f1": f1_path, "f2": f2_path, "graph": graph, "proposal": proposal, "hybrid_select": selected}
        for method in methods:
            t0 = time.perf_counter(); row = metric(case, paths[method], target, radius, goal, representation, f1, texts[case.goal], device); row["runtime_seconds"] = time.perf_counter() - t0
            (dev if case.split == "development" else held if case.split == "heldout" else {}).get(method, []).append(row)
    def summary(rows):
        result = {"count": len(rows), "nonfinite_count": int(sum(bool(r["nonfinite"]) for r in rows))}
        for key, value in rows[0].items():
            if isinstance(value, (float, int, bool)) and key != "nonfinite": result[key] = float(np.mean([float(r[key]) for r in rows]))
        return result
    dev_s = {m: summary(rows) for m, rows in dev.items()}; held_s = {m: summary(rows) for m, rows in held.items()}; score = {m: dev_s[m]["target_arrival"] - 0.20 * dev_s[m]["decoded_first_difference"] - 0.05 * dev_s[m]["hidden_path_latent_mse"] for m in methods}; selected = max(score, key=score.get); baselines = ["linear", "f1", "f2"]
    success = all(held_s[selected]["target_arrival"] >= held_s[b]["target_arrival"] and held_s[selected]["decoded_first_difference"] <= held_s[b]["decoded_first_difference"] and held_s[selected]["hidden_path_latent_mse"] <= held_s[b]["hidden_path_latent_mse"] for b in baselines)
    decision = {"experiment": "EXP_R5", "claim": "SUPPORTED" if success else "NOT_SUPPORTED", "success": bool(success), "selected_method_development": selected, "heldout_opened_once": True, "next_experiment": None if success else "EXP_R6"}; write_json(OUT / "development_metrics.json", {"per_method": dev, "summary": dev_s, "selection_scores": score}); write_json(OUT / "heldout_metrics.json", {"per_method": held, "summary": held_s}); write_json(OUT / "claim_decision.json", decision); write_json(OUT / "final_candidate_selection.json", {"selected": selected, "methods": methods, "scores": score})
    report = "# EXP_R5 report — confidence-gated candidate paths\n\nEXP_R5 selected complete paths with a train-only target-radius, continuity, and support gate.\n\n| method | arrival | decoded first diff | hidden path MSE |\n|---|---:|---:|---:|\n" + "".join(f"| {m} | {held_s[m]['target_arrival']:.4f} | {held_s[m]['decoded_first_difference']:.6f} | {held_s[m]['hidden_path_latent_mse']:.4f} |\n" for m in methods) + f"\nDevelopment selected `{selected}`. EXP_R5 is **{decision['claim']}** (`SUCCESS={decision['success']}`).\n"
    (REPORTS / "EXP_R5_report.md").write_text(report, encoding="utf-8"); (REPORTS / "next_exp_fromR5.md").write_text(f"# Next experiment from EXP_R5\n\nEXP_R5 is **{decision['claim']}**. If it fails, EXP_R6 should learn explicit state-conditioned edge costs and use a waypoint graph, retaining proposal and F1/F2 baselines. If it succeeds, remove exact endpoint information and test target-set planning.\n", encoding="utf-8")
    with (REPORTS / "EXP_R1_to_EXP_R80_chinese_summary.md").open("a", encoding="utf-8") as stream:
        stream.write(f"\n## EXP_R5\n\nEXP_R5 把 R4 的多步 proposal、线性路径、F1、F2 和图路径放进同一个 train-only confidence gate，根据目标邻域距离、动作连续性和训练支持性选择完整四步路径。它检验的是“候选路径筛选”能否补上 proposal 少量终点不到达的问题，而不是再训练更大的网络。EXP_R5 当前判定为 {decision['claim']}；如果仍失败，下一轮将显式学习状态条件的边代价并构造 waypoint 图。\n")
    print({"experiment": "EXP_R5", "selected": selected, "success": success, "next": decision["next_experiment"]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--stage", choices=("audit", "run"), required=True); parser.add_argument("--device", default="cpu"); parser.add_argument("--seed", type=int, default=4505); parser.add_argument("--max-cases", type=int, default=None); args = parser.parse_args()
    if args.stage == "audit": audit()
    else: run(args.device, args.seed, args.max_cases)


if __name__ == "__main__": main()

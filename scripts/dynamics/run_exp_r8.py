#!/usr/bin/env python3
"""Run EXP_R8 with an arrival-first development feasibility selector.

Purpose
-------
EXP_R7 showed that a stronger late terminal correction can satisfy the
held-out joint gate, while its single weighted development score selected a
weaker correction.  EXP_R8 tests the selection rule itself: a candidate must
first meet every frozen-baseline development lower/upper bound, then the
feasible candidate with the smallest development terminal distance is chosen
(decoded continuity breaks ties).  The selected candidate is evaluated once
on held-out episodes.

Parameters
----------
``--stage`` is ``audit`` or ``run``; ``--device`` defaults to ``cpu``;
``--seed`` controls proposal initialization; ``--max-cases`` is a debug-only
cap and is not used in the registered full run.

Usage
-----
PYTHONPATH=src:scripts/dynamics /home/jinjaguo/anaconda3/envs/libero/bin/python \\
  scripts/dynamics/run_exp_r8.py --stage audit --device cpu
PYTHONPATH=src:scripts/dynamics /home/jinjaguo/anaconda3/envs/libero/bin/python \\
  scripts/dynamics/run_exp_r8.py --stage run --device cpu

Outputs
-------
Artifacts are saved under ``results/EXP_R8/`` and ``reports/``.  The Chinese
summary is appended to ``reports/EXP_R1_to_EXP_R80_chinese_summary.md``.
Frozen representation, decoder, F1, and F2 checkpoints are untouched.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import torch

from run_exp_r1 import REPORTS, disk_audit, f_rollout, graph_plan, load_frozen_planners, write_json
from run_exp_r3 import build_cases, groups, metric
from run_exp_r4 import proposal_paths, target_texts, train_proposals
from run_exp_r7 import BETAS, SCHEDULES, repaired


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "EXP_R8"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def method_names() -> list[str]:
    return ["linear", "f1", "f2", "graph", "proposal_base", "proposal_repaired"] + [f"repair_{schedule}_{beta:.2f}" for schedule in SCHEDULES for beta in BETAS]


def audit() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases, _, _, info = build_cases(torch.device("cpu"))
    protocol = {
        "created_at": now(), "experiment": "EXP_R8", "parent": "EXP_R7", "disk": disk_audit(), "data": info,
        "selection": "development feasibility against each linear/F1/F2 bound, then minimum final_target_distance and decoded_first_difference",
        "methods": method_names(), "betas": list(BETAS), "schedules": {key: value.tolist() for key, value in SCHEDULES.items()},
        "frozen": ["representation", "decoder", "F1", "F2"], "heldout_opened": False,
    }
    write_json(OUT / "preregistration.json", protocol)
    (REPORTS / "EXP_R8_interface_audit.md").write_text("# EXP_R8 interface audit\n\nR8 changes only the train/development candidate-selection rule. It reuses the bounded four-step residual candidates from R7 and leaves all frozen model interfaces unchanged.\n", encoding="utf-8")
    (REPORTS / "EXP_R8_data_audit.md").write_text(f"# EXP_R8 data audit\n\nThe EXP_R3 episode-disjoint inventory is unchanged ({len(cases)} cases); train fits proposals, development selects a feasible candidate, and held-out opens once.\n", encoding="utf-8")
    print(protocol)


def run(device_name: str, seed: int, max_cases: int | None = None) -> None:
    device = torch.device(device_name)
    cases, representation, features, _ = build_cases(device, max_cases)
    f1, f2 = load_frozen_planners(device)
    train = [case for case in cases if case.split == "train"]
    development = [case for case in cases if case.split == "development"]
    pair, goal, radii = groups(cases)
    support = np.concatenate(list(goal.values()))
    texts = target_texts(cases, representation, features, device)
    models = train_proposals(train, development, representation, features, device, [seed, seed + 1, seed + 2, seed + 3])
    methods = method_names()
    dev = {method: [] for method in methods}; held = {method: [] for method in methods}
    for case in cases:
        target = pair.get((case.source_goal, case.goal), goal[case.goal])
        radius = radii.get((case.source_goal, case.goal), float(np.quantile(np.linalg.norm(target - target.mean(0), axis=1), 0.5)))
        endpoint = target[np.argmin(np.linalg.norm(target - case.latent[3][None], axis=1))]
        start = case.latent[3]
        linear = np.linspace(start, endpoint, 5, dtype=np.float32)[1:]
        base = proposal_paths(case, models, texts[case.goal], endpoint, support, device)[0]
        paths = {"linear": linear, "proposal_base": base, "proposal_repaired": repaired(base, endpoint, 0.20, np.asarray([0.10, 0.35, 0.70, 1.0], dtype=np.float32)), "f1": f_rollout(case, texts[case.goal], representation, f1, f2, "f1", device), "f2": f_rollout(case, texts[case.goal], representation, f1, f2, "f2", device), "graph": graph_plan(start, support, target)}
        for schedule_name, schedule in SCHEDULES.items():
            for beta in BETAS:
                paths[f"repair_{schedule_name}_{beta:.2f}"] = repaired(base, endpoint, beta, schedule)
        for method, planned in paths.items():
            t0 = time.perf_counter(); row = metric(case, planned, target, radius, goal, representation, f1, texts[case.goal], device); row["runtime_seconds"] = time.perf_counter() - t0
            if case.split == "development": dev[method].append(row)
            elif case.split == "heldout": held[method].append(row)

    def summarize(rows: list[dict]) -> dict:
        result = {"count": len(rows), "nonfinite_count": int(sum(bool(row["nonfinite"]) for row in rows))}
        for key, value in rows[0].items():
            if isinstance(value, (float, int, bool)) and key != "nonfinite": result[key] = float(np.mean([float(row[key]) for row in rows]))
        return result

    dev_s = {method: summarize(rows) for method, rows in dev.items()}; held_s = {method: summarize(rows) for method, rows in held.items()}
    baselines = ["linear", "f1", "f2"]
    lower_arrival = max(dev_s[baseline]["target_arrival"] for baseline in baselines)
    upper_diff = min(dev_s[baseline]["decoded_first_difference"] for baseline in baselines)
    upper_hidden = min(dev_s[baseline]["hidden_path_latent_mse"] for baseline in baselines)
    feasible = [method for method in methods if dev_s[method]["target_arrival"] >= lower_arrival and dev_s[method]["decoded_first_difference"] <= upper_diff and dev_s[method]["hidden_path_latent_mse"] <= upper_hidden]
    if not feasible:
        feasible = [method for method in methods if method not in baselines]
    selected = min(feasible, key=lambda method: (dev_s[method]["final_target_distance"], dev_s[method]["decoded_first_difference"], dev_s[method]["hidden_path_latent_mse"]))
    success = all(held_s[selected]["target_arrival"] >= held_s[baseline]["target_arrival"] and held_s[selected]["decoded_first_difference"] <= held_s[baseline]["decoded_first_difference"] and held_s[selected]["hidden_path_latent_mse"] <= held_s[baseline]["hidden_path_latent_mse"] for baseline in baselines)
    decision = {"experiment": "EXP_R8", "claim": "SUPPORTED" if success else "NOT_SUPPORTED", "success": bool(success), "selected_method_development": selected, "development_feasible": feasible, "heldout_opened_once": True, "next_experiment": None if success else "EXP_R9"}
    write_json(OUT / "development_metrics.json", {"per_method": dev, "summary": dev_s, "feasibility_bounds": {"arrival_lower": lower_arrival, "difference_upper": upper_diff, "hidden_mse_upper": upper_hidden}, "feasible": feasible})
    write_json(OUT / "heldout_metrics.json", {"per_method": held, "summary": held_s}); write_json(OUT / "claim_decision.json", decision); write_json(OUT / "final_candidate_selection.json", {"selected": selected, "feasible": feasible})
    report = "# EXP_R8 report — arrival-first feasible selection\n\nR8 first required development candidates to meet every frozen-baseline bound, then selected the smallest development terminal distance.\n\n| method | arrival | decoded first diff | hidden path MSE | F1 cost |\n|---|---:|---:|---:|---:|\n"
    for method in methods:
        row = held_s[method]; report += f"| {method} | {row['target_arrival']:.4f} | {row['decoded_first_difference']:.6f} | {row['hidden_path_latent_mse']:.4f} | {row['f1_dynamics_cost']:.4f} |\n"
    report += f"\nDevelopment feasible candidates: {', '.join(feasible)}. Selected `{selected}`. EXP_R8 is **{decision['claim']}** (`SUCCESS={decision['success']}`).\n"
    (REPORTS / "EXP_R8_report.md").write_text(report, encoding="utf-8")
    (REPORTS / "next_exp_fromR8.md").write_text(f"# Next experiment from EXP_R8\n\nEXP_R8 is **{decision['claim']}**. If it fails, EXP_R9 should learn a state-conditioned confidence/residual adapter and compare it with the fixed feasible selector. If it succeeds, remove oracle endpoint selection and test target-set planning.\n", encoding="utf-8")
    with (REPORTS / "EXP_R1_to_EXP_R80_chinese_summary.md").open("a", encoding="utf-8") as stream:
        stream.write(f"\n## EXP_R8\n\nEXP_R8 专门检验 R7 的选择规则：development 阶段先要求候选同时达到线性、F1、F2 的三项指标下界，再在可行候选中选择终点距离最小者，避免一个加权分数偏向过于保守的路径。EXP_R8 当前判定为 {decision['claim']}，development 选中 {selected}；如果仍失败，下一轮会训练状态条件的置信度/残差 adapter。\n")
    print({"experiment": "EXP_R8", "cases": len(cases), "selected": selected, "feasible": feasible, "success": success, "next": decision["next_experiment"]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--stage", choices=("audit", "run"), required=True); parser.add_argument("--device", default="cpu"); parser.add_argument("--seed", type=int, default=4808); parser.add_argument("--max-cases", type=int, default=None); args = parser.parse_args()
    if args.stage == "audit": audit()
    else: run(args.device, args.seed, args.max_cases)


if __name__ == "__main__": main()

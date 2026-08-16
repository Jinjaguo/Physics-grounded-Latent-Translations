#!/usr/bin/env python3
"""Run EXP_R7 bounded terminal-residual correction for latent paths.

Purpose
-------
Test whether EXP_R6's remaining arrival gap is caused only by a small
terminal residual.  A frozen four-step proposal is corrected toward the
train-only nearest target endpoint using fixed correction strengths and two
fixed schedules (distributed and late).  Development selects one candidate;
the episode-disjoint held-out split is then evaluated once.

Parameters
----------
``--stage`` is ``audit`` or ``run``; ``--device`` is ``cpu`` by default;
``--seed`` controls proposal initialization; ``--max-cases`` is a debug-only
case cap and is not used in the registered run.

Usage
-----
PYTHONPATH=src:scripts/dynamics /home/jinjaguo/anaconda3/envs/libero/bin/python \\
  scripts/dynamics/run_exp_r7.py --stage audit --device cpu
PYTHONPATH=src:scripts/dynamics /home/jinjaguo/anaconda3/envs/libero/bin/python \\
  scripts/dynamics/run_exp_r7.py --stage run --device cpu

Outputs
-------
Artifacts are written under ``results/EXP_R7/`` and ``reports/``; one
Chinese paragraph is appended to ``reports/EXP_R1_to_EXP_R80_chinese_summary.md``.
Frozen representation, decoder, F1, and F2 checkpoints are not modified.
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


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "EXP_R7"
BETAS = (0.10, 0.20, 0.35, 0.50, 0.75, 1.00)
SCHEDULES = {
    "distributed": np.asarray([0.10, 0.35, 0.70, 1.00], dtype=np.float32),
    "late": np.asarray([0.00, 0.10, 0.35, 1.00], dtype=np.float32),
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def repaired(base: np.ndarray, endpoint: np.ndarray, beta: float, schedule: np.ndarray) -> np.ndarray:
    correction = beta * schedule[:, None] * (endpoint[None] - base[-1][None])
    return (base + correction).astype(np.float32)


def audit() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases, _, _, info = build_cases(torch.device("cpu"))
    protocol = {
        "created_at": now(), "experiment": "EXP_R7", "parent": "EXP_R6",
        "disk": disk_audit(), "data": info,
        "betas": list(BETAS), "schedules": {key: value.tolist() for key, value in SCHEDULES.items()},
        "selection": "development-only arrival - 0.20 decoded first difference - 0.05 hidden path MSE",
        "methods": ["linear", "f1", "f2", "graph", "proposal_base", "proposal_repaired"] + [f"repair_{schedule}_{beta:.2f}" for schedule in SCHEDULES for beta in BETAS],
        "frozen": ["representation", "decoder", "F1", "F2"], "heldout_opened": False,
    }
    write_json(OUT / "preregistration.json", protocol)
    (REPORTS / "EXP_R7_interface_audit.md").write_text(
        "# EXP_R7 interface audit\n\nEXP_R7 only adds a bounded endpoint residual to the existing four-step proposal. "
        "The correction is computed from the train-only target region and is distributed over four waypoints by a fixed schedule. "
        "Representation, decoder, F1 and F2 remain frozen.\n", encoding="utf-8")
    (REPORTS / "EXP_R7_data_audit.md").write_text(
        f"# EXP_R7 data audit\n\nThe EXP_R3 complete-episode inventory is unchanged ({len(cases)} cases). "
        "Proposal fitting is train-only, correction selection is development-only, and held-out is opened once.\n", encoding="utf-8")
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
    methods = ["linear", "f1", "f2", "graph", "proposal_base", "proposal_repaired"] + [f"repair_{schedule}_{beta:.2f}" for schedule in SCHEDULES for beta in BETAS]
    dev = {method: [] for method in methods}
    held = {method: [] for method in methods}
    for case in cases:
        target = pair.get((case.source_goal, case.goal), goal[case.goal])
        radius = radii.get((case.source_goal, case.goal), float(np.quantile(np.linalg.norm(target - target.mean(0), axis=1), 0.5)))
        endpoint = target[np.argmin(np.linalg.norm(target - case.latent[3][None], axis=1))]
        start = case.latent[3]
        linear = np.linspace(start, endpoint, 5, dtype=np.float32)[1:]
        proposals = proposal_paths(case, models, texts[case.goal], endpoint, support, device)
        base = proposals[0]
        paths = {
            "linear": linear,
            "proposal_base": base,
            "proposal_repaired": proposals[-1],
            "f1": f_rollout(case, texts[case.goal], representation, f1, f2, "f1", device),
            "f2": f_rollout(case, texts[case.goal], representation, f1, f2, "f2", device),
            "graph": graph_plan(start, support, target),
        }
        for schedule_name, schedule in SCHEDULES.items():
            for beta in BETAS:
                paths[f"repair_{schedule_name}_{beta:.2f}"] = repaired(base, endpoint, beta, schedule)
        for method, planned in paths.items():
            t0 = time.perf_counter()
            row = metric(case, planned, target, radius, goal, representation, f1, texts[case.goal], device)
            row["runtime_seconds"] = time.perf_counter() - t0
            if case.split == "development":
                dev[method].append(row)
            elif case.split == "heldout":
                held[method].append(row)

    def summarize(rows: list[dict]) -> dict:
        result = {"count": len(rows), "nonfinite_count": int(sum(bool(row["nonfinite"]) for row in rows))}
        for key, value in rows[0].items():
            if isinstance(value, (float, int, bool)) and key != "nonfinite":
                result[key] = float(np.mean([float(row[key]) for row in rows]))
        return result

    dev_s = {method: summarize(rows) for method, rows in dev.items()}
    held_s = {method: summarize(rows) for method, rows in held.items()}
    scores = {method: dev_s[method]["target_arrival"] - 0.20 * dev_s[method]["decoded_first_difference"] - 0.05 * dev_s[method]["hidden_path_latent_mse"] for method in methods}
    selected = max(scores, key=scores.get)
    baselines = ["linear", "f1", "f2"]
    success = all(held_s[selected]["target_arrival"] >= held_s[baseline]["target_arrival"] and held_s[selected]["decoded_first_difference"] <= held_s[baseline]["decoded_first_difference"] and held_s[selected]["hidden_path_latent_mse"] <= held_s[baseline]["hidden_path_latent_mse"] for baseline in baselines)
    decision = {"experiment": "EXP_R7", "claim": "SUPPORTED" if success else "NOT_SUPPORTED", "success": bool(success), "selected_method_development": selected, "heldout_opened_once": True, "next_experiment": None if success else "EXP_R8"}
    write_json(OUT / "development_metrics.json", {"per_method": dev, "summary": dev_s, "selection_scores": scores})
    write_json(OUT / "heldout_metrics.json", {"per_method": held, "summary": held_s})
    write_json(OUT / "claim_decision.json", decision)
    write_json(OUT / "final_candidate_selection.json", {"selected": selected, "betas": list(BETAS), "schedules": {key: value.tolist() for key, value in SCHEDULES.items()}, "scores": scores})
    report = "# EXP_R7 report — bounded terminal residuals\n\nEXP_R7 compared distributed and late endpoint residual corrections against frozen baselines and the uncorrected proposal.\n\n| method | arrival | decoded first diff | hidden path MSE | F1 cost |\n|---|---:|---:|---:|---:|\n"
    for method in methods:
        row = held_s[method]
        report += f"| {method} | {row['target_arrival']:.4f} | {row['decoded_first_difference']:.6f} | {row['hidden_path_latent_mse']:.4f} | {row['f1_dynamics_cost']:.4f} |\n"
    report += f"\nDevelopment selected `{selected}`. EXP_R7 is **{decision['claim']}** (`SUCCESS={decision['success']}`).\n"
    (REPORTS / "EXP_R7_report.md").write_text(report, encoding="utf-8")
    (REPORTS / "next_exp_fromR7.md").write_text(f"# Next experiment from EXP_R7\n\nEXP_R7 is **{decision['claim']}**. If it fails, EXP_R8 should condition a residual adapter on source state, target language, and proposal confidence, and compare it with a conservative closed-loop correction. If it succeeds, remove oracle endpoint selection and test target-set planning.\n", encoding="utf-8")
    with (REPORTS / "EXP_R1_to_EXP_R80_chinese_summary.md").open("a", encoding="utf-8") as stream:
        stream.write(f"\n## EXP_R7\n\nEXP_R7 把 proposal 最后没有完全到达目标的问题单独拿出来，沿四个 waypoint 施加固定强度的终点残差，并比较均匀分布和最后集中两种修正方式。参数只在 development 选择，held-out 只评估一次。EXP_R7 当前判定为 {decision['claim']}，选中的方法是 {selected}；若仍失败，下一轮会让残差强度依赖当前状态、目标语言和 proposal 的置信度，而不是使用全局固定修正。\n")
    print({"experiment": "EXP_R7", "cases": len(cases), "selected": selected, "success": success, "next": decision["next_experiment"]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("audit", "run"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=4707)
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()
    if args.stage == "audit":
        audit()
    else:
        run(args.device, args.seed, args.max_cases)


if __name__ == "__main__":
    main()

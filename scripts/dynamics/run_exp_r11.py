#!/usr/bin/env python3
"""Run EXP_R11 robust latent-MPC selection under train-only disturbances.

Purpose
-------
Test whether EXP_R10's compliance-selection failure is caused by optimizing a
single average condition.  The proposal and R8 paths are evaluated across a
fixed compliance grid and positive/negative execution-space residual shocks
whose scale is estimated only from train transitions.  F1, old F2, and graph
remain mandatory controls at the central compliance.

Parameters
----------
``--stage`` is ``audit`` or ``run``; ``--device`` defaults to ``cpu``;
``--seed`` controls proposal initialization; ``--max-cases`` is a debug-only
cap and is not used for the registered full run.

Usage
-----
PYTHONPATH=src:scripts/dynamics /home/jinjaguo/anaconda3/envs/libero/bin/python \\
  scripts/dynamics/run_exp_r11.py --stage audit --device cpu
PYTHONPATH=src:scripts/dynamics /home/jinjaguo/anaconda3/envs/libero/bin/python \\
  scripts/dynamics/run_exp_r11.py --stage run --device cpu

Outputs
-------
Artifacts are written under ``results/EXP_R11/`` and ``reports/``; a Chinese
paragraph is appended to ``reports/EXP_R9_to_EXP_R58_chinese_summary.md``.
The frozen representation, decoder, F1, F2, and R8 planner are unchanged.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import torch

from run_exp_r1 import REPORTS, disk_audit, load_frozen_planners, sha, write_json
from run_exp_r3 import build_cases, groups
from run_exp_r4 import target_texts, train_proposals
from run_exp_r9 import initial_plan
from run_exp_r10 import build_nominal_index, cheap_metric, plant_step


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "EXP_R11"
COMPLIANCES = (0.25, 0.50, 0.75, 1.00)
ROBUST_METHODS = ("r8_open_loop", "proposal_h2_p2", "f1_closed_h4_p1", "old_f2_closed_h4_p1", "graph_mpc_h4_p1")
SHOCKS = ("negative", "positive")


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def train_sigma(index: dict) -> np.ndarray:
    currents, nexts = index["__all__"]
    sigma = np.std(nexts - currents, axis=0).astype(np.float32)
    sigma[:16] = 0.0
    return (0.15 * sigma).astype(np.float32)


def run_case(case: object, planner: str, compliance: float, shock: str, target: np.ndarray, radius: float, support: np.ndarray, text: np.ndarray, models: list[dict], representation: object, f1: object, f2: object, device: torch.device, index: dict, sigma: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    prefix = 4 if planner == "r8_open_loop" else (2 if planner == "proposal_h2_p2" else 1)
    sign = -1.0 if shock == "negative" else 1.0
    current = case.latent[3].copy(); previous = case.latent[2].copy(); planned_parts = []; actual_parts = []; consumed = 0; replans = 0; times = []
    while consumed < 4:
        t0 = time.perf_counter(); path = initial_plan(planner, current, previous, target, support, text, models, representation, f1, f2, device, 4, seed + consumed); take = min(prefix, 4 - consumed); commands = path[:take]; local = current; states = []
        for command in commands:
            nxt = plant_step(local, command, case, index, compliance) + sign * sigma
            states.append(nxt.astype(np.float32)); local = states[-1]
        planned_parts.append(commands.copy()); actual_parts.extend(states); previous, current = current, states[-1]; consumed += take; replans += 1; times.append(time.perf_counter() - t0)
    planned = np.concatenate(planned_parts, axis=0)[:4]; actual = np.stack(actual_parts[:4]); extra = {"actual_target_distance": float(np.min(np.linalg.norm(target - actual[-1][None], axis=1))), "actual_target_arrival": float(np.min(np.linalg.norm(target - actual[-1][None], axis=1)) <= radius), "actual_hidden_path_mse": float(np.mean((actual - case.latent[4:8]) ** 2)), "tracking_latent_mse": float(np.mean((planned - actual) ** 2)), "replans": replans, "mean_replan_runtime": float(np.mean(times)), "compliance": compliance, "shock": shock, "nonfinite": bool(not np.isfinite(planned).all() and not np.isfinite(actual).all())}
    return planned, actual, extra


def audit() -> None:
    OUT.mkdir(parents=True, exist_ok=True); cases, _, _, info = build_cases(torch.device("cpu")); frozen = {}
    for key, path in {"representation": ROOT / "checkpoints/representation/seed_810/correct_language/checkpoint_ema.pt", "f1": ROOT / "results/dynamics/fifteenth_wave/2026-08-12_dynamics_3/checkpoints/F1_execution_mlp.pt", "f2": ROOT / "results/dynamics/fifteenth_wave/2026-08-12_dynamics_3/checkpoints/F2_matched_refinement.pt", "r8_script": ROOT / "scripts/dynamics/run_exp_r8.py"}.items(): frozen[key] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
    protocol = {"created_at": now(), "experiment": "EXP_R11", "parent": "EXP_R10", "git_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "disk": disk_audit(), "data": info, "compliances": list(COMPLIANCES), "methods": list(ROBUST_METHODS), "shocks": list(SHOCKS), "shock_scale": "0.15 * train nominal execution-transition std; semantic dims zero", "selection": "development worst-case arrival minus worst continuity/path error", "frozen_manifest": frozen, "heldout_opened": False, "physical_claim": False}
    write_json(OUT / "preregistration.json", protocol); write_json(OUT / "frozen_manifest.json", frozen); (REPORTS / "EXP_R11_interface_audit.md").write_text("# EXP_R11 interface audit\n\nR11 preserves the R10 action-conditioned nominal plant and adds train-only execution residual shocks. Positive/negative shock scales are computed before development/held-out separation from train transitions only. R8, F1, old F2, graph, and proposal planners share the same frozen latent interface.\n", encoding="utf-8"); (REPORTS / "EXP_R11_data_audit.md").write_text(f"# EXP_R11 data audit\n\nThe EXP_R3 episode-disjoint inventory contains {len(cases)} cases. Train cases build the nominal plant and shock scale; development selects the robust candidate; held-out opens once.\n", encoding="utf-8"); print(protocol)


def run(device_name: str, seed: int, max_cases: int | None = None) -> None:
    device = torch.device(device_name); cases, representation, features, _ = build_cases(device, max_cases); f1, f2 = load_frozen_planners(device); train = [c for c in cases if c.split == "train"]; development = [c for c in cases if c.split == "development"]; eval_cases = [c for c in cases if c.split != "train"]; pair, goal, radii = groups(cases); support = np.concatenate(list(goal.values()))[::4].copy(); texts = target_texts(cases, representation, features, device); models = train_proposals(train, development, representation, features, device, [seed, seed + 1, seed + 2, seed + 3]); index = build_nominal_index(train); sigma = train_sigma(index)
    configs = [(method, compliance, shock) for method in ROBUST_METHODS for compliance in (COMPLIANCES if method in {"r8_open_loop", "proposal_h2_p2"} else (0.50,)) for shock in SHOCKS]; dev = {str(config): [] for config in configs}; held = {str(config): [] for config in configs}
    for case in eval_cases:
        target = pair.get((case.source_goal, case.goal), goal[case.goal]); radius = radii.get((case.source_goal, case.goal), float(np.quantile(np.linalg.norm(target - target.mean(0), axis=1), 0.5)))
        for method, compliance, shock in configs:
            key = str((method, compliance, shock)); t0 = time.perf_counter(); planned, actual, extra = run_case(case, method, compliance, shock, target, radius, support, texts[case.goal], models, representation, f1, f2, device, index, sigma, seed); row = cheap_metric(case, planned, actual, target, radius, representation, device, extra); row["runtime_seconds"] = time.perf_counter() - t0; (dev if case.split == "development" else held)[key].append(row)

    def summarize(rows: list[dict]) -> dict:
        result = {"count": len(rows), "nonfinite_count": int(sum(bool(r["nonfinite"]) for r in rows))}
        for key, value in rows[0].items():
            if isinstance(value, (float, int, bool)) and key != "nonfinite": result[key] = float(np.mean([float(r[key]) for r in rows]))
        return result
    dev_s = {key: summarize(rows) for key, rows in dev.items()}; held_s = {key: summarize(rows) for key, rows in held.items()}
    grouped_dev = {}
    grouped_held = {}
    for key, value in dev_s.items():
        method = eval(key)[0]; grouped_dev.setdefault(method, []).append(value)
    for key, value in held_s.items():
        method = eval(key)[0]; grouped_held.setdefault(method, []).append(value)
    robust_dev = {method: {"worst_arrival": min(v["actual_target_arrival"] for v in vals), "worst_diff": max(v["decoded_first_difference"] for v in vals), "worst_hidden": max(v["actual_hidden_path_mse"] for v in vals)} for method, vals in grouped_dev.items()}
    robust_held = {method: {"worst_arrival": min(v["actual_target_arrival"] for v in vals), "worst_diff": max(v["decoded_first_difference"] for v in vals), "worst_hidden": max(v["actual_hidden_path_mse"] for v in vals)} for method, vals in grouped_held.items()}
    scores = {method: values["worst_arrival"] - 0.20 * values["worst_diff"] - 0.05 * values["worst_hidden"] for method, values in robust_dev.items()}; selected = max(scores, key=scores.get); baseline = "r8_open_loop"; closed = selected != baseline and robust_held[selected]["worst_arrival"] >= robust_held[baseline]["worst_arrival"] and robust_held[selected]["worst_diff"] <= robust_held[baseline]["worst_diff"] and robust_held[selected]["worst_hidden"] <= robust_held[baseline]["worst_hidden"]
    decision = {"experiment": "EXP_R11", "claim": "SUPPORTED_ROBUST_SURROGATE" if closed else "NOT_SUPPORTED", "success": False, "surrogate_supported": bool(closed), "selected_method_development": selected, "heldout_opened_once": True, "overall_full_system": False, "next_experiment": "EXP_R12"}
    write_json(OUT / "development_metrics.json", {"per_config": dev, "summary": dev_s, "robust_summary": robust_dev, "selection_scores": scores}); write_json(OUT / "heldout_metrics.json", {"per_config": held, "summary": held_s, "robust_summary": robust_held}); write_json(OUT / "claim_decision.json", decision); write_json(OUT / "final_candidate_selection.json", {"selected": selected, "scores": scores, "robust_development": robust_dev})
    report = "# EXP_R11 report — robust latent MPC surrogate\n\nR11 selected candidates by worst-case development arrival, continuity, and hidden-path error across train-derived positive/negative execution shocks and compliance conditions.\n\n| method | worst arrival | worst decoded diff | worst hidden MSE |\n|---|---:|---:|---:|\n" + "".join(f"| {m} | {robust_held[m]['worst_arrival']:.4f} | {robust_held[m]['worst_diff']:.6f} | {robust_held[m]['worst_hidden']:.4f} |\n" for m in robust_held) + f"\nDevelopment selected `{selected}`. Robust surrogate claim: **{decision['claim']}**. Overall full-system success remains **false** because physical/exact feedback, learned F3, long-horizon sequencing, and return remain unavailable.\n"
    (REPORTS / "EXP_R11_report.md").write_text(report, encoding="utf-8"); (REPORTS / "next_exp_fromR11.md").write_text(f"# Next experiment from EXP_R11\n\nR11 tested worst-case selection under train-derived execution shocks. It selected `{selected}` with surrogate claim {decision['claim']}. If not supported, the next step should test an uncertainty-aware terminal-capture residual adapter and calibration of completion confidence while keeping representation, F1, F2, R8 and the nominal plant frozen. If supported, the next step may audit oracle F3 completion signals.\n", encoding="utf-8")
    summary = REPORTS / "EXP_R9_to_EXP_R58_chinese_summary.md"; with_open = summary.open("a", encoding="utf-8")
    with with_open as stream: stream.write(f"\n## EXP_R11\n\nEXP_R11 不再用单一 compliance 的平均分，而是在训练 transition 估计的执行残差正负扰动和多个 compliance 下，用 development 的最坏到达率、连续性和路径误差选择 proposal、R8、F1、旧 F2、graph。它测试了 robust latent MPC 的基本想法，但仍运行在 surrogate plant 上。development 选中 {selected}，held-out robust surrogate 判定为 {decision['claim']}；完整系统还没有达到成功条件，下一轮测试不确定性终点捕获和完成置信度。\n")
    print({"experiment": "EXP_R11", "cases": len(cases), "evaluated_cases": len(eval_cases), "selected": selected, "claim": decision["claim"], "overall_success": False, "next": decision["next_experiment"]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--stage", choices=("audit", "run"), required=True); parser.add_argument("--device", default="cpu"); parser.add_argument("--seed", type=int, default=5111); parser.add_argument("--max-cases", type=int, default=None); args = parser.parse_args(); audit() if args.stage == "audit" else run(args.device, args.seed, args.max_cases)


if __name__ == "__main__": main()

#!/usr/bin/env python3
"""Run EXP_R12 train-only target-set terminal-capture policies.

Purpose
-------
Test whether EXP_R11's robust arrival gap is caused by choosing one nearest
endpoint.  Proposal paths are generated toward several train-only target
points and selected by density, centroid margin, or an ensemble average.  R8,
F1, old F2, and graph remain controls on the same action-conditioned plant
with positive/negative train-derived shocks.

Parameters
----------
``--stage`` is ``audit`` or ``run``; ``--device`` defaults to ``cpu``;
``--seed`` controls proposal initialization; ``--max-cases`` is debug-only.

Usage
-----
PYTHONPATH=src:scripts/dynamics /home/jinjaguo/anaconda3/envs/libero/bin/python \\
  scripts/dynamics/run_exp_r12.py --stage audit --device cpu
PYTHONPATH=src:scripts/dynamics /home/jinjaguo/anaconda3/envs/libero/bin/python \\
  scripts/dynamics/run_exp_r12.py --stage run --device cpu

Outputs
-------
Artifacts are saved under ``results/EXP_R12/`` and ``reports/``; a Chinese
paragraph is appended to ``reports/EXP_R9_to_EXP_R58_chinese_summary.md``.
Frozen representation, decoder, F1, F2 and R8 are untouched.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import torch

from run_exp_r1 import REPORTS, disk_audit, graph_plan, load_frozen_planners, sha, write_json
from run_exp_r3 import build_cases, groups
from run_exp_r4 import target_texts, train_proposals
from run_exp_r9 import initial_plan
from run_exp_r10 import build_nominal_index, cheap_metric, plant_step
from run_exp_r11 import train_sigma


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "EXP_R12"
COMPLIANCES = (0.50, 0.75)
SHOCKS = ("negative", "positive")
POLICIES = ("r8_nearest", "proposal_density", "proposal_margin", "proposal_ensemble", "f1_closed_h4_p1", "old_f2_closed_h4_p1", "graph_mpc_h4_p1")
LATE_SCHEDULE = np.asarray([0.0, 0.10, 0.35, 1.0], dtype=np.float32)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def proposal_endpoint_path(current: np.ndarray, endpoint: np.ndarray, text: np.ndarray, models: list[dict], device: torch.device) -> np.ndarray:
    start = torch.tensor(current, dtype=torch.float32, device=device)[None]; txt = torch.tensor(text, dtype=torch.float32, device=device)[None]
    with torch.no_grad(): delta = models[0]["model"](start, txt)[0].cpu().numpy()
    base = (current[None] + np.cumsum(delta, axis=0)).astype(np.float32)
    return (base + 0.75 * LATE_SCHEDULE[:, None] * (endpoint[None] - base[-1][None])).astype(np.float32)


def choose_endpoints(current: np.ndarray, target: np.ndarray, policy: str) -> list[np.ndarray]:
    order = np.argsort(np.linalg.norm(target - current[None], axis=1)); nearest = target[order[: min(8, len(order))]]
    if policy == "r8_nearest": return [nearest[0]]
    if policy == "proposal_margin": return [nearest[-1]]
    if policy == "proposal_density":
        density = np.mean(np.linalg.norm(nearest[:, None] - nearest[None], axis=-1), axis=1)
        return [nearest[int(np.argmin(density))]]
    return [row for row in nearest]


def policy_path(current: np.ndarray, previous: np.ndarray, target: np.ndarray, policy: str, text: np.ndarray, models: list[dict], representation: object, f1: object, f2: object, support: np.ndarray, device: torch.device) -> np.ndarray:
    if policy in {"r8_nearest", "proposal_density", "proposal_margin", "proposal_ensemble"}:
        endpoints = choose_endpoints(current, target, policy); paths = [proposal_endpoint_path(current, endpoint, text, models, device) for endpoint in endpoints]
        if policy == "proposal_ensemble": return np.mean(np.stack(paths), axis=0).astype(np.float32)
        return paths[0]
    return initial_plan(policy, current, previous, target, support, text, models, representation, f1, f2, device, 4, 0)


def run_case(case: object, policy: str, compliance: float, shock: str, target: np.ndarray, radius: float, support: np.ndarray, text: np.ndarray, models: list[dict], representation: object, f1: object, f2: object, device: torch.device, index: dict, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    prefix = 4 if policy == "r8_nearest" else 2 if policy.startswith("proposal") else 1; sign = -1.0 if shock == "negative" else 1.0; current = case.latent[3].copy(); previous = case.latent[2].copy(); planned_parts = []; actual_parts = []; consumed = 0; replans = 0; times = []
    while consumed < 4:
        t0 = time.perf_counter(); path = policy_path(current, previous, target, policy, text, models, representation, f1, f2, support, device); take = min(prefix, 4 - consumed); commands = path[:take]; local = current; states = []
        for command in commands:
            nxt = plant_step(local, command, case, index, compliance) + sign * sigma; states.append(nxt.astype(np.float32)); local = states[-1]
        planned_parts.append(commands.copy()); actual_parts.extend(states); previous, current = current, states[-1]; consumed += take; replans += 1; times.append(time.perf_counter() - t0)
    planned = np.concatenate(planned_parts, axis=0)[:4]; actual = np.stack(actual_parts[:4]); extra = {"actual_target_distance": float(np.min(np.linalg.norm(target - actual[-1][None], axis=1))), "actual_target_arrival": float(np.min(np.linalg.norm(target - actual[-1][None], axis=1)) <= radius), "actual_hidden_path_mse": float(np.mean((actual - case.latent[4:8]) ** 2)), "tracking_latent_mse": float(np.mean((planned - actual) ** 2)), "replans": replans, "mean_replan_runtime": float(np.mean(times)), "compliance": compliance, "shock": shock, "nonfinite": bool(not np.isfinite(planned).all() or not np.isfinite(actual).all())}
    return planned, actual, extra


def audit() -> None:
    OUT.mkdir(parents=True, exist_ok=True); cases, _, _, info = build_cases(torch.device("cpu")); frozen = {}
    for key, path in {"representation": ROOT / "checkpoints/representation/seed_810/correct_language/checkpoint_ema.pt", "f1": ROOT / "results/dynamics/fifteenth_wave/2026-08-12_dynamics_3/checkpoints/F1_execution_mlp.pt", "f2": ROOT / "results/dynamics/fifteenth_wave/2026-08-12_dynamics_3/checkpoints/F2_matched_refinement.pt", "r8_script": ROOT / "scripts/dynamics/run_exp_r8.py"}.items(): frozen[key] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
    protocol = {"created_at": now(), "experiment": "EXP_R12", "parent": "EXP_R11", "git_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "disk": disk_audit(), "data": info, "policies": list(POLICIES), "compliances": list(COMPLIANCES), "shocks": list(SHOCKS), "target_candidates": "eight nearest train-only target points; density/margin/ensemble policies", "frozen_manifest": frozen, "heldout_opened": False, "physical_claim": False}
    write_json(OUT / "preregistration.json", protocol); write_json(OUT / "frozen_manifest.json", frozen); (REPORTS / "EXP_R12_interface_audit.md").write_text("# EXP_R12 interface audit\n\nR12 keeps the R11 nominal plant, train-only shock scale, and frozen model interfaces. It changes only how a target action region supplies terminal candidates: nearest, local-density, margin, or an average of eight train-only endpoints.\n", encoding="utf-8"); (REPORTS / "EXP_R12_data_audit.md").write_text(f"# EXP_R12 data audit\n\nThe EXP_R3 episode-disjoint inventory contains {len(cases)} cases. Target candidates, plant transitions, and shock scale use train cases only; development selects a policy and held-out opens once.\n", encoding="utf-8"); print(protocol)


def run(device_name: str, seed: int, max_cases: int | None = None) -> None:
    device = torch.device(device_name); cases, representation, features, _ = build_cases(device, max_cases); f1, f2 = load_frozen_planners(device); train = [c for c in cases if c.split == "train"]; development = [c for c in cases if c.split == "development"]; eval_cases = [c for c in cases if c.split != "train"]; pair, goal, radii = groups(cases); support = np.concatenate(list(goal.values()))[::4].copy(); texts = target_texts(cases, representation, features, device); models = train_proposals(train, development, representation, features, device, [seed, seed + 1, seed + 2, seed + 3]); index = build_nominal_index(train); sigma = train_sigma(index)
    configs = [(policy, compliance, shock) for policy in POLICIES for compliance in (COMPLIANCES if policy.startswith("proposal") or policy == "r8_nearest" else (0.50,)) for shock in SHOCKS]; dev = {str(c): [] for c in configs}; held = {str(c): [] for c in configs}
    for case in eval_cases:
        target = pair.get((case.source_goal, case.goal), goal[case.goal]); radius = radii.get((case.source_goal, case.goal), float(np.quantile(np.linalg.norm(target - target.mean(0), axis=1), 0.5)))
        for policy, compliance, shock in configs:
            key = str((policy, compliance, shock)); t0 = time.perf_counter(); planned, actual, extra = run_case(case, policy, compliance, shock, target, radius, support, texts[case.goal], models, representation, f1, f2, device, index, sigma); row = cheap_metric(case, planned, actual, target, radius, representation, device, extra); row["runtime_seconds"] = time.perf_counter() - t0; (dev if case.split == "development" else held)[key].append(row)

    def summarize(rows: list[dict]) -> dict:
        result = {"count": len(rows), "nonfinite_count": int(sum(bool(r["nonfinite"]) for r in rows))}
        for key, value in rows[0].items():
            if isinstance(value, (float, int, bool)) and key != "nonfinite": result[key] = float(np.mean([float(r[key]) for r in rows])
            )
        return result
    dev_s = {key: summarize(rows) for key, rows in dev.items()}; held_s = {key: summarize(rows) for key, rows in held.items()}; grouped_dev = {}; grouped_held = {}
    for key, value in dev_s.items(): grouped_dev.setdefault(eval(key)[0], []).append(value)
    for key, value in held_s.items(): grouped_held.setdefault(eval(key)[0], []).append(value)
    robust_dev = {m: {"worst_arrival": min(v["actual_target_arrival"] for v in vals), "worst_diff": max(v["decoded_first_difference"] for v in vals), "worst_hidden": max(v["actual_hidden_path_mse"] for v in vals)} for m, vals in grouped_dev.items()}; robust_held = {m: {"worst_arrival": min(v["actual_target_arrival"] for v in vals), "worst_diff": max(v["decoded_first_difference"] for v in vals), "worst_hidden": max(v["actual_hidden_path_mse"] for v in vals)} for m, vals in grouped_held.items()}; scores = {m: v["worst_arrival"] - 0.20 * v["worst_diff"] - 0.05 * v["worst_hidden"] for m, v in robust_dev.items()}; selected = max(scores, key=scores.get); baseline = "r8_nearest"; closed = selected != baseline and robust_held[selected]["worst_arrival"] >= robust_held[baseline]["worst_arrival"] and robust_held[selected]["worst_diff"] <= robust_held[baseline]["worst_diff"] and robust_held[selected]["worst_hidden"] <= robust_held[baseline]["worst_hidden"]
    decision = {"experiment": "EXP_R12", "claim": "SUPPORTED_TARGET_SET_SURROGATE" if closed else "NOT_SUPPORTED", "success": False, "surrogate_supported": bool(closed), "selected_method_development": selected, "heldout_opened_once": True, "overall_full_system": False, "next_experiment": "EXP_R13"}
    write_json(OUT / "development_metrics.json", {"per_config": dev, "summary": dev_s, "robust_summary": robust_dev, "selection_scores": scores}); write_json(OUT / "heldout_metrics.json", {"per_config": held, "summary": held_s, "robust_summary": robust_held}); write_json(OUT / "claim_decision.json", decision); write_json(OUT / "final_candidate_selection.json", {"selected": selected, "scores": scores, "robust_development": robust_dev})
    report = "# EXP_R12 report — target-set terminal capture\n\nR12 compared train-only endpoint policies under compliance and execution shocks.\n\n| policy | worst arrival | worst decoded diff | worst hidden MSE |\n|---|---:|---:|---:|\n" + "".join(f"| {m} | {robust_held[m]['worst_arrival']:.4f} | {robust_held[m]['worst_diff']:.6f} | {robust_held[m]['worst_hidden']:.4f} |\n" for m in robust_held) + f"\nDevelopment selected `{selected}`. Target-set surrogate claim: **{decision['claim']}**. Overall full-system success remains **false**.\n"
    (REPORTS / "EXP_R12_report.md").write_text(report, encoding="utf-8"); (REPORTS / "next_exp_fromR12.md").write_text(f"# Next experiment from EXP_R12\n\nR12 tested nearest, density, margin, and ensemble target-set terminal capture under the R11 surrogate plant. It selected `{selected}` with claim {decision['claim']}. The next experiment should test causal subgoal completion confidence from observable robot/latent history while retaining oracle F3 for F2, because endpoint choice is not yet a robust solution.\n", encoding="utf-8")
    summary = REPORTS / "EXP_R9_to_EXP_R58_chinese_summary.md"
    with summary.open("a", encoding="utf-8") as stream:
        stream.write(f"\n## EXP_R12\n\nEXP_R12 不再只选一个最近终点，而是从训练目标区域取多个 endpoint，比较最近点、局部密度、边界余量和 ensemble 平均路径，并在 R11 的 compliance 与正负扰动下做最坏情况选择。它检验了终点捕获是否是 R11 到达率下降的原因；development 选中 {selected}，held-out target-set surrogate 判定为 {decision['claim']}，完整系统仍未成功，下一轮转向可观测历史上的 subgoal completion confidence。\n")
    print({"experiment": "EXP_R12", "cases": len(cases), "evaluated_cases": len(eval_cases), "selected": selected, "claim": decision["claim"], "overall_success": False, "next": decision["next_experiment"]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--stage", choices=("audit", "run"), required=True); parser.add_argument("--device", default="cpu"); parser.add_argument("--seed", type=int, default=5212); parser.add_argument("--max-cases", type=int, default=None); args = parser.parse_args(); audit() if args.stage == "audit" else run(args.device, args.seed, args.max_cases)


if __name__ == "__main__": main()

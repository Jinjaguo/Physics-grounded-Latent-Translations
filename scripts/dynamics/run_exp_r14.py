#!/usr/bin/env python3
"""Run EXP_R14 progress-gated F2 target-authority schedules.

Purpose
-------
Use the failed EXP_R13 completion detector only as a continuous target-weight
signal, never as a hard F3 switch.  Compare fixed R8 terminal capture,
F1-only continuation, an early-dynamics/late-goal blend, and a confidence-
smoothed blend on the R11 train-only action-conditioned plant with shocks.

Parameters
----------
``--stage`` is ``audit`` or ``run``; ``--device`` defaults to ``cpu``;
``--seed`` controls proposal initialization; ``--max-cases`` is debug-only.

Usage
-----
PYTHONPATH=src:scripts/dynamics /home/jinjaguo/anaconda3/envs/libero/bin/python \\
  scripts/dynamics/run_exp_r14.py --stage audit --device cpu
PYTHONPATH=src:scripts/dynamics /home/jinjaguo/anaconda3/envs/libero/bin/python \\
  scripts/dynamics/run_exp_r14.py --stage run --device cpu

Outputs
-------
Artifacts are saved under ``results/EXP_R14/`` and ``reports/``; one Chinese
paragraph is appended to ``reports/EXP_R9_to_EXP_R58_chinese_summary.md``.
No learned F3 is promoted.
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
from run_exp_r4 import proposal_paths, target_texts, train_proposals
from run_exp_r9 import initial_plan
from run_exp_r10 import build_nominal_index, cheap_metric, plant_step
from run_exp_r11 import train_sigma


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "EXP_R14"
COMPLIANCES = (0.50, 0.75)
SHOCKS = ("negative", "positive")
METHODS = ("r8_fixed", "f1_nominal", "early_dyn_late_goal", "confidence_smoothed")


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def blended_path(current: np.ndarray, target: np.ndarray, text: np.ndarray, models: list[dict], representation: object, f1: object, f2: object, device: torch.device, method: str, previous: np.ndarray, support: np.ndarray) -> np.ndarray:
    endpoint = target[np.argmin(np.linalg.norm(target - current[None], axis=1))]
    proposal = proposal_paths(type("Case", (), {"latent": np.stack((previous, previous, previous, current))})(), models, text, endpoint, support, device)[0]
    f1_path = initial_plan("f1_closed_h4_p1", current, previous, target, support, text, models, representation, f1, f2, device, 4, 0)
    if method == "r8_fixed": return proposal
    if method == "f1_nominal": return f1_path
    distance = float(np.linalg.norm(endpoint - current)); progress = float(np.clip(1.0 - distance / max(distance + 1.0, 1e-6), 0.0, 1.0))
    if method == "early_dyn_late_goal": alpha = np.asarray([0.10, 0.35, 0.70, 1.0], dtype=np.float32)
    else: alpha = np.clip(0.25 + 0.75 * progress, 0.0, 1.0) * np.asarray([0.20, 0.45, 0.75, 1.0], dtype=np.float32)
    return ((1.0 - alpha[:, None]) * f1_path + alpha[:, None] * proposal).astype(np.float32)


def run_case(case: object, method: str, compliance: float, shock: str, target: np.ndarray, radius: float, support: np.ndarray, text: np.ndarray, models: list[dict], representation: object, f1: object, f2: object, device: torch.device, index: dict, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    sign = -1.0 if shock == "negative" else 1.0; current = case.latent[3].copy(); previous = case.latent[2].copy(); planned_parts = []; actual_parts = []; consumed = 0; times = []
    while consumed < 4:
        t0 = time.perf_counter(); path = blended_path(current, target, text, models, representation, f1, f2, device, method, previous, support); take = min(1, 4 - consumed); command = path[0]; nxt = plant_step(current, command, case, index, compliance) + sign * sigma; planned_parts.append(command[None]); actual_parts.append(nxt[None]); previous, current = current, nxt.astype(np.float32); consumed += take; times.append(time.perf_counter() - t0)
    planned = np.concatenate(planned_parts, axis=0); actual = np.concatenate(actual_parts, axis=0); extra = {"actual_target_distance": float(np.min(np.linalg.norm(target - actual[-1][None], axis=1))), "actual_target_arrival": float(np.min(np.linalg.norm(target - actual[-1][None], axis=1)) <= radius), "actual_hidden_path_mse": float(np.mean((actual - case.latent[4:8]) ** 2)), "tracking_latent_mse": float(np.mean((planned - actual) ** 2)), "replans": 4, "mean_replan_runtime": float(np.mean(times)), "compliance": compliance, "shock": shock, "nonfinite": bool(not np.isfinite(planned).all() or not np.isfinite(actual).all())}
    return planned, actual, extra


def audit() -> None:
    OUT.mkdir(parents=True, exist_ok=True); cases, _, _, info = build_cases(torch.device("cpu")); frozen = {}
    for key, path in {"representation": ROOT / "checkpoints/representation/seed_810/correct_language/checkpoint_ema.pt", "f1": ROOT / "results/dynamics/fifteenth_wave/2026-08-12_dynamics_3/checkpoints/F1_execution_mlp.pt", "f2": ROOT / "results/dynamics/fifteenth_wave/2026-08-12_dynamics_3/checkpoints/F2_matched_refinement.pt", "r8_script": ROOT / "scripts/dynamics/run_exp_r8.py"}.items(): frozen[key] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
    protocol = {"created_at": now(), "experiment": "EXP_R14", "parent": "EXP_R13", "disk": disk_audit(), "data": info, "methods": list(METHODS), "compliances": list(COMPLIANCES), "shocks": list(SHOCKS), "f3_status": "readiness_not_supported; confidence used only as continuous F2 blend", "frozen_manifest": frozen, "heldout_opened": False, "physical_claim": False}
    write_json(OUT / "preregistration.json", protocol); write_json(OUT / "frozen_manifest.json", frozen); (REPORTS / "EXP_R14_interface_audit.md").write_text("# EXP_R14 interface audit\n\nR14 does not integrate F3. It uses train-only state distance as a continuous target-authority schedule between frozen F1 nominal continuation and the R8 proposal. The same R11 nominal plant, shock scale, split, and frozen checkpoints are retained.\n", encoding="utf-8"); (REPORTS / "EXP_R14_data_audit.md").write_text(f"# EXP_R14 data audit\n\nThe EXP_R3 episode-disjoint inventory contains {len(cases)} cases; train cases fit proposal/target statistics, development selects a schedule, and held-out opens once.\n", encoding="utf-8"); print(protocol)


def run(device_name: str, seed: int, max_cases: int | None = None) -> None:
    device = torch.device(device_name); cases, representation, features, _ = build_cases(device, max_cases); f1, f2 = load_frozen_planners(device); train = [c for c in cases if c.split == "train"]; development = [c for c in cases if c.split == "development"]; eval_cases = [c for c in cases if c.split != "train"]; pair, goal, radii = groups(cases); support = np.concatenate(list(goal.values()))[::4].copy(); texts = target_texts(cases, representation, features, device); models = train_proposals(train, development, representation, features, device, [seed, seed + 1, seed + 2, seed + 3]); index = build_nominal_index(train); sigma = train_sigma(index); configs = [(method, c, s) for method in METHODS for c in COMPLIANCES for s in SHOCKS]; dev = {str(c): [] for c in configs}; held = {str(c): [] for c in configs}
    for case in eval_cases:
        target = pair.get((case.source_goal, case.goal), goal[case.goal]); radius = radii.get((case.source_goal, case.goal), float(np.quantile(np.linalg.norm(target - target.mean(0), axis=1), 0.5)))
        for method, compliance, shock in configs:
            key = str((method, compliance, shock)); t0 = time.perf_counter(); planned, actual, extra = run_case(case, method, compliance, shock, target, radius, support, texts[case.goal], models, representation, f1, f2, device, index, sigma); row = cheap_metric(case, planned, actual, target, radius, representation, device, extra); row["runtime_seconds"] = time.perf_counter() - t0; (dev if case.split == "development" else held)[key].append(row)
    def summarize(rows: list[dict]) -> dict:
        result = {"count": len(rows), "nonfinite_count": int(sum(bool(r["nonfinite"]) for r in rows))}
        for key, value in rows[0].items():
            if isinstance(value, (float, int, bool)) and key != "nonfinite": result[key] = float(np.mean([float(r[key]) for r in rows]))
        return result
    dev_s = {k: summarize(v) for k, v in dev.items()}; held_s = {k: summarize(v) for k, v in held.items()}; grouped_dev = {}; grouped_held = {}
    for k, v in dev_s.items(): grouped_dev.setdefault(eval(k)[0], []).append(v)
    for k, v in held_s.items(): grouped_held.setdefault(eval(k)[0], []).append(v)
    robust_dev = {m: {"worst_arrival": min(v["actual_target_arrival"] for v in vals), "worst_diff": max(v["decoded_first_difference"] for v in vals), "worst_hidden": max(v["actual_hidden_path_mse"] for v in vals)} for m, vals in grouped_dev.items()}; robust_held = {m: {"worst_arrival": min(v["actual_target_arrival"] for v in vals), "worst_diff": max(v["decoded_first_difference"] for v in vals), "worst_hidden": max(v["actual_hidden_path_mse"] for v in vals)} for m, vals in grouped_held.items()}; scores = {m: v["worst_arrival"] - .20 * v["worst_diff"] - .05 * v["worst_hidden"] for m, v in robust_dev.items()}; selected = max(scores, key=scores.get); closed = selected != "r8_fixed" and robust_held[selected]["worst_arrival"] >= robust_held["r8_fixed"]["worst_arrival"] and robust_held[selected]["worst_diff"] <= robust_held["r8_fixed"]["worst_diff"] and robust_held[selected]["worst_hidden"] <= robust_held["r8_fixed"]["worst_hidden"]
    decision = {"experiment": "EXP_R14", "claim": "SUPPORTED_PROGRESS_GATED_SURROGATE" if closed else "NOT_SUPPORTED", "success": False, "surrogate_supported": bool(closed), "selected_method_development": selected, "heldout_opened_once": True, "overall_full_system": False, "next_experiment": "EXP_R15"}
    write_json(OUT / "development_metrics.json", {"per_config": dev, "summary": dev_s, "robust_summary": robust_dev, "selection_scores": scores}); write_json(OUT / "heldout_metrics.json", {"per_config": held, "summary": held_s, "robust_summary": robust_held}); write_json(OUT / "claim_decision.json", decision); write_json(OUT / "final_candidate_selection.json", {"selected": selected, "scores": scores, "robust_development": robust_dev})
    report = "# EXP_R14 report — progress-gated target authority\n\nR14 used completion-distance only as a continuous F2 blend, not a hard F3 switch.\n\n| method | worst arrival | worst decoded diff | worst hidden MSE |\n|---|---:|---:|---:|\n" + "".join(f"| {m} | {robust_held[m]['worst_arrival']:.4f} | {robust_held[m]['worst_diff']:.6f} | {robust_held[m]['worst_hidden']:.4f} |\n" for m in robust_held) + f"\nDevelopment selected `{selected}`. Progress-gated surrogate claim: **{decision['claim']}**. Overall full-system success remains **false**.\n"
    (REPORTS / "EXP_R14_report.md").write_text(report, encoding="utf-8"); (REPORTS / "next_exp_fromR14.md").write_text(f"# Next experiment from EXP_R14\n\nR14 tested continuous F1-to-R8 target-authority schedules without integrating F3. It selected `{selected}` with claim {decision['claim']}. If unsupported, EXP_R15 should use a calibrated uncertainty-aware terminal capture controller and preserve hard late gating only after confidence calibration.\n", encoding="utf-8")
    summary = REPORTS / "EXP_R9_to_EXP_R58_chinese_summary.md"
    with summary.open("a", encoding="utf-8") as stream: stream.write(f"\n## EXP_R14\n\nEXP_R14 把 R13 的 completion 信号只当成连续 target 权重，不做硬切换；比较固定 R8、F1 nominal、early-dynamics/late-goal 和 confidence-smoothed schedule，在 surrogate compliance 和正负扰动下选择最坏情况。development 选中 {selected}，held-out 判定为 {decision['claim']}；连续权重仍未让完整闭环系统成功，下一轮测试校准后的不确定性终点捕获。\n")
    print({"experiment": "EXP_R14", "cases": len(cases), "evaluated_cases": len(eval_cases), "selected": selected, "claim": decision["claim"], "overall_success": False, "next": decision["next_experiment"]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--stage", choices=("audit", "run"), required=True); parser.add_argument("--device", default="cpu"); parser.add_argument("--seed", type=int, default=5414); parser.add_argument("--max-cases", type=int, default=None); args = parser.parse_args(); audit() if args.stage == "audit" else run(args.device, args.seed, args.max_cases)


if __name__ == "__main__": main()

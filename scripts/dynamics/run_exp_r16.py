#!/usr/bin/env python3
"""Run EXP_R16 history-conditioned latent plant surrogate.

Purpose
-------
Address the R10--R15 diagnosis that a current-state-only nominal plant misses
causal momentum/history.  The train-only surrogate now matches
previous/current latent pairs and source→goal before producing a nominal next
latent, then applies command compliance and fixed execution shocks.

Parameters
----------
``--stage`` is ``audit`` or ``run``; ``--device`` defaults to ``cpu``;
``--seed`` controls proposal initialization; ``--max-cases`` is debug-only.

Usage
-----
PYTHONPATH=src:scripts/dynamics /home/jinjaguo/anaconda3/envs/libero/bin/python \\
  scripts/dynamics/run_exp_r16.py --stage audit --device cpu
PYTHONPATH=src:scripts/dynamics /home/jinjaguo/anaconda3/envs/libero/bin/python \\
  scripts/dynamics/run_exp_r16.py --stage run --device cpu

Outputs
-------
Artifacts are saved under ``results/EXP_R16/`` and ``reports/``; a Chinese
paragraph is appended to ``reports/EXP_R9_to_EXP_R58_chinese_summary.md``.
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
from run_exp_r10 import cheap_metric
from run_exp_r11 import train_sigma

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "EXP_R16"
COMPLIANCES = (0.50, 0.75)
SHOCKS = ("negative", "positive")
METHODS = ("r8_fixed", "proposal_h2_p2", "f1_closed_h4_p1", "old_f2_closed_h4_p1", "graph_mpc_h4_p1")
SCHEDULE = np.asarray([0.0, 0.10, 0.35, 1.0], dtype=np.float32)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def build_history_index(train_cases: list) -> dict:
    grouped = {}
    for case in train_cases:
        for i in range(3, 7):
            grouped.setdefault((case.source_goal, case.goal), []).append((case.latent[i - 1], case.latent[i], case.latent[i + 1]))
    result = {}
    all_items = []
    for key, items in grouped.items():
        prev = np.stack([x[0] for x in items]); cur = np.stack([x[1] for x in items]); nxt = np.stack([x[2] for x in items]); result[key] = (prev, cur, nxt); all_items.extend(items)
    result["__all__"] = (np.stack([x[0] for x in all_items]), np.stack([x[1] for x in all_items]), np.stack([x[2] for x in all_items]))
    return result


def history_nominal(previous: np.ndarray, current: np.ndarray, case: object, index: dict) -> np.ndarray:
    key = (case.source_goal, case.goal); prev, cur, nxt = index.get(key, index["__all__"]); distance = np.mean((prev - previous[None]) ** 2, axis=1) + np.mean((cur - current[None]) ** 2, axis=1); return nxt[int(np.argmin(distance))].copy()


def run_case(case: object, method: str, compliance: float, shock: str, target: np.ndarray, radius: float, support: np.ndarray, text: np.ndarray, models: list[dict], representation: object, f1: object, f2: object, device: torch.device, index: dict, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    sign = -1.0 if shock == "negative" else 1.0; prefix = 4 if method == "r8_fixed" else 2 if method == "proposal_h2_p2" else 1; current = case.latent[3].copy(); previous = case.latent[2].copy(); planned_parts = []; actual_parts = []; consumed = 0; times = []
    while consumed < 4:
        t0 = time.perf_counter(); path = initial_plan("r8_open_loop" if method == "r8_fixed" else method, current, previous, target, support, text, models, representation, f1, f2, device, 4, 0); take = min(prefix, 4 - consumed); commands = path[:take]; local_prev = previous; local = current; states = []
        for command in commands:
            nominal = history_nominal(local_prev, local, case, index); nxt = nominal + compliance * (command - nominal) + sign * sigma; states.append(nxt.astype(np.float32)); local_prev, local = local, states[-1]
        planned_parts.append(commands.copy()); actual_parts.extend(states); previous, current = current, states[-1]; consumed += take; times.append(time.perf_counter() - t0)
    planned = np.concatenate(planned_parts, axis=0)[:4]; actual = np.stack(actual_parts[:4]); extra = {"actual_target_distance": float(np.min(np.linalg.norm(target - actual[-1][None], axis=1))), "actual_target_arrival": float(np.min(np.linalg.norm(target - actual[-1][None], axis=1)) <= radius), "actual_hidden_path_mse": float(np.mean((actual - case.latent[4:8]) ** 2)), "tracking_latent_mse": float(np.mean((planned - actual) ** 2)), "replans": int(4 / prefix), "mean_replan_runtime": float(np.mean(times)), "compliance": compliance, "shock": shock, "nonfinite": bool(not np.isfinite(planned).all() or not np.isfinite(actual).all())}
    return planned, actual, extra


def audit() -> None:
    OUT.mkdir(parents=True, exist_ok=True); cases, _, _, info = build_cases(torch.device("cpu")); frozen = {}
    for key, path in {"representation": ROOT / "checkpoints/representation/seed_810/correct_language/checkpoint_ema.pt", "f1": ROOT / "results/dynamics/fifteenth_wave/2026-08-12_dynamics_3/checkpoints/F1_execution_mlp.pt", "f2": ROOT / "results/dynamics/fifteenth_wave/2026-08-12_dynamics_3/checkpoints/F2_matched_refinement.pt", "r8_script": ROOT / "scripts/dynamics/run_exp_r8.py"}.items(): frozen[key] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
    protocol = {"created_at": now(), "experiment": "EXP_R16", "parent": "EXP_R15", "disk": disk_audit(), "data": info, "methods": list(METHODS), "compliances": list(COMPLIANCES), "shocks": list(SHOCKS), "plant": "nearest train previous/current/source/goal transition plus compliance command response", "frozen_manifest": frozen, "heldout_opened": False, "physical_claim": False}
    write_json(OUT / "preregistration.json", protocol); write_json(OUT / "frozen_manifest.json", frozen); (REPORTS / "EXP_R16_interface_audit.md").write_text("# EXP_R16 interface audit\n\nR16 changes only the surrogate plant's nominal lookup from current-only to previous/current history-conditioned matching. Planner, representation, decoder, F1, old F2, R8, split, compliance, and shock definitions remain frozen.\n", encoding="utf-8"); (REPORTS / "EXP_R16_data_audit.md").write_text(f"# EXP_R16 data audit\n\nThe EXP_R3 episode-disjoint inventory contains {len(cases)} cases; history index uses train cases only, development selects, and held-out opens once.\n", encoding="utf-8"); print(protocol)


def run(device_name: str, seed: int, max_cases: int | None = None) -> None:
    device = torch.device(device_name); cases, representation, features, _ = build_cases(device, max_cases); f1, f2 = load_frozen_planners(device); train = [c for c in cases if c.split == "train"]; development = [c for c in cases if c.split == "development"]; eval_cases = [c for c in cases if c.split != "train"]; pair, goal, radii = groups(cases); support = np.concatenate(list(goal.values()))[::4].copy(); texts = target_texts(cases, representation, features, device); models = train_proposals(train, development, representation, features, device, [seed, seed + 1, seed + 2, seed + 3]); index = build_history_index(train); sigma = train_sigma({"__all__": (index["__all__"][1], index["__all__"][2])})
    configs = [(m, c, s) for m in METHODS for c in COMPLIANCES for s in SHOCKS]; dev = {str(x): [] for x in configs}; held = {str(x): [] for x in configs}
    for case in eval_cases:
        target = pair.get((case.source_goal, case.goal), goal[case.goal]); radius = radii.get((case.source_goal, case.goal), float(np.quantile(np.linalg.norm(target - target.mean(0), axis=1), 0.5)))
        for method, compliance, shock in configs:
            key = str((method, compliance, shock)); t0 = time.perf_counter(); planned, actual, extra = run_case(case, method, compliance, shock, target, radius, support, texts[case.goal], models, representation, f1, f2, device, index, sigma); row = cheap_metric(case, planned, actual, target, radius, representation, device, extra); row["runtime_seconds"] = time.perf_counter() - t0; (dev if case.split == "development" else held)[key].append(row)
    def summarize(rows: list[dict]) -> dict:
        result = {"count": len(rows), "nonfinite_count": int(sum(bool(r["nonfinite"]) for r in rows))}
        for key, value in rows[0].items():
            if isinstance(value, (float, int, bool)) and key != "nonfinite": result[key] = float(np.mean([float(r[key]) for r in rows]))
        return result
    dev_s = {k: summarize(v) for k, v in dev.items()}; held_s = {k: summarize(v) for k, v in held.items()}; gd = {}; gh = {}
    for k, v in dev_s.items(): gd.setdefault(eval(k)[0], []).append(v)
    for k, v in held_s.items(): gh.setdefault(eval(k)[0], []).append(v)
    rd = {m: {"worst_arrival": min(v["actual_target_arrival"] for v in vals), "worst_diff": max(v["decoded_first_difference"] for v in vals), "worst_hidden": max(v["actual_hidden_path_mse"] for v in vals)} for m, vals in gd.items()}; rh = {m: {"worst_arrival": min(v["actual_target_arrival"] for v in vals), "worst_diff": max(v["decoded_first_difference"] for v in vals), "worst_hidden": max(v["actual_hidden_path_mse"] for v in vals)} for m, vals in gh.items()}; scores = {m: v["worst_arrival"] - .20 * v["worst_diff"] - .05 * v["worst_hidden"] for m, v in rd.items()}; selected = max(scores, key=scores.get); closed = selected != "r8_fixed" and rh[selected]["worst_arrival"] >= rh["r8_fixed"]["worst_arrival"] and rh[selected]["worst_diff"] <= rh["r8_fixed"]["worst_diff"] and rh[selected]["worst_hidden"] <= rh["r8_fixed"]["worst_hidden"]
    decision = {"experiment": "EXP_R16", "claim": "SUPPORTED_HISTORY_PLANT_SURROGATE" if closed else "NOT_SUPPORTED", "success": False, "surrogate_supported": bool(closed), "selected_method_development": selected, "heldout_opened_once": True, "overall_full_system": False, "next_experiment": "EXP_R17"}
    write_json(OUT / "development_metrics.json", {"per_config": dev, "summary": dev_s, "robust_summary": rd, "selection_scores": scores}); write_json(OUT / "heldout_metrics.json", {"per_config": held, "summary": held_s, "robust_summary": rh}); write_json(OUT / "claim_decision.json", decision); write_json(OUT / "final_candidate_selection.json", {"selected": selected, "scores": scores, "robust_development": rd})
    report = "# EXP_R16 report — history-conditioned latent plant\n\nR16 compared the previous/current history-conditioned plant against the current-only family.\n\n| method | worst arrival | worst decoded diff | worst hidden MSE |\n|---|---:|---:|---:|\n" + "".join(f"| {m} | {rh[m]['worst_arrival']:.4f} | {rh[m]['worst_diff']:.6f} | {rh[m]['worst_hidden']:.4f} |\n" for m in rh) + f"\nDevelopment selected `{selected}`. History-plant surrogate claim: **{decision['claim']}**. Overall full-system success remains **false**.\n"
    (REPORTS / "EXP_R16_report.md").write_text(report, encoding="utf-8"); (REPORTS / "next_exp_fromR16.md").write_text(f"# Next experiment from EXP_R16\n\nR16 changed only the surrogate plant to previous/current history matching. It selected `{selected}` with claim {decision['claim']}. If unsupported, EXP_R17 should test a small learned residual plant with train-only sequence augmentation; if supported, repeat the oracle-F3 closed-loop gate on the history plant.\n", encoding="utf-8")
    summary = REPORTS / "EXP_R9_to_EXP_R58_chinese_summary.md"
    with summary.open("a", encoding="utf-8") as stream: stream.write(f"\n## EXP_R16\n\nEXP_R16 把 surrogate plant 从只看当前 latent 改成匹配 previous/current latent history 和 source→goal，再按命令 compliance 推进，比较 R8、proposal、F1、旧 F2、graph。它直接检验历史/动量缺失是否导致前面闭环失败；development 选中 {selected}，held-out 判定为 {decision['claim']}，如果仍失败下一轮训练轻量 residual plant。\n")
    print({"experiment": "EXP_R16", "cases": len(cases), "evaluated_cases": len(eval_cases), "selected": selected, "claim": decision["claim"], "overall_success": False, "next": decision["next_experiment"]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--stage", choices=("audit", "run"), required=True); parser.add_argument("--device", default="cpu"); parser.add_argument("--seed", type=int, default=5616); parser.add_argument("--max-cases", type=int, default=None); args = parser.parse_args(); audit() if args.stage == "audit" else run(args.device, args.seed, args.max_cases)


if __name__ == "__main__": main()

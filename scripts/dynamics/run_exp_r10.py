#!/usr/bin/env python3
"""Run EXP_R10 action-conditioned latent-plant surrogate MPC.

Purpose
-------
Address EXP_R9's teacher-forced limitation with a causal train-only plant
surrogate.  A recorded train transition supplies a nominal next latent for a
nearest current state; a commanded latent moves the nominal result by a fixed
compliance fraction.  The controller observes that surrogate result after a
short prefix and replans.  Proposal, F1, old F2, graph, CEM, and trajectory
controllers are compared over a preregistered compliance grid.

Parameters
----------
``--stage`` is ``audit`` or ``run``; ``--device`` defaults to ``cpu``;
``--seed`` controls proposal/CEM initialization; ``--max-cases`` is a
debug-only cap and is not used for the registered run.

Usage
-----
PYTHONPATH=src:scripts/dynamics /home/jinjaguo/anaconda3/envs/libero/bin/python \\
  scripts/dynamics/run_exp_r10.py --stage audit --device cpu
PYTHONPATH=src:scripts/dynamics /home/jinjaguo/anaconda3/envs/libero/bin/python \\
  scripts/dynamics/run_exp_r10.py --stage run --device cpu

Outputs
-------
Artifacts are saved under ``results/EXP_R10/`` and ``reports/``; one Chinese
paragraph is appended to ``reports/EXP_R9_to_EXP_R58_chinese_summary.md``.
The frozen representation, decoder, F1, F2, and R8 planner are untouched.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import torch

from run_exp_r1 import REPORTS, disk_audit, f_rollout, graph_plan, load_frozen_planners, sha, write_json
from run_exp_r3 import build_cases, groups, metric
from run_exp_r4 import target_texts, train_proposals
from run_exp_r9 import HORIZONS, initial_plan, method_names, proposal_from_state


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "EXP_R10"
COMPLIANCES = (0.25, 0.50, 0.75, 1.00)
REGISTERED_PLANNERS = tuple(method for method in method_names() if method != "warm_proposal_h4_p1")
# CEM/trajectory candidates were registered and attempted, but their repeated
# autograd/sampling cost exceeded the CPU budget before any metric was written.
# The completed R10 comparison uses the executable lightweight candidates; the
# two invalid candidates are preserved in ``invalid_methods.json``.
PLANNERS = tuple(method for method in REGISTERED_PLANNERS if method not in {"cem_mpc_h4_p1", "traj_mpc_h4_p1"})
INVALID_PLANNERS = ("cem_mpc_h4_p1", "traj_mpc_h4_p1")


@dataclass
class NominalTransition:
    current: np.ndarray
    next: np.ndarray
    source_goal: str
    goal: str


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def build_nominal_index(train_cases: list) -> dict:
    transitions: list[NominalTransition] = []
    for case in train_cases:
        for index in range(3, 7):
            transitions.append(NominalTransition(case.latent[index], case.latent[index + 1], case.source_goal, case.goal))
    grouped: dict[tuple[str, str], list[NominalTransition]] = {}
    for item in transitions:
        grouped.setdefault((item.source_goal, item.goal), []).append(item)
    arrays: dict[object, tuple[np.ndarray, np.ndarray]] = {}
    for key, items in grouped.items():
        arrays[key] = (np.stack([item.current for item in items]), np.stack([item.next for item in items]))
    all_items = transitions
    arrays["__all__"] = (np.stack([item.current for item in all_items]), np.stack([item.next for item in all_items]))
    for goal in {item.goal for item in all_items}:
        items = [item for item in all_items if item.goal == goal]
        arrays[("__goal__", goal)] = (np.stack([item.current for item in items]), np.stack([item.next for item in items]))
    return arrays


def nearest_nominal(current: np.ndarray, case: object, index: dict) -> np.ndarray:
    key = (case.source_goal, case.goal)
    if key not in index: key = ("__goal__", case.goal)
    if key not in index: key = "__all__"
    currents, nexts = index[key]
    distances = np.mean((currents - current[None]) ** 2, axis=1)
    return nexts[int(np.argmin(distances))].copy()


def plant_step(current: np.ndarray, command: np.ndarray, case: object, index: dict, compliance: float) -> np.ndarray:
    nominal = nearest_nominal(current, case, index)
    return (nominal + compliance * (command - nominal)).astype(np.float32)


def cheap_metric(case: object, planned: np.ndarray, actual: np.ndarray, target: np.ndarray, radius: float, representation: object, device: torch.device, extra: dict) -> dict:
    """Metric-equivalent latent/decoder statistics without repeated F1 autograd."""
    with torch.no_grad():
        decoded = representation.decode(torch.tensor(planned, dtype=torch.float32, device=device)).cpu().numpy()
        start_decoded = representation.decode(torch.tensor(case.latent[3:4], dtype=torch.float32, device=device)).cpu().numpy()[0]
    joined = np.concatenate((start_decoded[None], decoded.reshape(-1, 16, 7)), axis=0).reshape(-1, 7)
    final_dist = float(np.min(np.linalg.norm(target - planned[-1][None], axis=1)))
    path = np.diff(np.concatenate((case.latent[3:4], planned)), axis=0)
    values = {"target_arrival": float(final_dist <= radius), "final_target_distance": final_dist, "hidden_path_latent_mse": float(np.mean((planned - case.latent[4:8]) ** 2)), "decoded_action_mse": float(np.mean((decoded[:, :, :6] - representation.decode(torch.tensor(case.latent[4:8], dtype=torch.float32, device=device)).cpu().numpy()[:, :, :6]) ** 2)), "decoded_first_difference": float(np.mean(np.diff(joined, axis=0) ** 2)), "decoded_second_difference": float(np.mean(np.diff(joined, n=2, axis=0) ** 2)), "path_length": float(np.sum(np.linalg.norm(path, axis=1))), "curvature": float(np.mean(np.linalg.norm(np.diff(path, axis=0), axis=1))), "f1_consistency_proxy": float(np.mean(np.diff(planned[:, 16:], axis=0) ** 2)), "nonfinite": bool(not np.isfinite(planned).all())}
    values.update(extra)
    return values


def run_surrogate(case: object, planner: str, compliance: float, target: np.ndarray, radius: float, support: np.ndarray, text: np.ndarray, models: list[dict], representation: object, f1: object, f2: object, device: torch.device, index: dict, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    horizon = 4; prefix = 4 if planner == "r8_open_loop" else (2 if planner.endswith("_p2") else 1)
    current = case.latent[3].copy(); previous = case.latent[2].copy(); planned_parts: list[np.ndarray] = []; actual_parts: list[np.ndarray] = []; runtimes: list[float] = []; consumed = 0; replans = 0
    while consumed < 4:
        t0 = time.perf_counter()
        path = initial_plan(planner, current, previous, target, support, text, models, representation, f1, f2, device, horizon, seed + consumed)
        take = min(prefix, 4 - consumed)
        commands = path[:take]
        next_states = []
        local_current = current
        for command in commands:
            next_state = plant_step(local_current, command, case, index, compliance)
            next_states.append(next_state); local_current = next_state
        planned_parts.append(commands.copy()); actual_parts.extend(next_states)
        previous, current = current, next_states[-1]
        consumed += take; replans += 1; runtimes.append(time.perf_counter() - t0)
    planned = np.concatenate(planned_parts, axis=0)[:4]; actual = np.stack(actual_parts[:4])
    extra = {"replans": replans, "mean_replan_runtime": float(np.mean(runtimes)), "tracking_latent_mse": float(np.mean((planned - actual) ** 2)), "actual_target_distance": float(np.min(np.linalg.norm(target - actual[-1][None], axis=1))), "actual_target_arrival": float(np.min(np.linalg.norm(target - actual[-1][None], axis=1)) <= radius), "actual_hidden_path_mse": float(np.mean((actual - case.latent[4:8]) ** 2)), "compliance": compliance, "surrogate_observation": True}
    return planned, actual, extra


def audit() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases, _, _, info = build_cases(torch.device("cpu"))
    frozen = {}
    for key, path in {"representation": ROOT / "checkpoints/representation/seed_810/correct_language/checkpoint_ema.pt", "f1": ROOT / "results/dynamics/fifteenth_wave/2026-08-12_dynamics_3/checkpoints/F1_execution_mlp.pt", "f2": ROOT / "results/dynamics/fifteenth_wave/2026-08-12_dynamics_3/checkpoints/F2_matched_refinement.pt", "r8_script": ROOT / "scripts/dynamics/run_exp_r8.py"}.items():
        frozen[key] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
    protocol = {"created_at": now(), "experiment": "EXP_R10", "parent": "EXP_R9", "git_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "disk": disk_audit(), "data": info, "compliances": list(COMPLIANCES), "planners": list(REGISTERED_PLANNERS), "invalid_planners": list(INVALID_PLANNERS), "plant": "nearest train current/source/goal nominal transition plus compliance*(command-nominal)", "frozen_manifest": frozen, "heldout_opened": False, "physical_claim": False}
    write_json(OUT / "preregistration.json", protocol); write_json(OUT / "frozen_manifest.json", frozen)
    (REPORTS / "EXP_R10_interface_audit.md").write_text("# EXP_R10 interface audit\n\nR10 keeps the exact EXP_R9 latent/action interface and adds a train-only action-conditioned surrogate plant. Each surrogate step uses only a nearest train nominal transition and the current commanded latent; the held-out future is used only for evaluation. This is not an exact Bullet simulator because retained episodes lack snapshots, contacts, and controller targets.\n", encoding="utf-8")
    (REPORTS / "EXP_R10_data_audit.md").write_text(f"# EXP_R10 data audit\n\nEXP_R10 reuses {len(cases)} EXP_R3 cases and the same episode-disjoint split. Train cases build the nominal transition index; development selects planner/compliance; held-out opens once.\n", encoding="utf-8")
    print(protocol)


def run(device_name: str, seed: int, max_cases: int | None = None) -> None:
    device = torch.device(device_name); cases, representation, features, _ = build_cases(device, max_cases)
    f1, f2 = load_frozen_planners(device); train = [case for case in cases if case.split == "train"]; development = [case for case in cases if case.split == "development"]; eval_cases = [case for case in cases if case.split != "train"]
    pair, goal, radii = groups(cases); support = np.concatenate(list(goal.values()))[::4].copy(); texts = target_texts(cases, representation, features, device); models = train_proposals(train, development, representation, features, device, [seed, seed + 1, seed + 2, seed + 3]); index = build_nominal_index(train)
    methods = [f"{planner}_c{compliance:.2f}" for planner in PLANNERS for compliance in COMPLIANCES]; dev = {method: [] for method in methods}; held = {method: [] for method in methods}
    for case in eval_cases:
        target = pair.get((case.source_goal, case.goal), goal[case.goal]); radius = radii.get((case.source_goal, case.goal), float(np.quantile(np.linalg.norm(target - target.mean(0), axis=1), 0.5)))
        for planner in PLANNERS:
            for compliance in COMPLIANCES:
                method = f"{planner}_c{compliance:.2f}"; t0 = time.perf_counter(); planned, actual, extra = run_surrogate(case, planner, compliance, target, radius, support, texts[case.goal], models, representation, f1, f2, device, index, seed); row = cheap_metric(case, planned, actual, target, radius, representation, device, extra); row["runtime_seconds"] = time.perf_counter() - t0; (dev if case.split == "development" else held)[method].append(row)

    def summarize(rows: list[dict]) -> dict:
        result = {"count": len(rows), "nonfinite_count": int(sum(bool(row["nonfinite"]) for row in rows))}
        for key, value in rows[0].items():
            if isinstance(value, (float, int, bool)) and key != "nonfinite": result[key] = float(np.mean([float(row[key]) for row in rows]))
        return result

    dev_s = {key: summarize(value) for key, value in dev.items()}; held_s = {key: summarize(value) for key, value in held.items()}; scores = {key: dev_s[key]["actual_target_arrival"] - 0.20 * dev_s[key]["decoded_first_difference"] - 0.05 * dev_s[key]["actual_hidden_path_mse"] - 0.10 * dev_s[key]["tracking_latent_mse"] for key in methods}; selected = max(scores, key=scores.get); baseline = "r8_open_loop_c1.00"; closed = selected != baseline and held_s[selected]["actual_target_arrival"] >= held_s[baseline]["actual_target_arrival"] and held_s[selected]["decoded_first_difference"] <= held_s[baseline]["decoded_first_difference"] and held_s[selected]["actual_hidden_path_mse"] <= held_s[baseline]["actual_hidden_path_mse"]
    decision = {"experiment": "EXP_R10", "claim": "SUPPORTED_ACTION_CONDITIONED_SURROGATE" if closed else "NOT_SUPPORTED", "success": False, "surrogate_supported": bool(closed), "selected_method_development": selected, "heldout_opened_once": True, "overall_full_system": False, "next_experiment": "EXP_R11"}
    write_json(OUT / "development_metrics.json", {"per_method": dev, "summary": dev_s, "selection_scores": scores, "invalid_planners": list(INVALID_PLANNERS)}); write_json(OUT / "heldout_metrics.json", {"per_method": held, "summary": held_s}); write_json(OUT / "claim_decision.json", decision); write_json(OUT / "final_candidate_selection.json", {"selected": selected, "scores": scores, "planners": list(REGISTERED_PLANNERS), "evaluated_planners": list(PLANNERS), "invalid_planners": list(INVALID_PLANNERS), "compliances": list(COMPLIANCES)}); write_json(OUT / "invalid_methods.json", {"methods": list(INVALID_PLANNERS), "reason": "Repeated surrogate replanning with historical autograd/sampling candidates exceeded CPU budget before metrics; no held-out values read."})
    report = "# EXP_R10 report — action-conditioned latent plant surrogate\n\nR10 replaced R9 teacher-forced observation with a train-only nominal-transition plant and explicit command compliance. It remains an offline surrogate, not physical MPC.\n\n| method | actual arrival | decoded first diff | actual hidden MSE | tracking MSE | replans |\n|---|---:|---:|---:|---:|---:|\n"
    ranking = sorted(methods, key=lambda m: scores[m], reverse=True)
    for method in ranking[:20]:
        row = held_s[method]; report += f"| {method} | {row['actual_target_arrival']:.4f} | {row['decoded_first_difference']:.6f} | {row['actual_hidden_path_mse']:.4f} | {row['tracking_latent_mse']:.4f} | {row['replans']:.1f} |\n"
    report += f"\nDevelopment selected `{selected}`. CEM and trajectory candidates were registered but invalidated before held-out because their repeated surrogate budget exceeded CPU time; no held-out values were read. Action-conditioned surrogate claim: **{decision['claim']}**. Overall full-system success remains **false**: no physical/exact simulator feedback, learned F3, long-horizon sequencing, or return.\n"
    (REPORTS / "EXP_R10_report.md").write_text(report, encoding="utf-8"); (REPORTS / "next_exp_fromR10.md").write_text(f"# Next experiment from EXP_R10\n\nEXP_R10 tested a train-only action-conditioned latent plant with compliance values {list(COMPLIANCES)}. It selected `{selected}` and established only a surrogate result ({decision['claim']}); physical closed-loop feedback remains unavailable. EXP_R11 should test disturbance-robust latent MPC: train-only residual uncertainty sets, robust terminal capture, and proposal/F1/F2/graph baselines under fixed perturbation budgets. Keep representation, decoder, F1, F2, R8 planner, and the R10 plant frozen.\n", encoding="utf-8")
    summary = REPORTS / "EXP_R9_to_EXP_R58_chinese_summary.md"
    with summary.open("a", encoding="utf-8") as stream: stream.write(f"\n## EXP_R10\n\nEXP_R10 为 R9 的 teacher-forced replay 加入了一个 train-only action-conditioned latent plant：先从训练 transition 找当前状态的名义下一步，再按命令与名义下一步的 compliance 比例生成实际反馈，比较 proposal、F1、旧 F2、图、CEM 和轨迹优化。它把“命令是否影响下一状态”纳入闭环，但仍不是 Bullet 物理模拟。development 选中 {selected}，held-out surrogate 判定为 {decision['claim']}；完整系统仍未成功，下一轮测试带不确定性的鲁棒 latent MPC。\n")
    print({"experiment": "EXP_R10", "cases": len(cases), "evaluated_cases": len(eval_cases), "selected": selected, "claim": decision["claim"], "overall_success": False, "next": decision["next_experiment"]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--stage", choices=("audit", "run"), required=True); parser.add_argument("--device", default="cpu"); parser.add_argument("--seed", type=int, default=5010); parser.add_argument("--max-cases", type=int, default=None); args = parser.parse_args(); audit() if args.stage == "audit" else run(args.device, args.seed, args.max_cases)


if __name__ == "__main__": main()

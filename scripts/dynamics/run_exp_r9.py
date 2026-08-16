#!/usr/bin/env python3
"""Run EXP_R9 closed-loop latent-control surrogate tournament.

Purpose
-------
Convert the successful EXP_R8 four-waypoint path into a causal receding-
horizon controller on the strongest executable repository interface.  Each
case is an episode-disjoint complete CALVIN episode transition: the
controller plans a finite latent path, consumes only a short prefix, then
observes the next recorded action window, re-encodes it, and replans.  The
retained complete episodes do not contain full robot/simulator snapshots, so
this is explicitly a teacher-forced latent replay surrogate, not a physical
or exact Bullet closed-loop claim.

Parameters
----------
``--stage`` is ``audit`` or ``run``; ``--device`` defaults to ``cpu``;
``--seed`` controls proposal/CEM initialization; ``--max-cases`` is a
debug-only deterministic case cap and is not used for the registered run.

Usage
-----
PYTHONPATH=src:scripts/dynamics /home/jinjaguo/anaconda3/envs/libero/bin/python \\
  scripts/dynamics/run_exp_r9.py --stage audit --device cpu
PYTHONPATH=src:scripts/dynamics /home/jinjaguo/anaconda3/envs/libero/bin/python \\
  scripts/dynamics/run_exp_r9.py --stage run --device cpu

Outputs
-------
Audits, frozen manifests, preregistration, development/held-out metrics,
claim decision, report, and next-experiment document are written under
``results/EXP_R9/`` and ``reports/``.  The Chinese R9 paragraph is appended
to ``reports/EXP_R9_to_EXP_R58_chinese_summary.md``.  No frozen checkpoint is
modified.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from run_exp_r1 import REPORTS, disk_audit, f_rollout, graph_plan, load_frozen_planners, sha, write_json
from run_exp_r2 import fast_traj
from run_exp_r3 import build_cases, groups, metric
from run_exp_r4 import proposal_paths, target_texts, train_proposals
from run_exp_r7 import repaired


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "EXP_R9"
HORIZONS = (2, 4)
PREFIXES = (1, 2)
LATE_SCHEDULE = np.asarray([0.0, 0.10, 0.35, 1.0], dtype=np.float32)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def method_names() -> list[str]:
    names = ["r8_open_loop"]
    names += [f"proposal_h{h}_p{p}" for h in HORIZONS for p in PREFIXES]
    names += ["warm_proposal_h4_p1", "f1_closed_h4_p1", "old_f2_closed_h4_p1", "graph_mpc_h4_p1", "cem_mpc_h4_p1", "traj_mpc_h4_p1"]
    return names


def project_texts(cases: list, representation: object, features: dict[str, np.ndarray], device: torch.device) -> dict[str, np.ndarray]:
    return target_texts(cases, representation, features, device)


def proposal_from_state(current: np.ndarray, target: np.ndarray, models: list[dict], text: np.ndarray, device: torch.device, horizon: int) -> np.ndarray:
    start = torch.tensor(current, dtype=torch.float32, device=device)[None]
    txt = torch.tensor(text, dtype=torch.float32, device=device)[None]
    with torch.no_grad():
        delta = models[0]["model"](start, txt)[0].cpu().numpy()
    base = current[None] + np.cumsum(delta, axis=0)
    base = base[:4].astype(np.float32)
    endpoint = target[np.argmin(np.linalg.norm(target - current[None], axis=1))]
    corrected = repaired(base, endpoint, 0.75, LATE_SCHEDULE)
    return corrected[:horizon]


def local_f_path(current: np.ndarray, previous: np.ndarray, text: np.ndarray, representation: object, f1: object, f2: object, which: str, device: torch.device, horizon: int) -> np.ndarray:
    prev = torch.tensor(previous[16:], dtype=torch.float32, device=device)[None]
    curr = torch.tensor(current[16:], dtype=torch.float32, device=device)[None]
    semantic = torch.tensor(current[:16], dtype=torch.float32, device=device)[None]
    txt = torch.tensor(text, dtype=torch.float32, device=device)[None]
    outputs = []
    for _ in range(horizon):
        context = torch.cat((semantic, txt), dim=-1)
        if which == "f1":
            with torch.no_grad():
                nxt = f1(prev, curr, context)
        else:
            # Historical matched refinement differentiates its internal energy
            # with respect to the candidate even during inference.
            with torch.enable_grad():
                nxt = f2(prev, curr, context)[0]
        full = torch.cat((semantic, nxt), dim=-1)
        outputs.append(full.detach().cpu().numpy()[0])
        prev, curr = curr.detach(), nxt.detach()
    return np.stack(outputs).astype(np.float32)


def cem_light_plan(start: np.ndarray, endpoint: np.ndarray, support: np.ndarray, seed: int) -> np.ndarray:
    """Low-cost CEM candidate used inside every replan, with fixed budget."""
    rng = np.random.default_rng(seed)
    mean = np.linspace(start, endpoint, 5, dtype=np.float32)[1:]
    std = np.full_like(mean, 0.25, dtype=np.float32)
    for _ in range(3):
        samples = rng.normal(mean, std, size=(48, 4, 32)).astype(np.float32)
        samples[:, 0] = 0.7 * samples[:, 0] + 0.3 * start
        terminal = np.mean((samples[:, -1] - endpoint) ** 2, axis=1)
        continuity = np.mean(np.diff(np.concatenate((np.repeat(start[None, None], 48, axis=0), samples), axis=1) ** 2), axis=(1, 2))
        support_cost = np.min(np.sum((samples[:, :, None, :] - support[None, None, :, :]) ** 2, axis=-1), axis=2).mean(axis=1)
        score = terminal + 0.25 * continuity + 0.10 * support_cost
        elite = samples[np.argsort(score)[:8]]
        mean = elite.mean(axis=0); std = np.maximum(elite.std(axis=0), 0.03)
    return mean.astype(np.float32)


def initial_plan(method: str, current: np.ndarray, previous: np.ndarray, target: np.ndarray, support: np.ndarray, text: np.ndarray, models: list[dict], representation: object, f1: object, f2: object, device: torch.device, horizon: int, seed: int) -> np.ndarray:
    if method.startswith("proposal") or method in {"r8_open_loop", "warm_proposal_h4_p1"}:
        return proposal_from_state(current, target, models, text, device, horizon)
    if method == "f1_closed_h4_p1":
        return local_f_path(current, previous, text, representation, f1, f2, "f1", device, horizon)
    if method == "old_f2_closed_h4_p1":
        return local_f_path(current, previous, text, representation, f1, f2, "f2", device, horizon)
    if method == "graph_mpc_h4_p1":
        return graph_plan(current, support, target)[:horizon]
    if method == "cem_mpc_h4_p1":
        endpoint = target[np.argmin(np.linalg.norm(target - current[None], axis=1))]
        return cem_light_plan(current, endpoint, support, seed)[:horizon]
    if method == "traj_mpc_h4_p1":
        endpoint = target[np.argmin(np.linalg.norm(target - current[None], axis=1))]
        return fast_traj(current, endpoint, support, representation, f1, text, "full", device, steps=4)[:horizon]
    raise ValueError(method)


def closed_loop_case(case: object, method: str, target: np.ndarray, radius: float, support: np.ndarray, texts: dict[str, np.ndarray], models: list[dict], representation: object, f1: object, f2: object, device: torch.device, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    if method == "r8_open_loop":
        h, p = 4, 4
    elif method.startswith("proposal_h"):
        h = int(method.split("_h")[1].split("_p")[0]); p = int(method.rsplit("_p", 1)[1])
    else:
        h, p = 4, 1
    current = case.latent[3].copy(); previous = case.latent[2].copy()
    planned_parts: list[np.ndarray] = []; actual_parts: list[np.ndarray] = []; replan_jumps: list[float] = []; runtimes: list[float] = []
    remaining: np.ndarray | None = None; consumed = 0
    while consumed < 4:
        t0 = time.perf_counter()
        fresh = initial_plan(method, current, previous, target, support, texts[case.goal], models, representation, f1, f2, device, h, seed + consumed)
        if method == "warm_proposal_h4_p1" and remaining is not None and len(remaining):
            tail = remaining[:h]
            if len(tail) < h: tail = np.concatenate((tail, np.repeat(tail[-1][None], h - len(tail), axis=0)), axis=0)
            tail = tail - tail[0][None] + current[None]
            fresh = (0.5 * tail + 0.5 * fresh).astype(np.float32)
        take = min(p, 4 - consumed)
        if planned_parts:
            replan_jumps.append(float(np.linalg.norm(fresh[0] - planned_parts[-1][-1])))
        planned_parts.append(fresh[:take].copy())
        actual_parts.append(case.latent[4 + consumed:4 + consumed + take].copy())
        remaining = fresh[take:]
        consumed += take
        previous = case.latent[2 + consumed].copy()
        current = case.latent[3 + consumed].copy()
        runtimes.append(time.perf_counter() - t0)
    planned = np.concatenate(planned_parts, axis=0)[:4]
    actual = np.concatenate(actual_parts, axis=0)[:4]
    extra = {"replans": len(planned_parts), "mean_replan_runtime": float(np.mean(runtimes)), "max_replan_jump": float(max(replan_jumps) if replan_jumps else 0.0), "mean_replan_jump": float(np.mean(replan_jumps) if replan_jumps else 0.0), "teacher_forced_observation": True, "observed_target_arrival": float(np.min(np.linalg.norm(target - actual[-1][None], axis=1)) <= radius), "tracking_latent_mse": float(np.mean((planned - actual) ** 2)), "tracking_terminal_distance": float(np.min(np.linalg.norm(target - actual[-1][None], axis=1))), "actual_path_latent_mse": float(np.mean((actual - case.latent[4:8]) ** 2))}
    return planned, actual, extra


def audit() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases, _, _, info = build_cases(torch.device("cpu"))
    frozen_paths = {
        "representation": ROOT / "checkpoints/representation/seed_810/correct_language/checkpoint_ema.pt",
        "f1": ROOT / "results/dynamics/fifteenth_wave/2026-08-12_dynamics_3/checkpoints/F1_execution_mlp.pt",
        "f2": ROOT / "results/dynamics/fifteenth_wave/2026-08-12_dynamics_3/checkpoints/F2_matched_refinement.pt",
        "r8_script": ROOT / "scripts/dynamics/run_exp_r8.py",
    }
    manifest = {key: {"path": str(path.relative_to(ROOT)), "sha256": sha(path)} for key, path in frozen_paths.items()}
    protocol = {
        "created_at": now(), "experiment": "EXP_R9", "parent": "EXP_R8", "git_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "disk": disk_audit(), "data": info,
        "closed_loop_interface": "teacher-forced complete-episode latent replay; no full Bullet snapshot exists",
        "supported_horizons": list(HORIZONS), "execution_prefixes": list(PREFIXES), "unsupported_horizon": "H=8: only four hidden chunks are available per complete-episode case",
        "methods": method_names(), "frozen_manifest": manifest, "oracle_f3": "annotation boundary and target action region; no learned completion",
        "heldout_opened": False, "overall_success_requires": "physical or exact simulator closed-loop plus learned F3/long-horizon/return; R9 surrogate alone cannot claim full system",
    }
    write_json(OUT / "preregistration.json", protocol); write_json(OUT / "frozen_manifest.json", manifest)
    (REPORTS / "EXP_R9_interface_audit.md").write_text("""# EXP_R9 interface audit

## Recovered interfaces

- Frozen representation: 32-D latent with 16 semantic and 16 execution coordinates; frozen decoder maps one latent to a normalized `(16, 7)` CALVIN action window.
- F1 consumes previous/current 16-D execution states, current 16-D semantic state, and projected target text. Historical F2 consumes the same causal context and returns an execution refinement.
- EXP_R8 supplies a four-waypoint path with a late terminal repair; its exact implementation is frozen as a baseline.
- Complete episode files contain `rel_actions (T,7)` and contiguous global frame indices. EXP_R3 encodes consecutive H16 action windows; the four post-boundary latent windows are the only exact next observations available for each case.
- Wave27 files additionally contain `robot_obs (128,15)` and `scene_obs (128,24)`, but the retained data do not contain a full Bullet snapshot, controller target state, contact state, or exact source branch. The historical CALVIN state audit records this reconstruction failure.

## Closed-loop surrogate

R9 therefore uses causal teacher-forced latent replay: plan H waypoints, consume P waypoints, expose only the next recorded post-boundary action window as the resulting observation, re-encode it with the frozen representation, and replan. This is a valid offline feedback/replanning surrogate for latent path tracking, but it is not physical MPC and does not claim that the planned action changed the recorded environment.

## Timing and oracle F3

The representation supplies one latent per 16-frame H16 window. H=2/4 and P=1/2 are supported by four hidden chunks; H=8 is not fabricated. The annotation boundary and target region provide oracle switching. No learned F3, waypoint controller, or return interface is available in R9.
""", encoding="utf-8")
    (REPORTS / "EXP_R9_data_audit.md").write_text(f"# EXP_R9 data audit\n\nEXP_R9 reuses the episode-disjoint EXP_R3 inventory with {len(cases)} cases. Train cases fit proposal models, development selects a closed-loop candidate, and held-out is opened once. Complete episodes provide four hidden H16 observations per case; no simulator state snapshot is present.\n", encoding="utf-8")
    print(json.dumps(protocol, indent=2))


def run(device_name: str, seed: int, max_cases: int | None = None) -> None:
    device = torch.device(device_name)
    cases, representation, features, _ = build_cases(device, max_cases)
    f1, f2 = load_frozen_planners(device)
    train = [case for case in cases if case.split == "train"]; development = [case for case in cases if case.split == "development"]
    eval_cases = [case for case in cases if case.split != "train"]
    pair, goal, radii = groups(cases); support = np.concatenate(list(goal.values())); support = support[::4].copy(); texts = project_texts(cases, representation, features, device)
    models = train_proposals(train, development, representation, features, device, [seed, seed + 1, seed + 2, seed + 3])
    methods = method_names(); dev = {method: [] for method in methods}; held = {method: [] for method in methods}
    for case in eval_cases:
        target = pair.get((case.source_goal, case.goal), goal[case.goal]); radius = radii.get((case.source_goal, case.goal), float(np.quantile(np.linalg.norm(target - target.mean(0), axis=1), 0.5)))
        for method in methods:
            t0 = time.perf_counter()
            planned, actual, extra = closed_loop_case(case, method, target, radius, support, texts, models, representation, f1, f2, device, seed)
            row = metric(case, planned, target, radius, goal, representation, f1, texts[case.goal], device); row.update(extra); row["runtime_seconds"] = time.perf_counter() - t0; row["planned_prefix_count"] = int(len(planned))
            (dev if case.split == "development" else held if case.split == "heldout" else {}).get(method, []).append(row)

    def summarize(rows: list[dict]) -> dict:
        result = {"count": len(rows), "nonfinite_count": int(sum(bool(row["nonfinite"]) for row in rows))}
        for key, value in rows[0].items():
            if isinstance(value, (float, int, bool)) and key != "nonfinite": result[key] = float(np.mean([float(row[key]) for row in rows]))
        return result

    dev_s = {method: summarize(rows) for method, rows in dev.items()}; held_s = {method: summarize(rows) for method, rows in held.items()}
    # The selection score rewards R8's joint latent metrics and explicitly rewards actual replanning.
    scores = {method: dev_s[method]["target_arrival"] - 0.20 * dev_s[method]["decoded_first_difference"] - 0.05 * dev_s[method]["hidden_path_latent_mse"] - 0.10 * dev_s[method]["tracking_latent_mse"] - 0.01 * dev_s[method]["max_replan_jump"] + 0.005 * min(dev_s[method]["replans"], 4.0) for method in methods}
    selected = max(scores, key=scores.get); baseline = "r8_open_loop"
    closed_loop_supported = selected != baseline and held_s[selected]["target_arrival"] >= held_s[baseline]["target_arrival"] and held_s[selected]["decoded_first_difference"] <= held_s[baseline]["decoded_first_difference"] and held_s[selected]["hidden_path_latent_mse"] <= held_s[baseline]["hidden_path_latent_mse"]
    decision = {"experiment": "EXP_R9", "claim": "SUPPORTED_CLOSED_LOOP_LATENT_SURROGATE" if closed_loop_supported else "NOT_SUPPORTED", "success": False, "closed_loop_surrogate_supported": bool(closed_loop_supported), "selected_method_development": selected, "heldout_opened_once": True, "overall_full_system": False, "next_experiment": "EXP_R10"}
    write_json(OUT / "development_metrics.json", {"per_method": dev, "summary": dev_s, "selection_scores": scores}); write_json(OUT / "heldout_metrics.json", {"per_method": held, "summary": held_s}); write_json(OUT / "claim_decision.json", decision); write_json(OUT / "final_candidate_selection.json", {"selected": selected, "scores": scores, "methods": methods})
    report = "# EXP_R9 report — closed-loop latent replay surrogate\n\nEXP_R9 converted the R8 four-waypoint path into a plan-short-prefix-observe-reencode-replan loop using complete-episode teacher-forced latent observations. This is not a physical MPC result because exact simulator snapshots are unavailable.\n\n| method | planned arrival | observed arrival | decoded first diff | hidden path MSE | tracking latent MSE | replans |\n|---|---:|---:|---:|---:|---:|---:|\n"
    for method in methods:
        row = held_s[method]; report += f"| {method} | {row['target_arrival']:.4f} | {row['observed_target_arrival']:.4f} | {row['decoded_first_difference']:.6f} | {row['hidden_path_latent_mse']:.4f} | {row['tracking_latent_mse']:.4f} | {row['replans']:.1f} |\n"
    report += f"\nDevelopment selected `{selected}`. Closed-loop latent-surrogate claim: **{decision['claim']}**. Overall full-system success remains **false** because learned F3, physical/exact simulator feedback, long-horizon sequencing, and return were not evaluated.\n"
    (REPORTS / "EXP_R9_report.md").write_text(report, encoding="utf-8")
    (REPORTS / "next_exp_fromR9.md").write_text(f"# Next experiment from EXP_R9\n\nEXP_R9 selected `{selected}` on development and established only a teacher-forced latent replay result: the controller replans from recorded next action windows, not from an action-conditioned simulator. The remaining bottleneck is causal plant feedback. EXP_R10 should build a train-only one-step action-conditioned latent plant surrogate from complete episode transitions and compare it with the teacher-forced replay, while keeping the representation, decoder, F1, historical F2, and R8 repair frozen. It must include proposal, F1, old F2, graph, and sampling baselines, then evaluate held-out once. Physical/exact Bullet closed loop remains a separate limitation, not a hidden claim.\n", encoding="utf-8")
    summary = REPORTS / "EXP_R9_to_EXP_R58_chinese_summary.md"
    if not summary.exists(): summary.write_text("# EXP_R9–EXP_R58 实验总结\n\n每个 EXP 一段通俗总结；成功前继续，EXP_R58 为上限。\n", encoding="utf-8")
    with summary.open("a", encoding="utf-8") as stream:
        stream.write(f"\n## EXP_R9\n\nEXP_R9 把 R8 的一次性四步路径改成了“规划 H 步、只执行 P 步、读取下一段真实动作窗口、重新编码、再规划”的闭环 latent replay。由于完整 CALVIN 文件没有 Bullet 快照、控制器目标和接触状态，这一轮不能声称真实物理 MPC，只能验证因果的离线重规划。实验比较了 H=2/4、P=1/2、warm-start、F1、旧 F2、图、CEM 和轨迹优化；development 选择 {selected}，held-out 的闭环 latent surrogate 判定为 {decision['claim']}，但完整系统仍未成功，下一轮转向 train-only action-conditioned latent plant surrogate。\n")
    print({"experiment": "EXP_R9", "cases": len(cases), "evaluated_cases": len(eval_cases), "selected": selected, "claim": decision["claim"], "overall_success": False, "next": decision["next_experiment"]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--stage", choices=("audit", "run"), required=True); parser.add_argument("--device", default="cpu"); parser.add_argument("--seed", type=int, default=4909); parser.add_argument("--max-cases", type=int, default=None); args = parser.parse_args()
    if args.stage == "audit": audit()
    else: run(args.device, args.seed, args.max_cases)


if __name__ == "__main__": main()

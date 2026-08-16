#!/usr/bin/env python3
"""Run EXP_R4 learned local-edge and multi-hypothesis path planning.

Purpose
-------
Address EXP_R3's bottleneck by training a small multi-step edge proposal on
complete episode transitions.  The frozen action representation, decoder, F1,
and historical F2 remain baselines; new proposal paths are selected using
train-only target-region, continuity, and empirical-support costs.

Parameters
----------
``--stage`` is ``audit`` or ``run``; ``--device`` is a PyTorch device
(default ``cpu``); ``--seed`` controls deterministic proposal initialization;
``--max-cases`` is a debug-only cap and is not used for the registered run.

Usage
-----
PYTHONPATH=src:scripts/dynamics /home/jinjaguo/anaconda3/envs/libero/bin/python \\
  scripts/dynamics/run_exp_r4.py --stage audit --device cpu
PYTHONPATH=src:scripts/dynamics /home/jinjaguo/anaconda3/envs/libero/bin/python \\
  scripts/dynamics/run_exp_r4.py --stage run --device cpu

Outputs
-------
Artifacts are written under ``results/EXP_R4/`` and ``reports/``.  The
per-experiment Chinese summary is appended to
``reports/EXP_R1_to_EXP_R80_chinese_summary.md``.  No frozen checkpoint is
overwritten.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from run_exp_r1 import ROOT, REPORTS, disk_audit, f_rollout, graph_plan, load_frozen_planners, select_representation, write_json
from run_exp_r3 import build_cases, groups, metric


OUT = ROOT / "results" / "EXP_R4"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class EdgeProposal(nn.Module):
    """Goal-conditioned four-step latent path proposal network."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(48, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128))

    def forward(self, start: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((start, text), dim=-1)).reshape(-1, 4, 32)


def train_proposals(train_cases: list[Any], dev_cases: list[Any], representation: Any, features: dict[str, np.ndarray], device: torch.device, seeds: list[int]) -> list[dict[str, Any]]:
    models = []
    for seed in seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model = EdgeProposal().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
        starts = torch.tensor(np.stack([case.latent[3] for case in train_cases]), dtype=torch.float32, device=device)
        texts = []
        for case in train_cases:
            with torch.no_grad():
                text = F.normalize(representation.project_text(torch.tensor(features[case.text], device=device)[None]), dim=-1).cpu().numpy()[0]
            texts.append(text)
        text_tensor = torch.tensor(np.stack(texts), dtype=torch.float32, device=device)
        targets = torch.tensor(np.stack([case.latent[4:8] - case.latent[3] for case in train_cases]), dtype=torch.float32, device=device)
        best = None; best_dev = float("inf")
        for _ in range(70):
            model.train(); prediction = model(starts, text_tensor); path = starts[:, None] + torch.cumsum(prediction, dim=1)
            loss = F.mse_loss(path, targets + starts[:, None]) + 0.08 * torch.diff(path, dim=1).square().mean()
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            model.eval()
            with torch.no_grad():
                dev_starts = torch.tensor(np.stack([case.latent[3] for case in dev_cases]), dtype=torch.float32, device=device)
                dev_texts = torch.tensor(np.stack([F.normalize(representation.project_text(torch.tensor(features[case.text], device=device)[None]), dim=-1).cpu().numpy()[0] for case in dev_cases]), dtype=torch.float32, device=device)
                dev_path = dev_starts[:, None] + torch.cumsum(model(dev_starts, dev_texts), dim=1)
                score = float(F.mse_loss(dev_path, torch.tensor(np.stack([case.latent[4:8] for case in dev_cases]), dtype=torch.float32, device=device)).cpu())
            if score < best_dev:
                best_dev = score; best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        model.load_state_dict(best); models.append({"model": model, "dev_mse": best_dev, "seed": seed})
    return models


def target_texts(cases: list[Any], representation: Any, features: dict[str, np.ndarray], device: torch.device) -> dict[str, np.ndarray]:
    result = {}
    with torch.no_grad():
        for case in cases:
            if case.goal not in result:
                result[case.goal] = F.normalize(representation.project_text(torch.tensor(features[case.text], device=device)[None]), dim=-1).cpu().numpy()[0]
    return result


def proposal_paths(case: Any, models: list[dict[str, Any]], text: np.ndarray, target: np.ndarray, support: np.ndarray, device: torch.device) -> list[np.ndarray]:
    start = torch.tensor(case.latent[3], dtype=torch.float32, device=device)[None]
    txt = torch.tensor(text, dtype=torch.float32, device=device)[None]
    paths = []
    with torch.no_grad():
        for entry in models:
            delta = entry["model"](start, txt)[0]
            paths.append((start + torch.cumsum(delta, 0)).cpu().numpy())
    # Add a small terminal repair candidate, still as a multi-step path.
    base = paths[0]; repaired = base + np.linspace(0.0, 0.20, 4)[:, None] * (target - base[-1])[None]
    paths.append(repaired.astype(np.float32))
    return paths


def score_path(path: np.ndarray, target: np.ndarray, support: np.ndarray) -> float:
    terminal = float(np.min(np.linalg.norm(target - path[-1], axis=1)))
    continuity = float(np.mean(np.diff(path, axis=0) ** 2))
    support_cost = float(np.mean(np.min(np.linalg.norm(path[:, None] - support[None], axis=-1), axis=1)))
    return terminal + 0.20 * continuity + 0.10 * support_cost


def audit() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases, _, _, info = build_cases(torch.device("cpu"))
    protocol = {"created_at": now(), "experiment": "EXP_R4", "parent": "EXP_R3", "disk": disk_audit(), "data": info, "methods": ["linear", "f1", "f2", "graph", "edge_proposal", "multi_hypothesis", "edge_proposal_repaired"], "proposal": "4-step residual path proposal conditioned on current 32-D latent and projected target text; train cases only", "frozen": ["representation", "decoder", "F1", "F2"], "heldout_opened": False}
    write_json(OUT / "preregistration.json", protocol)
    (REPORTS / "EXP_R4_interface_audit.md").write_text("# EXP_R4 interface audit\n\nEXP_R4 adds only a new trainable multi-step edge proposal. The released representation, decoder, F1 and historical F2 remain frozen. The proposal consumes the current latent and exact target language feature, and emits four latent waypoints; it never receives hidden future actions at evaluation.\n", encoding="utf-8")
    (REPORTS / "EXP_R4_data_audit.md").write_text(f"# EXP_R4 data audit\n\nThe complete-episode EXP_R3 cases are reused with the same episode-disjoint split. There are {len(cases)} cases; proposal fitting uses train cases, development selects the candidate, and held-out is opened once.\n", encoding="utf-8")
    print(json.dumps(protocol, indent=2))


def run(device_name: str, seed: int, max_cases: int | None = None) -> None:
    device = torch.device(device_name)
    cases, representation, features, _ = build_cases(device, max_cases)
    f1, f2 = load_frozen_planners(device); train = [case for case in cases if case.split == "train"]; development = [case for case in cases if case.split == "development"]
    pair, goal, radii = groups(cases); all_support = np.concatenate(list(goal.values())); texts = target_texts(cases, representation, features, device)
    models = train_proposals(train, development, representation, features, device, [seed, seed + 1, seed + 2, seed + 3])
    methods = ["linear", "f1", "f2", "graph", "edge_proposal", "multi_hypothesis", "edge_proposal_repaired"]
    dev: dict[str, list[dict[str, Any]]] = {m: [] for m in methods}; held: dict[str, list[dict[str, Any]]] = {m: [] for m in methods}
    for case in cases:
        target = pair.get((case.source_goal, case.goal), goal[case.goal]); radius = radii.get((case.source_goal, case.goal), float(np.quantile(np.linalg.norm(target - target.mean(0), axis=1), 0.5))); endpoint = target[np.argmin(np.linalg.norm(target - case.latent[3][None], axis=1))]
        candidates = proposal_paths(case, models, texts[case.goal], endpoint, all_support, device)
        selected_path = candidates[int(np.argmin([score_path(path, target, all_support) for path in candidates]))]
        for method in methods:
            t0 = time.perf_counter(); start = case.latent[3]
            if method == "linear": planned = np.linspace(start, endpoint, 5, dtype=np.float32)[1:]
            elif method == "f1": planned = f_rollout(case, texts[case.goal], representation, f1, f2, "f1", device)
            elif method == "f2": planned = f_rollout(case, texts[case.goal], representation, f1, f2, "f2", device)
            elif method == "graph": planned = graph_plan(start, all_support, target)
            elif method == "edge_proposal": planned = candidates[0]
            elif method == "multi_hypothesis": planned = selected_path
            else: planned = candidates[-1]
            row = metric(case, planned, target, radius, goal, representation, f1, texts[case.goal], device); row["runtime_seconds"] = time.perf_counter() - t0
            (dev if case.split == "development" else held if case.split == "heldout" else {}).get(method, []).append(row)
    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        result = {"count": len(rows), "nonfinite_count": int(sum(bool(row["nonfinite"]) for row in rows))}
        for key, value in rows[0].items():
            if isinstance(value, (float, int, bool)) and key != "nonfinite": result[key] = float(np.mean([float(row[key]) for row in rows]))
        return result
    dev_s = {m: summarize(v) for m, v in dev.items()}; held_s = {m: summarize(v) for m, v in held.items()}
    scores = {m: dev_s[m]["target_arrival"] - 0.20 * dev_s[m]["decoded_first_difference"] - 0.08 * dev_s[m]["f1_dynamics_cost"] - 0.05 * dev_s[m]["hidden_path_latent_mse"] for m in methods}; selected = max(scores, key=scores.get); baselines = ["linear", "f1", "f2"]
    success = all(held_s[selected]["target_arrival"] >= held_s[b]["target_arrival"] and held_s[selected]["decoded_first_difference"] <= held_s[b]["decoded_first_difference"] and held_s[selected]["hidden_path_latent_mse"] <= held_s[b]["hidden_path_latent_mse"] for b in baselines)
    decision = {"experiment": "EXP_R4", "claim": "SUPPORTED" if success else "NOT_SUPPORTED", "success": bool(success), "selected_method_development": selected, "heldout_opened_once": True, "next_experiment": None if success else "EXP_R5"}
    write_json(OUT / "development_metrics.json", {"per_method": dev, "summary": dev_s, "selection_scores": scores}); write_json(OUT / "heldout_metrics.json", {"per_method": held, "summary": held_s}); write_json(OUT / "claim_decision.json", decision); write_json(OUT / "final_candidate_selection.json", {"selected": selected, "scores": scores, "methods": methods}); torch.save({"models": [{"seed": item["seed"], "dev_mse": item["dev_mse"], "state_dict": item["model"].state_dict()} for item in models]}, OUT / "proposal_models.pt")
    report = "# EXP_R4 report — learned local edge proposals\n\nEXP_R4 trained a small four-step proposal on train complete-episode transitions and compared single proposals, multi-hypothesis selection, and terminal repair with frozen baselines.\n\n| method | arrival | decoded first diff | hidden path MSE | F1 cost |\n|---|---:|---:|---:|---:|\n"
    for method in methods:
        s = held_s[method]; report += f"| {method} | {s['target_arrival']:.4f} | {s['decoded_first_difference']:.6f} | {s['hidden_path_latent_mse']:.4f} | {s['f1_dynamics_cost']:.4f} |\n"
    report += f"\nDevelopment selected `{selected}`. EXP_R4 is **{decision['claim']}** (`SUCCESS={decision['success']}`). The proposal remains offline; no F3 learning, closed-loop MPC, or return claim is made.\n"
    (REPORTS / "EXP_R4_report.md").write_text(report, encoding="utf-8"); (REPORTS / "next_exp_fromR4.md").write_text(f"# Next experiment from EXP_R4\n\nEXP_R4 is **{decision['claim']}**. If it fails, EXP_R5 should add explicit state-conditioned local edge costs and a graph planner over complete episode waypoints, keeping the representation and historical F1/F2 frozen. If it succeeds, remove exact endpoint knowledge and test target-set/language grounding.\n", encoding="utf-8")
    with (REPORTS / "EXP_R1_to_EXP_R80_chinese_summary.md").open("a", encoding="utf-8") as stream:
        stream.write(f"\n## EXP_R4\n\nEXP_R4 第一次训练了一个真正的多步局部边 proposal：输入当前 latent 和目标语言，一次提出四个未来 latent，再用目标区域、连续性和训练支持性选出候选。它与线性路径、冻结 F1/F2 和图搜索进行了同一 held-out 对比。这样可以检验“目标切换信息是否能进入一条完整路径”，而不是只推一个点。EXP_R4 当前判定为 {decision['claim']}；如果仍失败，下一轮将把重点放到状态条件的局部边代价和完整 waypoint 图上。\n")
    print(json.dumps({"experiment": "EXP_R4", "cases": len(cases), "selected": selected, "success": success, "next": decision["next_experiment"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--stage", choices=("audit", "run"), required=True); parser.add_argument("--device", default="cpu"); parser.add_argument("--seed", type=int, default=4404); parser.add_argument("--max-cases", type=int, default=None); args = parser.parse_args()
    if args.stage == "audit": audit()
    else: run(args.device, args.seed, args.max_cases)


if __name__ == "__main__": main()

#!/usr/bin/env python3
"""Run EXP_R6: train-only continuous proposal/linear path blending.

Purpose
-------
Resolve the EXP_R4--R5 trade-off between a smooth learned four-step proposal
and the high-arrival but jumpy linear path.  A fixed, preregistered grid of
continuous blend weights is evaluated on the development split; exactly one
weight is selected there and then evaluated once on the episode-disjoint
held-out split.  The frozen action representation, decoder, F1, and F2 are
unchanged.

Parameters
----------
``--stage`` is ``audit`` or ``run``; ``--device`` selects PyTorch (default
``cpu``); ``--seed`` controls proposal initialization; ``--max-cases`` is a
debug-only deterministic cap and is not used for the registered full run.

Usage
-----
PYTHONPATH=src:scripts/dynamics /home/jinjaguo/anaconda3/envs/libero/bin/python \\
  scripts/dynamics/run_exp_r6.py --stage audit --device cpu
PYTHONPATH=src:scripts/dynamics /home/jinjaguo/anaconda3/envs/libero/bin/python \\
  scripts/dynamics/run_exp_r6.py --stage run --device cpu

Outputs
-------
Audits, preregistration, metrics, proposal weights, the report, and the next
experiment document are written to ``results/EXP_R6/`` and ``reports/``.
One plain-language Chinese paragraph is appended to
``reports/EXP_R1_to_EXP_R80_chinese_summary.md``.  No frozen checkpoint is
overwritten.
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
OUT = ROOT / "results" / "EXP_R6"
ALPHAS = (0.0, 0.25, 0.50, 0.75, 0.90, 1.0)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def blend(linear: np.ndarray, proposal: np.ndarray, alpha: float) -> np.ndarray:
    """Return a continuous convex path; alpha=1 is proposal, alpha=0 linear."""
    return ((1.0 - alpha) * linear + alpha * proposal).astype(np.float32)


def audit() -> None:
    """Freeze the protocol and perform the one-per-EXP disk audit."""
    OUT.mkdir(parents=True, exist_ok=True)
    cases, _, _, info = build_cases(torch.device("cpu"))
    protocol = {
        "created_at": now(),
        "experiment": "EXP_R6",
        "parent": "EXP_R5",
        "disk": disk_audit(),
        "data": info,
        "blend_alphas": list(ALPHAS),
        "proposal": "EXP_R4 four-step proposal repaired toward train-only nearest endpoint",
        "selection": "development-only score: arrival - 0.20 decoded first difference - 0.05 hidden path MSE",
        "methods": ["linear", "f1", "f2", "graph", "proposal"] + [f"blend_{alpha:.2f}" for alpha in ALPHAS],
        "frozen": ["representation", "decoder", "F1", "F2"],
        "heldout_opened": False,
    }
    write_json(OUT / "preregistration.json", protocol)
    (REPORTS / "EXP_R6_interface_audit.md").write_text(
        "# EXP_R6 interface audit\n\n"
        "EXP_R6 adds no new interface to the frozen action-coordinate model. "
        "It convexly mixes a four-step proposal and a four-step linear path; "
        "the alpha grid is fixed before held-out evaluation. F1/F2 and the "
        "decoder are loaded from the historical checkpoints and remain frozen.\n",
        encoding="utf-8",
    )
    (REPORTS / "EXP_R6_data_audit.md").write_text(
        f"# EXP_R6 data audit\n\nThe episode-disjoint EXP_R3 inventory is reused unchanged ({len(cases)} cases). "
        "Proposal fitting uses train cases, alpha selection uses development cases, "
        "and held-out paths are opened once after selection.\n",
        encoding="utf-8",
    )
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
    methods = ["linear", "f1", "f2", "graph", "proposal"] + [f"blend_{alpha:.2f}" for alpha in ALPHAS]
    dev = {method: [] for method in methods}
    held = {method: [] for method in methods}

    for case in cases:
        target = pair.get((case.source_goal, case.goal), goal[case.goal])
        radius = radii.get((case.source_goal, case.goal), float(np.quantile(np.linalg.norm(target - target.mean(0), axis=1), 0.5)))
        endpoint = target[np.argmin(np.linalg.norm(target - case.latent[3][None], axis=1))]
        start = case.latent[3]
        linear = np.linspace(start, endpoint, 5, dtype=np.float32)[1:]
        proposals = proposal_paths(case, models, texts[case.goal], endpoint, support, device)
        proposal = proposals[-1]
        paths = {
            "linear": linear,
            "proposal": proposal,
            "f1": f_rollout(case, texts[case.goal], representation, f1, f2, "f1", device),
            "f2": f_rollout(case, texts[case.goal], representation, f1, f2, "f2", device),
            "graph": graph_plan(start, support, target),
        }
        for alpha in ALPHAS:
            paths[f"blend_{alpha:.2f}"] = blend(linear, proposal, alpha)
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
    scores = {
        method: dev_s[method]["target_arrival"]
        - 0.20 * dev_s[method]["decoded_first_difference"]
        - 0.05 * dev_s[method]["hidden_path_latent_mse"]
        for method in methods
    }
    selected = max(scores, key=scores.get)
    baselines = ["linear", "f1", "f2"]
    success = all(
        held_s[selected]["target_arrival"] >= held_s[baseline]["target_arrival"]
        and held_s[selected]["decoded_first_difference"] <= held_s[baseline]["decoded_first_difference"]
        and held_s[selected]["hidden_path_latent_mse"] <= held_s[baseline]["hidden_path_latent_mse"]
        for baseline in baselines
    )
    decision = {
        "experiment": "EXP_R6",
        "claim": "SUPPORTED" if success else "NOT_SUPPORTED",
        "success": bool(success),
        "selected_method_development": selected,
        "heldout_opened_once": True,
        "next_experiment": None if success else "EXP_R7",
    }
    write_json(OUT / "development_metrics.json", {"per_method": dev, "summary": dev_s, "selection_scores": scores})
    write_json(OUT / "heldout_metrics.json", {"per_method": held, "summary": held_s})
    write_json(OUT / "claim_decision.json", decision)
    write_json(OUT / "final_candidate_selection.json", {"selected": selected, "alphas": list(ALPHAS), "scores": scores})
    report = "# EXP_R6 report — continuous proposal/linear blends\n\n"
    report += "EXP_R6 selected a fixed convex blend on development data and evaluated that choice once on held-out episodes.\n\n"
    report += "| method | arrival | decoded first diff | hidden path MSE | F1 cost |\n|---|---:|---:|---:|---:|\n"
    for method in methods:
        row = held_s[method]
        report += f"| {method} | {row['target_arrival']:.4f} | {row['decoded_first_difference']:.6f} | {row['hidden_path_latent_mse']:.4f} | {row['f1_dynamics_cost']:.4f} |\n"
    report += f"\nDevelopment selected `{selected}`. EXP_R6 is **{decision['claim']}** (`SUCCESS={decision['success']}`).\n"
    (REPORTS / "EXP_R6_report.md").write_text(report, encoding="utf-8")
    (REPORTS / "next_exp_fromR6.md").write_text(
        f"# Next experiment from EXP_R6\n\nEXP_R6 is **{decision['claim']}**. "
        "If it fails, EXP_R7 should add a state-conditioned residual correction with a bounded terminal target, while preserving the smooth proposal and frozen F1/F2. "
        "If it succeeds, remove oracle endpoint selection and test target-set planning.\n",
        encoding="utf-8",
    )
    with (REPORTS / "EXP_R1_to_EXP_R80_chinese_summary.md").open("a", encoding="utf-8") as stream:
        stream.write(
            f"\n## EXP_R6\n\nEXP_R6 针对前几轮暴露出的矛盾，把 R4 的平滑四步 proposal 和高到达率但跳变明显的线性路径按预先固定的比例做连续混合。比例只在 development 数据上选择，held-out 只打开一次。实验判定为 {decision['claim']}，development 选出的方法是 {selected}；如果仍失败，说明简单的全局比例不能同时控制终点和连续性，下一轮将加入受限的状态条件残差修正。\n"
        )
    print({"experiment": "EXP_R6", "cases": len(cases), "selected": selected, "success": success, "next": decision["next_experiment"]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("audit", "run"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=4606)
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()
    if args.stage == "audit":
        audit()
    else:
        run(args.device, args.seed, args.max_cases)


if __name__ == "__main__":
    main()

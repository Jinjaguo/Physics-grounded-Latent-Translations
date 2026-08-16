#!/usr/bin/env python3
"""Run EXP_R13 oracle-boundary F3 completion-readiness diagnostics.

Purpose
-------
Measure whether subgoal completion is learnable from causal latent/text
history before integrating F3.  Samples are built from the real annotation
boundary cases used in EXP_R3: pre-boundary chunks are negative and target
chunks are positive.  This is a readiness diagnostic only; because closed-loop
F2 is not yet supported, no learned F3 is connected to control.

Parameters
----------
``--stage`` is ``audit`` or ``run``; ``--device`` defaults to ``cpu``;
``--seed`` controls MLP initialization; ``--max-cases`` is debug-only.

Usage
-----
PYTHONPATH=src:scripts/dynamics /home/jinjaguo/anaconda3/envs/libero/bin/python \\
  scripts/dynamics/run_exp_r13.py --stage audit --device cpu
PYTHONPATH=src:scripts/dynamics /home/jinjaguo/anaconda3/envs/libero/bin/python \\
  scripts/dynamics/run_exp_r13.py --stage run --device cpu

Outputs
-------
Results are saved under ``results/EXP_R13/`` and ``reports/``.  A Chinese
paragraph is appended to ``reports/EXP_R9_to_EXP_R58_chinese_summary.md``.
No F3 checkpoint is promoted to the control stack.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from run_exp_r1 import REPORTS, disk_audit, load_frozen_planners, sha, write_json
from run_exp_r3 import build_cases, groups
from run_exp_r4 import target_texts


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "EXP_R13"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class CompletionMLP(nn.Module):
    def __init__(self, input_dim: int, width: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, width), nn.GELU(), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value).squeeze(-1)


def make_samples(cases: list, representation: object, features: dict[str, np.ndarray], device: torch.device) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    texts = target_texts(cases, representation, features, device); pair, goal, _ = groups(cases); split_x = {"train": [], "development": [], "heldout": []}; split_y = {"train": [], "development": [], "heldout": []}
    for case in cases:
        target = pair.get((case.source_goal, case.goal), goal[case.goal]); text = texts[case.goal]
        for index in range(8):
            current = case.latent[index]; previous = case.latent[max(0, index - 1)]; distance = float(np.min(np.linalg.norm(target - current[None], axis=1))); value = np.concatenate((current, text, np.asarray([distance, np.linalg.norm(current[16:]), np.linalg.norm(current[:16])], dtype=np.float32)))
            split_x[case.split].append(value); split_y[case.split].append(float(index >= 4))
    return {key: np.asarray(value, dtype=np.float32) for key, value in split_x.items()}, {key: np.asarray(value, dtype=np.float32) for key, value in split_y.items()}


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores); ranks = np.empty_like(order, dtype=np.float64); ranks[order] = np.arange(len(order)) + 1; positives = labels > 0.5; n_pos = positives.sum(); n_neg = len(labels) - n_pos
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / max(n_pos * n_neg, 1))


def classification_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    pred = scores >= threshold; positive = labels > 0.5; return {"auroc": auc_score(labels, scores), "balanced_accuracy": float(0.5 * ((pred[positive].mean() if positive.any() else 0.0) + ((~pred[~positive]).mean() if (~positive).any() else 0.0))), "early_switch_rate": float(pred[~positive].mean() if (~positive).any() else 0.0), "late_miss_rate": float((~pred[positive]).mean() if positive.any() else 0.0), "threshold": threshold}


def audit() -> None:
    OUT.mkdir(parents=True, exist_ok=True); cases, _, _, info = build_cases(torch.device("cpu")); frozen = {}
    for key, path in {"representation": ROOT / "checkpoints/representation/seed_810/correct_language/checkpoint_ema.pt", "f1": ROOT / "results/dynamics/fifteenth_wave/2026-08-12_dynamics_3/checkpoints/F1_execution_mlp.pt", "f2": ROOT / "results/dynamics/fifteenth_wave/2026-08-12_dynamics_3/checkpoints/F2_matched_refinement.pt"}.items(): frozen[key] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
    protocol = {"created_at": now(), "experiment": "EXP_R13", "parent": "EXP_R12", "disk": disk_audit(), "data": info, "labels": "oracle annotation boundary: chunks 0-3 negative, chunks 4-7 positive", "methods": ["distance_score", "linear_mlp", "mlp_fusion"], "frozen_manifest": frozen, "f3_promoted": False, "heldout_opened": False}
    write_json(OUT / "preregistration.json", protocol); write_json(OUT / "frozen_manifest.json", frozen); (REPORTS / "EXP_R13_interface_audit.md").write_text("# EXP_R13 interface audit\n\nF3 is not integrated. The only supported causal inputs are the frozen current latent, projected target text, train-derived target distance, and latent norms. Oracle annotation boundary labels are used for readiness evaluation only; hidden future actions are not inputs.\n", encoding="utf-8"); (REPORTS / "EXP_R13_data_audit.md").write_text(f"# EXP_R13 data audit\n\nBuilt causal samples from {len(cases)} episode-disjoint cases. Train/development/held-out splits remain episode-disjoint; held-out opens once after threshold selection.\n", encoding="utf-8"); print(protocol)


def run(device_name: str, seed: int, max_cases: int | None = None) -> None:
    device = torch.device(device_name); cases, representation, features, _ = build_cases(device, max_cases); x, y = make_samples(cases, representation, features, device); rng = np.random.default_rng(seed)
    train_x, train_y = x["train"], y["train"]; dev_x, dev_y = x["development"], y["development"]; held_x, held_y = x["heldout"], y["heldout"]
    train_mean = train_x.mean(0); train_std = np.maximum(train_x.std(0), 1e-4); train_xn = (train_x - train_mean) / train_std; dev_xn = (dev_x - train_mean) / train_std; held_xn = (held_x - train_mean) / train_std
    methods: dict[str, dict] = {}
    distance_index = train_x.shape[1] - 3; distance_train = train_x[:, distance_index]; distance_dev = dev_x[:, distance_index]; distance_held = held_x[:, distance_index]; thresholds = np.quantile(distance_train[train_y < 0.5], np.linspace(0.05, 0.95, 19)); best = max(thresholds, key=lambda threshold: classification_metrics(dev_y, 1.0 - distance_dev / max(float(np.max(distance_train)), 1e-6), float(threshold / max(float(np.max(distance_train)), 1e-6)))["balanced_accuracy"]); methods["distance_score"] = {"train": classification_metrics(train_y, 1.0 - distance_train / max(float(np.max(distance_train)), 1e-6), float(best / max(float(np.max(distance_train)), 1e-6))), "development": classification_metrics(dev_y, 1.0 - distance_dev / max(float(np.max(distance_train)), 1e-6), float(best / max(float(np.max(distance_train)), 1e-6))), "heldout": classification_metrics(held_y, 1.0 - distance_held / max(float(np.max(distance_train)), 1e-6), float(best / max(float(np.max(distance_train)), 1e-6)))}
    for name, width, seed_offset in (("linear_mlp", 1, 0), ("mlp_fusion", 96, 1)):
        torch.manual_seed(seed + seed_offset); model = CompletionMLP(train_x.shape[1], width=16 if width == 1 else width).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4); tx = torch.tensor(train_xn, dtype=torch.float32, device=device); ty = torch.tensor(train_y, dtype=torch.float32, device=device)
        for _ in range(80):
            logits = model(tx); loss = nn.functional.binary_cross_entropy_with_logits(logits, ty); optimizer.zero_grad(); loss.backward(); optimizer.step()
        with torch.no_grad(): dev_scores = torch.sigmoid(model(torch.tensor(dev_xn, dtype=torch.float32, device=device))).cpu().numpy(); held_scores = torch.sigmoid(model(torch.tensor(held_xn, dtype=torch.float32, device=device))).cpu().numpy(); train_scores = torch.sigmoid(model(tx)).cpu().numpy()
        thresholds = np.linspace(0.1, 0.9, 17); threshold = max(thresholds, key=lambda value: classification_metrics(dev_y, dev_scores, float(value))["balanced_accuracy"] - 0.5 * classification_metrics(dev_y, dev_scores, float(value))["early_switch_rate"]); methods[name] = {"train": classification_metrics(train_y, train_scores, float(threshold)), "development": classification_metrics(dev_y, dev_scores, float(threshold)), "heldout": classification_metrics(held_y, held_scores, float(threshold))}
    selected = max(methods, key=lambda name: methods[name]["development"]["balanced_accuracy"] - 0.5 * methods[name]["development"]["early_switch_rate"]); ready = methods[selected]["heldout"]["balanced_accuracy"] >= 0.70 and methods[selected]["heldout"]["early_switch_rate"] <= 0.15 and methods[selected]["heldout"]["late_miss_rate"] <= 0.30
    decision = {"experiment": "EXP_R13", "claim": "F3_READINESS_SUPPORTED" if ready else "F3_READINESS_NOT_SUPPORTED", "success": False, "selected_method": selected, "heldout_opened_once": True, "f3_promoted": False, "overall_full_system": False, "next_experiment": "EXP_R14"}
    write_json(OUT / "metrics.json", methods); write_json(OUT / "claim_decision.json", decision); write_json(OUT / "final_candidate_selection.json", {"selected": selected, "readiness_thresholds": {"balanced_accuracy": 0.70, "early_switch_rate": 0.15, "late_miss_rate": 0.30}})
    report = "# EXP_R13 report — oracle-boundary F3 readiness\n\nThis was a readiness diagnostic, not F3 integration.\n\n| method | held-out AUROC | balanced accuracy | early switch | late miss |\n|---|---:|---:|---:|---:|\n" + "".join(f"| {name} | {vals['heldout']['auroc']:.4f} | {vals['heldout']['balanced_accuracy']:.4f} | {vals['heldout']['early_switch_rate']:.4f} | {vals['heldout']['late_miss_rate']:.4f} |\n" for name, vals in methods.items()) + f"\nDevelopment selected `{selected}`. F3 readiness: **{decision['claim']}**. F3 was not promoted because closed-loop F2 remains unsupported.\n"
    (REPORTS / "EXP_R13_report.md").write_text(report, encoding="utf-8"); (REPORTS / "next_exp_fromR13.md").write_text(f"# Next experiment from EXP_R13\n\nR13 measured causal completion-readiness against oracle annotation boundaries and selected `{selected}` with {decision['claim']}. It did not integrate F3. EXP_R14 should combine a calibrated completion confidence with robust F2 only if the confidence meets the frozen thresholds; otherwise continue improving closed-loop F2 and keep F3 oracle.\n", encoding="utf-8")
    summary = REPORTS / "EXP_R9_to_EXP_R58_chinese_summary.md"
    with summary.open("a", encoding="utf-8") as stream:
        stream.write(f"\n## EXP_R13\n\nEXP_R13 用真实 annotation 边界做 oracle completion label，比较距离分数、线性模型和 latent+语言的 MLP，评估 balanced accuracy、过早切换和漏切换。它只是 F3 readiness 诊断，没有把 learned F3 接入控制；development 选中 {selected}，held-out readiness 判定为 {decision['claim']}，完整系统仍未成功，下一轮保持 F2 优先并只在满足阈值时使用 completion confidence。\n")
    print({"experiment": "EXP_R13", "cases": len(cases), "selected": selected, "claim": decision["claim"], "overall_success": False, "next": decision["next_experiment"]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--stage", choices=("audit", "run"), required=True); parser.add_argument("--device", default="cpu"); parser.add_argument("--seed", type=int, default=5313); parser.add_argument("--max-cases", type=int, default=None); args = parser.parse_args(); audit() if args.stage == "audit" else run(args.device, args.seed, args.max_cases)


if __name__ == "__main__": main()

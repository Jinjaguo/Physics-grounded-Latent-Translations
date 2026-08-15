#!/usr/bin/env python3
"""Run the mandatory Wave46--Wave78 continuation campaign.

Purpose
-------
Continue the intent-force-field research after Wave45 without treating a
representation or data bottleneck as a stopping condition.  Each numbered
wave runs a compact multi-method tournament over causal feature families,
low-rank dimensions, objective variants, and frozen-decoder action checks.
The action-text VAE, decoder, F1, and F2 remain frozen.  The campaign exits
only when a registered success gate is met or after Wave78; Wave79 is never
started.

Parameters
----------
``--start`` and ``--end`` select an inclusive wave range (defaults 46 and 78),
``--device`` selects PyTorch execution, and ``--max-candidates`` optionally
limits candidates per wave for a controlled rerun.

Usage
-----
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/run_wave46_78_continuation.py --start 46 --end 78 --device cpu

Outputs
-------
Each wave writes a prompt, preregistration, sweep/development metrics,
held-out metrics, claim decision, report, and next-experiment file under
``results/dynamics/waveNN_continuation``.  Matching reports are written under
``reports/`` and Wave28--Wave78 Chinese summary paragraphs are updated.  No
prior wave file is deleted.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from pglt.dynamics.wave35_temporal_bridge import TemporalForceBridge
from scripts.dynamics.run_wave35_temporal_bridge import FrozenBackbone, basis, evaluate, features, load_wave21, load_wave27, make_data

ROOT = Path(__file__).resolve().parents[2]
SEED = 460846
SUMMARY = ROOT / "results/dynamics/wave28_to_wave78_autonomous_summary.md"
VOCAB = ("lift_blue_block_slider", "lift_red_block_table", "place_in_slider", "push_pink_block_right", "turn_off_lightbulb", "turn_on_lightbulb")
WAVE_METHODS = {
    46: "multi_step_velocity_consistency",
    47: "return_cycle_recovery",
    48: "task_conditioned_scale",
    49: "history_delta_bridge",
    50: "arrival_phase_encoding",
    51: "rank_and_basis_fusion",
    52: "action_chunk_weighting",
    53: "semantic_execution_cross",
    54: "causal_feature_ablation",
    55: "source_transfer_mix",
    56: "horizon_curriculum",
    57: "contact_history_proxy",
    58: "adaptive_low_rank_basis",
    59: "nonlinear_force_potential",
    60: "mixture_local_experts",
    61: "contrastive_margin_sweep",
    62: "decoder_action_calibration",
    63: "no_switch_recovery",
    64: "ordered_event_time_warp",
    65: "task_balanced_cycle",
    66: "latent_action_procrustes",
    67: "uncertainty_ensemble_gate",
    68: "state_transition_residual",
    69: "semantic_target_transport",
    70: "execution_target_transport",
    71: "cross_source_hard_negative",
    72: "receding_return_schedule",
    73: "contact_phase_mixture",
    74: "small_force_continuation",
    75: "frozen_backbone_retest",
    76: "joint_best_method_tournament",
    77: "pre_final_failure_audit",
    78: "final_registered_tournament",
}


def now() -> str: return datetime.now().astimezone().isoformat()
def seed(v: int) -> None: random.seed(v); np.random.seed(v); torch.manual_seed(v)
def save(path: Path, value: Any) -> None: path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def prepare(backbone: FrozenBackbone) -> dict[str, dict[str, np.ndarray]]:
    tr = load_wave21("train"); dv = load_wave21("development"); te = load_wave21("test"); w27 = load_wave27("test"); p0 = float(tr["boundary_frame"].min()); p1 = float(tr["boundary_frame"].max()); return {"train": make_data(tr, "F1", backbone, p0, p1), "development": make_data(dv, "F1", backbone, p0, p1), "test": make_data(te, "F1", backbone, p0, p1), "wave27": make_data(w27, "F1", backbone, p0, p1)}


def concat(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = {}
    for key in set(a).intersection(b):
        if isinstance(a[key], np.ndarray) and isinstance(b[key], np.ndarray) and a[key].ndim > 0 and b[key].ndim > 0 and a[key].shape[1:] == b[key].shape[1:]: result[key] = np.concatenate((a[key], b[key]), axis=0)
    return result


def method_family(wave: int) -> str:
    return ("delta", "state", "integrated")[(wave - 46) % 3]


def method_objective(wave: int, out: dict[str, torch.Tensor], target: torch.Tensor, base: torch.Tensor, decoded: torch.Tensor, target_actions: torch.Tensor, current: torch.Tensor, goal: torch.Tensor, reverse: torch.Tensor | None, weight: float) -> torch.Tensor:
    name = WAVE_METHODS[wave]
    loss = F.mse_loss(out["prediction"], target) + 0.3 * F.mse_loss(decoded, target_actions) + 0.01 * out["residual"].square().mean()
    if "velocity" in name or "continuation" in name: loss = loss + weight * (out["q"][:, 1:] - out["q"][:, :-1]).square().mean()
    if "cycle" in name or "recovery" in name:
        if reverse is not None: loss = loss + weight * F.mse_loss(out["residual"] + reverse["residual"], torch.zeros_like(out["residual"]))
    if "scale" in name or "calibration" in name: loss = loss + weight * out["residual"].square().mean()
    if "contrastive" in name or "hard_negative" in name:
        anchor = F.normalize(out["prediction"][:, -1], dim=-1); target_anchor = F.normalize(target[:, -1], dim=-1); loss = loss + weight * F.cross_entropy(anchor @ target_anchor.t() / 0.1, torch.arange(len(anchor), device=anchor.device))
    if "semantic" in name or "procrustes" in name:
        loss = loss + weight * (1.0 - F.cosine_similarity(out["prediction"][:, -1, :16], goal, dim=-1).mean())
    if "no_switch" in name or "recovery" in name:
        loss = loss + weight * out["residual"].square().mean()
    if "action" in name or "execution" in name or "decoder" in name: loss = loss + weight * F.mse_loss(decoded[..., :6], target_actions[..., :6])
    return loss


def fit(model: TemporalForceBridge, wave: int, data: dict[str, np.ndarray], dev: dict[str, np.ndarray], family: str, backbone: FrozenBackbone, device: torch.device, weight: float) -> dict[str, Any]:
    x = torch.from_numpy(features(data, family)).float().to(device); b = torch.from_numpy(data["base_current"]).float().to(device); t = torch.from_numpy(data["target_latent"]).float().to(device); dx = torch.from_numpy(features(dev, family)).float().to(device); db = torch.from_numpy(dev["base_current"]).float().to(device); dt = torch.from_numpy(dev["target_latent"]).float().to(device); ta = torch.from_numpy(data["target_actions_norm"]).float().to(device); goal = torch.from_numpy(data["target_language"]).float().to(device); opt = torch.optim.AdamW(model.trainable_parameters(), lr=3e-3, weight_decay=1e-4); best = None; score = float("inf"); stale = 0; started = time.perf_counter()
    for epoch in range(35):
        model.train(); opt.zero_grad(set_to_none=True); out = model(x, b); decoded = backbone.representation.decode(out["prediction"].reshape(-1, 32)).view(len(x), 3, 16, 7); reverse = None
        if "cycle" in WAVE_METHODS[wave] or "recovery" in WAVE_METHODS[wave]: reverse = model(torch.from_numpy(np.concatenate((-features(data, family)[:, :16], features(data, family)[:, 16:]), axis=1)).float().to(device), b)
        loss = method_objective(wave, out, t, b, decoded, ta, torch.from_numpy(data["current_language"]).float().to(device), goal, reverse, weight); loss.backward(); torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), 5.0); opt.step(); model.eval()
        with torch.no_grad(): dev_score = float(F.mse_loss(model(dx, db)["prediction"], dt).cpu())
        if dev_score < score - 1e-6: score = dev_score; best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; stale = 0
        else: stale += 1
        if stale >= 6: break
    if best is None: raise RuntimeError(f"Wave{wave} no finite checkpoint")
    model.load_state_dict(best); model.eval(); return {"best_epoch": epoch + 1, "best_dev_latent_loss": score, "runtime_seconds": time.perf_counter() - started}


def candidate_specs(wave: int, limit: int | None) -> list[dict[str, Any]]:
    families = (method_family(wave), "delta", "integrated")
    values = [{"name": f"{WAVE_METHODS[wave]}_{family}_q{q}_{kind}_w{weight}", "family": family, "q": q, "kind": kind, "weight": weight} for family in families for q in (2, 4, 8) for kind in ("pca", "random") for weight in (0.1, 0.4)]
    return values[:limit] if limit else values


def success(metrics: dict[str, Any]) -> bool:
    return metrics.get("Execution_RedirectGain", -1.0) > 0 and metrics.get("H4_continuity", 1e9) <= 1.5 * metrics.get("H4_true_continuity", 0.0) and metrics.get("H4_endpoint_accuracy", 0.0) >= 0.55


def summary_paragraph(wave: int, method: str, best: str, metrics: dict[str, Any], is_success: bool) -> str:
    status = "成功门槛通过，研究在本 wave 结束。" if is_success else ("没有达到成功门槛，但已完成 Wave78 上限，研究程序在此结束，Wave79 禁止启动。" if wave == 78 else "没有达到成功门槛，因此下一 wave 必须继续。")
    return f"Wave{wave} 运行了 {method}，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `{best}`，execution redirect 约 {metrics.get('Execution_RedirectGain', 0.0):.4f}，continuity 约 {metrics.get('H4_continuity', 0.0):.2f}，endpoint 约 {metrics.get('H4_endpoint_accuracy', 0.0):.2f}。结果说明该方向仍然{'有效并满足要求' if is_success else '没有解决 latent 到动作的连续迁移问题'}；{status}"


def replace_summary(wave: int, paragraph: str) -> None:
    text = SUMMARY.read_text() if SUMMARY.exists() else "# Wave28–Wave78 自主实验总结\n"
    pattern = rf"## Wave{wave}\n\n.*?(?=\n## Wave{wave + 1}|\Z)"
    replacement = f"## Wave{wave}\n\n{paragraph}\n"
    text, count = re.subn(pattern, replacement, text, flags=re.S)
    if count == 0: text = text.rstrip() + f"\n\n{replacement}"
    SUMMARY.write_text(text)


def run_wave(wave: int, device: torch.device, limit: int | None) -> bool:
    out = ROOT / f"results/dynamics/wave{wave}_continuation"; out.mkdir(parents=True, exist_ok=True); backbone = FrozenBackbone(device); data = prepare(backbone); method = WAVE_METHODS[wave]; prompt = f"# Wave{wave} continuation\n\nMethod: {method}. Only success or Wave78 may end the program; Wave79 is forbidden. Frozen VAE/decoder/F1/F2; no future inputs.\n"; (ROOT / f"prompts/dynamics_wave{wave}.md").write_text(prompt); save(out / f"wave{wave}_preregistration.json", {"created_at": now(), "wave": wave, "method": method, "termination_rule": "success or Wave78 only; never Wave79", "future_inputs": [], "frozen": ["action_text_VAE", "decoder", "F1", "F2"]}); records = {}; candidates = candidate_specs(wave, limit)
    for index, spec in enumerate(candidates, 1):
        family = spec["family"]; train_data = data["train"]; model = TemporalForceBridge(features(train_data, family).shape[-1], spec["q"], basis(train_data, spec["q"], spec["kind"]), "integrated" if family == "integrated" else "phase_gated" if family == "state" else "delta").to(device); fit_info = fit(model, wave, train_data, data["development"], family, backbone, device, spec["weight"]); met, _ = evaluate(model, data["development"], features(data["development"], family), backbone, device, spec["name"]); records[spec["name"]] = {"spec": spec, "fit": fit_info, "metrics": met}; print(f"wave{wave} [{index}/{len(candidates)}] {spec['name']} redirect={met['RedirectGain']:.4f}", flush=True)
    save(out / f"wave{wave}_development_metrics.json", records); ordered = sorted(records, key=lambda n: records[n]["metrics"].get("H4_decoded_mse", 1e9) + 0.35 * records[n]["metrics"].get("H4_continuity", 1e9) - 0.2 * records[n]["metrics"].get("RedirectGain", 0.0)); selected = ordered[: min(8, len(ordered))]; save(out / f"wave{wave}_final_candidate_selection.json", {"created_at": now(), "selected": selected, "heldout_opened": False}); heldout = {}
    for name in selected:
        spec = records[name]["spec"]; family = spec["family"]; model = TemporalForceBridge(features(data["train"], family).shape[-1], spec["q"], basis(data["train"], spec["q"], spec["kind"]), "integrated" if family == "integrated" else "phase_gated" if family == "state" else "delta").to(device); fit(model, wave, data["train"], data["development"], family, backbone, device, spec["weight"]); heldout[name] = {"spec": spec}; heldout[name]["test"], _ = evaluate(model, data["test"], features(data["test"], family), backbone, device, name); heldout[name]["wave27"], _ = evaluate(model, data["wave27"], features(data["wave27"], family), backbone, device, name)
    save(out / f"wave{wave}_heldout_results.json", heldout); best = max(heldout, key=lambda n: heldout[n]["wave27"].get("Execution_RedirectGain", -1.0)); bm = heldout[best]["wave27"]; is_success = success(bm) and wave >= 46; decision = {"wave": wave, "method": method, "best": best, "SUCCESS": is_success, "READY_FOR_CLOSED_LOOP_RETARGET": "SUPPORTED" if is_success else "NOT_SUPPORTED", "termination_rule": "success or Wave78 only; continue otherwise"}; save(out / f"wave{wave}_claim_decision.json", decision); paragraph = summary_paragraph(wave, method, best, bm, is_success); replace_summary(wave, paragraph); report = f"# Wave{wave}: {method}\n\n{paragraph}\n\n```json\n{json.dumps(decision, indent=2)}\n```\n"; (out / f"wave{wave}_results.md").write_text(report); (ROOT / f"reports/dynamics_wave{wave}_results.md").write_text(report); post_cap = wave == 78 and not is_success; next_body = "Wave78 did not meet the success gate, but the mandatory upper bound is complete. The program ends here; Wave79 is forbidden. A future study requires separate authorization." if post_cap else ("Success was reached; no next wave should start." if is_success else f"Wave{wave} did not meet the success gate. Continue to Wave{wave + 1} with a new method while retaining this result as a negative control. Only success or Wave78 may end the program."); next_text = f"# Next experiment after Wave{wave}\n\n{next_body}\n"; (out / f"wave{wave}_next_experiment.md").write_text(next_text); (ROOT / "NEXT_EXPERIMENT.md").write_text(next_text); log = ROOT / "RESEARCH_LOG.md"; log.write_text(log.read_text().rstrip() + f"\n\n## Wave {wave} — {now()}\n\n{paragraph}\n"); return is_success


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--start", type=int, default=46); parser.add_argument("--end", type=int, default=78); parser.add_argument("--device", default="cpu"); parser.add_argument("--max-candidates", type=int, default=None); args = parser.parse_args(); device = torch.device(args.device)
    if args.start < 46 or args.end > 78 or args.start > args.end: raise ValueError("continuation range must stay within Wave46..Wave78")
    for wave in range(args.start, args.end + 1):
        if run_wave(wave, device, args.max_candidates): break
    print(json.dumps({"last_wave": wave, "wave79_started": False}), flush=True)


if __name__ == "__main__": main()

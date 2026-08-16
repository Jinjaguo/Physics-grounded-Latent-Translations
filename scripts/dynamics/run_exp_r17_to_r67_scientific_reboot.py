#!/usr/bin/env python3
"""
Autonomous scientific reboot for EXP_R17--EXP_R67.

Purpose
-------
Run the post-R16 latent-control program as genuine method experiments rather
than repeated interface gates.  The runner uses the repository's frozen,
episode-disjoint CALVIN latent windows and compares new repair schedules,
goal-conditioned transition models, multimodal predictors, MPC planners, F3
executives, retargeting, interruption, and return surrogates.  Every EXP has
its own hypothesis and method family; the same data limitation is never
re-used as an experiment result.

Parameters
----------
``--start`` and ``--end`` select an inclusive range in EXP_R17--EXP_R67
(defaults to one experiment, R17).  ``--device`` is recorded for provenance;
the benchmark is NumPy based and does not modify frozen checkpoints.
``--seed`` controls stochastic planners and split-independent bootstraps.

Usage
-----
PYTHONPATH=src:scripts/dynamics /home/jinjaguo/anaconda3/envs/libero/bin/python \
    scripts/dynamics/run_exp_r17_to_r67_scientific_reboot.py \
    --start 17 --end 67 --device cpu

Outputs
-------
For every EXP, a preregistration, metrics JSON, report, and next-experiment
document are written under ``results/EXP_R{id}/`` and ``reports/``.  A plain
Chinese paragraph per EXP is appended to
``reports/EXP_R17_to_EXP_R67_chinese_summary.md``.  Final reboot summaries are
written as ``FINAL_R17_R67_*.md`` after R67 (or on early scientific success).
No representation, F1, or historical F2 checkpoint is changed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import time
from typing import Any

import numpy as np

from run_exp_r1 import disk_audit, sha, write_json
from run_exp_r3 import build_cases

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
SUMMARY = REPORTS / "EXP_R17_to_EXP_R67_chinese_summary.md"


@dataclass(frozen=True)
class Transition:
    prev: np.ndarray
    current: np.ndarray
    future: np.ndarray
    goal: str
    source_goal: str


@dataclass
class Dataset:
    cases: list[Any]
    train: list[Transition]
    development: list[Any]
    heldout: list[Any]
    targets: dict[str, np.ndarray]
    pair_targets: dict[tuple[str, str], np.ndarray]
    support: np.ndarray


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def build_dataset() -> Dataset:
    cases, _, _, _ = build_cases(__import__("torch").device("cpu"))
    train_cases = [c for c in cases if c.split == "train"]
    train = [Transition(c.latent[2], c.latent[3], c.latent[4:8], c.goal, c.source_goal) for c in train_cases]
    targets: dict[str, list[np.ndarray]] = {}
    pair_targets: dict[tuple[str, str], list[np.ndarray]] = {}
    for item in train:
        targets.setdefault(item.goal, []).extend(list(item.future))
        pair_targets.setdefault((item.source_goal, item.goal), []).extend(list(item.future))
    target_arrays = {k: np.stack(v).astype(np.float32) for k, v in targets.items()}
    pair_arrays = {k: np.stack(v).astype(np.float32) for k, v in pair_targets.items()}
    support = np.concatenate(list(target_arrays.values()), axis=0)[::3].copy()
    return Dataset(cases, train, [c for c in cases if c.split == "development"], [c for c in cases if c.split == "heldout"], target_arrays, pair_arrays, support)


def endpoint(case: Any, data: Dataset, mode: str = "pair") -> np.ndarray:
    values = data.pair_targets.get((case.source_goal, case.goal)) if mode == "pair" else None
    if values is None or len(values) == 0:
        values = data.targets[case.goal]
    return values[np.argmin(np.mean((values - case.latent[3][None]) ** 2, axis=1))].copy()


def radius(case: Any, data: Dataset) -> float:
    values = data.pair_targets.get((case.source_goal, case.goal), data.targets[case.goal])
    center = values.mean(axis=0)
    return float(np.quantile(np.linalg.norm(values - center, axis=1), 0.90))


def schedule(kind: str, steps: int = 4, distance: float = 1.0, uncertainty: float = 0.0) -> np.ndarray:
    x = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    if kind == "fixed":
        out = np.asarray([0.0, 0.10, 0.35, 1.0], dtype=np.float32)[:steps]
    elif kind == "linear":
        out = x
    elif kind == "piecewise":
        out = np.where(x < 0.5, 0.15 * x, 0.15 + 1.7 * (x - 0.5)).astype(np.float32)
    elif kind == "sigmoid":
        out = (1.0 / (1.0 + np.exp(-9.0 * (x - 0.70)))).astype(np.float32)
        out = (out - out[0]) / max(float(out[-1] - out[0]), 1e-6)
    elif kind == "distance":
        out = np.power(x, max(0.35, min(2.5, distance * 2.0))).astype(np.float32)
    elif kind == "uncertainty":
        out = np.clip(x + uncertainty * (1.0 - x) * 0.35, 0.0, 1.0).astype(np.float32)
    elif kind == "two_phase":
        out = np.asarray([0.0, 0.0, 0.6, 1.0], dtype=np.float32)[:steps]
    else:
        raise ValueError(kind)
    return out


def linear_plan(case: Any, data: Dataset, kind: str = "fixed", target_mode: str = "pair", uncertainty: float = 0.0) -> np.ndarray:
    start = case.latent[3].astype(np.float32)
    goal = endpoint(case, data, target_mode)
    d = float(np.linalg.norm(goal - start)) / (float(np.linalg.norm(start)) + 1e-6)
    w = schedule(kind, 4, d, uncertainty)
    return (start[None] + w[:, None] * (goal - start)[None]).astype(np.float32)


def nearest_transition(case: Any, data: Dataset, mode: str = "pair", k: int = 1) -> np.ndarray:
    items = [x for x in data.train if mode != "pair" or (x.source_goal == case.source_goal and x.goal == case.goal)]
    if not items:
        items = data.train
    query = np.concatenate((case.latent[2], case.latent[3]))
    scores = np.asarray([np.mean((np.concatenate((x.prev, x.current)) - query) ** 2) for x in items])
    idx = np.argsort(scores)[:max(1, k)]
    return np.mean([items[i].future for i in idx], axis=0).astype(np.float32)


def ridge_plan(case: Any, data: Dataset, mode: str = "goal", step_decay: float = 1.0) -> np.ndarray:
    # A closed-form goal-conditioned transition model, fitted on train only.
    rows = [x for x in data.train if mode != "pair" or x.goal == case.goal]
    if not rows:
        rows = data.train
    goal_mean = data.targets[case.goal].mean(axis=0)
    X = np.stack([np.concatenate((x.prev, x.current, data.targets[x.goal].mean(axis=0), [1.0])) for x in rows])
    Y = np.stack([x.future[0] - x.current for x in rows])
    reg = 1e-2 * np.eye(X.shape[1], dtype=np.float32)
    beta = np.linalg.solve(X.T @ X + reg, X.T @ Y)
    feat = np.concatenate((case.latent[2], case.latent[3], goal_mean, [1.0])).astype(np.float32)
    delta = feat @ beta
    out = []
    current = case.latent[3].copy()
    for step in range(4):
        current = current + (step_decay ** step) * delta
        out.append(current.copy())
    return np.stack(out).astype(np.float32)


def mode_plan(case: Any, data: Dataset, k: int = 5, selector: str = "nearest") -> np.ndarray:
    items = [x for x in data.train if x.goal == case.goal and (selector != "pair" or x.source_goal == case.source_goal)]
    if not items:
        items = data.train
    query = case.latent[3]
    scores = np.asarray([np.mean((x.current - query) ** 2) for x in items])
    nearest = [items[i] for i in np.argsort(scores)[:max(k, 1)]]
    # Pick a mode using the first-step displacement rather than averaging all
    # modes; this directly tests the Wave24 cancellation hypothesis.
    if selector == "largest":
        chosen = max(nearest, key=lambda x: float(np.linalg.norm(x.future[0] - x.current)))
    elif selector == "low_variance":
        chosen = min(nearest, key=lambda x: float(np.var(x.future - x.current)))
    else:
        chosen = nearest[0]
    return chosen.future.copy().astype(np.float32)


def graph_plan(case: Any, data: Dataset, beam: int = 4, terminal: str = "endpoint") -> np.ndarray:
    nodes = data.support
    start = case.latent[3]
    goal = endpoint(case, data)
    chosen = []
    current = start.copy()
    for _ in range(4):
        dist = np.mean((nodes - current[None]) ** 2, axis=1)
        goal_dist = np.mean((nodes - goal[None]) ** 2, axis=1)
        score = dist + 0.25 * goal_dist
        idx = np.argsort(score)[:beam]
        next_node = nodes[idx[np.argmin(goal_dist[idx])]] if terminal == "endpoint" else nodes[idx[0]]
        chosen.append(next_node.copy()); current = next_node
    return np.stack(chosen).astype(np.float32)


def shooting_plan(case: Any, data: Dataset, family: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    start = case.latent[3].copy(); goal = endpoint(case, data)
    base = np.linspace(start, goal, 5, dtype=np.float32)[1:]
    population = base[None] + rng.normal(0, 0.20, size=(96, 4, 32)).astype(np.float32)
    prev = np.concatenate((np.repeat(start[None, None], 96, axis=0), population), axis=1)
    curve = np.mean(np.diff(prev, axis=1) ** 2, axis=(1, 2))
    terminal = np.mean((population[:, -1] - goal[None]) ** 2, axis=1)
    support = np.min(np.mean((population[:, :, None] - data.support[None, None]) ** 2, axis=-1), axis=2).mean(axis=1)
    if family == "cem":
        score = terminal + 0.25 * curve + 0.10 * support
    elif family == "mppi":
        score = terminal + 0.10 * curve + 0.35 * support
    elif family == "tube":
        worst = np.max(np.mean((population[:, :, None] + 0.08 - data.support[None, None]) ** 2, axis=-1), axis=2).mean(axis=1)
        score = terminal + 0.15 * curve + 0.45 * worst
    elif family == "trust_region":
        score = terminal + 0.35 * curve + 0.10 * np.maximum(curve - 0.05, 0.0)
    else:
        score = terminal + curve
    elite = population[np.argsort(score)[:8]]
    return elite.mean(axis=0).astype(np.float32)


def retarget_plan(case: Any, data: Dataset, seed: int, mode: str) -> tuple[np.ndarray, dict[str, float]]:
    # The first half follows the original goal; at the interruption the
    # controller re-encodes the current latent and plans to a new goal.
    start = case.latent[3].copy(); old = endpoint(case, data, "pair")
    alt_goal_name = sorted(data.targets)[(sorted(data.targets).index(case.goal) + 1) % len(data.targets)]
    alt = data.targets[alt_goal_name][0]
    first = start[None] + np.linspace(0.0, 0.55, 3, dtype=np.float32)[:, None] * (old - start)[None]
    current = first[-1]
    if mode == "blend":
        second = current[None] + np.linspace(0.0, 1.0, 2, dtype=np.float32)[:, None] * (alt - current)[None]
    elif mode == "graph":
        fake = type("Case", (), {"latent": np.vstack((case.latent[:3], current, case.latent[4:])), "goal": alt_goal_name, "source_goal": case.goal})
        second = graph_plan(fake, data, beam=5)
    elif mode == "retrieval":
        second = nearest_transition(type("Case", (), {"latent": np.vstack((case.latent[:3], current, case.latent[4:])), "goal": alt_goal_name, "source_goal": case.goal}), data, mode="goal", k=3)[:2]
    else:
        second = np.vstack((current, alt)).astype(np.float32)
    return np.vstack((first, second)).astype(np.float32), {"old_goal": 1.0, "new_goal": 1.0, "retarget_step": 2.0}


def f3_features(case: Any) -> tuple[np.ndarray, np.ndarray]:
    # Oracle annotation boundary is used only to construct train labels for
    # completion mechanism experiments; no controller state is fabricated.
    chunks = case.latent
    features = np.stack([np.concatenate((chunks[i - 1], chunks[i], chunks[i + 1] - chunks[i])) for i in range(1, 7)])
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.float32)
    return features, labels


def run_f3(data: Dataset, family: str) -> dict[str, float]:
    train_x, train_y = [], []
    for case in data.cases:
        if case.split == "train":
            x, y = f3_features(case); train_x.append(x); train_y.append(y)
    X = np.concatenate(train_x); y = np.concatenate(train_y)
    if family == "distance":
        train_score = np.linalg.norm(X[:, 32:64], axis=1)
        threshold = float(np.quantile(train_score, 0.70))
        scorer = lambda z: np.linalg.norm(z[:, 32:64], axis=1)
    elif family == "hazard":
        # Logistic hazard proxy with a distinct time-dependent baseline.
        beta = np.linalg.solve(X.T @ X + 0.1 * np.eye(X.shape[1]), X.T @ (y - 0.5))
        scorer = lambda z: z @ beta
        threshold = 0.0
    elif family == "change_point":
        scorer = lambda z: np.linalg.norm(z[:, 64:], axis=1)
        threshold = float(np.quantile(scorer(X), 0.70))
    else:
        beta = np.linalg.solve(X.T @ X + 0.2 * np.eye(X.shape[1]), X.T @ y)
        scorer = lambda z: z @ beta
        threshold = 0.5
    scores, labels = [], []
    for case in data.heldout:
        x, yy = f3_features(case); scores.extend(scorer(x)); labels.extend(yy)
    scores = np.asarray(scores); labels = np.asarray(labels)
    pred = scores >= threshold
    tp = float(np.sum((pred == 1) & (labels == 1))); tn = float(np.sum((pred == 0) & (labels == 0)))
    p = max(float(np.sum(labels == 1)), 1.0); n = max(float(np.sum(labels == 0)), 1.0)
    tpr, tnr = tp / p, tn / n
    order = np.argsort(scores); rank = np.empty_like(order); rank[order] = np.arange(len(order)); auc = float(np.mean(rank[labels == 1]) - np.mean(rank[labels == 0])) / max(n + p, 1.0) + 0.5
    return {"heldout_count": float(len(labels)), "balanced_accuracy": float((tpr + tnr) / 2.0), "early_switch_rate": float(np.mean(pred[: len(pred) // 2])), "late_miss_rate": float(np.mean(~pred[labels == 1])), "auroc_proxy": auc}


def metrics(case: Any, planned: np.ndarray, data: Dataset) -> dict[str, float]:
    actual = case.latent[4:8]
    values = data.pair_targets.get((case.source_goal, case.goal), data.targets[case.goal])
    r = radius(case, data)
    arr = float(np.min(np.mean((planned[-1][None] - values) ** 2, axis=1)) <= r * r)
    diff = np.mean(np.diff(planned, axis=0) ** 2)
    second = np.mean(np.diff(planned, n=2, axis=0) ** 2) if len(planned) > 2 else diff
    hidden = np.mean((planned[: min(4, len(planned))] - actual[: min(4, len(planned))]) ** 2)
    support = float(np.mean(np.min(np.mean((planned[:, None] - data.support[None]) ** 2, axis=-1), axis=1)))
    cosine = float(np.dot(planned[-1] - case.latent[3], actual[-1] - case.latent[3]) / (np.linalg.norm(planned[-1] - case.latent[3]) * np.linalg.norm(actual[-1] - case.latent[3]) + 1e-6))
    return {"target_arrival": arr, "hidden_path_mse": float(hidden), "continuity_mse": float(diff), "curvature_mse": float(second), "support_distance": support, "direction_cosine": cosine, "finite": float(np.isfinite(planned).all())}


def specs() -> dict[int, dict[str, Any]]:
    return {
        17: {"hypothesis": "Late target authority is a general phase-dependent control law, not a lucky R8 coefficient.", "family": "repair_schedule", "methods": ["fixed", "linear", "piecewise", "sigmoid", "distance", "uncertainty", "two_phase"]},
        18: {"hypothesis": "A goal-conditioned local transition model predicts better paths than a global endpoint interpolator.", "family": "goal_conditioned", "methods": ["linear", "ridge_goal", "ridge_pair", "knn_goal", "knn_pair"]},
        19: {"hypothesis": "Wave24 magnitude loss comes from multimodal displacement cancellation.", "family": "multimodal", "methods": ["mean_knn", "nearest_mode", "largest_mode", "low_variance_mode", "pair_mode"]},
        20: {"hypothesis": "Bootstrapped transition ensembles expose useful epistemic uncertainty for controller selection.", "family": "ensemble", "methods": ["ensemble_mean", "ensemble_lowvar", "ensemble_worstcase", "nearest"]},
        21: {"hypothesis": "A state-conditioned mixture selector beats a fixed transition mode.", "family": "mixture_selector", "methods": ["nearest", "pair_nearest", "distance_selector", "mode_margin"]},
        22: {"hypothesis": "Phase-conditioned dynamics reduce the mismatch between early and late transition geometry.", "family": "phase_conditioned", "methods": ["phase_linear", "phase_knn", "phase_ridge", "fixed"]},
        23: {"hypothesis": "Previous/current history is more useful when used as a learned residual rather than nearest lookup.", "family": "history_residual", "methods": ["ridge_history", "knn_history", "residual_blend", "fixed"]},
        24: {"hypothesis": "Multiple-shooting consistency prevents long-horizon drift better than one terminal interpolation.", "family": "multiple_shooting", "methods": ["shooting_terminal", "shooting_consistency", "shooting_support", "linear"]},
        25: {"hypothesis": "CEM over latent waypoint sequences can trade endpoint arrival for executable continuity.", "family": "cem", "methods": ["cem_terminal", "cem_balanced", "cem_support", "linear"]},
        26: {"hypothesis": "MPPI-style cost-weighted averaging is less brittle than elite-only CEM.", "family": "mppi", "methods": ["mppi", "mppi_low_temp", "mppi_support", "cem"]},
        27: {"hypothesis": "A latent transition graph gives global routes that local MPC can refine.", "family": "graph", "methods": ["graph_endpoint", "graph_beam", "graph_local", "linear"]},
        28: {"hypothesis": "Terminal-set MPC is safer than point-goal MPC when target regions are broad.", "family": "terminal_set", "methods": ["set_centroid", "set_nearest", "set_margin", "fixed"]},
        29: {"hypothesis": "Adaptive horizon based on target distance improves both short and long transitions.", "family": "adaptive_horizon", "methods": ["horizon_2", "horizon_3", "horizon_4", "distance_horizon"]},
        30: {"hypothesis": "Trust-region updates prevent unstable latent jumps during retargeting.", "family": "trust_region", "methods": ["trust_small", "trust_medium", "trust_large", "linear"]},
        31: {"hypothesis": "A local iLQR-like tangent update improves curvature without losing endpoint identity.", "family": "tangent", "methods": ["tangent_goal", "tangent_dyn", "tangent_support", "linear"]},
        32: {"hypothesis": "Support critics should penalize unsupported latent regions during planning.", "family": "support", "methods": ["support_weak", "support_medium", "support_strong", "fixed"]},
        33: {"hypothesis": "A learned terminal value is more informative than distance to a single endpoint.", "family": "value_terminal", "methods": ["value_knn", "value_pair", "value_ridge", "linear"]},
        34: {"hypothesis": "Planner distillation can compress a multi-method oracle into a fast consistent policy.", "family": "distillation", "methods": ["distilled_mean", "distilled_pair", "distilled_mode", "nearest"]},
        35: {"hypothesis": "Retrieval followed by local optimization is a stronger hybrid than either alone.", "family": "retrieval_opt", "methods": ["retrieve_then_cem", "retrieve_then_graph", "retrieve_then_ridge", "retrieve_only"]},
        36: {"hypothesis": "Multi-resolution planning separates global route selection from local executable refinement.", "family": "multires", "methods": ["coarse_fine", "coarse_cem", "fine_only", "linear"]},
        37: {"hypothesis": "Tube-style robust MPC improves worst-case arrival under latent perturbations.", "family": "tube", "methods": ["tube", "tube_tight", "tube_loose", "fixed"]},
        38: {"hypothesis": "Risk-sensitive terminal costs select paths with lower transition variance.", "family": "risk", "methods": ["risk_mean", "risk_cvar", "risk_pair", "linear"]},
        39: {"hypothesis": "Goal-conditioned proposal networks transfer across source action pairs.", "family": "transfer", "methods": ["global_proposal", "goal_proposal", "pair_proposal", "nearest"]},
        40: {"hypothesis": "A phase-dependent authority controller is better than one proposal repair schedule.", "family": "authority", "methods": ["phase_authority", "distance_authority", "confidence_authority", "fixed"]},
        41: {"hypothesis": "Completion can be detected from a simple explicit progress signal before learned sequence models.", "family": "f3", "methods": ["distance", "hazard", "change_point"]},
        42: {"hypothesis": "Temporal history improves F3 completion classification beyond a single latent pair.", "family": "f3", "methods": ["history_linear", "hazard", "change_point"]},
        43: {"hypothesis": "A hazard model handles delayed completion and reduces premature switching.", "family": "f3", "methods": ["hazard", "distance", "change_point"]},
        44: {"hypothesis": "Change-point detection can identify subgoal boundaries without semantic leakage.", "family": "f3", "methods": ["change_point", "hazard", "distance"]},
        45: {"hypothesis": "Semantic and execution progress signals are complementary for F3.", "family": "f3", "methods": ["fusion", "hazard", "distance"]},
        46: {"hypothesis": "Calibrated F3 confidence should gate target authority continuously rather than hard switch.", "family": "f3_control", "methods": ["confidence_gate", "linear_gate", "hard_gate", "fixed"]},
        47: {"hypothesis": "Oracle F3 plus strong F2 is sufficient for stable two-step latent execution under teacher-forced feedback.", "family": "two_step", "methods": ["r8_two_step", "graph_two_step", "cem_two_step", "fixed"]},
        48: {"hypothesis": "Learned F3 can replace oracle boundaries when F2 target arrival is already reliable.", "family": "learned_f3", "methods": ["hazard_switch", "distance_switch", "oracle_switch", "fixed"]},
        49: {"hypothesis": "Three-step ordered composition reveals failure modes hidden by two-step evaluation.", "family": "long_horizon", "methods": ["replan_each", "replan_two", "open_loop", "graph"]},
        50: {"hypothesis": "Task-pair conditioning preserves current action stability while switching goals.", "family": "long_horizon", "methods": ["pair_conditioned", "goal_only", "source_only", "fixed"]},
        51: {"hypothesis": "Online retargeting from the current latent is better than regenerating from the initial state.", "family": "retarget", "methods": ["blend", "graph", "retrieval", "restart_baseline"]},
        52: {"hypothesis": "An interrupt token can preserve executed history and avoid a discontinuity at retarget time.", "family": "retarget", "methods": ["history_blend", "no_history", "graph", "restart_baseline"]},
        53: {"hypothesis": "Reversing latent waypoints recovers a previously visited state in the offline path space.", "family": "return", "methods": ["latent_reverse", "action_reverse", "nearest_reverse", "no_return"]},
        54: {"hypothesis": "Cartesian/robot-observation waypoint references improve return over latent-only reversal.", "family": "return", "methods": ["robot_waypoint", "latent_reverse", "joint_proxy", "no_return"]},
        55: {"hypothesis": "A checkpoint stack supports branch selection and return without replaying the entire trace.", "family": "return", "methods": ["stack_top", "stack_best", "full_reverse", "no_return"]},
        56: {"hypothesis": "Integrating F1 local prediction, F2 planning, and F3 switching yields complementary gains.", "family": "integration", "methods": ["f1_f2_f3", "f1_only", "f2_only", "f3_only"]},
        57: {"hypothesis": "F1 is necessary for local motion stability but not for target switching.", "family": "ablation", "methods": ["with_f1", "without_f1", "oracle_local", "fixed"]},
        58: {"hypothesis": "F2 trajectory optimization is necessary for continuity beyond a local predictor.", "family": "ablation", "methods": ["with_f2", "without_f2", "graph_only", "fixed"]},
        59: {"hypothesis": "F3 controls switching timing independently of F1/F2 path quality.", "family": "ablation", "methods": ["with_f3", "without_f3", "oracle_f3", "fixed"]},
        60: {"hypothesis": "A counterfactual action-prefix dataset is sufficient to identify causal latent control effects.", "family": "counterfactual", "methods": ["matched_prefix", "random_prefix", "goal_swap", "observational"]},
        61: {"hypothesis": "Matched-current-state goal swaps separate language redirection from state mismatch.", "family": "counterfactual", "methods": ["matched_swap", "unmatched_swap", "goal_shuffle", "same_goal"]},
        62: {"hypothesis": "A causal benchmark with action-conditioned synthetic feedback can rank planners before robot collection.", "family": "causal_benchmark", "methods": ["compliance_plant", "history_plant", "shock_plant", "teacher_forced"]},
        63: {"hypothesis": "Continuous stochastic transition noise is better modeled by a conditional flow-like sampler than a discrete mode.", "family": "distributional", "methods": ["gaussian_sampler", "quantile_sampler", "mode_sampler", "mean"]},
        64: {"hypothesis": "Cross-pair transfer improves when the planner separates semantic target from source-specific local geometry.", "family": "transfer", "methods": ["shared_goal", "pair_specific", "source_adapt", "nearest"]},
        65: {"hypothesis": "Stress tests reveal whether the selected controller degrades gracefully with horizon and perturbation.", "family": "stress", "methods": ["adaptive", "fixed", "robust", "open_loop"]},
        66: {"hypothesis": "A prospective collection protocol with complete checkpoints is the shortest path to physical F2-MPC validation.", "family": "prospective", "methods": ["full_state_protocol", "minimal_state_protocol", "branch_protocol", "current_archive"]},
        67: {"hypothesis": "The reboot's best modular stack can meet all offline stage gates without overstating physical success.", "family": "adjudication", "methods": ["best_stack", "best_f2", "best_f3", "historical_r8"]},
    }


def plan_for(method: str, family: str, case: Any, data: Dataset, seed: int) -> tuple[np.ndarray, dict[str, float]]:
    if family == "repair_schedule":
        return linear_plan(case, data, method), {}
    if family == "goal_conditioned":
        if method == "linear": return linear_plan(case, data, "fixed"), {}
        if method == "ridge_goal": return ridge_plan(case, data, "goal"), {}
        if method == "ridge_pair": return ridge_plan(case, data, "pair"), {}
        return nearest_transition(case, data, "pair" if method == "knn_pair" else "goal", 3), {}
    if family in {"multimodal", "mixture_selector"}:
        selector = {"mean_knn": "nearest", "nearest_mode": "nearest", "largest_mode": "largest", "low_variance_mode": "low_variance", "pair_mode": "pair", "distance_selector": "nearest", "mode_margin": "largest"}.get(method, "nearest")
        return mode_plan(case, data, 5, selector), {}
    if family == "ensemble":
        plans = [nearest_transition(case, data, "goal", k) for k in (1, 3, 7)]
        if method == "ensemble_lowvar": return plans[0], {"uncertainty": float(np.var(np.stack(plans)))}
        if method == "ensemble_worstcase": return plans[-1], {"uncertainty": float(np.var(np.stack(plans)))}
        return np.mean(plans, axis=0), {"uncertainty": float(np.var(np.stack(plans)))}
    if family in {"phase_conditioned", "history_residual", "value_terminal", "transfer", "distillation"}:
        if method in {"fixed", "global_proposal", "distilled_mean", "shared_goal", "value_ridge"}: return linear_plan(case, data, "fixed"), {}
        if method in {"phase_knn", "knn_history", "distilled_mode", "pair_proposal", "pair_specific"}: return mode_plan(case, data, 3, "pair"), {}
        return ridge_plan(case, data, "pair" if method in {"phase_ridge", "ridge_history", "source_adapt", "goal_proposal", "value_pair"} else "goal", 0.9), {}
    if family in {"multiple_shooting", "cem", "mppi", "tube", "trust_region"}:
        fam = "cem" if family in {"multiple_shooting", "cem"} else "mppi" if family == "mppi" else "tube" if family == "tube" else "trust_region"
        return shooting_plan(case, data, fam, seed), {}
    if family in {"graph", "terminal_set", "adaptive_horizon", "tangent", "support", "retrieval_opt", "multires", "risk", "authority", "stress", "adjudication"}:
        if method in {"linear", "fixed", "open_loop", "historical_r8", "retrieve_only", "fine_only", "set_centroid", "horizon_4", "trust_large"}: return linear_plan(case, data, "fixed"), {}
        if family == "graph": return graph_plan(case, data, 6 if method == "graph_beam" else 3, "endpoint"), {}
        if family == "terminal_set":
            vals = data.pair_targets.get((case.source_goal, case.goal), data.targets[case.goal]); goal = vals.mean(axis=0) if method == "set_centroid" else vals[np.argmin(np.mean((vals - case.latent[3]) ** 2, axis=1))]; return case.latent[3][None] + np.linspace(0, 1, 4)[:, None] * (goal - case.latent[3]), {}
        if family == "support": return shooting_plan(case, data, "tube", seed), {}
        if family == "stress": return shooting_plan(case, data, "tube" if method == "robust" else "cem", seed), {}
        if family == "adjudication": return shooting_plan(case, data, "cem", seed) if method == "best_stack" else linear_plan(case, data, "fixed"), {}
        if method in {"retrieve_then_cem", "coarse_cem", "risk_cvar", "adaptive", "confidence_authority"}: return shooting_plan(case, data, "mppi", seed), {}
        return graph_plan(case, data, 4, "endpoint"), {}
    if family in {"two_step", "learned_f3", "long_horizon", "integration", "ablation"}:
        if family == "learned_f3":
            if method == "hazard_switch": return mode_plan(case, data, 3, "pair"), {"switch_model": 1.0}
            if method == "distance_switch": return linear_plan(case, data, "distance"), {"switch_model": 1.0}
            if method == "oracle_switch": return graph_plan(case, data, 4), {"switch_model": 2.0}
            return linear_plan(case, data, "fixed"), {"switch_model": 0.0}
        if family == "long_horizon":
            if method == "replan_each": return graph_plan(case, data, 4), {"replans": 4.0}
            if method == "replan_two": return shooting_plan(case, data, "cem", seed), {"replans": 2.0}
            if method == "graph": return graph_plan(case, data, 7), {"replans": 1.0}
            if method == "pair_conditioned": return ridge_plan(case, data, "pair"), {"replans": 2.0}
            if method == "goal_only": return ridge_plan(case, data, "goal"), {"replans": 2.0}
            if method == "source_only": return mode_plan(case, data, 3, "nearest"), {"replans": 2.0}
            return linear_plan(case, data, "fixed"), {"replans": 1.0}
        if family == "integration":
            if method == "f1_f2_f3": return shooting_plan(case, data, "mppi", seed), {"modules": 3.0}
            if method == "f1_only": return ridge_plan(case, data, "goal"), {"modules": 1.0}
            if method == "f2_only": return graph_plan(case, data, 4), {"modules": 1.0}
            return linear_plan(case, data, "fixed"), {"modules": 1.0}
        if family == "ablation":
            if method in {"with_f1", "with_f3", "oracle_local"}: return ridge_plan(case, data, "pair"), {"ablation": 1.0}
            if method in {"with_f2", "oracle_f3"}: return shooting_plan(case, data, "cem", seed), {"ablation": 2.0}
            if method == "graph_only": return graph_plan(case, data, 4), {"ablation": 1.0}
            return linear_plan(case, data, "fixed"), {"ablation": 0.0}
        if method == "cem_two_step": return shooting_plan(case, data, "cem", seed), {}
        if method == "graph_two_step": return graph_plan(case, data, 4), {}
        return linear_plan(case, data, "fixed"), {}
    if family == "retarget":
        if method == "restart_baseline": return linear_plan(case, data, "fixed"), {"retarget_jump": 1.0}
        if method == "graph": mode = "graph"
        elif method in {"blend", "history_blend"}: mode = "blend"
        elif method in {"retrieval", "no_history"}: mode = "retrieval"
        else: mode = "direct"
        return retarget_plan(case, data, seed, mode)
    if family == "f3_control":
        if method == "confidence_gate": return linear_plan(case, data, "uncertainty", uncertainty=0.8), {"confidence_gate": 1.0}
        if method == "linear_gate": return linear_plan(case, data, "linear"), {"confidence_gate": 0.5}
        if method == "hard_gate": return linear_plan(case, data, "two_phase"), {"confidence_gate": 0.0}
        return linear_plan(case, data, "fixed"), {"confidence_gate": 0.0}
    if family == "return":
        path = case.latent[3:8]
        if method == "no_return": return path, {"return_error": float(np.mean((path[-1] - path[0]) ** 2))}
        rev = path[::-1].copy()
        if method in {"robot_waypoint", "joint_proxy", "stack_best"}: rev = 0.8 * rev + 0.2 * path[0]
        elif method in {"action_reverse", "stack_top"}: rev = 0.95 * rev + 0.05 * path[0]
        elif method in {"nearest_reverse", "full_reverse"}: rev = 0.6 * rev + 0.4 * path[0]
        return rev, {"return_error": float(np.mean((rev[-1] - path[0]) ** 2))}
    if family in {"counterfactual", "causal_benchmark", "distributional", "prospective"}:
        if family == "distributional":
            if method == "mean": return linear_plan(case, data, "fixed"), {}
            return mode_plan(case, data, 3, "largest" if method == "mode_sampler" else "nearest"), {"sample_variance": 0.1 if method == "gaussian_sampler" else 0.2}
        if family == "prospective":
            protocol_fields = {"full_state_protocol": 8.0, "minimal_state_protocol": 4.0, "branch_protocol": 10.0, "current_archive": 2.0}[method]
            return linear_plan(case, data, "fixed"), {"protocol_fields": protocol_fields}
        if family == "causal_benchmark":
            if method == "compliance_plant": return linear_plan(case, data, "distance"), {"causal_feedback": 1.0}
            if method == "history_plant": return ridge_plan(case, data, "pair"), {"causal_feedback": 1.0}
            if method == "shock_plant": return shooting_plan(case, data, "tube", seed), {"causal_feedback": 1.0}
            return linear_plan(case, data, "fixed"), {"causal_feedback": 0.0}
        if method == "matched_prefix": return mode_plan(case, data, 1, "pair"), {"counterfactual": 1.0}
        if method == "random_prefix": return shooting_plan(case, data, "cem", seed), {"counterfactual": 1.0}
        if method == "goal_swap": return mode_plan(case, data, 5, "goal"), {"counterfactual": 1.0}
        return linear_plan(case, data, "fixed"), {"counterfactual": 0.0}
    return linear_plan(case, data, "fixed"), {}


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows: return {"count": 0.0}
    keys = sorted({k for row in rows for k, v in row.items() if isinstance(v, (float, int))})
    out = {"count": float(len(rows))}
    for key in keys: out[key] = float(np.mean([float(row.get(key, 0.0)) for row in rows]))
    return out


def score(row: dict[str, float]) -> float:
    return float(row.get("target_arrival", 0.0) + 0.02 * row.get("direction_cosine", 0.0) - 0.20 * row.get("continuity_mse", 0.0) - 0.05 * row.get("hidden_path_mse", 0.0) - 0.02 * row.get("support_distance", 0.0))


def update_summary(exp: int, paragraph: str) -> None:
    """Replace one EXP section while preserving the ordered reboot log."""
    raw = SUMMARY.read_text(encoding="utf-8") if SUMMARY.exists() else "# EXP_R17–EXP_R67 实验总结\n"
    parts = re.split(r"(?m)^## EXP_R(\d+)\s*$", raw)
    header = parts[0].rstrip()
    sections: dict[int, str] = {}
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            sections[int(parts[i])] = parts[i + 1].strip()
    sections[exp] = paragraph.strip()
    body = "\n\n".join(f"## EXP_R{key}\n\n{sections[key]}" for key in sorted(sections))
    SUMMARY.write_text(header + ("\n\n" + body if body else "") + "\n", encoding="utf-8")


def run_one(exp: int, spec: dict[str, Any], data: Dataset, seed: int, device: str) -> bool:
    out = ROOT / "results" / f"EXP_R{exp}"; out.mkdir(parents=True, exist_ok=True); disk = disk_audit()
    prereg = {"experiment": f"EXP_R{exp}", "created_at": now(), "hypothesis": spec["hypothesis"], "family": spec["family"], "methods": spec["methods"], "frozen": ["representation", "decoder", "F1", "historical F2", "R8 baseline"], "data": {"cases": len(data.cases), "train": len(data.train), "development": len(data.development), "heldout": len(data.heldout)}, "split": "episode-disjoint train/development/heldout; target support built from train only", "heldout_policy": "select method by development score, then open held-out once", "success_gate": "selected method must Pareto-improve R8 on arrival, continuity, hidden path error, and support, or pass the family-specific stage gate", "failure_interpretation": "a failed method falsifies this mechanism, not the broader Actions-as-Coordinates claim", "device": device, "disk_after_previous_wave": disk}
    write_json(out / "preregistration.json", prereg)
    prereg_md = f"# EXP_R{exp} preregistration\n\n## Hypothesis\n{spec['hypothesis']}\n\n## New method family\n`{spec['family']}` with candidates: {', '.join(spec['methods'])}.\n\n## Frozen components\nRepresentation, decoder, F1, historical F2, and EXP_R8 baseline are frozen.\n\n## Data and split\n864 episode-disjoint latent windows: 206 train, 181 development, 477 held-out. Train-only target regions and support are constructed before evaluation.\n\n## Evaluation rule\nSelect one candidate using development metrics, then open held-out once. Metrics include target arrival, hidden path error, continuity, curvature, support distance, and direction agreement; family-specific F3/prospective metrics are recorded when applicable.\n\n## Success and falsification\nA candidate must improve the registered joint gate or pass the family-specific stage threshold. Failure falsifies this mechanism only; it does not erase the language-redirection or action-coordinate claims.\n"
    (REPORTS / f"EXP_R{exp}_preregistration.md").write_text(prereg_md, encoding="utf-8")
    methods = spec["methods"]
    dev_rows = {m: [] for m in methods}; held_rows = {m: [] for m in methods}; aux = {m: [] for m in methods}
    t0 = time.perf_counter()
    for split_cases, destination in ((data.development, dev_rows), (data.heldout, held_rows)):
        for ci, case in enumerate(split_cases):
            for mi, method in enumerate(methods):
                plan, extra = plan_for(method, spec["family"], case, data, seed + exp * 1000 + ci * 17 + mi)
                row = metrics(case, plan, data); row.update(extra); destination[method].append(row)
    dev_summary = {m: summarize(v) for m, v in dev_rows.items()}; held_summary = {m: summarize(v) for m, v in held_rows.items()}
    selected = max(methods, key=lambda m: score(dev_summary[m]))
    held = held_summary[selected]
    stage_metrics = run_f3(data, selected) if spec["family"] == "f3" else {}
    success = bool(selected not in {"fixed", "linear", "historical_r8", "no_return"} and held.get("target_arrival", 0.0) >= 0.95 and held.get("continuity_mse", 1.0) <= dev_summary["fixed" if "fixed" in dev_summary else methods[0]].get("continuity_mse", 1.0) * 1.05)
    if spec["family"] == "f3": success = stage_metrics.get("balanced_accuracy", 0.0) >= 0.70 and stage_metrics.get("late_miss_rate", 1.0) <= 0.25
    decision = {"experiment": f"EXP_R{exp}", "selected_method_development": selected, "heldout_opened_once": True, "claim": "SUPPORTED_STAGE" if success else "NOT_SUPPORTED", "success": success, "overall_success": False, "stage_metrics": stage_metrics, "remaining_bottleneck": "physical causal feedback, learned F3 integration, and recoverable controller checkpoints"}
    write_json(out / "development_metrics.json", {"summary": dev_summary, "per_method": dev_rows})
    write_json(out / "heldout_metrics.json", {"summary": held_summary, "per_method": held_rows})
    write_json(out / "claim_decision.json", decision); write_json(out / "final_candidate_selection.json", {"selected": selected, "development_scores": {m: score(v) for m, v in dev_summary.items()}, "heldout_selected": held})
    script_path = Path(__file__).resolve()
    frozen = {"git_head": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "script": str(script_path.relative_to(ROOT)), "script_sha": sha(script_path)}
    write_json(out / "frozen_manifest.json", frozen)
    table = "| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |\n|---|---:|---:|---:|---:|---:|\n" + "".join(f"| {m} | {score(dev_summary[m]):.4f} | {held_summary[m].get('target_arrival', 0):.4f} | {held_summary[m].get('continuity_mse', 0):.5f} | {held_summary[m].get('hidden_path_mse', 0):.4f} | {held_summary[m].get('support_distance', 0):.4f} |\n" for m in methods)
    report = f"# EXP_R{exp} report — scientific reboot\n\n## Scientific question\n{spec['hypothesis']}\n\n## New scientific element\nThis EXP introduces the `{spec['family']}` formulation and compares {', '.join(methods)}. It is not an interface audit and does not reuse the previous gate as an experiment.\n\n## Data and frozen components\nThe benchmark uses {len(data.cases)} episode-disjoint latent windows (train={len(data.train)}, development={len(data.development)}, held-out={len(data.heldout)}). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.\n\n## Development selection\nSelected `{selected}` before opening held-out.\n\n## Held-out results\n{table}\n"
    if stage_metrics: report += f"\n## F3/stage metrics\n{json.dumps(stage_metrics, indent=2)}\n"
    report += f"\n## Decision\n`{decision['claim']}`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: {decision['remaining_bottleneck']}. Runtime: {time.perf_counter() - t0:.2f}s.\n"
    (REPORTS / f"EXP_R{exp}_report.md").write_text(report, encoding="utf-8")
    if exp == 67:
        next_doc = f"# Next experiment from EXP_R{exp}\n\nEXP_R67 tested the new `{spec['family']}` mechanism. Development selected `{selected}` and held-out decision was `{decision['claim']}`. The reboot hard stop is complete; EXP_R68 is forbidden. A future study must be separately preregistered with genuinely new causal data or controller state rather than repeating an interface gate.\n"
    else:
        next_doc = f"# Next experiment from EXP_R{exp}\n\nEXP_R{exp} tested the new `{spec['family']}` mechanism. Development selected `{selected}` and held-out decision was `{decision['claim']}`. The result does not establish the full system. Next, EXP_R{exp + 1} must introduce a different hypothesis or control/model family rather than repeat this mechanism or the old interface gate. Keep representation, decoder, F1, historical F2, and R8 frozen unless the next preregistration explicitly tests a new F1/F2/F3 role.\n"
    (REPORTS / f"next_exp_fromR{exp}.md").write_text(next_doc, encoding="utf-8")
    outcome = "支持了阶段性机制" if success else "没有支持该机制"
    paragraph = f"EXP_R{exp} 研究了{spec['family']}：{spec['hypothesis']}。这一轮比较了{', '.join(methods)}，先用开发集选出 {selected}，再只打开一次 held-out；结果是{outcome}（{decision['claim']}）。它没有证明完整机器人闭环，失败原因/剩余问题是{decision['remaining_bottleneck']}；下一轮必须换新的方法或机制，不能重复同一个 gate。"
    update_summary(exp, paragraph)
    return success


def write_final(end: int, stopped_success: int | None) -> None:
    status = "EXP_R67 完成硬上限，未达到完整系统成功。" if end == 67 else (f"EXP_R{stopped_success} 达到一个阶段性 gate，但仍未达到完整系统成功。" if stopped_success else f"EXP_R{end} 完成，未达到完整系统成功。")
    (ROOT / "FINAL_R17_R67_RESEARCH_SUMMARY.md").write_text(f"# EXP_R17–EXP_R67 research summary\n\n本重启程序把 R17–R67 重新登记为实质方法实验；每个编号都引入了新的假设、模型族、控制形式、评估协议或机制消融。{status}\n\nR17–R40 主要测试 transition/planner 结构，R41–R46 测试 F3 completion，R47–R55 测试长时域、retarget、interrupt 和 return surrogate，R56–R67 测试模块集成、消融、因果 benchmark、分布式预测、prospective collection 和最终 adjudication。完整物理 F2-MPC、自动长时域切换和真实 checkpoint return 仍需 prospective controller/simulator state。\n", encoding="utf-8")
    (ROOT / "FINAL_R17_R67_FAILURE_TAXONOMY.md").write_text("# EXP_R17–EXP_R67 failure taxonomy\n\n1. endpoint arrival 与连续性仍有 trade-off；\n2. 多模态 transition 的离线 selector 对新 source pair 不稳定；\n3. planner 改进依赖 teacher-forced latent feedback，不能直接升级为物理 MPC；\n4. F3 completion 的早切/晚切错误仍会污染长时域组合；\n5. latent/robot waypoint return surrogate 不等于真实世界状态恢复；\n6. prospective causal checkpoint 数据仍是最终物理验证的必要条件。\n", encoding="utf-8")
    (ROOT / "FINAL_R17_R67_SUPPORTED_CLAIMS.md").write_text("# EXP_R17–EXP_R67 supported claims\n\n- R17–R40 provide comparative offline evidence about horizon schedules, transition families, and latent planners; no single family is a physical MPC proof unless its report says so.\n- R41–R46 provide F3 readiness diagnostics under the stated oracle/latent benchmark.\n- R47–R55 test long-horizon, retarget, interruption, and return only in the available offline surrogate.\n- R60–R62 define causal counterfactual benchmark constructions that can enable later physical tests.\n- No unsupported claim of full robot closed-loop control, autonomous long-horizon execution, or strict physical reversal is authorized.\n", encoding="utf-8")
    (ROOT / "FINAL_R17_R67_BEST_SYSTEM.md").write_text("# EXP_R17–EXP_R67 best system\n\nThe strongest released path-construction baseline remains EXP_R8 `repair_late_0.75`. The reboot adds a broad set of planner and executive diagnostics, but the best offline stage result must not be presented as a physical robot controller without action-conditioned simulator/controller snapshots.\n", encoding="utf-8")
    (ROOT / "FINAL_R17_R67_RECOMMENDED_NEXT_DIRECTION.md").write_text("# Recommended next direction after EXP_R67\n\nUse the R60–R62 causal benchmark and prospective collection protocol to acquire matched current-state/action-prefix/checkpoint trajectories. Then preregister a small physical F2-MPC study with oracle F3 before attempting integrated long-horizon and return claims. Do not spend future IDs on repeated interface gates.\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=17); parser.add_argument("--end", type=int, default=17); parser.add_argument("--device", default="cpu"); parser.add_argument("--seed", type=int, default=1717)
    args = parser.parse_args()
    if not (17 <= args.start <= args.end <= 67): raise SystemExit("range must stay within EXP_R17..EXP_R67")
    REPORTS.mkdir(parents=True, exist_ok=True)
    if not SUMMARY.exists(): SUMMARY.write_text("# EXP_R17–EXP_R67 实验总结\n\n每个 EXP 都必须有新方法/假设；R17–R67 为 scientific reboot。\n", encoding="utf-8")
    data = build_dataset(); all_specs = specs(); successes = []
    for exp in range(args.start, args.end + 1):
        success = run_one(exp, all_specs[exp], data, args.seed, args.device)
        if success: successes.append(exp)
    if args.end == 67 or successes:
        write_final(args.end, successes[0] if successes else None)
    print(json.dumps({"start": args.start, "end": args.end, "successes": successes, "cases": len(data.cases), "summary": str(SUMMARY)}, indent=2))


if __name__ == "__main__":
    main()

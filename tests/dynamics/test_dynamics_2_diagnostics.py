"""Frozen DEL adjudication tests for exact residuals and diagnostic solvers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from pglt.dynamics.del_diagnostics import exact_residual, historical_iteration, residual_jacobian, robust_lbfgs
from pglt.dynamics.variational import DELTransition


def frozen_model() -> DELTransition:
    """Create a small finite DEL model with every learned tensor frozen."""

    torch.manual_seed(14)
    model = DELTransition(forced=False, solver_iterations=4, solver_step_size=0.25)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def test_diagnostic_residual_and_iteration_exactly_reproduce_forward() -> None:
    model = frozen_model()
    previous = torch.randn(3, 32) * 0.05
    current = torch.randn(3, 32) * 0.05
    context = torch.randn(3, 16) * 0.05
    prediction, info = model(previous, current, context, 16 / 30)
    diagnostic = historical_iteration(model, previous, current, context, 16 / 30, None, 4)
    reproduced = exact_residual(model, previous, current, prediction, context, 16 / 30, None).norm(dim=-1)
    assert torch.equal(prediction, diagnostic.root)
    assert torch.equal(info.residual_trace, diagnostic.residual_trace)
    assert torch.equal(info.residual_norm, reproduced)


def test_robust_solver_updates_only_q_next_and_remains_finite() -> None:
    model = frozen_model()
    previous = torch.randn(2, 32) * 0.01
    current = torch.randn(2, 32) * 0.01
    context = torch.randn(2, 16) * 0.01
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    result = robust_lbfgs(
        model, previous, current, context, 16 / 30, None, current,
        max_iterations=4, tolerance_gradient=1e-7, tolerance_change=1e-9,
        history_size=4, line_search="strong_wolfe", convergence_tolerance=1e-3,
    )
    assert result.finite.all()
    assert all(torch.equal(before[name], tensor) for name, tensor in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_residual_jacobian_is_finite_32_by_32() -> None:
    model = frozen_model()
    previous = torch.randn(1, 32) * 0.01
    current = torch.randn(1, 32) * 0.01
    following = torch.randn(1, 32) * 0.01
    context = torch.randn(1, 16) * 0.01
    jacobian = residual_jacobian(model, previous, current, following, context, 16 / 30, lambda _value: None)
    assert jacobian.shape == (32, 32)
    assert torch.isfinite(jacobian).all()


def test_wave13_frozen_checkpoint_hashes_still_match() -> None:
    root = Path("results/dynamics/thirteenth_wave/2026-08-12_dynamics_1")
    manifest = json.loads((root / "dynamics_confirmation_manifest.json").read_text())
    for name, expected in manifest["checkpoint_sha256"].items():
        assert hashlib.sha256((root / "checkpoints" / name).read_bytes()).hexdigest() == expected


def test_gt_near_and_validation_roles_are_preregistered() -> None:
    config = Path("configs/dynamics_2.yaml").read_text(encoding="utf-8")
    source = Path("scripts/dynamics/run_dynamics_2.py").read_text(encoding="utf-8")
    assert "gt_near_perturbation_scale_training_std" in config
    assert "ground_truth_near_ORACLE_LOCAL_SOLVABILITY_ONLY" in source
    assert "descriptive replication only; not held-out and not used for settings" in source

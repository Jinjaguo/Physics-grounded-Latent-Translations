"""Frozen-model DEL failure-adjudication primitives for dynamics_2.

The functions reproduce the wave-13 residual exactly, run the unchanged
time-scaled iteration at fixed budgets, optimize only q_next with deterministic
LBFGS, and compute residual Jacobians.  No learned parameter is updated and no
future target action enters forced DEL diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from pglt.dynamics.variational import ControlPacket, DELTransition, require_finite


def exact_residual(
    model: DELTransition,
    q_previous: torch.Tensor,
    q_current: torch.Tensor,
    q_next: torch.Tensor,
    context: torch.Tensor,
    step_size: float,
    packet: ControlPacket | None,
    *,
    create_graph: bool = True,
) -> torch.Tensor:
    """Reproduce the trained wave-13 DEL residual and force sign exactly."""

    force = model.generalized_force(q_previous, q_current, context, packet)
    return model.residual(
        q_previous, q_current, q_next, context, step_size, force,
        create_graph=create_graph,
    )


@dataclass
class IterativeDiagnostic:
    """Fixed-budget historical solver output with complete norm traces."""

    root: torch.Tensor
    residual_trace: torch.Tensor
    step_norm_trace: torch.Tensor
    finite: torch.Tensor
    converged: torch.Tensor


def historical_iteration(
    model: DELTransition,
    q_previous: torch.Tensor,
    q_current: torch.Tensor,
    context: torch.Tensor,
    step_size: float,
    packet: ControlPacket | None,
    iterations: int,
) -> IterativeDiagnostic:
    """Run the exact corrected wave-13 update from constant velocity."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    candidate = q_current + (q_current - q_previous)
    residual_trace = []
    step_trace = []
    for iteration in range(iterations):
        residual = exact_residual(
            model, q_previous, q_current, candidate, context, step_size, packet
        )
        residual_trace.append(residual.norm(dim=-1))
        mass = model.lagrangian.mass_diag(q_current, context)
        correction = model.solver_step_size * float(step_size) * residual / mass
        require_finite(f"diagnostic historical correction {iteration}", correction)
        step_trace.append(correction.norm(dim=-1))
        candidate = candidate + correction
        require_finite(f"diagnostic historical coordinate {iteration}", candidate)
    final = exact_residual(
        model, q_previous, q_current, candidate, context, step_size, packet
    ).norm(dim=-1)
    residual_trace.append(final)
    finite = torch.isfinite(candidate).all(dim=-1) & torch.isfinite(final)
    return IterativeDiagnostic(
        candidate,
        torch.stack(residual_trace, dim=-1),
        torch.stack(step_trace, dim=-1),
        finite,
        final <= model.solver_tolerance,
    )


@dataclass
class RobustDiagnostic:
    """LBFGS least-squares root result and closure-level traces."""

    root: torch.Tensor
    residual_norm: torch.Tensor
    converged: torch.Tensor
    finite: torch.Tensor
    residual_trace: list[float]
    step_norm_trace: list[float]
    closure_calls: int


def robust_lbfgs(
    model: DELTransition,
    q_previous: torch.Tensor,
    q_current: torch.Tensor,
    context: torch.Tensor,
    step_size: float,
    packet: ControlPacket | None,
    initial: torch.Tensor,
    *,
    max_iterations: int,
    tolerance_gradient: float,
    tolerance_change: float,
    history_size: int,
    line_search: str,
    convergence_tolerance: float,
) -> RobustDiagnostic:
    """Minimize 0.5||R||^2 while exposing only q_next to the optimizer."""

    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("All learned DEL parameters must be frozen")
    candidate = initial.detach().clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [candidate],
        lr=1.0,
        max_iter=int(max_iterations),
        max_eval=int(max_iterations) * 2,
        tolerance_grad=float(tolerance_gradient),
        tolerance_change=float(tolerance_change),
        history_size=int(history_size),
        line_search_fn=line_search,
    )
    residual_trace: list[float] = []
    step_trace: list[float] = []
    previous_candidate = candidate.detach().clone()

    def closure() -> torch.Tensor:
        nonlocal previous_candidate
        optimizer.zero_grad(set_to_none=True)
        residual = exact_residual(
            model, q_previous, q_current, candidate, context, step_size, packet
        )
        loss = 0.5 * residual.square().sum(dim=-1).mean()
        require_finite("robust root objective", loss)
        loss.backward()
        if candidate.grad is None or not torch.isfinite(candidate.grad).all():
            raise FloatingPointError("Robust root solver produced a non-finite q_next gradient")
        residual_trace.append(float(residual.norm(dim=-1).mean().detach().cpu()))
        step_trace.append(float((candidate.detach() - previous_candidate).norm(dim=-1).mean().cpu()))
        previous_candidate = candidate.detach().clone()
        return loss

    optimizer.step(closure)
    final_residual = exact_residual(
        model, q_previous, q_current, candidate, context, step_size, packet
    ).norm(dim=-1)
    finite = torch.isfinite(candidate).all(dim=-1) & torch.isfinite(final_residual)
    return RobustDiagnostic(
        candidate.detach(),
        final_residual.detach(),
        final_residual.detach() <= float(convergence_tolerance),
        finite.detach(),
        residual_trace,
        step_trace,
        len(residual_trace),
    )


def residual_jacobian(
    model: DELTransition,
    q_previous: torch.Tensor,
    q_current: torch.Tensor,
    q_next: torch.Tensor,
    context: torch.Tensor,
    step_size: float,
    packet_factory: Callable[[torch.Tensor], ControlPacket | None],
) -> torch.Tensor:
    """Compute dR/dq_next for one sample without physical interpretation."""

    if q_next.shape != (1, q_next.shape[-1]):
        raise ValueError("Jacobian diagnostic requires one sample")
    fixed_previous = q_previous.detach()
    fixed_current = q_current.detach()
    fixed_context = context.detach()

    def residual_of_next(value: torch.Tensor) -> torch.Tensor:
        shaped = value.reshape(1, -1)
        packet = packet_factory(shaped)
        return exact_residual(
            model, fixed_previous, fixed_current, shaped, fixed_context,
            step_size, packet, create_graph=True,
        ).reshape(-1)

    jacobian = torch.autograd.functional.jacobian(
        residual_of_next, q_next.detach().reshape(-1), create_graph=False, vectorize=True
    )
    require_finite("DEL residual Jacobian", jacobian)
    return jacobian


def singular_summary(
    jacobian: torch.Tensor,
    *,
    absolute_epsilon: float,
    relative_rank_epsilon: float,
    nearly_singular_threshold: float,
) -> dict[str, Any]:
    """Summarize finite singular values without calling them stiffness."""

    singular = torch.linalg.svdvals(jacobian).detach().cpu().numpy()
    maximum = float(singular.max())
    minimum = float(singular.min())
    condition = maximum / max(minimum, float(absolute_epsilon))
    effective_rank = int(np.sum(singular > maximum * float(relative_rank_epsilon)))
    return {
        "singular_values": singular.tolist(),
        "minimum_singular_value": minimum,
        "maximum_singular_value": maximum,
        "condition_number_epsilon_stabilized": condition,
        "effective_rank": effective_rank,
        "nearly_singular": bool(minimum < absolute_epsilon or condition > nearly_singular_threshold),
        "interpretation": "numerical residual conditioning only; not physical stiffness",
    }


def deterministic_indices(count: int, subset_size: int) -> list[int]:
    """Choose preregistered evenly spaced indices including both endpoints."""

    if count <= 0 or subset_size <= 0:
        raise ValueError("count and subset_size must be positive")
    size = min(count, subset_size)
    return sorted(set(np.linspace(0, count - 1, num=size, dtype=np.int64).tolist()))


def cluster_roots(roots: np.ndarray, threshold: float) -> tuple[list[int], list[np.ndarray]]:
    """Greedily cluster roots in fixed input order using a frozen threshold."""

    assignments: list[int] = []
    centers: list[np.ndarray] = []
    members: list[list[np.ndarray]] = []
    for root in roots:
        distances = [float(np.linalg.norm(root - center)) for center in centers]
        if distances and min(distances) <= threshold:
            cluster = int(np.argmin(distances))
            members[cluster].append(root)
            centers[cluster] = np.mean(np.stack(members[cluster]), axis=0)
        else:
            cluster = len(centers)
            centers.append(root.copy())
            members.append([root])
        assignments.append(cluster)
    return assignments, centers

"""Differentiable equal-information latent transition models for dynamics_1.

The module implements modest MLP baselines, a conservative diagonal-mass
discrete Lagrangian, corrected time-scaled DEL refinement, audited causal
control packets, and a generic matched-refinement control.  It never imports
or updates the frozen representation and never replaces non-finite values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def build_mlp(input_dim: int, hidden_dim: int, output_dim: int, depth: int) -> nn.Sequential:
    """Build a SiLU MLP with an exact number of linear layers."""

    if depth < 1:
        raise ValueError("depth must be positive")
    layers: list[nn.Module] = []
    current = input_dim
    for _ in range(depth - 1):
        layers.extend((nn.Linear(current, hidden_dim), nn.SiLU()))
        current = hidden_dim
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


def require_finite(name: str, value: torch.Tensor) -> None:
    """Reject non-finite solver or model values without masking them."""

    if not torch.isfinite(value).all():
        bad = int((~torch.isfinite(value)).sum().detach().cpu())
        raise FloatingPointError(f"{name} contains {bad}/{value.numel()} non-finite values")


@dataclass(frozen=True)
class ControlPacket:
    """Already executed commands with explicit causal availability metadata."""

    values: torch.Tensor
    command_frame_indices: torch.Tensor
    prediction_issue_frame: torch.Tensor
    availability_source: str
    available_before_prediction: bool

    def validate(self) -> None:
        """Reject target-window or otherwise unavailable logged commands."""

        if not self.available_before_prediction:
            raise ValueError("Control packet was not available before prediction")
        if self.availability_source != "logged_executed_history":
            raise ValueError(f"Unsupported causal availability source: {self.availability_source}")
        if self.values.ndim != 3 or self.values.shape[-1] != 7:
            raise ValueError("Control values must have shape (B,H,7)")
        if self.command_frame_indices.shape != self.values.shape[:2]:
            raise ValueError("Control values and frame indices are not aligned")
        issue = self.prediction_issue_frame.reshape(-1, 1)
        if len(issue) != len(self.values):
            raise ValueError("One prediction issue frame is required per packet")
        if torch.any(self.command_frame_indices >= issue):
            raise ValueError("Future target action rejected by causal mask")

    def flattened(self) -> torch.Tensor:
        """Return values only after the causal mask has passed."""

        self.validate()
        return self.values.flatten(start_dim=1)


class MLPTransition(nn.Module):
    """Residual MLP transition over exactly (q_prev, q_curr, context)."""

    information_fields = ("q_previous", "q_current", "context")

    def __init__(self, q_dim: int = 32, context_dim: int = 16, hidden_dim: int = 64, depth: int = 3) -> None:
        super().__init__()
        self.network = build_mlp(2 * q_dim + context_dim, hidden_dim, q_dim, depth)

    def forward(self, q_previous: torch.Tensor, q_current: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return q_current + self.network(torch.cat((q_previous, q_current, context), dim=-1))


class HistoryMLPTransition(nn.Module):
    """MLP receiving the same audited causal packet as the forced DEL model."""

    information_fields = ("q_previous", "q_current", "context", "causal_history_packet")

    def __init__(self, control_dim: int = 112, q_dim: int = 32, context_dim: int = 16, hidden_dim: int = 64, depth: int = 3) -> None:
        super().__init__()
        self.network = build_mlp(2 * q_dim + context_dim + control_dim, hidden_dim, q_dim, depth)

    def forward(self, q_previous: torch.Tensor, q_current: torch.Tensor, context: torch.Tensor, packet: ControlPacket) -> torch.Tensor:
        features = torch.cat((q_previous, q_current, context, packet.flattened()), dim=-1)
        return q_current + self.network(features)


class OracleFutureMLPTransition(nn.Module):
    """Leakage upper bound that consumes target-window actions by explicit name."""

    information_fields = ("q_previous", "q_current", "context", "future_target_actions_ORACLE")

    def __init__(self, control_dim: int = 112, q_dim: int = 32, context_dim: int = 16, hidden_dim: int = 64, depth: int = 3) -> None:
        super().__init__()
        self.network = build_mlp(2 * q_dim + context_dim + control_dim, hidden_dim, q_dim, depth)

    def forward(self, q_previous: torch.Tensor, q_current: torch.Tensor, context: torch.Tensor, future_actions: torch.Tensor) -> torch.Tensor:
        features = torch.cat((q_previous, q_current, context, future_actions.flatten(start_dim=1)), dim=-1)
        return q_current + self.network(features)


class ConservativeDiscreteLagrangian(nn.Module):
    """Midpoint discrete Lagrangian with a positive diagonal latent metric."""

    def __init__(self, q_dim: int = 32, context_dim: int = 16, mass_hidden_dim: int = 32, potential_hidden_dim: int = 64, depth: int = 3, mass_epsilon: float = 1e-3) -> None:
        super().__init__()
        # First-wave conservative restriction: the positive diagonal metric is
        # context-conditioned but constant with respect to q.  This is a valid
        # special case of M(q_bar,c) and makes the corrected mass preconditioner
        # exact for the kinetic part instead of ignoring dM/dq terms.
        self.mass_network = build_mlp(context_dim, mass_hidden_dim, q_dim, depth)
        self.potential_network = build_mlp(q_dim + context_dim, potential_hidden_dim, 1, depth)
        self.mass_epsilon = float(mass_epsilon)

    def mass_diag(self, coordinate: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if coordinate.shape[:-1] != context.shape[:-1]:
            raise ValueError("Coordinate and context batch shapes differ")
        return F.softplus(self.mass_network(context)) + self.mass_epsilon

    def potential(self, coordinate: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.potential_network(torch.cat((coordinate, context), dim=-1)).squeeze(-1)

    def forward(self, q_left: torch.Tensor, q_right: torch.Tensor, context: torch.Tensor, step_size: float) -> torch.Tensor:
        midpoint = 0.5 * (q_left + q_right)
        velocity = (q_right - q_left) / float(step_size)
        kinetic = 0.5 * (self.mass_diag(midpoint, context) * velocity.square()).sum(dim=-1)
        return float(step_size) * (kinetic - self.potential(midpoint, context))


@dataclass
class DELSolveInfo:
    """Per-sample final residuals and the full fixed-iteration trace."""

    residual_norm: torch.Tensor
    residual_trace: torch.Tensor
    iterations: int
    converged: torch.Tensor
    failed: torch.Tensor


class DELTransition(nn.Module):
    """Corrected differentiable implicit DEL transition, optionally forced."""

    autonomous_information_fields = ("q_previous", "q_current", "context")
    forced_information_fields = ("q_previous", "q_current", "context", "causal_history_packet")

    def __init__(self, *, forced: bool, q_dim: int = 32, context_dim: int = 16, control_dim: int = 112, mass_hidden_dim: int = 32, potential_hidden_dim: int = 64, force_hidden_dim: int = 64, depth: int = 3, solver_iterations: int = 4, solver_step_size: float = 0.25, solver_tolerance: float = 1e-3, mass_epsilon: float = 1e-3) -> None:
        super().__init__()
        self.forced = bool(forced)
        self.lagrangian = ConservativeDiscreteLagrangian(q_dim, context_dim, mass_hidden_dim, potential_hidden_dim, depth, mass_epsilon)
        self.force_network = build_mlp(2 * q_dim + context_dim + control_dim, force_hidden_dim, q_dim, depth) if forced else None
        self.solver_iterations = int(solver_iterations)
        self.solver_step_size = float(solver_step_size)
        self.solver_tolerance = float(solver_tolerance)

    @property
    def information_fields(self) -> tuple[str, ...]:
        return self.forced_information_fields if self.forced else self.autonomous_information_fields

    def generalized_force(self, q_previous: torch.Tensor, q_current: torch.Tensor, context: torch.Tensor, packet: ControlPacket | None) -> torch.Tensor:
        if not self.forced:
            return torch.zeros_like(q_current)
        if packet is None or self.force_network is None:
            raise ValueError("Forced DEL requires an audited causal ControlPacket")
        return self.force_network(torch.cat((q_previous, q_current, context, packet.flattened()), dim=-1))

    def residual(self, q_previous: torch.Tensor, q_current: torch.Tensor, q_next: torch.Tensor, context: torch.Tensor, step_size: float, force: torch.Tensor, *, create_graph: bool = True) -> torch.Tensor:
        q_prev_v = q_previous if q_previous.requires_grad else q_previous.detach().requires_grad_(True)
        q_curr_v = q_current if q_current.requires_grad else q_current.detach().requires_grad_(True)
        q_next_v = q_next if q_next.requires_grad else q_next.detach().requires_grad_(True)
        left = self.lagrangian(q_prev_v, q_curr_v, context, step_size)
        right = self.lagrangian(q_curr_v, q_next_v, context, step_size)
        d2 = torch.autograd.grad(left.sum(), q_curr_v, create_graph=create_graph, retain_graph=True)[0]
        d1 = torch.autograd.grad(right.sum(), q_curr_v, create_graph=create_graph, retain_graph=True)[0]
        result = d2 + d1 + force
        require_finite("DEL residual", result)
        return result

    def forward(self, q_previous: torch.Tensor, q_current: torch.Tensor, context: torch.Tensor, step_size: float, packet: ControlPacket | None = None) -> tuple[torch.Tensor, DELSolveInfo]:
        force = self.generalized_force(q_previous, q_current, context, packet)
        q_next = q_current + (q_current - q_previous)
        trace = []
        for iteration in range(self.solver_iterations):
            residual = self.residual(q_previous, q_current, q_next, context, step_size, force)
            trace.append(residual.norm(dim=-1))
            mass = self.lagrangian.mass_diag(q_current, context)
            require_finite(f"DEL mass iteration {iteration}", mass)
            correction = self.solver_step_size * float(step_size) * residual / mass
            require_finite(f"DEL correction iteration {iteration}", correction)
            q_next = q_next + correction
            require_finite(f"DEL coordinate iteration {iteration}", q_next)
        final_residual = self.residual(q_previous, q_current, q_next, context, step_size, force)
        final_norm = final_residual.norm(dim=-1)
        trace.append(final_norm)
        failed = ~torch.isfinite(final_norm)
        return q_next, DELSolveInfo(final_norm, torch.stack(trace, dim=-1), self.solver_iterations, final_norm <= self.solver_tolerance, failed)


class GenericRefinementTransition(nn.Module):
    """Matched generic energy refinement initialized from a frozen MLP."""

    information_fields = MLPTransition.information_fields

    def __init__(self, initializer: MLPTransition, q_dim: int = 32, context_dim: int = 16, hidden_dim: int = 64, depth: int = 3, iterations: int = 4, step_size: float = 0.01) -> None:
        super().__init__()
        self.initializer = initializer
        for parameter in self.initializer.parameters():
            parameter.requires_grad_(False)
        self.energy_network = build_mlp(3 * q_dim + context_dim, hidden_dim, 1, depth)
        self.iterations = int(iterations)
        self.step_size = float(step_size)

    def forward(self, q_previous: torch.Tensor, q_current: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with torch.no_grad():
            initial = self.initializer(q_previous, q_current, context)
        candidate = initial.detach().requires_grad_(True)
        trace = []
        fixed = torch.cat((q_previous, q_current, context), dim=-1)
        for _ in range(self.iterations):
            energy = self.energy_network(torch.cat((fixed, candidate), dim=-1)).squeeze(-1)
            gradient = torch.autograd.grad(energy.sum(), candidate, create_graph=True)[0]
            require_finite("generic refinement gradient", gradient)
            trace.append(gradient.norm(dim=-1))
            candidate = candidate - self.step_size * gradient
            require_finite("generic refined coordinate", candidate)
        return candidate, {"gradient_trace": torch.stack(trace, dim=-1), "iterations": torch.tensor(self.iterations, device=candidate.device)}


def trainable_parameter_count(model: nn.Module) -> int:
    """Count parameters that are actually optimized for a model."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def model_spec(model: nn.Module) -> dict[str, Any]:
    """Return machine-readable capacity and information-set provenance."""

    return {
        "class": type(model).__name__,
        "trainable_parameters": trainable_parameter_count(model),
        "information_fields": list(getattr(model, "information_fields", ())),
    }

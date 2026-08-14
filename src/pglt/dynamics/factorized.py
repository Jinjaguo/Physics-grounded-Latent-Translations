"""Factorized executable-subspace dynamics for the fifteenth-wave experiment.

The frozen 32-D action latent is split exactly into a 16-D semantic prefix and
a 16-D executable suffix.  This module supplies the shared semantic predictor,
equal-information executable MLP/refinement models, a free execution-only DEL
ablation, and the primary DEL model whose kinetic quadratic form is induced by
JVPs through the frozen action decoder.  Decoder parameters are never updated.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from pglt.dynamics.variational import (
    DELSolveInfo,
    DELTransition,
    GenericRefinementTransition,
    MLPTransition,
    build_mlp,
    require_finite,
    trainable_parameter_count,
)


class SemanticPredictor(MLPTransition):
    """Shared residual predictor over exactly ``(s_previous, s_current, context)``."""

    information_fields = ("s_previous", "s_current", "context")

    def __init__(self, context_dim: int = 16, hidden_dim: int = 64, depth: int = 3) -> None:
        super().__init__(q_dim=16, context_dim=context_dim, hidden_dim=hidden_dim, depth=depth)


class ExecutionMLP(MLPTransition):
    """Residual execution predictor with the registered F-block information set."""

    information_fields = ("e_previous", "e_current", "s_current", "context")

    def __init__(self, context_dim: int = 32, hidden_dim: int = 64, depth: int = 3) -> None:
        super().__init__(q_dim=16, context_dim=context_dim, hidden_dim=hidden_dim, depth=depth)


class ExecutionMatchedRefinement(GenericRefinementTransition):
    """Generic refinement initialized by the frozen selected execution MLP."""

    information_fields = ExecutionMLP.information_fields

    def __init__(
        self,
        initializer: ExecutionMLP,
        *,
        context_dim: int = 32,
        hidden_dim: int = 64,
        depth: int = 3,
        iterations: int = 4,
        step_size: float = 0.01,
    ) -> None:
        super().__init__(
            deepcopy(initializer), q_dim=16, context_dim=context_dim,
            hidden_dim=hidden_dim, depth=depth, iterations=iterations,
            step_size=step_size,
        )


class FreeExecutionDEL(DELTransition):
    """Execution-only version of the historical free learned-metric DEL family."""

    autonomous_information_fields = ExecutionMLP.information_fields

    def __init__(
        self,
        *,
        context_dim: int = 32,
        mass_hidden_dim: int = 32,
        potential_hidden_dim: int = 64,
        depth: int = 3,
        solver_iterations: int = 4,
        solver_step_size: float = 0.25,
        solver_tolerance: float = 1e-3,
        mass_epsilon: float = 1e-3,
    ) -> None:
        super().__init__(
            forced=False, q_dim=16, context_dim=context_dim,
            mass_hidden_dim=mass_hidden_dim,
            potential_hidden_dim=potential_hidden_dim, depth=depth,
            solver_iterations=solver_iterations,
            solver_step_size=solver_step_size,
            solver_tolerance=solver_tolerance, mass_epsilon=mass_epsilon,
        )


class DecoderGeometryLagrangian(nn.Module):
    """Discrete Lagrangian with decoder-Jacobian pullback kinetic geometry."""

    def __init__(
        self,
        decoder: nn.Module,
        *,
        context_dim: int = 16,
        hidden_dim: int = 64,
        depth: int = 3,
        metric_epsilon: float = 1e-3,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(False)
        self.potential_network = build_mlp(16 + 16 + context_dim, hidden_dim, 1, depth)
        self.metric_epsilon = float(metric_epsilon)

    def potential(self, execution: torch.Tensor, semantic: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.potential_network(torch.cat((execution, semantic, context), dim=-1)).squeeze(-1)

    def decoder_jvp(
        self, semantic: torch.Tensor, execution: torch.Tensor, tangent: torch.Tensor
    ) -> torch.Tensor:
        """Return ``J_e(s,e) tangent`` for continuous decoder outputs/logits."""

        def decode_exec(value: torch.Tensor) -> torch.Tensor:
            return self.decoder(torch.cat((semantic, value), dim=-1))

        _, product = torch.autograd.functional.jvp(
            decode_exec, execution, tangent, create_graph=True, strict=False
        )
        return product.flatten(start_dim=1)

    def metric_quadratic(
        self, semantic: torch.Tensor, execution: torch.Tensor, tangent: torch.Tensor
    ) -> torch.Tensor:
        product = self.decoder_jvp(semantic, execution, tangent)
        return product.square().sum(dim=-1) + self.metric_epsilon * tangent.square().sum(dim=-1)

    def forward(
        self,
        e_left: torch.Tensor,
        e_right: torch.Tensor,
        semantic: torch.Tensor,
        context: torch.Tensor,
        step_size: float,
    ) -> torch.Tensor:
        midpoint = 0.5 * (e_left + e_right)
        velocity = (e_right - e_left) / float(step_size)
        kinetic = 0.5 * self.metric_quadratic(semantic, midpoint, velocity)
        return float(step_size) * (kinetic - self.potential(midpoint, semantic, context))


class DecoderGeometryDEL(nn.Module):
    """F4 decoder-geometry DEL refinement from the exact frozen F1 prediction."""

    information_fields = ExecutionMLP.information_fields

    def __init__(
        self,
        initializer: ExecutionMLP,
        decoder: nn.Module,
        *,
        context_dim: int = 16,
        potential_hidden_dim: int = 64,
        depth: int = 3,
        iterations: int = 4,
        step_size: float = 0.01,
        tolerance: float = 1e-3,
        metric_epsilon: float = 1e-3,
    ) -> None:
        super().__init__()
        self.initializer = deepcopy(initializer)
        for parameter in self.initializer.parameters():
            parameter.requires_grad_(False)
        self.lagrangian = DecoderGeometryLagrangian(
            decoder, context_dim=context_dim, hidden_dim=potential_hidden_dim,
            depth=depth, metric_epsilon=metric_epsilon,
        )
        self.iterations = int(iterations)
        self.step_size = float(step_size)
        self.tolerance = float(tolerance)

    def residual(
        self,
        e_previous: torch.Tensor,
        e_current: torch.Tensor,
        e_next: torch.Tensor,
        semantic: torch.Tensor,
        context: torch.Tensor,
        step_size: float,
        *,
        create_graph: bool = True,
    ) -> torch.Tensor:
        previous = e_previous if e_previous.requires_grad else e_previous.detach().requires_grad_(True)
        current = e_current if e_current.requires_grad else e_current.detach().requires_grad_(True)
        following = e_next if e_next.requires_grad else e_next.detach().requires_grad_(True)
        left = self.lagrangian(previous, current, semantic, context, step_size)
        right = self.lagrangian(current, following, semantic, context, step_size)
        d2 = torch.autograd.grad(left.sum(), current, create_graph=create_graph, retain_graph=True)[0]
        d1 = torch.autograd.grad(right.sum(), current, create_graph=create_graph, retain_graph=True)[0]
        result = d2 + d1
        require_finite("decoder-geometry DEL residual", result)
        return result

    def forward(
        self,
        e_previous: torch.Tensor,
        e_current: torch.Tensor,
        semantic: torch.Tensor,
        context: torch.Tensor,
        physical_step_size: float,
    ) -> tuple[torch.Tensor, DELSolveInfo]:
        combined_context = torch.cat((semantic, context), dim=-1)
        with torch.no_grad():
            candidate = self.initializer(e_previous, e_current, combined_context)
        candidate = candidate.detach().requires_grad_(True)
        trace = []
        for iteration in range(self.iterations):
            residual = self.residual(
                e_previous, e_current, candidate, semantic, context,
                physical_step_size,
            )
            trace.append(residual.norm(dim=-1))
            # The decoder pullback is much larger than the unit latent metric.
            # A per-sample Rayleigh quotient supplies the matching scalar metric
            # preconditioner without materializing J^T J.  It is frozen for the
            # update, while gradients still flow through the DEL residual.
            metric_scale = self.lagrangian.metric_quadratic(
                semantic, e_current, residual
            ) / residual.square().sum(dim=-1).clamp_min(1e-12)
            metric_scale = metric_scale.detach().clamp_min(self.lagrangian.metric_epsilon)
            candidate = candidate + self.step_size * float(physical_step_size) * residual / metric_scale.unsqueeze(-1)
            require_finite(f"decoder-geometry coordinate iteration {iteration}", candidate)
        final = self.residual(
            e_previous, e_current, candidate, semantic, context,
            physical_step_size,
        )
        norm = final.norm(dim=-1)
        trace.append(norm)
        failed = ~torch.isfinite(norm)
        return candidate, DELSolveInfo(
            residual_norm=norm,
            residual_trace=torch.stack(trace, dim=-1),
            iterations=self.iterations,
            converged=norm <= self.tolerance,
            failed=failed,
        )


def factorized_model_spec(model: nn.Module) -> dict[str, Any]:
    """Return exact capacity and registered causal information fields."""

    return {
        "class": type(model).__name__,
        "trainable_parameters": trainable_parameter_count(model),
        "information_fields": list(getattr(model, "information_fields", ())),
    }

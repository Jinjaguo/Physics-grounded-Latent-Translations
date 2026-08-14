"""Numerically diagnosed LaWM adapters kept outside the upstream repository.

This module intentionally does not make a physical claim about PGLT latents.
It exists only to test the official LaWM parabolic toy problem after the
upstream fixed-point DEL solver exhibited nonfinite optimizer gradients.  The
adapter preserves the upstream Lagrangian, context network, DEL residual, and
training objective while replacing only the residual iteration used to solve
for the next coordinate.
"""

from __future__ import annotations

from typing import Optional

import torch

from lawm.dynamics import DELSolveInfo, LatentVariationalDynamics
from lawm.model import LeastActionWorldModel


def require_finite(name: str, tensor: torch.Tensor) -> None:
    """Raise at the first nonfinite solver quantity instead of masking it.

    The upstream solver applies ``nan_to_num`` and clamps residuals and states.
    Those operations keep the forward tensor printable but can hide an
    unstable differentiated iteration.  A numerical regression harness needs
    an explicit failure at the originating quantity, so the local adapter
    checks finiteness without replacing values.
    """

    if not torch.isfinite(tensor).all():
        bad = int((~torch.isfinite(tensor)).sum().detach().cpu())
        raise FloatingPointError(f"{name} contains {bad}/{tensor.numel()} nonfinite values")


class TimeScaledStableVariationalDynamics(LatentVariationalDynamics):
    """Use a dimensionless damped DEL correction with the stable sign.

    For a constant diagonal mass and uniform step ``h``, the upstream residual
    is ``R = m/h * (2*q_k - q_{k-1} - q_{k+1})``.  Its upstream update
    ``q_next <- q_next - alpha * R/m`` has local Jacobian
    ``1 + alpha/h`` and is expansive.  With the README values this is 13.5 per
    solver iteration before composing eight iterations and 62 rollout steps.

    Solving ``R=0`` with the diagonal approximation to
    ``dR/dq_next = -m/h`` instead gives
    ``q_next <- q_next + alpha*h*R/m``.  Its free-particle Jacobian is
    ``1-alpha`` for ``0 < alpha <= 1``.  This implementation makes that single
    mathematically motivated sign/time-scale correction.  It does not clamp,
    normalize, discard losses, or replace nonfinite values.
    """

    def step(
        self,
        q_prev: torch.Tensor,
        q_curr: torch.Tensor,
        h_prev: torch.Tensor | float,
        h_next: Optional[torch.Tensor | float],
        eta: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, DELSolveInfo]:
        """Solve one DEL transition with a finite, time-scaled correction."""

        if h_next is None:
            h_next = h_prev
        if not torch.is_tensor(h_next):
            h_scale = q_curr.new_tensor(float(h_next))
        else:
            h_scale = h_next.to(device=q_curr.device, dtype=q_curr.dtype)
        h_scale = h_scale.abs().clamp_min(1e-8)
        q_next = q_curr + (q_curr - q_prev)
        residual = torch.zeros_like(q_next)
        for iteration in range(max(self.solver_iters, 1)):
            residual = self.del_residual(
                q_prev, q_curr, q_next, h_prev, h_next, eta, create_graph=True
            )
            require_finite(f"DEL residual at solver iteration {iteration}", residual)
            mass = self.lagrangian.mass_diag(q_curr, eta).detach()
            require_finite(f"mass at solver iteration {iteration}", mass)
            if torch.any(mass <= 0):
                raise FloatingPointError("Positive diagonal mass invariant was violated")
            correction = self.solver_step_size * h_scale * residual / mass
            require_finite(f"DEL correction at solver iteration {iteration}", correction)
            q_next = q_next + correction
            require_finite(f"next coordinate at solver iteration {iteration}", q_next)
        info = DELSolveInfo(
            residual_norm=residual.norm(dim=-1).mean(),
            iterations=max(self.solver_iters, 1),
        )
        return q_next, info


class StableToyLeastActionWorldModel(LeastActionWorldModel):
    """Upstream state-space model with only the diagnosed solver replacement."""

    def __init__(
        self,
        state_dim: int = 9,
        latent_dim: int | None = None,
        context_dim: int = 16,
        hidden_dim: int = 128,
        depth: int = 3,
        solver_iters: int = 8,
        solver_step_size: float = 0.25,
    ) -> None:
        super().__init__(
            state_dim=state_dim,
            latent_dim=latent_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            solver_iters=solver_iters,
            solver_step_size=solver_step_size,
        )
        self.dynamics = TimeScaledStableVariationalDynamics(
            q_dim=self.latent_dim,
            context_dim=self.context_dim,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            solver_iters=self.solver_iters,
            solver_step_size=self.solver_step_size,
        )

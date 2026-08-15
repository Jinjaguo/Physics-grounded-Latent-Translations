"""Compact trajectory models used by the prospective Wave 27 study.

The module keeps all predictors on the same three-horizon, 32-dimensional
latent trajectory interface.  It intentionally contains no data loading or
experiment policy; those live in ``scripts/dynamics/run_dynamics_15.py``.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TrajectoryMLP(nn.Module):
    """Predict a joint H1/H2/H4 latent displacement trajectory."""

    def __init__(self, input_dim: int, hidden: int, residual_anchor: bool = False):
        super().__init__()
        self.residual_anchor = residual_anchor
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 96),
        )

    def forward(self, features: torch.Tensor, anchor: torch.Tensor | None = None) -> torch.Tensor:
        value = self.net(features).view(-1, 3, 32)
        if self.residual_anchor:
            if anchor is None:
                raise ValueError("residual-anchor model requires an anchor")
            value = value + anchor
        return value


class TrajectoryMoE(nn.Module):
    """Small hard/soft mixture of trajectory regressors."""

    def __init__(self, input_dim: int, hidden: int, experts: int = 4):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU(), nn.Linear(hidden, experts))
        self.experts = nn.ModuleList([TrajectoryMLP(input_dim, hidden) for _ in range(experts)])

    def forward(self, features: torch.Tensor, hard: bool = False) -> torch.Tensor:
        weights = self.gate(features).softmax(-1)
        values = torch.stack([expert(features) for expert in self.experts], dim=1)
        if hard:
            selected = weights.argmax(-1)
            return values[torch.arange(len(values), device=values.device), selected]
        return (values * weights[:, :, None, None]).sum(1)

    def loss(self, features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = self(features)
        balance = self.gate(features).softmax(-1).mean(0)
        return F.mse_loss(prediction, target) + 0.01 * (balance - 1 / len(balance)).square().mean()


class HeteroscedasticTrajectory(nn.Module):
    """Trajectory mean and diagonal aleatoric scale."""

    def __init__(self, input_dim: int, hidden: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.mean = nn.Linear(hidden, 96)
        self.log_scale = nn.Linear(hidden, 96)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(features)
        return self.mean(hidden).view(-1, 3, 32), self.log_scale(hidden).clamp(-5, 3).view(-1, 3, 32)

    def loss(self, features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mean, log_scale = self(features)
        return (0.5 * (target - mean).square() * torch.exp(-2 * log_scale) + log_scale).mean()


class ConditionalTrajectoryFlow(nn.Module):
    """Conditional flow-matching field over a joint three-horizon trajectory."""

    def __init__(self, input_dim: int, hidden: int, anchor_dim: int = 0):
        super().__init__()
        self.anchor_dim = anchor_dim
        self.field = nn.Sequential(
            nn.Linear(input_dim + anchor_dim + 96 + 1, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 96),
        )

    def velocity(self, features: torch.Tensor, state: torch.Tensor, time: torch.Tensor, anchor: torch.Tensor | None) -> torch.Tensor:
        parts = [features, state.reshape(len(state), 96), time]
        if self.anchor_dim:
            if anchor is None:
                raise ValueError("anchored flow requires an anchor")
            parts.insert(1, anchor.reshape(len(anchor), 96))
        return self.field(torch.cat(parts, dim=-1)).view(-1, 3, 32)

    def loss(self, features: torch.Tensor, target: torch.Tensor, generator: torch.Generator, anchor: torch.Tensor | None = None) -> torch.Tensor:
        noise = torch.randn(target.shape, device=target.device, generator=generator)
        time = torch.rand((len(target), 1), device=target.device, generator=generator)
        state = (1 - time[:, :, None]) * noise + time[:, :, None] * target
        velocity = self.velocity(features, state, time, anchor)
        return F.mse_loss(velocity, target - noise)

    def sample(
        self, features: torch.Tensor, samples: int, steps: int,
        generator: torch.Generator, anchor: torch.Tensor | None = None,
        initialize_from_anchor: bool = False,
    ) -> torch.Tensor:
        repeated = features[:, None].expand(-1, samples, -1).reshape(-1, features.shape[-1])
        repeated_anchor = None if anchor is None else anchor[:, None].expand(-1, samples, -1, -1).reshape(-1, 3, 32)
        noise = torch.randn((len(repeated), 3, 32), device=features.device, generator=generator)
        state = noise if not initialize_from_anchor else repeated_anchor + 0.25 * noise
        dt = 1.0 / steps
        for index in range(steps):
            time = torch.full((len(repeated), 1), (index + 0.5) * dt, device=features.device)
            state = state + dt * self.velocity(repeated, state, time, repeated_anchor)
        return state.view(len(features), samples, 3, 32)

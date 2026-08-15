"""Phase-aware continuous-transition models used by the Wave 26 study.

Purpose
-------
Provide compact state-selected-prior, retrieval/warm-start, heteroscedastic,
multi-path flow, temporal flow, and learned VQ transition heads.  These heads
operate only on frozen causal action-latent/history features and never update
the representation encoder, decoder, or text projection.

Parameters
----------
Constructors take the causal feature dimension, hidden width/depth, and where
needed the number of source branches or VQ codes.  ``loss`` consumes causal
features and TRAIN displacement targets; ``sample`` additionally receives a
fixed torch generator and integration-step count.

Usage
-----
Imported by ``scripts/dynamics/run_dynamics_14.py``.  This module has no
standalone command.

Outputs
-------
Returns transition tensors in memory.  The runner saves local ``.pt``
checkpoints and tracked reports under
``results/dynamics/twenty_sixth_wave/2026-08-14_dynamics_14``.
"""
from __future__ import annotations

import torch
from torch import nn

from pglt.dynamics.wave25_models import Backbone


class PriorFlow(nn.Module):
    """Conditional flow with learned mean and optionally learned source scale."""

    def __init__(self, input_dim: int, hidden: int, depth: int = 2, heteroscedastic: bool = False):
        super().__init__()
        self.heteroscedastic = heteroscedastic
        self.prior = Backbone(input_dim, hidden, depth)
        self.prior_head = nn.Linear(hidden, 64 if heteroscedastic else 32)
        self.field = Backbone(input_dim + 33, hidden, depth)
        self.velocity_head = nn.Linear(hidden, 32)

    def prior_parameters(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.prior_head(self.prior(features))
        mean = value[:, :32]
        log_scale = value[:, 32:].clamp(-3.0, 1.0) if self.heteroscedastic else torch.zeros_like(mean)
        return mean, log_scale

    def velocity(self, value: torch.Tensor, time: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return self.velocity_head(self.field(torch.cat((value, time, features), dim=-1)))

    def loss(self, features: torch.Tensor, target: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        mean, log_scale = self.prior_parameters(features)
        source = mean + torch.randn(target.shape, device=target.device, generator=generator) * log_scale.exp()
        time = torch.rand((len(target), 1), device=target.device, generator=generator)
        point = (1 - time) * source + time * target
        flow = (self.velocity(point, time, features) - (target - source)).square().mean()
        prior_fit = (mean - target).square().mean()
        if self.heteroscedastic:
            calibrated = (((target - mean) / log_scale.exp()).square() + 2 * log_scale).mean()
            return flow + 0.05 * prior_fit + 0.01 * calibrated
        return flow + 0.05 * prior_fit

    def sample(self, features: torch.Tensor, samples: int, steps: int, generator: torch.Generator) -> torch.Tensor:
        repeated = features[:, None].expand(-1, samples, -1).reshape(-1, features.shape[-1])
        mean, log_scale = self.prior_parameters(repeated)
        value = mean + torch.randn(mean.shape, device=mean.device, generator=generator) * log_scale.exp()
        dt = 1.0 / steps
        for step in range(steps):
            time = torch.full((len(value), 1), (step + 0.5) / steps, device=value.device)
            value = value + dt * self.velocity(value, time, repeated)
        return value.view(len(features), samples, 32)


class AnchoredFlow(nn.Module):
    """Flow initialized by a causal 32-D anchor appended to the feature vector."""

    def __init__(self, input_dim: int, hidden: int, depth: int = 2, learned_scale: bool = False):
        super().__init__()
        self.learned_scale = learned_scale
        self.scale = nn.Sequential(Backbone(input_dim, hidden, depth), nn.Linear(hidden, 32)) if learned_scale else None
        self.field = Backbone(input_dim + 33, hidden, depth)
        self.head = nn.Linear(hidden, 32)

    def source(self, features: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        anchor = features[:, -32:]
        log_scale = self.scale(features).clamp(-3.0, 0.5) if self.scale is not None else torch.full_like(anchor, -1.5)
        return anchor + torch.randn(anchor.shape, device=anchor.device, generator=generator) * log_scale.exp()

    def velocity(self, value: torch.Tensor, time: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.field(torch.cat((value, time, features), dim=-1)))

    def loss(self, features: torch.Tensor, target: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        source = self.source(features, generator)
        time = torch.rand((len(target), 1), device=target.device, generator=generator)
        point = (1 - time) * source + time * target
        return (self.velocity(point, time, features) - (target - source)).square().mean()

    def sample(self, features: torch.Tensor, samples: int, steps: int, generator: torch.Generator) -> torch.Tensor:
        repeated = features[:, None].expand(-1, samples, -1).reshape(-1, features.shape[-1])
        value = self.source(repeated, generator)
        dt = 1.0 / steps
        for step in range(steps):
            time = torch.full((len(value), 1), (step + 0.5) / steps, device=value.device)
            value = value + dt * self.velocity(value, time, repeated)
        return value.view(len(features), samples, 32)


class MultiPathFlow(nn.Module):
    """Small learned continuous prior family with a causal soft branch gate."""

    def __init__(self, input_dim: int, hidden: int, branches: int = 3, depth: int = 2):
        super().__init__()
        self.branches = branches
        self.prior = Backbone(input_dim, hidden, depth)
        self.gate = nn.Linear(hidden, branches)
        self.means = nn.Linear(hidden, branches * 32)
        self.log_scales = nn.Linear(hidden, branches * 32)
        self.field = Backbone(input_dim + 33, hidden, depth)
        self.head = nn.Linear(hidden, 32)

    def parameters_out(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.prior(features)
        return (self.gate(hidden), self.means(hidden).view(-1, self.branches, 32),
                self.log_scales(hidden).view(-1, self.branches, 32).clamp(-3.0, 0.5))

    def source(self, features: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        logits, means, scales = self.parameters_out(features)
        index = logits.argmax(-1)
        row = torch.arange(len(features), device=features.device)
        return means[row, index] + torch.randn((len(features), 32), device=features.device, generator=generator) * scales[row, index].exp()

    def velocity(self, value: torch.Tensor, time: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.field(torch.cat((value, time, features), dim=-1)))

    def loss(self, features: torch.Tensor, target: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        logits, means, _ = self.parameters_out(features)
        branch_error = (means - target[:, None]).square().mean(-1)
        soft = (logits.softmax(-1) * branch_error).sum(-1).mean()
        source = self.source(features, generator)
        time = torch.rand((len(target), 1), device=target.device, generator=generator)
        point = (1 - time) * source + time * target
        flow = (self.velocity(point, time, features) - (target - source)).square().mean()
        balance = (logits.softmax(-1).mean(0) - 1 / self.branches).square().mean()
        return flow + 0.05 * soft + 0.01 * balance

    def sample(self, features: torch.Tensor, samples: int, steps: int, generator: torch.Generator) -> torch.Tensor:
        repeated = features[:, None].expand(-1, samples, -1).reshape(-1, features.shape[-1])
        value = self.source(repeated, generator)
        dt = 1.0 / steps
        for step in range(steps):
            time = torch.full((len(value), 1), (step + 0.5) / steps, device=value.device)
            value = value + dt * self.velocity(value, time, repeated)
        return value.view(len(features), samples, 32)


class TemporalFlow(nn.Module):
    """Joint H1/H2/H4 flow with data-relative velocity/acceleration losses."""

    def __init__(self, input_dim: int, hidden: int, depth: int = 2, auxiliary: str = "base"):
        super().__init__()
        self.auxiliary = auxiliary
        self.field = Backbone(input_dim + 97, hidden, depth)
        self.head = nn.Linear(hidden, 96)

    def velocity(self, value: torch.Tensor, time: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.field(torch.cat((value, time, features), dim=-1)))

    def loss(self, features: torch.Tensor, target: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        source = torch.randn(target.shape, device=target.device, generator=generator)
        time = torch.rand((len(target), 1), device=target.device, generator=generator)
        point = (1 - time) * source + time * target
        prediction = source + self.velocity(point, time, features)
        loss = (self.velocity(point, time, features) - (target - source)).square().mean()
        path = prediction.view(-1, 3, 32)
        truth = target.view(-1, 3, 32)
        if self.auxiliary in {"multi_horizon", "decoded", "combined"}:
            loss = loss + 0.15 * (path - truth).square().mean()
        if self.auxiliary in {"contrastive", "combined"}:
            correct = torch.cosine_similarity(path[:, -1], truth[:, -1], dim=-1)
            wrong = torch.cosine_similarity(path[:, -1], truth.roll(1, 0)[:, -1], dim=-1)
            loss = loss + 0.05 * torch.relu(0.1 - correct + wrong).mean()
        if self.auxiliary in {"adaptive_continuity", "combined"}:
            pred_velocity = path[:, 1:] - path[:, :-1]
            true_velocity = truth[:, 1:] - truth[:, :-1]
            threshold = true_velocity.norm(dim=-1).detach().quantile(0.90)
            excess = torch.relu(pred_velocity.norm(dim=-1) - threshold)
            loss = loss + 0.02 * excess.square().mean()
        return loss

    def sample(self, features: torch.Tensor, samples: int, steps: int, generator: torch.Generator) -> torch.Tensor:
        repeated = features[:, None].expand(-1, samples, -1).reshape(-1, features.shape[-1])
        value = torch.randn((len(repeated), 96), device=features.device, generator=generator)
        dt = 1.0 / steps
        for step in range(steps):
            time = torch.full((len(value), 1), (step + 0.5) / steps, device=value.device)
            value = value + dt * self.velocity(value, time, repeated)
        return value.view(len(features), samples, 3, 32)


class VQTransition(nn.Module):
    """Learned code prediction plus a causal residual displacement head."""

    def __init__(self, input_dim: int, hidden: int, codebook: torch.Tensor, depth: int = 2):
        super().__init__()
        self.register_buffer("codebook", codebook)
        self.backbone = Backbone(input_dim, hidden, depth)
        self.logits = nn.Linear(hidden, len(codebook))
        self.residual = nn.Linear(hidden, 32)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(features)
        index = self.logits(hidden).argmax(-1)
        return self.codebook[index] + self.residual(hidden)

    def loss(self, features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(features)
        distances = (target[:, None] - self.codebook[None]).square().mean(-1)
        label = distances.argmin(-1)
        prediction = self.codebook[label] + self.residual(hidden)
        return nn.functional.cross_entropy(self.logits(hidden), label) + (prediction - target).square().mean()

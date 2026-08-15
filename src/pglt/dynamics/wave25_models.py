"""Compact conditional transition models used by the Wave 25 sweep.

Purpose
-------
Provide small direction/magnitude, mode-selection, mixture-density,
mixture-of-experts, cVAE, flow-matching, diffusion, and retrieval-scoring
modules for 32-D latent displacement prediction.

Parameters
----------
All constructors receive causal feature dimension, hidden width, and the
family-specific component/latent count. Forward methods consume causal
features and, only during training, the observed displacement target.

Usage
-----
Imported by ``scripts/dynamics/run_dynamics_13.py``; it is not a standalone
command-line entry point.

Outputs
-------
Returns tensors or distribution parameters in memory. Checkpoints and metrics
are saved by the Wave 25 runner under
``results/dynamics/twenty_fifth_wave/2026-08-14_dynamics_13``.
"""
from __future__ import annotations

import math
import random

import numpy as np
import torch
from torch import nn


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unit(value: torch.Tensor) -> torch.Tensor:
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def join_direction_magnitude(direction: torch.Tensor, log_magnitude: torch.Tensor) -> torch.Tensor:
    return unit(direction) * log_magnitude.clamp(-6.0, 4.0).exp()


class Backbone(nn.Module):
    def __init__(self, input_dim: int, hidden: int, depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        width = input_dim
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.GELU(), nn.LayerNorm(hidden)]
            width = hidden
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class FactoredRegressor(nn.Module):
    """Joint or block-separated direction/log-magnitude regressor."""

    def __init__(self, input_dim: int, hidden: int, depth: int = 2, separate_blocks: bool = False):
        super().__init__()
        self.separate_blocks = separate_blocks
        self.backbone = Backbone(input_dim, hidden, depth)
        if separate_blocks:
            self.semantic = nn.Linear(hidden, 17)
            self.execution = nn.Linear(hidden, 17)
        else:
            self.head = nn.Linear(hidden, 33)

    def parameters_out(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(features)
        if not self.separate_blocks:
            value = self.head(hidden)
            return value[:, :32], value[:, 32:33]
        semantic, execution = self.semantic(hidden), self.execution(hidden)
        direction = torch.cat((unit(semantic[:, :16]), unit(execution[:, :16])), dim=-1)
        magnitude = torch.log(torch.sqrt(semantic[:, 16:17].exp().square() + execution[:, 16:17].exp().square()).clamp_min(1e-8))
        return direction, magnitude

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return join_direction_magnitude(*self.parameters_out(features))


class ModeSelector(nn.Module):
    def __init__(self, input_dim: int, hidden: int, modes: int, depth: int = 2):
        super().__init__()
        self.backbone = Backbone(input_dim, hidden, depth)
        self.logits = nn.Linear(hidden, modes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.logits(self.backbone(features))


class ModeResidual(nn.Module):
    def __init__(self, input_dim: int, hidden: int, depth: int = 2):
        super().__init__()
        self.backbone = Backbone(input_dim + 33, hidden, depth)
        self.head = nn.Linear(hidden, 33)

    def forward(self, features: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        base_direction = unit(base)
        base_logmag = base.norm(dim=-1, keepdim=True).clamp_min(1e-8).log()
        residual = self.head(self.backbone(torch.cat((features, base_direction, base_logmag), dim=-1)))
        return join_direction_magnitude(base_direction + residual[:, :32], base_logmag + residual[:, 32:33])


class MDN(nn.Module):
    def __init__(self, input_dim: int, hidden: int, components: int, depth: int = 2):
        super().__init__()
        self.components = components
        self.backbone = Backbone(input_dim, hidden, depth)
        self.head = nn.Linear(hidden, components * 35)

    def parameters_out(self, features: torch.Tensor) -> tuple[torch.Tensor, ...]:
        value = self.head(self.backbone(features)).view(-1, self.components, 35)
        return value[..., 0], unit(value[..., 1:33]), value[..., 33], value[..., 34].clamp(-4.0, 2.0)

    def loss(self, features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits, direction, logmag, logscale = self.parameters_out(features)
        target_direction = unit(target)[:, None]
        target_logmag = target.norm(dim=-1).clamp_min(1e-8).log()[:, None]
        direction_energy = (direction - target_direction).square().sum(-1) / (2 * 0.25**2)
        scale = logscale.exp()
        magnitude_energy = 0.5 * ((target_logmag - logmag) / scale).square() + logscale
        return -torch.logsumexp(logits.log_softmax(-1) - direction_energy - magnitude_energy, dim=-1).mean()

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        logits, direction, logmag, _ = self.parameters_out(features)
        index = logits.argmax(-1)
        row = torch.arange(len(features), device=features.device)
        return direction[row, index] * logmag[row, index, None].exp()

    def sample(self, features: torch.Tensor, samples: int, generator: torch.Generator) -> torch.Tensor:
        logits, direction, logmag, logscale = self.parameters_out(features)
        indices = torch.multinomial(logits.softmax(-1), samples, replacement=True, generator=generator)
        row = torch.arange(len(features), device=features.device)[:, None]
        noise = torch.randn((len(features), samples), device=features.device, generator=generator)
        magnitude = (logmag[row, indices] + noise * logscale[row, indices].exp()).exp()
        return direction[row, indices] * magnitude[..., None]


class MoE(nn.Module):
    def __init__(self, input_dim: int, hidden: int, experts: int, depth: int = 2):
        super().__init__()
        self.experts = experts
        self.backbone = Backbone(input_dim, hidden, depth)
        self.gate = nn.Linear(hidden, experts)
        self.head = nn.Linear(hidden, experts * 33)

    def outputs(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(features)
        logits = self.gate(hidden)
        values = self.head(hidden).view(-1, self.experts, 33)
        delta = unit(values[..., :32]) * values[..., 32:33].clamp(-6, 4).exp()
        return logits, delta

    def predict(self, features: torch.Tensor, hard: bool) -> torch.Tensor:
        logits, delta = self.outputs(features)
        if hard:
            index = logits.argmax(-1)
            return delta[torch.arange(len(features), device=features.device), index]
        return (logits.softmax(-1)[..., None] * delta).sum(1)

    def loss(self, features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits, delta = self.outputs(features)
        error = (delta - target[:, None]).square().mean(-1)
        soft = (logits.softmax(-1) * error).sum(-1).mean()
        best = error.min(-1).values.mean()
        load = logits.softmax(-1).mean(0)
        balance = (load - 1.0 / self.experts).square().mean()
        return soft + 0.25 * best + 0.02 * balance


class ConditionalVAE(nn.Module):
    def __init__(self, input_dim: int, hidden: int, latent_dim: int, depth: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Backbone(input_dim + 33, hidden, depth)
        self.posterior = nn.Linear(hidden, latent_dim * 2)
        self.decoder = Backbone(input_dim + latent_dim, hidden, depth)
        self.head = nn.Linear(hidden, 33)

    def decode(self, features: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        value = self.head(self.decoder(torch.cat((features, latent), dim=-1)))
        return join_direction_magnitude(value[:, :32], value[:, 32:33])

    def loss(self, features: torch.Tensor, target: torch.Tensor, generator: torch.Generator) -> tuple[torch.Tensor, dict[str, float]]:
        encoded_target = torch.cat((unit(target), target.norm(dim=-1, keepdim=True).clamp_min(1e-8).log()), dim=-1)
        posterior = self.posterior(self.encoder(torch.cat((features, encoded_target), dim=-1)))
        mean, logvar = posterior.chunk(2, dim=-1)
        latent = mean + torch.randn(mean.shape, device=mean.device, generator=generator) * (0.5 * logvar).exp()
        prediction = self.decode(features, latent)
        recon = (prediction - target).square().mean()
        kl = -0.5 * (1 + logvar - mean.square() - logvar.exp()).mean()
        return recon + 0.01 * kl, {"reconstruction": float(recon.detach()), "kl": float(kl.detach())}

    def sample(self, features: torch.Tensor, samples: int, generator: torch.Generator) -> torch.Tensor:
        repeated = features[:, None].expand(-1, samples, -1).reshape(-1, features.shape[-1])
        latent = torch.randn((len(repeated), self.latent_dim), device=features.device, generator=generator)
        return self.decode(repeated, latent).view(len(features), samples, 32)


class FlowMatcher(nn.Module):
    def __init__(self, input_dim: int, hidden: int, depth: int = 2):
        super().__init__()
        self.field = Backbone(input_dim + 33, hidden, depth)
        self.head = nn.Linear(hidden, 32)

    def velocity(self, x: torch.Tensor, time: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.field(torch.cat((x, time, features), dim=-1)))

    def loss(self, features: torch.Tensor, target: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        noise = torch.randn(target.shape, device=target.device, generator=generator)
        time = torch.rand((len(target), 1), device=target.device, generator=generator)
        point = (1 - time) * noise + time * target
        return (self.velocity(point, time, features) - (target - noise)).square().mean()

    def sample(self, features: torch.Tensor, samples: int, steps: int, generator: torch.Generator) -> torch.Tensor:
        feat = features[:, None].expand(-1, samples, -1).reshape(-1, features.shape[-1])
        value = torch.randn((len(feat), 32), device=features.device, generator=generator)
        dt = 1.0 / steps
        for step in range(steps):
            time = torch.full((len(value), 1), (step + 0.5) / steps, device=value.device)
            value = value + dt * self.velocity(value, time, feat)
        return value.view(len(features), samples, 32)


class Diffusion(nn.Module):
    def __init__(self, input_dim: int, hidden: int, train_steps: int, depth: int = 2):
        super().__init__()
        self.train_steps = train_steps
        beta = torch.linspace(1e-4, 0.02, train_steps)
        self.register_buffer("alpha_bar", torch.cumprod(1 - beta, dim=0))
        self.denoiser = Backbone(input_dim + 33, hidden, depth)
        self.head = nn.Linear(hidden, 32)

    def noise(self, value: torch.Tensor, time: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.denoiser(torch.cat((value, time, features), dim=-1)))

    def loss(self, features: torch.Tensor, target: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        index = torch.randint(self.train_steps, (len(target),), device=target.device, generator=generator)
        alpha = self.alpha_bar[index, None]
        epsilon = torch.randn(target.shape, device=target.device, generator=generator)
        noisy = alpha.sqrt() * target + (1 - alpha).sqrt() * epsilon
        time = index[:, None].float() / max(self.train_steps - 1, 1)
        return (self.noise(noisy, time, features) - epsilon).square().mean()

    def sample(self, features: torch.Tensor, samples: int, steps: int, generator: torch.Generator) -> torch.Tensor:
        feat = features[:, None].expand(-1, samples, -1).reshape(-1, features.shape[-1])
        value = torch.randn((len(feat), 32), device=features.device, generator=generator)
        indices = torch.linspace(self.train_steps - 1, 0, steps, device=features.device).long()
        for position, index in enumerate(indices):
            alpha = self.alpha_bar[index]
            time = torch.full((len(value), 1), float(index) / max(self.train_steps - 1, 1), device=value.device)
            epsilon = self.noise(value, time, feat)
            clean = (value - (1 - alpha).sqrt() * epsilon) / alpha.sqrt().clamp_min(1e-8)
            if position + 1 < len(indices):
                next_alpha = self.alpha_bar[indices[position + 1]]
                value = next_alpha.sqrt() * clean + (1 - next_alpha).sqrt() * epsilon
            else:
                value = clean
        return value.view(len(features), samples, 32)


class RetrievalScorer(nn.Module):
    def __init__(self, feature_dim: int, hidden: int, depth: int = 2):
        super().__init__()
        self.scorer = Backbone(feature_dim + 64, hidden, depth)
        self.score = nn.Linear(hidden, 1)
        self.residual = Backbone(feature_dim + 32, hidden, depth)
        self.residual_head = nn.Linear(hidden, 33)

    def scores(self, features: torch.Tensor, source_difference: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        repeated = features[:, None].expand(-1, candidates.shape[1], -1)
        value = torch.cat((repeated, source_difference, candidates), dim=-1)
        return self.score(self.scorer(value)).squeeze(-1)

    def predict(self, features: torch.Tensor, source_difference: torch.Tensor, candidates: torch.Tensor, mode: str) -> torch.Tensor:
        scores = self.scores(features, source_difference, candidates)
        if mode == "soft":
            return (scores.softmax(-1)[..., None] * candidates).sum(1)
        selected = candidates[torch.arange(len(features), device=features.device), scores.argmax(-1)]
        if mode == "hard":
            return selected
        correction = self.residual_head(self.residual(torch.cat((features, selected), dim=-1)))
        return join_direction_magnitude(unit(selected) + correction[:, :32], selected.norm(dim=-1, keepdim=True).clamp_min(1e-8).log() + correction[:, 32:33])

    def loss(self, features: torch.Tensor, source_difference: torch.Tensor, candidates: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        soft = self.predict(features, source_difference, candidates, "soft")
        residual = self.predict(features, source_difference, candidates, "residual")
        return (soft - target).square().mean() + (residual - target).square().mean()

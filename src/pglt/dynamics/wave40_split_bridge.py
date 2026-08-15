"""Semantic/execution disentangled force bridge for Wave40.

Purpose
-------
Separate a small semantic instruction force from a state-conditioned execution
force before projecting both into the frozen 32-D action latent.  The original
action-text VAE, decoder, and F1/F2 dynamics remain frozen.

Parameters
----------
``input_dim`` is the causal feature width, ``q_dim`` is each branch width,
``basis`` is the frozen latent projection, and ``family`` chooses independent,
shared-gate, calibrated, or cross-coupled branches.

Usage
-----
The Wave40 driver constructs ``SplitForceBridge`` and calls ``forward`` with
features and a frozen base trajectory.

Outputs
-------
Returns adapted latent trajectories, branch residuals, q trajectories, and
branch scales.  Files are written by the Wave40 driver.
"""
from __future__ import annotations

import torch
from torch import nn


def _mlp(input_dim: int, hidden: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU(), nn.Linear(hidden, output_dim))


class SplitForceBridge(nn.Module):
    """Two-branch semantic/execution force transport."""

    def __init__(self, input_dim: int, q_dim: int, basis: torch.Tensor, family: str, latent_dim: int = 32) -> None:
        super().__init__(); self.q_dim = q_dim; self.family = family; self.register_buffer("basis", basis[:, :q_dim].clone()); self.semantic = _mlp(input_dim, 64, 3 * q_dim); self.execution = _mlp(input_dim, 64, 3 * q_dim); self.scale = nn.Parameter(torch.zeros(2)); self.gate = _mlp(input_dim, 64, 1) if family in ("shared_gate", "cross") else None; self.cross = _mlp(input_dim, 64, 3 * q_dim) if family == "cross" else None

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def forward(self, features: torch.Tensor, base: torch.Tensor) -> dict[str, torch.Tensor]:
        sem = self.semantic(features).view(features.shape[0], 3, self.q_dim); exe = self.execution(features).view(features.shape[0], 3, self.q_dim)
        if self.cross is not None: exe = exe + 0.25 * self.cross(features).view(features.shape[0], 3, self.q_dim)
        gate = torch.sigmoid(self.gate(features)).unsqueeze(1) if self.gate is not None else torch.ones_like(sem[..., :1]); scales = torch.sigmoid(self.scale).view(2, 1, 1) * 0.2 + 0.02; sem = sem * gate * scales[0]; exe = exe * (2.0 - gate) * scales[1]; q = sem + exe; residual = 0.12 * torch.tanh(torch.matmul(q, self.basis.t())); return {"prediction": base + residual, "residual": residual, "q": q, "direction": q[:, -1], "semantic_q": sem, "execution_q": exe, "scales": scales.squeeze(-1).squeeze(-1)}

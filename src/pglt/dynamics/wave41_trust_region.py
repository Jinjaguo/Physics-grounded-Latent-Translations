"""Confidence-calibrated local trust-region force bridge for Wave41.

Purpose
-------
Predict a compact latent force and a causal confidence/radius so that the
frozen behavior is changed only when the event/state input is reliable.

Parameters
----------
``input_dim`` is the causal feature width, ``q_dim`` the force width, ``basis``
the frozen projection, ``family`` the radius rule, and ``radius`` the maximum
residual scale.

Usage
-----
The Wave41 driver calls ``TrustRegionBridge.forward`` with causal features and
the frozen base trajectory.

Outputs
-------
Returns prediction, residual, q, direction, and confidence tensors. Artifacts
are written by the Wave41 driver.
"""
from __future__ import annotations

import torch
from torch import nn


def _mlp(input_dim: int, hidden: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU(), nn.Linear(hidden, output_dim))


class TrustRegionBridge(nn.Module):
    """Causal direction plus learned confidence-scaled local update."""

    def __init__(self, input_dim: int, q_dim: int, basis: torch.Tensor, family: str, radius: float) -> None:
        super().__init__(); self.q_dim = q_dim; self.family = family; self.radius = radius; self.register_buffer("basis", basis[:, :q_dim].clone()); self.direction = _mlp(input_dim, 64, 3 * q_dim); self.uncertainty = _mlp(input_dim, 64, 3 if family != "fixed" else 1); self.second = _mlp(input_dim, 64, 3 * q_dim) if family == "two_head" else None

    def trainable_parameters(self): return (p for p in self.parameters() if p.requires_grad)

    def forward(self, features: torch.Tensor, base: torch.Tensor) -> dict[str, torch.Tensor]:
        q = self.direction(features).view(features.shape[0], 3, self.q_dim)
        if self.second is not None: q = 0.5 * (q + self.second(features).view(features.shape[0], 3, self.q_dim))
        confidence = torch.sigmoid(self.uncertainty(features))
        if self.family == "fixed": confidence = confidence.mean(-1, keepdim=True).expand(-1, 3)
        elif self.family == "adaptive": confidence = confidence
        elif self.family == "two_head": confidence = confidence
        residual = self.radius * confidence.unsqueeze(-1) * torch.tanh(torch.matmul(q, self.basis.t())); return {"prediction": base + residual, "residual": residual, "q": q, "direction": q[:, -1], "confidence": confidence}

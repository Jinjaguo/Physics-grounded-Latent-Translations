"""Causal phase/contact transition-gated force bridge for Wave38.

Purpose
-------
Release a compact low-dimensional latent force only when the current ordered
instruction and past state/history indicate a transition.  The frozen
action-text representation, decoder, and F1/F2 behavior models are unchanged.

Parameters
----------
``input_dim`` is the causal feature width, ``q_dim`` is the force width,
``basis`` is a frozen latent projection, and ``gate_family`` chooses hazard,
contact/history, monotonic, or two-stage gating.

Usage
-----
The Wave38 driver constructs ``TransitionGateBridge`` and calls ``forward``
with causal features and the frozen base trajectory.

Outputs
-------
Returns adapted latent trajectories, residuals, q trajectories, direction, and
gate values.  Files are written by the Wave38 driver.
"""
from __future__ import annotations

import torch
from torch import nn


def _mlp(input_dim: int, hidden: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU(), nn.Linear(hidden, output_dim))


class TransitionGateBridge(nn.Module):
    """Force bridge with a learned causal transition-release gate."""

    def __init__(self, input_dim: int, q_dim: int, basis: torch.Tensor, gate_family: str, latent_dim: int = 32) -> None:
        super().__init__(); self.q_dim = q_dim; self.gate_family = gate_family; self.register_buffer("basis", basis[:, :q_dim].clone()); self.direction = _mlp(input_dim, 64, 3 * q_dim); self.gate = _mlp(input_dim, 64, 3); self.latent_dim = latent_dim

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def forward(self, features: torch.Tensor, base: torch.Tensor) -> dict[str, torch.Tensor]:
        direction = self.direction(features).view(features.shape[0], 3, self.q_dim); logits = self.gate(features).view(features.shape[0], 3)
        gate = torch.sigmoid(logits)
        if self.gate_family == "monotonic": gate = torch.cumprod(gate.clamp_min(0.05), dim=1)
        if self.gate_family == "hazard": gate = gate * torch.linspace(0.35, 1.0, 3, device=features.device).view(1, 3)
        q = direction * gate.unsqueeze(-1); residual = 0.12 * torch.tanh(torch.matmul(q, self.basis.t())); return {"prediction": base + residual, "residual": residual, "q": q, "direction": direction[:, -1], "gate": gate}

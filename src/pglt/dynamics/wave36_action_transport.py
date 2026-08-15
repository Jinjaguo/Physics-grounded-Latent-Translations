"""Frozen-decoder Jacobian action-transport bridge for Wave36.

Purpose
-------
Predict a compact action-space correction from event/state features and map it
through the local Jacobian of the frozen action decoder to a small latent force.
This tests whether the Wave35 failure came from choosing directions directly
in latent coordinates.

Parameters
----------
``input_dim`` is the causal feature width, ``q_dim`` is the compact action
coordinate width, ``action_basis`` is a frozen 6-by-q basis, and ``transport``
selects Jacobian-transpose, damped-pseudoinverse, or execution-only mapping.

Usage
-----
The Wave36 driver supplies features, frozen base trajectories, and precomputed
decoder Jacobians to ``ActionTransportBridge.forward``.

Outputs
-------
Returns latent predictions, latent residuals, q trajectories, and predicted
action corrections.  Experiment artifacts are written by the Wave36 driver.
"""
from __future__ import annotations

import torch
from torch import nn


def _mlp(input_dim: int, hidden: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU(), nn.Linear(hidden, output_dim))


class ActionTransportBridge(nn.Module):
    """Causal action correction transported through a frozen decoder Jacobian."""

    def __init__(self, input_dim: int, q_dim: int, action_basis: torch.Tensor, transport: str, damping: float, phase_gate: bool, latent_dim: int = 32) -> None:
        super().__init__()
        self.q_dim = q_dim
        self.latent_dim = latent_dim
        self.transport = transport
        self.damping = damping
        self.phase_gate = phase_gate
        self.register_buffer("action_basis", action_basis[:, :q_dim].clone())
        self.head = _mlp(input_dim, 64, 3 * q_dim)
        self.gate = _mlp(input_dim, 64, 1) if phase_gate else None

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def _transport(self, jacobian: torch.Tensor, action_delta: torch.Tensor) -> torch.Tensor:
        # jacobian: B,H,6,32; action_delta: B,H,6.
        if self.transport == "transpose":
            return torch.einsum("bhad,bha->bhd", jacobian, action_delta)
        gram = torch.matmul(jacobian, jacobian.transpose(-1, -2))
        eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype).view(1, 1, 6, 6)
        solved = torch.linalg.solve(gram + self.damping * eye, action_delta.unsqueeze(-1)).squeeze(-1)
        return torch.matmul(jacobian.transpose(-1, -2), solved.unsqueeze(-1)).squeeze(-1)

    def forward(self, features: torch.Tensor, base: torch.Tensor, jacobian: torch.Tensor) -> dict[str, torch.Tensor]:
        q = self.head(features).view(features.shape[0], 3, self.q_dim)
        if self.gate is not None:
            q = q * torch.sigmoid(self.gate(features)).unsqueeze(1)
        action_delta = torch.einsum("bhq,aq->bha", q, self.action_basis)
        residual = self._transport(jacobian, action_delta)
        if self.transport == "execution_only":
            residual = torch.cat((torch.zeros_like(residual[..., :16]), residual[..., 16:]), dim=-1)
        residual = 0.12 * torch.tanh(residual)
        return {"prediction": base + residual, "residual": residual, "q": q, "direction": q[:, -1], "action_delta": action_delta}

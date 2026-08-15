"""Temporal and state-action conditioned low-dimensional force bridge for Wave35.

Purpose
-------
Map ordered language events and past robot/latent state into a small continuous
force in a frozen action latent.  The action-text VAE and F1/F2 behavior model
remain frozen; this module only learns the bridge and a low-rank projection.

Parameters
----------
``input_dim`` is the selected event/state feature width, ``q_dim`` is the
low-dimensional force width, ``latent_dim`` is the frozen latent width, and
``family`` selects the registered bridge (delta, state, history, phase-gated,
or integrated).

Usage
-----
The Wave35 driver constructs ``TemporalForceBridge`` and calls ``forward``
with a feature tensor, a frozen base trajectory, and a low-rank basis.

Outputs
-------
The forward pass returns adapted latent trajectories, residuals, q trajectories,
and the learned force direction.  Files are written by
``scripts/dynamics/run_wave35_temporal_bridge.py`` under the Wave35 results
directory.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _mlp(input_dim: int, hidden: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU(), nn.Linear(hidden, output_dim))


class TemporalForceBridge(nn.Module):
    """Causal event/state bridge with a compact q-space force."""

    def __init__(self, input_dim: int, q_dim: int, basis: torch.Tensor, family: str, hidden: int = 64, latent_dim: int = 32) -> None:
        super().__init__()
        self.q_dim = q_dim
        self.latent_dim = latent_dim
        self.family = family
        self.register_buffer("basis", basis[:, :q_dim].clone())
        self.encoder = _mlp(input_dim, hidden, q_dim)
        if family == "phase_gated":
            self.gate = _mlp(input_dim, hidden, 1)
        else:
            self.gate = None
        if family == "integrated":
            self.velocity = _mlp(input_dim + q_dim, hidden, q_dim)
        else:
            self.velocity = None
        if family == "history_contact":
            self.contact = _mlp(input_dim, hidden, q_dim)
        else:
            self.contact = None

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def forward(self, features: torch.Tensor, base: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 2 or base.ndim != 3:
            raise ValueError("features must be (B,F), base must be (B,H,latent)")
        direction = self.encoder(features)
        if self.contact is not None:
            direction = direction + 0.5 * self.contact(features)
        phase = torch.sigmoid(self.gate(features)) if self.gate is not None else torch.ones_like(direction[:, :1])
        q = torch.zeros_like(direction)
        outputs, residuals, qs = [], [], []
        previous = base[:, 0]
        for index in range(base.shape[1]):
            if self.velocity is not None:
                velocity = self.velocity(torch.cat((features, q), dim=-1))
                q = q + 0.35 * (direction - q) + 0.15 * velocity
            else:
                q = q + (0.45 / float(index + 1)) * (direction - q)
            q_step = q * phase * (float(index + 1) / float(base.shape[1]))
            residual = q_step @ self.basis.t()
            # A small continuous force is the intervention; scale is learned
            # through the loss but bounded to prevent replacing the backbone.
            residual = 0.15 * torch.tanh(residual)
            outputs.append(base[:, index] + residual)
            residuals.append(residual)
            qs.append(q_step)
            previous = base[:, index]
        return {
            "prediction": torch.stack(outputs, dim=1),
            "residual": torch.stack(residuals, dim=1),
            "q": torch.stack(qs, dim=1),
            "direction": direction,
        }

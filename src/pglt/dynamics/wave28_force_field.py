"""Low-dimensional latent intent force-field adapters for Wave 28.

Purpose
-------
Provide the small trainable intervention layer used by Wave 28.  The released
action/text representation and its decoder remain frozen; this module learns a
compact intention coordinate, a causal field update, and a low-rank residual
projection into the original 32-D action latent.

Parameters
----------
``latent_dim`` is the frozen action-latent width, ``q_dim`` is the force-field
width, ``language_dim`` is the frozen text-coordinate width, and ``encoding``,
``field`` and ``composition`` select the preregistered ablations.

Usage
-----
The experiment driver constructs ``IntentForceField`` and calls ``forward``
with frozen base trajectories, current/target language embeddings, and the
current latent.  Only parameters returned by ``trainable_parameters`` are
optimized.

Outputs
-------
The forward pass returns adapted latent trajectories, residuals, force-field
states, and language displacement vectors.  No files are written by this
library; Wave 28 artifacts are written by
``scripts/dynamics/run_wave28_force_field.py``.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


def _mlp(input_dim: int, hidden: int, output_dim: int, depth: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for _ in range(max(1, depth - 1)):
        layers.extend((nn.Linear(current, hidden), nn.SiLU()))
        current = hidden
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class IntentForceField(nn.Module):
    """Frozen-latent residual steering through a compact learned q-space."""

    def __init__(
        self,
        latent_dim: int = 32,
        language_dim: int = 16,
        q_dim: int = 2,
        hidden: int = 48,
        encoding: str = "E0_linear",
        field: str = "FF3_attractor",
        composition: str = "COMP0_additive",
        subspace: str = "C2_learned",
        basis: torch.Tensor | None = None,
        semantic_dim: int = 16,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.language_dim = language_dim
        self.q_dim = q_dim
        self.encoding = encoding
        self.field = field
        self.composition = composition
        self.subspace = subspace
        self.semantic_dim = semantic_dim
        if encoding in ("E0_linear", "E1_normalized_linear"):
            self.language_map = nn.Linear(language_dim, q_dim, bias=False)
        elif encoding == "E2_mlp":
            self.language_map = _mlp(language_dim, hidden, q_dim, 2)
        elif encoding in ("E3_pairwise", "E4_antisymmetric"):
            self.language_map = _mlp(language_dim * 3, hidden, q_dim, 2)
        elif encoding == "E6_dictionary":
            self.language_map = nn.Parameter(torch.zeros(6, q_dim))
        else:
            raise KeyError(encoding)
        if encoding == "E6_dictionary":
            nn.init.normal_(self.language_map, std=0.02)
        elif hasattr(self.language_map, "weight"):
            nn.init.normal_(self.language_map.weight, std=0.08)
            if hasattr(self.language_map, "bias") and self.language_map.bias is not None:
                nn.init.zeros_(self.language_map.bias)
        if field in ("FF4_state_conditioned", "FF5_nonlinear", "FF7_gated", "FF8_velocity"):
            extra = latent_dim + (latent_dim if field == "FF8_velocity" else 0)
            self.field_net = _mlp(q_dim * 2 + extra, hidden, q_dim, 2)
        else:
            self.field_net = None
        if field == "FF7_gated":
            self.gate_net = _mlp(latent_dim + q_dim, hidden, 1, 2)
        else:
            self.gate_net = None
        if subspace == "C2_learned" or subspace == "C6_state_dependent":
            self.B = nn.Parameter(torch.empty(latent_dim, q_dim))
            nn.init.orthogonal_(self.B)
        else:
            if basis is None:
                generator = torch.Generator().manual_seed(seed)
                basis = torch.randn(latent_dim, q_dim, generator=generator)
                basis = torch.linalg.qr(basis, mode="reduced").Q
            self.register_buffer("B", basis[:, :q_dim].clone())
        if subspace == "C6_state_dependent":
            self.state_B = _mlp(latent_dim, hidden, q_dim * q_dim, 2)
        else:
            self.state_B = None

    def trainable_parameters(self):
        """Return the intervention parameters, excluding frozen B controls."""

        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def p(self, language: torch.Tensor, ids: torch.Tensor | None = None) -> torch.Tensor:
        """Map frozen language coordinates to q-space without future inputs."""
        if self.encoding == "E6_dictionary":
            if ids is None:
                raise ValueError("dictionary encoding requires language ids")
            return self.language_map[ids]
        if self.encoding in ("E3_pairwise", "E4_antisymmetric"):
            raise RuntimeError("pair encodings define relative_d directly")
        value = self.language_map(language)
        if self.encoding == "E1_normalized_linear":
            value = F.normalize(value, dim=-1) * (self.q_dim ** 0.5)
        return value

    def relative(self, current: torch.Tensor, target: torch.Tensor, ids_current: torch.Tensor | None, ids_target: torch.Tensor | None) -> torch.Tensor:
        if self.encoding in ("E3_pairwise", "E4_antisymmetric"):
            pair = torch.cat((current, target, target - current), dim=-1)
            return self.language_map(pair)
        return self.p(target, ids_target) - self.p(current, ids_current)

    def project(self, q: torch.Tensor, base: torch.Tensor, residual_scale: torch.Tensor | None = None) -> torch.Tensor:
        """Project q into the frozen latent, optionally using a state basis."""
        if self.state_B is not None:
            rotation = self.state_B(base).view(-1, self.q_dim, self.q_dim)
            B = torch.matmul(self.B.unsqueeze(0), torch.matrix_exp(0.05 * rotation))
            value = torch.bmm(B, q.unsqueeze(-1)).squeeze(-1)
        else:
            value = q @ self.B.t()
        if self.subspace == "C3_block_separable":
            mask = torch.zeros_like(value); mask[:, : self.semantic_dim] = value[:, : self.semantic_dim]
            value = mask
        elif self.subspace == "C4_execution_only":
            value = F.pad(value[:, self.semantic_dim :], (self.semantic_dim, 0))
        elif self.subspace == "C5_semantic_only":
            value = F.pad(value[:, : self.semantic_dim], (0, self.latent_dim - self.semantic_dim))
        if residual_scale is not None:
            value = value * residual_scale
        return value

    def _field_step(self, d: torch.Tensor, q: torch.Tensor, base: torch.Tensor, velocity: torch.Tensor | None) -> torch.Tensor:
        if self.field == "FF1_direct":
            return d
        if self.field == "FF2_accumulating":
            return q + d
        if self.field == "FF3_attractor":
            return q + 0.5 * (d - q)
        if self.field in ("FF4_state_conditioned", "FF5_nonlinear", "FF8_velocity"):
            pieces = [q, d, base]
            if self.field == "FF8_velocity":
                pieces.append(torch.zeros_like(base) if velocity is None else velocity)
            return q + 0.35 * self.field_net(torch.cat(pieces, dim=-1))
        if self.field == "FF7_gated":
            gate = torch.sigmoid(self.gate_net(torch.cat((base, d), dim=-1)))
            return q + gate * 0.5 * (d - q)
        if self.field == "FF6_potential":
            return q + 0.5 * (d - q)
        if self.field == "FF9_retrieval":
            return q + 0.65 * (d - q)
        raise KeyError(self.field)

    def forward(
        self,
        base: torch.Tensor,
        current_latent: torch.Tensor,
        current_language: torch.Tensor,
        target_language: torch.Tensor,
        current_ids: torch.Tensor | None = None,
        target_ids: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Return adapted trajectory and causal q-field diagnostics."""
        if base.ndim != 3 or base.shape[-1] != self.latent_dim:
            raise ValueError(f"base must be (B,H,{self.latent_dim})")
        d = self.relative(current_language, target_language, current_ids, target_ids)
        q = torch.zeros_like(d)
        outputs: list[torch.Tensor] = []
        residuals: list[torch.Tensor] = []
        qs: list[torch.Tensor] = []
        previous = current_latent
        for index in range(base.shape[1]):
            velocity = base[:, index] - previous
            q = self._field_step(d, q, base[:, index], velocity)
            if self.field == "FF2_accumulating":
                q = (index + 1) / base.shape[1] * d
            if self.field in ("FF1_direct", "FF3_attractor", "FF6_potential", "FF7_gated", "FF9_retrieval"):
                q = ((index + 1) / base.shape[1]) * q + (1 - (index + 1) / base.shape[1]) * d * 0.05
            residual = self.project(q, base[:, index])
            if self.composition == "COMP1_gated_additive":
                residual = torch.sigmoid((q.square().sum(-1, keepdim=True) - 0.2)) * residual
            elif self.composition == "COMP2_normalized":
                scale = base[:, index].norm(dim=-1, keepdim=True) / residual.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                residual = residual * (0.05 * scale.clamp(max=2.0))
            elif self.composition == "COMP3_film":
                residual = torch.tanh(residual) * (0.1 + 0.05 * base[:, index].abs())
            elif self.composition == "COMP4_rotation":
                residual = torch.tanh(residual) * 0.1
            outputs.append(base[:, index] + residual)
            residuals.append(residual)
            qs.append(q)
            previous = base[:, index]
        return {
            "prediction": torch.stack(outputs, dim=1),
            "residual": torch.stack(residuals, dim=1),
            "q": torch.stack(qs, dim=1),
            "direction": d,
        }

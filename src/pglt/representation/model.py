"""Action-only language-grounded continuous representation model."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


def mlp(input_dim: int, hidden_dim: int, output_dim: int, depth: int) -> nn.Sequential:
    """Build a GELU MLP with exactly ``depth`` linear layers."""

    if depth < 1:
        raise ValueError("MLP depth must be at least one")
    layers: list[nn.Module] = []
    current = input_dim
    for _ in range(depth - 1):
        layers.extend([nn.Linear(current, hidden_dim), nn.GELU()])
        current = hidden_dim
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class ActionRepresentationModel(nn.Module):
    """Encode H-step CALVIN actions into a 16-D semantic + 16-D execution latent."""

    def __init__(
        self,
        *,
        input_mode: Literal["action_only"],
        chunk_length: int,
        action_dim: int = 7,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        depth: int = 3,
        text_feature_dim: int = 768,
        semantic_dim: int = 16,
    ) -> None:
        super().__init__()
        if input_mode != "action_only":
            raise ValueError("The released representation is action-only")
        self.input_mode = input_mode
        self.chunk_length = int(chunk_length)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.semantic_dim = int(semantic_dim)
        if not 0 < self.semantic_dim < self.latent_dim:
            raise ValueError("semantic_dim must be a strict prefix of latent_dim")
        self.encoder = mlp(
            self.chunk_length * self.action_dim, hidden_dim, self.latent_dim, depth
        )
        self.decoder = mlp(
            self.latent_dim, hidden_dim, self.chunk_length * self.action_dim, depth
        )
        self.text_projection = nn.Linear(text_feature_dim, self.semantic_dim, bias=False)

    def encoder_features(self, actions: torch.Tensor) -> torch.Tensor:
        """Return the shared encoder feature before the final latent head."""

        if actions.ndim != 3 or actions.shape[1:] != (self.chunk_length, self.action_dim):
            raise ValueError(
                f"Expected actions (B,{self.chunk_length},{self.action_dim}), got {tuple(actions.shape)}"
            )
        return self.encoder[:-1](actions.flatten(start_dim=1))

    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        """Encode one normalized action window."""

        return self.encoder[-1](self.encoder_features(actions))

    def isolated_clip_semantic_latent(self, actions: torch.Tensor) -> torch.Tensor:
        """Apply the semantic head to a trunk feature detached only for language loss."""

        return self.encoder[-1](self.encoder_features(actions).detach())[:, : self.semantic_dim]

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode normalized continuous actions and a gripper logit."""

        return self.decoder(latent).view(-1, self.chunk_length, self.action_dim)

    def project_text(self, text_features: torch.Tensor) -> torch.Tensor:
        """Project frozen OpenCLIP features into the semantic subspace."""

        return self.text_projection(text_features)

    def split_latent(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return semantic prefix and execution residual."""

        if latent.ndim != 2 or latent.shape[-1] != self.latent_dim:
            raise ValueError(f"Expected latent (B,{self.latent_dim}), got {tuple(latent.shape)}")
        return latent[:, : self.semantic_dim], latent[:, self.semantic_dim :]

    def forward(
        self, actions: torch.Tensor, isolate_clip_shared: bool = False
    ) -> dict[str, torch.Tensor]:
        """Return the full latent, both slices, reconstruction, and optional CLIP view."""

        latent = self.encode(actions)
        semantic, execution = self.split_latent(latent)
        result = {
            "latent": latent,
            "semantic_latent": semantic,
            "execution_latent": execution,
            "reconstruction": self.decode(latent),
        }
        if isolate_clip_shared:
            result["clip_semantic_latent"] = self.isolated_clip_semantic_latent(actions)
        return result

"""Mathematically explicit losses for the released PGLT representation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def symmetric_contrastive_loss(
    action_latents: torch.Tensor,
    text_latents: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute symmetric diagonal CLIP cross entropy and its logits matrix."""

    if action_latents.shape != text_latents.shape or action_latents.ndim != 2:
        raise ValueError(
            f"Action and text latents must share shape (B,D), got {action_latents.shape} and {text_latents.shape}"
        )
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    action_normalized = F.normalize(action_latents, dim=-1)
    text_normalized = F.normalize(text_latents, dim=-1)
    logits = action_normalized @ text_normalized.T / temperature
    targets = torch.arange(logits.shape[0], device=logits.device)
    loss = 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))
    return loss, logits


def reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    continuous_weight: float = 1.0,
    gripper_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Separate normalized continuous MSE from binary CALVIN gripper BCE.

    ``prediction[..., :6]`` reconstructs standardized relative Cartesian and
    Euler components. ``prediction[..., 6]`` is a logit for the audited gripper
    command, whose target is mapped explicitly from ``{-1, 1}`` to ``{0, 1}``.
    """

    if prediction.shape != target.shape or prediction.shape[-1] != 7:
        raise ValueError(f"Expected matching (...,7) tensors, got {prediction.shape} and {target.shape}")
    continuous = F.mse_loss(prediction[..., :6], target[..., :6])
    gripper_target = (target[..., 6] + 1.0) / 2.0
    if not torch.all((gripper_target == 0.0) | (gripper_target == 1.0)):
        raise ValueError("Observed gripper target is not exactly -1 or 1")
    gripper = F.binary_cross_entropy_with_logits(prediction[..., 6], gripper_target)
    total = continuous_weight * continuous + gripper_weight * gripper
    return {"total": total, "continuous_mse": continuous, "gripper_bce": gripper}

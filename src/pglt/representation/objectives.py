"""Released representation model builder and condition-specific objectives."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from pglt.representation.data import build_representation_model
from pglt.representation.losses import reconstruction_loss, symmetric_contrastive_loss
from pglt.representation.model import ActionRepresentationModel


CONDITIONS = ("correct_language", "shuffled_language", "reconstruction_only")


def build_model(config: Mapping[str, Any], device: torch.device) -> ActionRepresentationModel:
    """Build the exact released 32-D action-only representation."""

    return build_representation_model(config, device)


def batch_objective(
    model: ActionRepresentationModel,
    batch: Mapping[str, Any],
    condition: str,
    optimization: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Compute reconstruction plus optional isolated symmetric contrastive loss."""

    if condition not in CONDITIONS:
        raise ValueError(condition)
    isolate_language = condition != "reconstruction_only"
    output = model(batch["actions"], isolate_clip_shared=isolate_language)
    reconstruction = reconstruction_loss(
        output["reconstruction"],
        batch["actions"],
        continuous_weight=float(optimization["continuous_reconstruction_weight"]),
        gripper_weight=float(optimization["gripper_reconstruction_weight"]),
    )
    if condition == "reconstruction_only":
        clip = output["latent"].new_tensor(0.0)
    else:
        text = model.project_text(batch["text_feature"])
        clip, _ = symmetric_contrastive_loss(
            output["clip_semantic_latent"], text, float(optimization["temperature"])
        )
    total = (
        float(optimization["lambda_reconstruction"]) * reconstruction["total"]
        + float(optimization["lambda_clip"]) * clip
    )
    return {
        "total": total,
        "clip": clip,
        "reconstruction": reconstruction["total"],
        "continuous_mse": reconstruction["continuous_mse"],
        "gripper_bce": reconstruction["gripper_bce"],
    }

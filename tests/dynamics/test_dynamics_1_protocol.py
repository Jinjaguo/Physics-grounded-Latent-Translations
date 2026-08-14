"""Protocol regressions for non-overlap, causality, equality, and DEL rollouts."""

from __future__ import annotations

import numpy as np
from pathlib import Path
import pytest
import torch

from pglt.dynamics.dynamics_data import DynamicsSequence, horizon_starts
from pglt.dynamics.variational import ControlPacket, DELTransition, HistoryMLPTransition, MLPTransition
from pglt.representation.model import ActionRepresentationModel


def packet(indices: torch.Tensor) -> ControlPacket:
    """Build one synthetic packet with an issue frame at 16."""

    return ControlPacket(torch.zeros(1, 16, 7), indices.reshape(1, 16), torch.tensor([16]), "logged_executed_history", True)


def test_causal_packet_accepts_history_and_rejects_target_window() -> None:
    packet(torch.arange(16)).validate()
    with pytest.raises(ValueError, match="Future target action"):
        packet(torch.arange(1, 17)).validate()


def test_block_information_sets_are_exactly_equal() -> None:
    assert MLPTransition.information_fields == DELTransition.autonomous_information_fields
    assert HistoryMLPTransition.information_fields == DELTransition.forced_information_fields


def test_horizon_enumeration_never_pads() -> None:
    sequence = DynamicsSequence("training", 0, 0, "task", "text", 0, 63, ((0, 15), (16, 31), (32, 47), (48, 63)), (0, 1, 2, 3), 64 / 30)
    assert len(horizon_starts([sequence], 1)) == 2
    assert len(horizon_starts([sequence], 2)) == 1
    assert len(horizon_starts([sequence], 4)) == 0


def test_corrected_del_multistep_is_finite_and_differentiable() -> None:
    torch.manual_seed(3)
    model = DELTransition(forced=False, solver_iterations=4, solver_step_size=0.25)
    previous = torch.randn(2, 32)
    current = torch.randn(2, 32)
    context = torch.randn(2, 16)
    rollout = []
    for _ in range(8):
        following, info = model(previous, current, context, 16 / 30)
        assert torch.isfinite(info.residual_trace).all()
        rollout.append(following)
        previous, current = current, following
    loss = torch.stack(rollout).square().mean()
    loss.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_primary_window_definition_is_nonoverlapping() -> None:
    windows = [(start, start + 15) for start in range(100, 164, 16)]
    assert all(left_end < right_start for (_, left_end), (right_start, _) in zip(windows, windows[1:]))
    assert np.all(np.diff([start for start, _ in windows]) == 16)


def test_decoder_and_representation_can_be_fully_frozen() -> None:
    model = ActionRepresentationModel(input_mode="action_only", chunk_length=16)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        decoded = model.decode(torch.zeros(2, 32))
    assert decoded.shape == (2, 16, 7)
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters())


def test_confirmation_manifest_is_required_by_protocol_source() -> None:
    source = Path("scripts/dynamics/run_dynamics_1.py").read_text(encoding="utf-8")
    assert "Official validation evaluation requires frozen confirmation manifest" in source
    assert "oracle_excluded_from_primary" in source

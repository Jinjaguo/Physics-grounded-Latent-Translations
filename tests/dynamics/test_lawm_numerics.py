"""Regression tests for the locally diagnosed LaWM toy solver correction."""

from pathlib import Path

import torch

from lawm.train import batch_objective
from lawm.utils import load_state_tensor, make_state_weights, make_time_grid
from pglt.dynamics.lawm_adapter import StableToyLeastActionWorldModel


def test_stable_lawm_official_toy_optimizer_step_is_finite() -> None:
    """Require finite losses, gradients, and parameters around one real toy step.

    The input is the official upstream generator output, not a PGLT synthetic
    latent trajectory.  The test uses the README dimensions, 64 time steps,
    eight solver iterations, solver coefficient 0.25, AdamW learning rate
    1e-4, and the exact upstream objective weights.
    """

    toy_path = Path("results/dynamics/software_validation/upstream_reference/toy_parabolic.pt")
    assert toy_path.is_file(), "Generate the official LaWM toy artifact before this regression test"
    states = load_state_tensor(toy_path, state_dim=9, max_samples=2)
    ts = make_time_grid(states.shape[1], 0.02, "cpu")
    torch.manual_seed(0)
    model = StableToyLeastActionWorldModel(
        state_dim=9,
        latent_dim=9,
        context_dim=16,
        hidden_dim=128,
        depth=3,
        solver_iters=8,
        solver_step_size=0.25,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    output = batch_objective(
        model,
        states,
        ts,
        make_state_weights(9, "cpu"),
        lambda_del=1e-2,
        lambda_reg=1e-4,
    )
    assert all(torch.isfinite(output[name]).all() for name in ("loss", "traj", "del", "reg"))
    optimizer.zero_grad(set_to_none=True)
    output["loss"].backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    optimizer.step()
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())

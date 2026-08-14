"""Protocol tests for the third dynamics experiment (wave 15).

Purpose
-------
Detect exact latent-slice, information-set, matched-initialization,
decoder-freezing, Jacobian/metric, split-reuse, and confirmation-order errors.

Parameters
----------
No command-line parameters; pytest supplies its standard options.

Usage
-----
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  tests/dynamics/test_dynamics_3_factorized.py -q

Outputs
-------
Pytest status on stdout; the formal run writes JUnit XML to the wave-15 result
directory through the command recorded in ``executed_commands.txt``.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from pglt.dynamics.dynamics_data import load_frozen_representation
from pglt.dynamics.factorized import (
    DecoderGeometryDEL,
    ExecutionMLP,
    ExecutionMatchedRefinement,
    FreeExecutionDEL,
    SemanticPredictor,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((ROOT / "configs/dynamics_3.yaml").read_text(encoding="utf-8"))


def representation() -> torch.nn.Module:
    manifest = json.loads((ROOT / CONFIG["representation"]["checkpoint_manifest"]).read_text(encoding="utf-8"))
    entry = next(item for item in manifest["checkpoints"] if item["condition"] == "correct_language")
    model, _ = load_frozen_representation(
        yaml.safe_load((ROOT / CONFIG["representation"]["config"]).read_text(encoding="utf-8")),
        ROOT / entry["path"],
    )
    return model


def test_exact_frozen_16_16_slicing_and_wave13_split_reuse() -> None:
    with np.load(ROOT / CONFIG["data"]["frozen_latents"], allow_pickle=False) as saved:
        assert saved["latents"].shape[1] == 32
        assert np.array_equal(saved["latents"][:, :16], saved["semantic_latents"])
        assert np.array_equal(saved["latents"][:, 16:], saved["execution_latents"])
    assert CONFIG["data"]["train_sequences"].startswith(CONFIG["experiment"]["wave13_root"])
    assert CONFIG["data"]["development_sequences"].startswith(CONFIG["experiment"]["wave13_root"])
    assert CONFIG["data"]["validation_sequences"].startswith(CONFIG["experiment"]["wave13_root"])


def test_information_set_and_shared_semantic_design() -> None:
    semantic = SemanticPredictor()
    f1 = ExecutionMLP()
    f3 = FreeExecutionDEL()
    expected = ("e_previous", "e_current", "s_current", "context")
    assert semantic.information_fields == ("s_previous", "s_current", "context")
    assert f1.information_fields == expected
    assert f3.information_fields == expected
    assert all("target" not in field and "future" not in field for field in expected)


def test_f2_f4_identical_initializer_and_iteration_budget() -> None:
    torch.manual_seed(1503)
    f1 = ExecutionMLP()
    decoder = deepcopy(representation().decoder)
    f2 = ExecutionMatchedRefinement(f1, iterations=4)
    f4 = DecoderGeometryDEL(f1, decoder, iterations=4)
    for key, value in f1.state_dict().items():
        assert torch.equal(value, f2.initializer.state_dict()[key])
        assert torch.equal(value, f4.initializer.state_dict()[key])
    assert f2.iterations == f4.iterations == CONFIG["models"]["refinement_iterations"]


def test_decoder_jvp_finite_difference_metric_psd_and_no_decoder_grad() -> None:
    torch.manual_seed(1503)
    decoder = deepcopy(representation().decoder)
    f4 = DecoderGeometryDEL(ExecutionMLP(), decoder, metric_epsilon=1e-3)
    semantic = torch.randn(2, 16)
    execution = torch.randn(2, 16, requires_grad=True)
    tangent = torch.randn_like(execution)
    jvp = f4.lagrangian.decoder_jvp(semantic, execution, tangent)
    delta = 1e-3
    with torch.no_grad():
        plus = decoder(torch.cat((semantic, execution + delta * tangent), dim=-1)).flatten(1)
        minus = decoder(torch.cat((semantic, execution - delta * tangent), dim=-1)).flatten(1)
    finite_difference = (plus - minus) / (2 * delta)
    relative_error = (jvp.detach() - finite_difference).norm() / finite_difference.norm().clamp_min(1e-12)
    assert float(relative_error) < 0.01
    quadratic = f4.lagrangian.metric_quadratic(semantic, execution, tangent)
    assert torch.all(quadratic >= -1e-7)
    quadratic.mean().backward()
    assert all(parameter.grad is None for parameter in f4.lagrangian.decoder.parameters())


def test_confirmation_manifest_precedes_validation_when_results_exist() -> None:
    out = ROOT / CONFIG["experiment"]["output_root"]
    result_path = out / "one_shot_official_validation_results.json"
    if not result_path.exists():
        return
    manifest = json.loads((out / "factorized_dynamics_confirmation_manifest.json").read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert manifest["frozen_before_official_validation_metrics"] is True
    assert result["one_shot"] is True
    assert result["manifest_created_at"] < result["evaluated_at"]


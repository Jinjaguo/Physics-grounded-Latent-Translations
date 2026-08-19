#!/usr/bin/env python3
"""Serve the fixed official π0.5 LIBERO source policy for Wave 20.

Purpose
-------
Load the local official ``pi05_libero`` PyTorch checkpoint with its frozen
normalization assets and serve websocket inference. Each response contains the
standard postprocessed 10x7 LIBERO action chunk plus ``raw_model_actions``
captured before OpenPI output transforms. No π0.5 hidden state is exposed,
saved, or used by the experiment.

Parameters
----------
``--checkpoint_dir`` is the local checkpoint containing ``model.safetensors``.
``--norm_assets_dir`` contains the official LIBERO normalization asset tree.
``--config_name`` defaults to ``pi05_libero``, ``--seed`` controls the fixed
PyTorch sampling stream, and ``--port`` defaults to 8000.

Usage
-----
cd /home/jinjaguo/openpi
source .venv/bin/activate
python /home/jinjaguo/Actions_As_Coordinates/scripts/dynamics/start_wave20_pi05_server.py \
  --checkpoint_dir /home/jinjaguo/.cache/openpi/pytorch_checkpoints/pi05_libero \
  --norm_assets_dir /home/jinjaguo/.cache/openpi/openpi-assets/checkpoints/pi05_libero/assets \
  --port 8000

Outputs
-------
The process listens on ``0.0.0.0:<port>`` and writes no files. Collection
artifacts are written only by ``collect_wave20_libero.py``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import socket
import time
from typing import Any

import jax
import numpy as np
import torch

from openpi.models import model as model_lib
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import checkpoints
from openpi.training import config as training_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("/home/jinjaguo/.cache/openpi/pytorch_checkpoints/pi05_libero"),
    )
    parser.add_argument(
        "--norm_assets_dir",
        type=Path,
        default=Path("/home/jinjaguo/.cache/openpi/openpi-assets/checkpoints/pi05_libero/assets"),
    )
    parser.add_argument("--config_name", default="pi05_libero")
    parser.add_argument("--seed", type=int, default=200820)
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


class RawActionPolicy:
    """Expose model-space actions while preserving the official output transform."""

    def __init__(self, policy: Any, *, seed: int):
        if not getattr(policy, "_is_pytorch_model", False):
            raise TypeError("Wave-20 source server requires the local PyTorch π0.5 checkpoint")
        self._policy = policy
        self._seed = seed

    @property
    def metadata(self) -> dict[str, Any]:
        return {**self._policy.metadata, "wave20_batch_inference": True, "wave20_policy_seed": self._seed}

    def _infer_batch(self, observations: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float]:
        if not observations:
            raise ValueError("wave20_observation_batch must not be empty")
        transformed = [
            self._policy._input_transform(jax.tree.map(lambda value: value, observation))
            for observation in observations
        ]
        inputs = jax.tree.map(
            lambda *values: torch.from_numpy(np.stack([np.asarray(value) for value in values])).to(
                self._policy._pytorch_device
            ),
            *transformed,
        )
        observation = model_lib.Observation.from_dict(inputs)
        started = time.monotonic()
        raw = self._policy._sample_actions(
            self._policy._pytorch_device,
            observation,
            **dict(self._policy._sample_kwargs),
        )
        infer_ms = (time.monotonic() - started) * 1000.0
        state = inputs["state"]
        raw_numpy = np.asarray(raw.detach().cpu())
        state_numpy = np.asarray(state.detach().cpu())
        results = []
        for index in range(len(observations)):
            result = self._policy._output_transform(
                {
                    "state": state_numpy[index],
                    "actions": raw_numpy[index].copy(),
                }
            )
            result["raw_model_actions"] = raw_numpy[index]
            result["policy_timing"] = {"infer_ms": infer_ms, "batch_size": len(observations)}
            results.append(result)
        return results, infer_ms

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        if "wave20_observation_batch" in obs:
            if set(obs) != {"wave20_observation_batch"}:
                raise ValueError("Batch request may only contain wave20_observation_batch")
            results, infer_ms = self._infer_batch(list(obs["wave20_observation_batch"]))
            return {
                "wave20_batch_results": results,
                "policy_timing": {"infer_ms": infer_ms, "batch_size": len(results)},
            }
        results, _ = self._infer_batch([obs])
        return results[0]


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train = training_config.get_config(args.config_name)
    data_config = train.data.create(train.assets_dirs, train.model)
    if data_config.asset_id is None:
        raise RuntimeError("pi05_libero data config has no normalization asset id")
    norm_stats = checkpoints.load_norm_stats(args.norm_assets_dir, data_config.asset_id)
    policy = policy_config.create_trained_policy(
        train,
        args.checkpoint_dir,
        norm_stats=norm_stats,
        pytorch_device="cuda",
    )
    # Model construction may consume the global generator. Reset immediately
    # before serving so the recorded seed names the source-policy noise stream.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    source_policy = RawActionPolicy(policy, seed=args.seed)
    hostname = socket.gethostname()
    logging.info("Creating Wave-20 π0.5 source server on %s:%d", hostname, args.port)
    logging.info("Responses contain raw model actions and postprocessed LIBERO actions; hidden states are disabled")
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=source_policy,
        host="0.0.0.0",
        port=args.port,
        metadata=source_policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()

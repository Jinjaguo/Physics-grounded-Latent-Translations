#!/usr/bin/env bash
: <<'WAVE27_COMMANDS'
Purpose
-------
Reproduce Wave27 prospective collection, encoding, sweep, selection, final
evaluation, reporting, and tests from the repository root.

Parameters
----------
The commands use configs/dynamics_15.yaml, the OpenPI environment for remote
Parquet/Hugging Face acquisition, and the libero environment with cuda:0 for
representation/model work.

Usage
-----
bash results/dynamics/twenty_seventh_wave/2026-08-15_dynamics_15/exact_commands.sh

Outputs
-------
Tracked experiment artifacts are written under
results/dynamics/twenty_seventh_wave/2026-08-15_dynamics_15; local compact
records and checkpoints remain ignored.
WAVE27_COMMANDS

set -euo pipefail

PYTHONPATH=.:src /home/jinjaguo/openpi/.venv/bin/python \
  scripts/dynamics/acquire_dynamics_15.py --config configs/dynamics_15.yaml \
  --stage all

PYTHONPATH=.:src MPLCONFIGDIR=/tmp/wave27_mpl \
  /home/jinjaguo/anaconda3/envs/libero/bin/python \
  scripts/dynamics/run_dynamics_15.py --config configs/dynamics_15.yaml \
  --stage all --device cuda:0

PYTHONPATH=.:src MPLCONFIGDIR=/tmp/wave27_mpl \
  /home/jinjaguo/anaconda3/envs/libero/bin/python -m pytest -q \
  tests/test_dynamics_15.py

NUMBA_DISABLE_JIT=1 \
PYTHONPATH=.:src:/home/jinjaguo/LIBERO:third_party/LaWM \
MPLCONFIGDIR=/tmp/wave27_mpl \
  /home/jinjaguo/anaconda3/envs/libero/bin/python -m pytest -q \
  -k 'not project_remains_below_20_gib'

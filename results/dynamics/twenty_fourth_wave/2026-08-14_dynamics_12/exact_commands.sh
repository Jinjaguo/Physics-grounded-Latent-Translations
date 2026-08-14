#!/usr/bin/env bash
set -euo pipefail
# The first direct report invocation lacked PYTHONPATH and stopped at ModuleNotFoundError; no experiment code ran.
/home/jinjaguo/anaconda3/envs/libero/bin/python -m pip install pyarrow==17.0.0
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_12.py --config configs/dynamics_12.yaml --stage prepare --device cuda:0
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_12.py --config configs/dynamics_12.yaml --stage phasea --device cuda:0
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_12.py --config configs/dynamics_12.yaml --stage report --device cuda:0
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python -m pytest tests/dynamics/test_dynamics_12_displacement_family.py -q

#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_10.py --config configs/dynamics_10.yaml --stage prepare --device cuda:0
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_10.py --config configs/dynamics_10.yaml --stage phasea --device cuda:0
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_10.py --config configs/dynamics_10.yaml --stage geometry --device cuda:0
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_10.py --config configs/dynamics_10.yaml --stage report --device cuda:0
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python -m pytest tests/dynamics/test_dynamics_10_cycle_consistency.py -q

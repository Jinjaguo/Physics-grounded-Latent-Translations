#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_11.py --config configs/dynamics_11.yaml --stage prepare --device cuda:0
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_11.py --config configs/dynamics_11.yaml --stage phasea --device cuda:0
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_11.py --config configs/dynamics_11.yaml --stage train --device cuda:0
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_11.py --config configs/dynamics_11.yaml --stage report --device cuda:0
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python -m pytest tests/dynamics/test_dynamics_11_goal_alignment.py -q

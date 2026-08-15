#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_14.py --config configs/dynamics_14.yaml --stage prepare --device cuda:0
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_14.py --config configs/dynamics_14.yaml --stage sweep --device cuda:0
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_14.py --config configs/dynamics_14.yaml --stage select --device cuda:0
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_14.py --config configs/dynamics_14.yaml --stage final --device cuda:0
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_14.py --config configs/dynamics_14.yaml --stage report --device cuda:0
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python -m pytest tests/dynamics/test_dynamics_14_phase_flow.py -q

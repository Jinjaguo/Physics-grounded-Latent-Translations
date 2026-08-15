#!/usr/bin/env bash
set -euo pipefail
# See wave25_execution_log.md for discarded diagnostic/sweep/report attempts and fixes.
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_13.py --config configs/dynamics_13.yaml --stage prepare --device cuda:0
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_13.py --config configs/dynamics_13.yaml --stage diagnose --device cuda:0
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_13.py --config configs/dynamics_13.yaml --stage sweep --device cuda:0
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_13.py --config configs/dynamics_13.yaml --stage select --device cuda:0
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_13.py --config configs/dynamics_13.yaml --stage final --device cuda:0
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_13.py --config configs/dynamics_13.yaml --stage report --device cuda:0
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python -m pytest tests/dynamics/test_dynamics_13_broad_sweep.py -q

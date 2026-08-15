#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_wave28_force_field.py --stage audit --device cpu
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_wave28_force_field.py --stage sweep --device cpu
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_wave28_force_field.py --stage select --device cpu
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_wave28_force_field.py --stage final --device cpu
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_wave28_force_field.py --stage report --device cpu
PYTHONPATH=.:src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_wave34_representation_stop.py --stage all

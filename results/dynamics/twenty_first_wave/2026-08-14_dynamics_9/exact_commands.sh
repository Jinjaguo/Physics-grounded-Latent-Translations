#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/fetch_wave21_annotation_metadata.py --manifest data/representation/calvin_task_D_D/metadata/fetch_manifest.json --split training
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_9.py --config configs/dynamics_9.yaml --stage all --device cuda:0
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python -m pytest tests/dynamics/test_dynamics_9_language_transition.py -q

# 2026-08-14T15:37:26-04:00 phase=preregister
PYTHONPATH=src:/home/jinjaguo/LIBERO /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_8.py --config configs/dynamics_8.yaml --stage preregister

# 2026-08-14 phase=fresh-source-policy-server
# First restricted launch failed before serving because CUDA was not visible; the identical command was relaunched with GPU access.
/home/jinjaguo/openpi/.venv/bin/python /home/jinjaguo/Actions_As_Coordinates/scripts/dynamics/start_wave20_pi05_server.py --checkpoint_dir /home/jinjaguo/.cache/openpi/pytorch_checkpoints/pi05_libero --norm_assets_dir /home/jinjaguo/.cache/openpi/openpi-assets/checkpoints/pi05_libero/assets --seed 200820 --port 8001

# 2026-08-14 phase=fresh-confirmation-collection-and-certification
PYTHONPATH=src:/home/jinjaguo/LIBERO:/home/jinjaguo/openpi/packages/openpi-client/src MUJOCO_GL=egl /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/collect_wave20_libero.py --config configs/dynamics_8.yaml --host localhost --port 8001

# 2026-08-14 phase=pre-representation-confirmation-tests
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 NUMBA_DISABLE_JIT=1 PYTHONPATH=src:/home/jinjaguo/LIBERO /home/jinjaguo/anaconda3/envs/libero/bin/python -m pytest -q tests/dynamics/test_dynamics_8_motor_margin.py --tb=short

# 2026-08-14 phase=dynamics-training-and-offline-O1-O8
PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/train_wave20_dynamics.py --config configs/dynamics_8.yaml --device cuda:0

# 2026-08-14 phase=final-wave19-wave20-protocol-tests
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 NUMBA_DISABLE_JIT=1 PYTHONPATH=src:/home/jinjaguo/LIBERO:/home/jinjaguo/openpi/packages/openpi-client/src MUJOCO_GL=egl /home/jinjaguo/anaconda3/envs/libero/bin/python -m pytest -q tests/dynamics/test_dynamics_8_motor_margin.py tests/dynamics/test_dynamics_7_libero_protocol.py --tb=short

# 2026-08-14 phase=offline-stop
# O1/O3/O5/O8 failed. Final-test opening, B0-B5, direction controls, and perturbation recovery were NOT RUN.

# 2026-08-14T15:42:37-04:00 phase=source-collection-start-or-resume
PYTHONPATH=src:/home/jinjaguo/LIBERO:/home/jinjaguo/openpi/packages/openpi-client/src MUJOCO_GL=egl /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/collect_wave20_libero.py --config configs/dynamics_8.yaml --host localhost --port 8001

# 2026-08-14T16:19:49-04:00 phase=source-collection
PYTHONPATH=src:/home/jinjaguo/LIBERO:/home/jinjaguo/openpi/packages/openpi-client/src MUJOCO_GL=egl /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/collect_wave20_libero.py --config configs/dynamics_8.yaml --host localhost --port 8001

# 2026-08-14T16:20:55-04:00 phase=R0-R1-shuffled-control-training-confirmation-evaluation-bootstrap-gate
PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/train_wave20_representation.py --config configs/dynamics_8.yaml --device cuda:0

# 2026-08-14T16:39:01-04:00 phase=dynamics-training-and-offline-gate
PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/train_wave20_dynamics.py --config configs/dynamics_8.yaml --device cuda:0

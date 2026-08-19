
# 2026-08-14T06:00:20-04:00 phase=audit-interrupted
# FAILED after preregistration write: OpenPI uv venv has no pip module
PYTHONPATH=src:/home/jinjaguo/LIBERO /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_7.py --config configs/dynamics_7.yaml --stage audit

# 2026-08-14T06:00:20-04:00 phase=audit-resume
PYTHONPATH=src:/home/jinjaguo/LIBERO /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_7.py --config configs/dynamics_7.yaml --stage audit-resume

# 2026-08-14T06:11:02-04:00 phase=dev-certify
PYTHONPATH=src:/home/jinjaguo/LIBERO MUJOCO_GL=egl /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/run_dynamics_7.py --config configs/dynamics_7.yaml --stage dev-certify

# 2026-08-14T06:41:49-04:00 phase=source-policy-server
/home/jinjaguo/openpi/.venv/bin/python /home/jinjaguo/Actions_As_Coordinates/scripts/dynamics/start_wave19_pi05_server.py --checkpoint_dir /home/jinjaguo/.cache/openpi/pytorch_checkpoints/pi05_libero --norm_assets_dir /home/jinjaguo/.cache/openpi/openpi-assets/checkpoints/pi05_libero/assets --seed 190819 --port 8000

# 2026-08-14T06:41:49-04:00 phase=source-collection-start
PYTHONPATH=src:/home/jinjaguo/LIBERO:/home/jinjaguo/openpi/packages/openpi-client/src MUJOCO_GL=egl /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/collect_wave19_libero.py --config configs/dynamics_7.yaml --host localhost --port 8000

# 2026-08-14T06:48:31-04:00 phase=source-policy-server
/home/jinjaguo/openpi/.venv/bin/python /home/jinjaguo/Actions_As_Coordinates/scripts/dynamics/start_wave19_pi05_server.py --checkpoint_dir /home/jinjaguo/.cache/openpi/pytorch_checkpoints/pi05_libero --norm_assets_dir /home/jinjaguo/.cache/openpi/openpi-assets/checkpoints/pi05_libero/assets --seed 190819 --port 8000

# 2026-08-14T06:48:31-04:00 phase=source-collection-start
PYTHONPATH=src:/home/jinjaguo/LIBERO:/home/jinjaguo/openpi/packages/openpi-client/src MUJOCO_GL=egl /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/collect_wave19_libero.py --config configs/dynamics_7.yaml --host localhost --port 8000

# 2026-08-14T06:51:03-04:00 phase=text-feature-freeze
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/prepare_wave19_text_features.py --config configs/dynamics_7.yaml --weights /home/jinjaguo/.cache/huggingface/hub/models--laion--CLIP-ViT-L-14-DataComp.XL-s13B-b90K/snapshots/84c9828e63dc9a9351d1fe637c346d4c1c4db341/open_clip_pytorch_model.bin

# 2026-08-14T07:24:39-04:00 phase=source-collection-invalid-numba-mode
PYTHONPATH=src:/home/jinjaguo/LIBERO:/home/jinjaguo/openpi/packages/openpi-client/src MUJOCO_GL=egl NUMBA_DISABLE_JIT=1 /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/collect_wave19_libero.py --config configs/dynamics_7.yaml --host localhost --port 8000

# 2026-08-14T07:25:29-04:00 phase=source-collection-resume-original-jit-mode
PYTHONPATH=src:/home/jinjaguo/LIBERO:/home/jinjaguo/openpi/packages/openpi-client/src MUJOCO_GL=egl /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/collect_wave19_libero.py --config configs/dynamics_7.yaml --host localhost --port 8000

# 2026-08-14T07:29:13-04:00 phase=source-collection-start-or-resume
PYTHONPATH=src:/home/jinjaguo/LIBERO:/home/jinjaguo/openpi/packages/openpi-client/src MUJOCO_GL=egl /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/collect_wave19_libero.py --config configs/dynamics_7.yaml --host localhost --port 8000

# 2026-08-14T10:11:08-04:00 phase=source-collection
PYTHONPATH=src:/home/jinjaguo/LIBERO:/home/jinjaguo/openpi/packages/openpi-client/src MUJOCO_GL=egl /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/collect_wave19_libero.py --config configs/dynamics_7.yaml --host localhost --port 8000

# 2026-08-14T10:12:52-04:00 phase=contact-materialization
PYTHONPATH=src:/home/jinjaguo/LIBERO MUJOCO_GL=egl /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/derive_wave19_contacts.py --config configs/dynamics_7.yaml

# timestamp not retained phase=dependency-install
/home/jinjaguo/anaconda3/envs/libero/bin/python -m pip install open_clip_torch

# 2026-08-14T10:14:00-04:00 phase=representation-training-invalid-determinism-environment
# FAILED before the first optimizer step: deterministic CUDA matmul required CUBLAS_WORKSPACE_CONFIG.
PYTHONPATH=src /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/train_wave19_representation.py --config configs/dynamics_7.yaml --device cuda:0

# 2026-08-14T10:14:43-04:00 phase=representation-training-and-gate
PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/jinjaguo/anaconda3/envs/libero/bin/python scripts/dynamics/train_wave19_representation.py --config configs/dynamics_7.yaml --device cuda:0

# 2026-08-14 phase=final-environment-freeze
PYTHONPATH=src:/home/jinjaguo/LIBERO /home/jinjaguo/anaconda3/envs/libero/bin/python -c "import yaml; from scripts.dynamics.run_dynamics_7 import write_environment_freeze; c=yaml.safe_load(open('configs/dynamics_7.yaml')); write_environment_freeze(c)"

# 2026-08-14 phase=targeted-tests
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 NUMBA_DISABLE_JIT=1 PYTHONPATH=src:/home/jinjaguo/LIBERO:/home/jinjaguo/openpi/packages/openpi-client/src MUJOCO_GL=egl /home/jinjaguo/anaconda3/envs/libero/bin/python -m pytest -q tests/dynamics/test_dynamics_7_libero_protocol.py tests/dynamics/test_dynamics_6_reconstruction_gate.py --tb=short

# 2026-08-14 phase=stop-condition
# F1/F2 training, offline evaluation, closed-loop evaluation, and final-test statistics were NOT RUN because the representation R-gate failed.
# No clone, checkpoint download, or LIBERO dataset download was needed; the official local repositories and π0.5 checkpoint were audited before collection.

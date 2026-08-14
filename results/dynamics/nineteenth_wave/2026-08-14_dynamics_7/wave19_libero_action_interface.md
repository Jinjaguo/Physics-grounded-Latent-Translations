# Wave-19 official LIBERO action interface

- frozen_at: `2026-08-14T05:59:11-04:00`
- action_dim: `7`
- controller: `OSC_POSE`
- control_frequency_hz: `20`
- physics_frequency_hz: `500`
- translation: `relative_delta_scaled_to_plus_minus_0.05_m`
- rotation: `relative_axis_angle_scaled_to_plus_minus_0.5_rad`
- gripper: `one_dimensional sign command integrated by Panda gripper; OpenPI value is not client-clipped`
- OSC input bounds, first six dimensions: `[-1.0, 1.0]`
- formal task-0 gripper example range: `[-1.00527336, 1.00590591]`; only its sign controls the gripper
- policy_action_horizon: `10`
- policy_replan_steps: `5`
- action_repeat: `1`
- physics_steps_per_control_step: `25`
- clipping: `OSC clips each of first six normalized components to [-1,1] before scaling`
- network_vs_executed_boundary: `policy_output_actions stores websocket output after OpenPI unnormalization; executed_actions stores an independent copy passed to env.step; no additional client clipping`
- immutable_action_boundary: `safe_env_step passes action.copy() and checks caller bytes unchanged`

This wording was amended before representation/dynamics training and before any final-test access. The original
global `action_bounds` label incorrectly implied that the gripper magnitude was clipped; the execution code never
performed that clipping, so the clarification does not alter any collected action or rollout.

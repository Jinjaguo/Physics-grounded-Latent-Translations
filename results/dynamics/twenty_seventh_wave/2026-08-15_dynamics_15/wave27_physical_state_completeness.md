# Wave 27 physical-state completeness

```json
{
  "records": 407,
  "sessions": 52,
  "per_goal": {
    "lift_blue_block_slider": 78,
    "turn_off_lightbulb": 56,
    "turn_on_lightbulb": 57,
    "lift_red_block_table": 83,
    "push_pink_block_right": 54,
    "place_in_slider": 79
  },
  "fields": {
    "action_available": 407,
    "robot_obs_available": 407,
    "scene_obs_available": 407,
    "gripper_width_available": 407,
    "tcp_pose_available": 407,
    "joint_position_available": 407,
    "measured_joint_velocity_available": 0,
    "measured_tcp_velocity_available": 0,
    "true_contact_available": 0
  },
  "velocity_policy": "finite differences may be used only as explicitly derived causal features; they are not labeled measured velocity",
  "contact_policy": "unavailable, not replaced by proxy in true-contact claims"
}
```

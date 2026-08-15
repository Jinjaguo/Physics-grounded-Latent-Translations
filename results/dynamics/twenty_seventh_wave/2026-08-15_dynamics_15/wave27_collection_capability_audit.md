# Wave 27 collection capability audit

```json
{
  "created_at": "2026-08-14T23:03:37.050397-04:00",
  "routes": {
    "official_human_play_archive": {
      "available": true,
      "selected": true,
      "physical_fields": [
        "rel_actions",
        "robot_obs",
        "scene_obs"
      ],
      "true_contact": false
    },
    "official_debug_archive": {
      "available": true,
      "selected": false,
      "reason": "only two source sessions and 17 annotations"
    },
    "trained_policy_collector": {
      "available": false,
      "selected": false,
      "reason": "no verified CALVIN policy checkpoint with the frozen six-goal interface"
    },
    "scripted_controller": {
      "available": false,
      "selected": false,
      "reason": "no verified six-goal primitive controller"
    },
    "manual_teleoperation": {
      "available": false,
      "selected": false,
      "reason": "VR/SpaceMouse hardware and operator loop unavailable in this run"
    }
  },
  "selected_route": "official human-play continuous source shards 005+",
  "collector_type": "official_human_play_archive",
  "collector_version": "c0ddd6e9cf1463d8a6023a18cc608a6dff6a136d"
}
```

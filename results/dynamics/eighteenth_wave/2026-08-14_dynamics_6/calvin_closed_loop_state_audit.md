# CALVIN closed-loop state audit

## Installed implementation

- Commit: `fa03f01f19c65920e18cf37398a9ce859274af76` under `third_party/calvin`.
- `PlayTableSimEnv.reset` in `calvin_env/envs/play_table_env.py` resets `scene_obs`, then `robot_obs`, then advances Bullet one physics step.
- `Robot.reset` in `calvin_env/robot/robot.py` restores seven arm positions and gripper opening. It does not receive source joint velocities or prior motor/controller state; it recomputes `target_pos/target_orn` from the reset TCP pose.
- `PlayTableScene.reset` in `calvin_env/scene/play_table_scene.py` restores door/button/switch/light values and three movable-object poses. It does not receive movable-object linear/angular velocity or contact manifolds.
- `PlayTableSimEnv.serialize` delegates to `Robot.serialize` and `PlayTableScene.serialize`. Robot serialization includes joint values/velocities, while scene serialization omits movable-object velocities. `reset_from_storage` restores movable poses but not those velocities. It also does not explicitly restore the robot's Python-side `target_pos/target_orn`.
- Bullet is configured at 240 Hz with eight physics steps per 30 Hz control action and `deterministicOverlappingPairs=1`.
- The merged dataset config uses seed 0; scene randomness is used for initial object placement. Observation-based reset bypasses pose sampling but does not supply prior velocities/contact state. `serialize` records wall time, but `reset_from_storage` does not restore the environment timing fields.
- Relative action preprocessing is `Robot.relative_to_absolute`: position ×0.02, Euler orientation ×0.05, accumulated on `target_pos/target_orn`; gripper is signed.
- Official success is `Tasks.get_task_info_for_set` from `calvin_env/envs/tasks.py` with `new_playtable_tasks.yaml`/the merged dataset config.
- PyBullet 3.2.7 exposes in-process `saveState/restoreState`, which is the preferred strategy for new prospective episodes. No such source snapshot or restorable state ID/file is present in the retained rendered data.

## State required for valid branching

An exact source continuation needs at least all robot joint position/velocity state, TCP/controller targets, gripper command/state, fixed-joint state, movable-object pose and linear/angular velocity, logical scene state, physics timing/configuration, and the contact/constraint state needed by Bullet. Python-side task/controller state must also correspond to the snapshot.

## What the retained public files contain

Official rendered frames and retained VyoJ files contain `robot_obs`, `scene_obs`, absolute/relative actions, and images or compact equivalents. They contain no full Bullet snapshot. The observation reset is therefore an explicitly **approximate** reset, not strategy A, B, or C from the preregistration: the retained continuous play ranges do not begin at a known simulator reset and the original raw recorder pickles are absent.

## Decision

Exact source branch reconstruction is unavailable. After the same approximate reset, twin continuous-state components and terminal predicates agree, but their exposed contact sets differ by one point in all six diagnostics. Even a perfect approximate-reset twin match could not repair or certify correspondence to the recorded source branch. The planned closed-loop causal continuation study therefore could not be executed; this says nothing about whether refinement would succeed or fail.

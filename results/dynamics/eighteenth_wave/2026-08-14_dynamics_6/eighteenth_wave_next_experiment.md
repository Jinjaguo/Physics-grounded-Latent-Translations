# Next experiment after wave 18

The planned wave-18 closed-loop causal continuation study could not be executed because the retained public CALVIN artifacts do not permit exact reconstruction of source branch states. Closed-loop refinement was not evaluated. The next experiment is not another replay of rendered `robot_obs/scene_obs` files.

Prospectively collect CALVIN episodes while saving a branchable state at every candidate point: Bullet `saveState` (or an equivalent complete engine snapshot), robot joint positions/velocities, Python-side controller targets, gripper command, fixed-joint state, movable-object pose and linear/angular velocity, logical scene state, seeds/timing, and the official task start-info. Immediately validate each saved snapshot by restoring two twins and replaying the same expert continuation.

Collect at least 30 independent successful source episodes for each of the same six tasks (≥180 total; no task >35%), from fresh simulator resets. Freeze the 25/50/75% branch manifest only after 100% terminal-predicate agreement, finite trajectories, and median/P95 twin errors within the frozen 1e-9 diagnostic tolerance. Then, and only then, run the already specified frozen F1/F2/random/shuffled/negative/perturbation protocol exactly once.

The single most important experiment outside CALVIN is the same design in a genuinely independent embodied simulator or robot domain with exact resettable state and prospective causal logging.

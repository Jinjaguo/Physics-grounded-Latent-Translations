# EXP_R40 report — interface-gated continuation

## Scientific question
Can the next waypoint memory, branch checkpoints, and robot-state return be evaluated without violating the closed-loop causal interface?

## Audit result
**NOT_RUN_INTERFACE_GATE**. This is a bounded gate audit, not a positive or negative physical task result.

## Concrete evidence
- Disk audit: available bytes=911885479936 (floor=300000000000, passed=True).
- The repository's retained complete CALVIN episode schema is action-only (`rel_actions`, `global_frame_indices`); Wave27 observation windows contain `robot_obs` and `scene_obs` but do not contain a full Bullet snapshot.
- The historical closed-loop state audit is preserved at `results/dynamics/eighteenth_wave/2026-08-14_dynamics_6/calvin_closed_loop_state_audit.md` and its not-run manifest at `results/dynamics/eighteenth_wave/2026-08-14_dynamics_6/closed_loop_not_run_manifest.json`.

## Why this EXP cannot claim held-out control
robot_obs/scene_obs omit Bullet contacts, controller targets, movable-object velocities, and exact branch state. Opening a held-out physical evaluation under these conditions would not be causal and would repeat the documented reconstruction-gate failure, so no held-out metrics are fabricated. Frozen representation, decoder, F1, old F2, and R8 results remain unchanged.

## Required change
record full serialize/saveState snapshots and waypoint fields during new rollouts.

## Decision
`SUCCESS=false`; EXP_R41 is the next bounded audit.

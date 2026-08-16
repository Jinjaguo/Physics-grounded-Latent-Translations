# EXP_R9–EXP_R58 failure taxonomy

1. **Causal state missing:** complete episodes omit Bullet snapshots, contacts, controller targets, and object velocities.
2. **Teacher-forced feedback:** recorded next states do not respond to the planned command, so they cannot certify physical MPC.
3. **Arrival–continuity trade-off:** proposal paths are smooth but slightly miss target regions; R8/linear paths arrive more often but are less smooth.
4. **Surrogate model mismatch:** current-only, history-conditioned, compliance, shock, target-set, and repair variants did not jointly dominate R8.
5. **Completion detection:** oracle-boundary F3 readiness did not meet the balanced-accuracy/early-switch/late-miss thresholds.
6. **Return unavailable:** no exact branch checkpoint or waypoint/controller snapshot exists for a robot-state return claim.

# EXP_R2 report — source-conditioned local target planning

EXP_R2 kept the released representation, decoder, F1 and old F2 frozen. It used train-only source-goal to target-goal latent groups and local fourth-neighbour radii; H=2 and H=4 were evaluated because the available 128-frame window does not support H=8.

## Held-out summary

| method | arrival | decoded first diff | support distance | hidden path MSE |
|---|---:|---:|---:|---:|
| local_interpolation | 0.9419 | 4.905813 | 2.5088 | 2.3557 |
| f1_free_rollout | 0.7849 | 1.591124 | 4.4499 | 2.2941 |
| f2_old_refinement | 0.8198 | 1.519761 | 4.2916 | 2.1811 |
| graph_smooth | 0.9419 | 4.917414 | 2.5464 | 2.3573 |
| traj_target | 0.9419 | 4.905813 | 2.5088 | 2.3557 |
| traj_smooth | 0.9419 | 4.917414 | 2.5464 | 2.3573 |
| traj_full_local | 0.9419 | 4.089100 | 0.7096 | 2.3158 |
| cem_local | 0.9419 | 5.175911 | 2.3321 | 2.4485 |

Development selected `traj_full_local`. EXP_R2 is **NOT_SUPPORTED** (`SUCCESS=False`). Source-conditioned targets and stricter local radii were not sufficient to establish a joint planning advantage. F3 remained oracle; no closed-loop MPC or return claim is made.

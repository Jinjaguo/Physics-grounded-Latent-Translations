# EXP_R9 report — closed-loop latent replay surrogate

EXP_R9 converted the R8 four-waypoint path into a plan-short-prefix-observe-reencode-replan loop using complete-episode teacher-forced latent observations. This is not a physical MPC result because exact simulator snapshots are unavailable.

| method | planned arrival | observed arrival | decoded first diff | hidden path MSE | tracking latent MSE | replans |
|---|---:|---:|---:|---:|---:|---:|
| r8_open_loop | 1.0000 | 0.9329 | 1.049618 | 0.9681 | 0.9681 | 1.0 |
| proposal_h2_p1 | 0.9476 | 0.9329 | 0.914636 | 0.9112 | 0.9112 | 4.0 |
| proposal_h2_p2 | 0.9790 | 0.9329 | 0.847130 | 0.9650 | 0.9650 | 2.0 |
| proposal_h4_p1 | 0.9476 | 0.9329 | 0.914636 | 0.9112 | 0.9112 | 4.0 |
| proposal_h4_p2 | 0.9790 | 0.9329 | 0.847130 | 0.9650 | 0.9650 | 2.0 |
| warm_proposal_h4_p1 | 0.9434 | 0.9329 | 0.996743 | 1.0972 | 1.0972 | 4.0 |
| f1_closed_h4_p1 | 0.9287 | 0.9329 | 1.701111 | 1.2817 | 1.2817 | 4.0 |
| old_f2_closed_h4_p1 | 0.9329 | 0.9329 | 1.746045 | 1.2307 | 1.2307 | 4.0 |
| graph_mpc_h4_p1 | 0.9623 | 0.9329 | 2.823320 | 1.6441 | 1.6441 | 4.0 |
| cem_mpc_h4_p1 | 0.9434 | 0.9329 | 3.162941 | 1.7842 | 1.7842 | 4.0 |
| traj_mpc_h4_p1 | 0.9539 | 0.9329 | 2.920584 | 1.6465 | 1.6465 | 4.0 |

Development selected `r8_open_loop`. Closed-loop latent-surrogate claim: **NOT_SUPPORTED**. Overall full-system success remains **false** because learned F3, physical/exact simulator feedback, long-horizon sequencing, and return were not evaluated.

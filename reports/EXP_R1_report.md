# EXP_R1 report — hierarchical latent path planning

## Scientific question

Can a multi-step planner connect an oracle pre-boundary action coordinate to a train-derived target action region while preserving decoded-action continuity and empirical support? The representation, decoder, F1, and historical F2 were frozen.

## Data and protocol

The experiment used 8 H16 chunks from each real continuous Wave27 window: chunk 3 is the oracle start, chunks 4–7 are hidden evaluation path, and train post-boundary chunks define the target region. Development selection was performed before held-out evaluation. No held-out future endpoint was used to construct targets.

## Methods

Compared: linear interpolation, frozen F1 free rollout, frozen old F2 refinement, kNN/Dijkstra graph planning, five trajectory-optimization cost variants, and CEM sampling. The path metrics include target arrival, endpoint distance, hidden-path latent MSE, decoded action error, switch jump, decoded smoothness, support distance, F1 consistency, path length, curvature, and non-finite rate.

## Development summary

| method | arrival | decoded first diff | support distance | hidden path MSE |
|---|---:|---:|---:|---:|
| linear_interpolation | 1.0000 | 1.799392 | 3.2120 | 1.3985 |
| f1_free_rollout | 0.8736 | 1.362049 | 4.6319 | 2.3525 |
| f2_old_refinement | 0.9195 | 1.308608 | 4.5035 | 2.2363 |
| graph_dijkstra | 1.0000 | 4.736836 | 1.3315 | 2.3196 |
| traj_terminal | 1.0000 | 1.799392 | 3.2120 | 1.3985 |
| traj_terminal_dynamics | 1.0000 | 1.537111 | 3.4676 | 1.4431 |
| traj_terminal_continuity | 1.0000 | 1.808612 | 3.2150 | 1.4034 |
| traj_terminal_support | 1.0000 | 2.529537 | 1.1635 | 1.5782 |
| traj_full | 1.0000 | 2.266237 | 1.4633 | 1.5208 |
| cem | 1.0000 | 2.540820 | 2.6451 | 1.5398 |

## Held-out summary

| method | arrival | decoded first diff | support distance | hidden path MSE |
|---|---:|---:|---:|---:|
| linear_interpolation | 1.0000 | 2.181805 | 3.3078 | 1.6124 |
| f1_free_rollout | 0.9186 | 1.448699 | 4.5852 | 2.3133 |
| f2_old_refinement | 0.9419 | 1.356168 | 4.4000 | 2.1863 |
| graph_dijkstra | 1.0000 | 6.357655 | 1.3110 | 2.5148 |
| traj_terminal | 1.0000 | 2.181805 | 3.3078 | 1.6124 |
| traj_terminal_dynamics | 1.0000 | 1.840389 | 3.4997 | 1.6354 |
| traj_terminal_continuity | 1.0000 | 2.193445 | 3.3119 | 1.6153 |
| traj_terminal_support | 1.0000 | 3.033071 | 1.1904 | 1.8530 |
| traj_full | 1.0000 | 2.701812 | 1.5105 | 1.7878 |
| cem | 1.0000 | 3.114204 | 2.7435 | 1.7891 |

## Decision

The development-selected method was `traj_full`. EXP_R1 is **NOT_SUPPORTED** under the preregistered Pareto gate (`SUCCESS=False`). This does not upgrade F2 to MPC: only a later plan–execute-prefix–observe–replan loop may make that claim. F3 remained oracle and return was not tested.

## Interpretation

A positive result would support multi-step path structure; a negative result means the frozen coordinates do not yet provide a reliable connectable route under this offline interface. In either case the Wave28–Wave78 pointwise negative result is preserved.

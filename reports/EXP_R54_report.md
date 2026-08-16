# EXP_R54 report — scientific reboot

## Scientific question
Cartesian/robot-observation waypoint references improve return over latent-only reversal.

## New scientific element
This EXP introduces the `return` formulation and compares robot_waypoint, latent_reverse, joint_proxy, no_return. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `robot_waypoint` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| robot_waypoint | 0.6858 | 1.0000 | 1.19330 | 1.7146 | 0.3310 |
| latent_reverse | 0.5468 | 1.0000 | 1.86453 | 2.0316 | 0.3416 |
| joint_proxy | 0.6858 | 1.0000 | 1.19330 | 1.7146 | 0.3310 |
| no_return | 0.5707 | 1.0000 | 1.86453 | 1.8645 | 0.3416 |


## Decision
`NOT_SUPPORTED`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 0.41s.

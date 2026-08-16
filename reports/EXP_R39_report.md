# EXP_R39 report — scientific reboot

## Scientific question
Goal-conditioned proposal networks transfer across source action pairs.

## New scientific element
This EXP introduces the `transfer` formulation and compares global_proposal, goal_proposal, pair_proposal, nearest. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `global_proposal` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| global_proposal | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |
| goal_proposal | -0.2756 | 0.8952 | 1.77257 | 13.5953 | 8.8979 |
| pair_proposal | 0.6047 | 1.0000 | 1.72891 | 1.3208 | 0.2063 |
| nearest | 0.2760 | 0.9874 | 1.28741 | 8.7115 | 4.9478 |


## Decision
`NOT_SUPPORTED`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 15.62s.

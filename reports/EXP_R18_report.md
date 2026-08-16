# EXP_R18 report — scientific reboot

## Scientific question
A goal-conditioned local transition model predicts better paths than a global endpoint interpolator.

## New scientific element
This EXP introduces the `goal_conditioned` formulation and compares linear, ridge_goal, ridge_pair, knn_goal, knn_pair. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `linear` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| linear | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |
| ridge_goal | -0.0173 | 0.9392 | 1.93348 | 11.2499 | 6.8944 |
| ridge_pair | -0.8250 | 0.8008 | 2.66213 | 17.3210 | 11.9547 |
| knn_goal | 0.7921 | 1.0000 | 0.82505 | 0.9612 | 0.2521 |
| knn_pair | 0.7574 | 1.0000 | 1.01935 | 1.0082 | 0.2433 |


## Decision
`NOT_SUPPORTED`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 13.33s.

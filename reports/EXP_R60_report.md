# EXP_R60 report — scientific reboot

## Scientific question
A counterfactual action-prefix dataset is sufficient to identify causal latent control effects.

## New scientific element
This EXP introduces the `counterfactual` formulation and compares matched_prefix, random_prefix, goal_swap, observational. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `random_prefix` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| matched_prefix | 0.6047 | 1.0000 | 1.72891 | 1.3208 | 0.2063 |
| random_prefix | 0.9006 | 1.0000 | 0.06614 | 1.8422 | 0.2625 |
| goal_swap | 0.5677 | 1.0000 | 1.75785 | 1.1697 | 0.1986 |
| observational | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |


## Decision
`SUPPORTED_STAGE`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 2.46s.

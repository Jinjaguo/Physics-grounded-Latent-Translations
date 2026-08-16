# EXP_R50 report — scientific reboot

## Scientific question
Task-pair conditioning preserves current action stability while switching goals.

## New scientific element
This EXP introduces the `long_horizon` formulation and compares pair_conditioned, goal_only, source_only, fixed. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `fixed` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| pair_conditioned | -0.8250 | 0.8008 | 2.66213 | 17.3210 | 11.9547 |
| goal_only | -0.0173 | 0.9392 | 1.93348 | 11.2499 | 6.8944 |
| source_only | 0.5677 | 1.0000 | 1.75785 | 1.1697 | 0.1986 |
| fixed | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |


## Decision
`NOT_SUPPORTED`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 22.97s.

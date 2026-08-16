# EXP_R55 report — scientific reboot

## Scientific question
A checkpoint stack supports branch selection and return without replaying the entire trace.

## New scientific element
This EXP introduces the `return` formulation and compares stack_top, stack_best, full_reverse, no_return. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `full_reverse` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| stack_top | 0.5882 | 1.0000 | 1.68274 | 1.9356 | 0.3295 |
| stack_best | 0.6858 | 1.0000 | 1.19330 | 1.7146 | 0.3310 |
| full_reverse | 0.7889 | 1.0000 | 0.67123 | 1.5760 | 0.3905 |
| no_return | 0.5707 | 1.0000 | 1.86453 | 1.8645 | 0.3416 |


## Decision
`SUPPORTED_STAGE`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 0.39s.

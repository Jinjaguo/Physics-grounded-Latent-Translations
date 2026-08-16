# EXP_R28 report — scientific reboot

## Scientific question
Terminal-set MPC is safer than point-goal MPC when target regions are broad.

## New scientific element
This EXP introduces the `terminal_set` formulation and compares set_centroid, set_nearest, set_margin, fixed. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `set_nearest` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| set_centroid | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |
| set_nearest | 0.8900 | 1.0000 | 0.10434 | 1.9283 | 0.3056 |
| set_margin | 0.8900 | 1.0000 | 0.10434 | 1.9283 | 0.3056 |
| fixed | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |


## Decision
`SUPPORTED_STAGE`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 0.41s.

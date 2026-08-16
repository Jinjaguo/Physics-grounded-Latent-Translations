# EXP_R29 report — scientific reboot

## Scientific question
Adaptive horizon based on target distance improves both short and long transitions.

## New scientific element
This EXP introduces the `adaptive_horizon` formulation and compares horizon_2, horizon_3, horizon_4, distance_horizon. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `horizon_2` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| horizon_2 | 0.9045 | 1.0000 | 0.08039 | 1.8250 | 0.0000 |
| horizon_3 | 0.9045 | 1.0000 | 0.08039 | 1.8250 | 0.0000 |
| horizon_4 | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |
| distance_horizon | 0.9045 | 1.0000 | 0.08039 | 1.8250 | 0.0000 |


## Decision
`NOT_SUPPORTED`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 0.61s.

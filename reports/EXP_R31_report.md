# EXP_R31 report — scientific reboot

## Scientific question
A local iLQR-like tangent update improves curvature without losing endpoint identity.

## New scientific element
This EXP introduces the `tangent` formulation and compares tangent_goal, tangent_dyn, tangent_support, linear. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `tangent_goal` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| tangent_goal | 0.9045 | 1.0000 | 0.08039 | 1.8250 | 0.0000 |
| tangent_dyn | 0.9045 | 1.0000 | 0.08039 | 1.8250 | 0.0000 |
| tangent_support | 0.9045 | 1.0000 | 0.08039 | 1.8250 | 0.0000 |
| linear | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |


## Decision
`NOT_SUPPORTED`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 0.63s.

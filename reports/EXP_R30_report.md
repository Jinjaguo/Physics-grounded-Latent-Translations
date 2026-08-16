# EXP_R30 report — scientific reboot

## Scientific question
Trust-region updates prevent unstable latent jumps during retargeting.

## New scientific element
This EXP introduces the `trust_region` formulation and compares trust_small, trust_medium, trust_large, linear. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `trust_medium` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| trust_small | 0.9004 | 1.0000 | 0.06547 | 1.8428 | 0.2667 |
| trust_medium | 0.9006 | 1.0000 | 0.06552 | 1.8438 | 0.2670 |
| trust_large | 0.9005 | 1.0000 | 0.06528 | 1.8437 | 0.2672 |
| linear | 0.9005 | 1.0000 | 0.06530 | 1.8452 | 0.2673 |


## Decision
`NOT_SUPPORTED`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 7.90s.

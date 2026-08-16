# EXP_R56 report — scientific reboot

## Scientific question
Integrating F1 local prediction, F2 planning, and F3 switching yields complementary gains.

## New scientific element
This EXP introduces the `integration` formulation and compares f1_f2_f3, f1_only, f2_only, f3_only. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `f2_only` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| f1_f2_f3 | 0.9008 | 1.0000 | 0.06631 | 1.8367 | 0.2530 |
| f1_only | -0.0173 | 0.9392 | 1.93348 | 11.2499 | 6.8944 |
| f2_only | 0.9045 | 1.0000 | 0.08039 | 1.8250 | 0.0000 |
| f3_only | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |


## Decision
`NOT_SUPPORTED`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 24.21s.

# EXP_R65 report — scientific reboot

## Scientific question
Stress tests reveal whether the selected controller degrades gracefully with horizon and perturbation.

## New scientific element
This EXP introduces the `stress` formulation and compares adaptive, fixed, robust, open_loop. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `robust` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| adaptive | 0.9004 | 1.0000 | 0.06591 | 1.8442 | 0.2624 |
| fixed | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |
| robust | 0.9012 | 1.0000 | 0.06617 | 1.8245 | 0.2673 |
| open_loop | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |


## Decision
`SUPPORTED_STAGE`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 6.07s.

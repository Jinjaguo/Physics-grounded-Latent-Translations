# EXP_R32 report — scientific reboot

## Scientific question
Support critics should penalize unsupported latent regions during planning.

## New scientific element
This EXP introduces the `support` formulation and compares support_weak, support_medium, support_strong, fixed. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `support_weak` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| support_weak | 0.9014 | 1.0000 | 0.06643 | 1.8245 | 0.2672 |
| support_medium | 0.9012 | 1.0000 | 0.06615 | 1.8245 | 0.2678 |
| support_strong | 0.9011 | 1.0000 | 0.06634 | 1.8248 | 0.2676 |
| fixed | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |


## Decision
`SUPPORTED_STAGE`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 11.21s.

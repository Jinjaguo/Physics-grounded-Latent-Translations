# EXP_R53 report — scientific reboot

## Scientific question
Reversing latent waypoints recovers a previously visited state in the offline path space.

## New scientific element
This EXP introduces the `return` formulation and compares latent_reverse, action_reverse, nearest_reverse, no_return. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `nearest_reverse` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| latent_reverse | 0.5468 | 1.0000 | 1.86453 | 2.0316 | 0.3416 |
| action_reverse | 0.5882 | 1.0000 | 1.68274 | 1.9356 | 0.3295 |
| nearest_reverse | 0.7889 | 1.0000 | 0.67123 | 1.5760 | 0.3905 |
| no_return | 0.5707 | 1.0000 | 1.86453 | 1.8645 | 0.3416 |


## Decision
`SUPPORTED_STAGE`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 0.42s.

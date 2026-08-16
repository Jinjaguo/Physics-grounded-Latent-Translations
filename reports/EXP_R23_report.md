# EXP_R23 report — scientific reboot

## Scientific question
Previous/current history is more useful when used as a learned residual rather than nearest lookup.

## New scientific element
This EXP introduces the `history_residual` formulation and compares ridge_history, knn_history, residual_blend, fixed. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `fixed` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| ridge_history | -0.2756 | 0.8952 | 1.77257 | 13.5953 | 8.8979 |
| knn_history | 0.6047 | 1.0000 | 1.72891 | 1.3208 | 0.2063 |
| residual_blend | 0.2760 | 0.9874 | 1.28741 | 8.7115 | 4.9478 |
| fixed | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |


## Decision
`NOT_SUPPORTED`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 11.34s.

# EXP_R33 report — scientific reboot

## Scientific question
A learned terminal value is more informative than distance to a single endpoint.

## New scientific element
This EXP introduces the `value_terminal` formulation and compares value_knn, value_pair, value_ridge, linear. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `value_ridge` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| value_knn | 0.2760 | 0.9874 | 1.28741 | 8.7115 | 4.9478 |
| value_pair | -0.2756 | 0.8952 | 1.77257 | 13.5953 | 8.8979 |
| value_ridge | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |
| linear | 0.2760 | 0.9874 | 1.28741 | 8.7115 | 4.9478 |


## Decision
`SUPPORTED_STAGE`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 17.77s.

# EXP_R19 report — scientific reboot

## Scientific question
Wave24 magnitude loss comes from multimodal displacement cancellation.

## New scientific element
This EXP introduces the `multimodal` formulation and compares mean_knn, nearest_mode, largest_mode, low_variance_mode, pair_mode. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `low_variance_mode` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| mean_knn | 0.5677 | 1.0000 | 1.75785 | 1.1697 | 0.1986 |
| nearest_mode | 0.5677 | 1.0000 | 1.75785 | 1.1697 | 0.1986 |
| largest_mode | 0.4930 | 1.0000 | 2.10019 | 1.3587 | 0.2347 |
| low_variance_mode | 0.6816 | 1.0000 | 1.38381 | 1.1416 | 0.1505 |
| pair_mode | 0.6047 | 1.0000 | 1.72891 | 1.3208 | 0.2063 |


## Decision
`SUPPORTED_STAGE`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 0.90s.

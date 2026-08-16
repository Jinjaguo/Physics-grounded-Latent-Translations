# EXP_R35 report — scientific reboot

## Scientific question
Retrieval followed by local optimization is a stronger hybrid than either alone.

## New scientific element
This EXP introduces the `retrieval_opt` formulation and compares retrieve_then_cem, retrieve_then_graph, retrieve_then_ridge, retrieve_only. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `retrieve_then_graph` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| retrieve_then_cem | 0.9009 | 1.0000 | 0.06610 | 1.8377 | 0.2528 |
| retrieve_then_graph | 0.9045 | 1.0000 | 0.08039 | 1.8250 | 0.0000 |
| retrieve_then_ridge | 0.9045 | 1.0000 | 0.08039 | 1.8250 | 0.0000 |
| retrieve_only | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |


## Decision
`NOT_SUPPORTED`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 2.44s.

# EXP_R51 report — scientific reboot

## Scientific question
Online retargeting from the current latent is better than regenerating from the initial state.

## New scientific element
This EXP introduces the `retarget` formulation and compares blend, graph, retrieval, restart_baseline. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `graph` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| blend | 0.8214 | 1.0000 | 0.45128 | 1.9308 | 0.3077 |
| graph | 0.8876 | 1.0000 | 0.13130 | 1.9344 | 0.1544 |
| retrieval | 0.8130 | 1.0000 | 0.49793 | 1.9624 | 0.3100 |
| restart_baseline | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |


## Decision
`SUPPORTED_STAGE`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 1.15s.

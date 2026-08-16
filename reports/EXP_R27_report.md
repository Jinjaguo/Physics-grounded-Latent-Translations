# EXP_R27 report — scientific reboot

## Scientific question
A latent transition graph gives global routes that local MPC can refine.

## New scientific element
This EXP introduces the `graph` formulation and compares graph_endpoint, graph_beam, graph_local, linear. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `graph_beam` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| graph_endpoint | 0.9040 | 1.0000 | 0.06899 | 1.8425 | 0.0000 |
| graph_beam | 0.9044 | 1.0000 | 0.07855 | 1.8179 | 0.0000 |
| graph_local | 0.9040 | 1.0000 | 0.06899 | 1.8425 | 0.0000 |
| linear | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |


## Decision
`NOT_SUPPORTED`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 0.61s.

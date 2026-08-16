# EXP_R67 report — scientific reboot

## Scientific question
The reboot's best modular stack can meet all offline stage gates without overstating physical success.

## New scientific element
This EXP introduces the `adjudication` formulation and compares best_stack, best_f2, best_f3, historical_r8. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `best_stack` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| best_stack | 0.9005 | 1.0000 | 0.06608 | 1.8413 | 0.2620 |
| best_f2 | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |
| best_f3 | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |
| historical_r8 | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |


## Decision
`NOT_SUPPORTED`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 2.30s.

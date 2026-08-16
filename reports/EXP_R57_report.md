# EXP_R57 report — scientific reboot

## Scientific question
F1 is necessary for local motion stability but not for target switching.

## New scientific element
This EXP introduces the `ablation` formulation and compares with_f1, without_f1, oracle_local, fixed. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `without_f1` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| with_f1 | -0.8250 | 0.8008 | 2.66213 | 17.3210 | 11.9547 |
| without_f1 | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |
| oracle_local | -0.8250 | 0.8008 | 2.66213 | 17.3210 | 11.9547 |
| fixed | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |


## Decision
`NOT_SUPPORTED`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 10.25s.

# EXP_R59 report — scientific reboot

## Scientific question
F3 controls switching timing independently of F1/F2 path quality.

## New scientific element
This EXP introduces the `ablation` formulation and compares with_f3, without_f3, oracle_f3, fixed. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `oracle_f3` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| with_f3 | -0.8250 | 0.8008 | 2.66213 | 17.3210 | 11.9547 |
| without_f3 | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |
| oracle_f3 | 0.9006 | 1.0000 | 0.06594 | 1.8432 | 0.2625 |
| fixed | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |


## Decision
`SUPPORTED_STAGE`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 17.59s.

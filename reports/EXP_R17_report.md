# EXP_R17 report — scientific reboot

## Scientific question
Late target authority is a general phase-dependent control law, not a lucky R8 coefficient.

## New scientific element
This EXP introduces the `repair_schedule` formulation and compares fixed, linear, piecewise, sigmoid, distance, uncertainty, two_phase. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `linear` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| fixed | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |
| linear | 0.8900 | 1.0000 | 0.10434 | 1.9283 | 0.3056 |
| piecewise | 0.8784 | 1.0000 | 0.14729 | 2.0172 | 0.3496 |
| sigmoid | 0.8781 | 1.0000 | 0.14847 | 2.0191 | 0.3500 |
| distance | 0.8860 | 1.0000 | 0.12000 | 1.9636 | 0.3224 |
| uncertainty | 0.8900 | 1.0000 | 0.10434 | 1.9283 | 0.3056 |
| two_phase | 0.8757 | 1.0000 | 0.16277 | 2.0179 | 0.3423 |


## Decision
`NOT_SUPPORTED`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 0.72s.

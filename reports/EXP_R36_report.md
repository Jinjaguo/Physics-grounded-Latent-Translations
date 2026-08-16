# EXP_R36 report — scientific reboot

## Scientific question
Multi-resolution planning separates global route selection from local executable refinement.

## New scientific element
This EXP introduces the `multires` formulation and compares coarse_fine, coarse_cem, fine_only, linear. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `coarse_fine` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| coarse_fine | 0.9045 | 1.0000 | 0.08039 | 1.8250 | 0.0000 |
| coarse_cem | 0.9011 | 1.0000 | 0.06646 | 1.8382 | 0.2531 |
| fine_only | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |
| linear | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |


## Decision
`NOT_SUPPORTED`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 2.45s.

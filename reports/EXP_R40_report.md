# EXP_R40 report — scientific reboot

## Scientific question
A phase-dependent authority controller is better than one proposal repair schedule.

## New scientific element
This EXP introduces the `authority` formulation and compares phase_authority, distance_authority, confidence_authority, fixed. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `phase_authority` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| phase_authority | 0.9045 | 1.0000 | 0.08039 | 1.8250 | 0.0000 |
| distance_authority | 0.9045 | 1.0000 | 0.08039 | 1.8250 | 0.0000 |
| confidence_authority | 0.9010 | 1.0000 | 0.06614 | 1.8389 | 0.2525 |
| fixed | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |


## Decision
`SUPPORTED_STAGE`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 2.46s.

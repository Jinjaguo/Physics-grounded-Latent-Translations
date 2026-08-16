# EXP_R47 report — scientific reboot

## Scientific question
Oracle F3 plus strong F2 is sufficient for stable two-step latent execution under teacher-forced feedback.

## New scientific element
This EXP introduces the `two_step` formulation and compares r8_two_step, graph_two_step, cem_two_step, fixed. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `graph_two_step` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| r8_two_step | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |
| graph_two_step | 0.9045 | 1.0000 | 0.08039 | 1.8250 | 0.0000 |
| cem_two_step | 0.9004 | 1.0000 | 0.06621 | 1.8436 | 0.2622 |
| fixed | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |


## Decision
`SUPPORTED_STAGE`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 2.39s.

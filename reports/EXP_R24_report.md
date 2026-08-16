# EXP_R24 report — scientific reboot

## Scientific question
Multiple-shooting consistency prevents long-horizon drift better than one terminal interpolation.

## New scientific element
This EXP introduces the `multiple_shooting` formulation and compares shooting_terminal, shooting_consistency, shooting_support, linear. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `linear` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| shooting_terminal | 0.9005 | 1.0000 | 0.06574 | 1.8428 | 0.2624 |
| shooting_consistency | 0.9004 | 1.0000 | 0.06622 | 1.8416 | 0.2623 |
| shooting_support | 0.9004 | 1.0000 | 0.06596 | 1.8420 | 0.2620 |
| linear | 0.9006 | 1.0000 | 0.06608 | 1.8414 | 0.2619 |


## Decision
`NOT_SUPPORTED`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 7.93s.

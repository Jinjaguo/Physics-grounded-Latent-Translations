# EXP_R49 report — scientific reboot

## Scientific question
Three-step ordered composition reveals failure modes hidden by two-step evaluation.

## New scientific element
This EXP introduces the `long_horizon` formulation and compares replan_each, replan_two, open_loop, graph. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `replan_each` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| replan_each | 0.9045 | 1.0000 | 0.08039 | 1.8250 | 0.0000 |
| replan_two | 0.9005 | 1.0000 | 0.06616 | 1.8420 | 0.2618 |
| open_loop | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |
| graph | 0.9029 | 1.0000 | 0.07804 | 1.8068 | 0.0000 |


## Decision
`NOT_SUPPORTED`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 2.52s.

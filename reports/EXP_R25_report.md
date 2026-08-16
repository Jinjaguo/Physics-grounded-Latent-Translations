# EXP_R25 report — scientific reboot

## Scientific question
CEM over latent waypoint sequences can trade endpoint arrival for executable continuity.

## New scientific element
This EXP introduces the `cem` formulation and compares cem_terminal, cem_balanced, cem_support, linear. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `cem_balanced` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| cem_terminal | 0.9006 | 1.0000 | 0.06604 | 1.8412 | 0.2617 |
| cem_balanced | 0.9006 | 1.0000 | 0.06584 | 1.8428 | 0.2623 |
| cem_support | 0.9005 | 1.0000 | 0.06600 | 1.8422 | 0.2617 |
| linear | 0.9006 | 1.0000 | 0.06610 | 1.8418 | 0.2622 |


## Decision
`NOT_SUPPORTED`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 7.88s.

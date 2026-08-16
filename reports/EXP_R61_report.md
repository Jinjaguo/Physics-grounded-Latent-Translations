# EXP_R61 report — scientific reboot

## Scientific question
Matched-current-state goal swaps separate language redirection from state mismatch.

## New scientific element
This EXP introduces the `counterfactual` formulation and compares matched_swap, unmatched_swap, goal_shuffle, same_goal. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `matched_swap` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| matched_swap | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |
| unmatched_swap | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |
| goal_shuffle | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |
| same_goal | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |


## Decision
`NOT_SUPPORTED`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 0.40s.

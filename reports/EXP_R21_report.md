# EXP_R21 report — scientific reboot

## Scientific question
A state-conditioned mixture selector beats a fixed transition mode.

## New scientific element
This EXP introduces the `mixture_selector` formulation and compares nearest, pair_nearest, distance_selector, mode_margin. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `nearest` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| nearest | 0.5677 | 1.0000 | 1.75785 | 1.1697 | 0.1986 |
| pair_nearest | 0.5677 | 1.0000 | 1.75785 | 1.1697 | 0.1986 |
| distance_selector | 0.5677 | 1.0000 | 1.75785 | 1.1697 | 0.1986 |
| mode_margin | 0.4930 | 1.0000 | 2.10019 | 1.3587 | 0.2347 |


## Decision
`SUPPORTED_STAGE`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 0.72s.

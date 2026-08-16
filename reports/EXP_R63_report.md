# EXP_R63 report — scientific reboot

## Scientific question
Continuous stochastic transition noise is better modeled by a conditional flow-like sampler than a discrete mode.

## New scientific element
This EXP introduces the `distributional` formulation and compares gaussian_sampler, quantile_sampler, mode_sampler, mean. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `mean` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| gaussian_sampler | 0.5677 | 1.0000 | 1.75785 | 1.1697 | 0.1986 |
| quantile_sampler | 0.5677 | 1.0000 | 1.75785 | 1.1697 | 0.1986 |
| mode_sampler | 0.5240 | 1.0000 | 2.02845 | 1.3140 | 0.2253 |
| mean | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |


## Decision
`SUPPORTED_STAGE`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 0.66s.

# EXP_R38 report — scientific reboot

## Scientific question
Risk-sensitive terminal costs select paths with lower transition variance.

## New scientific element
This EXP introduces the `risk` formulation and compares risk_mean, risk_cvar, risk_pair, linear. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `risk_mean` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| risk_mean | 0.9045 | 1.0000 | 0.08039 | 1.8250 | 0.0000 |
| risk_cvar | 0.9008 | 1.0000 | 0.06625 | 1.8367 | 0.2530 |
| risk_pair | 0.9045 | 1.0000 | 0.08039 | 1.8250 | 0.0000 |
| linear | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |


## Decision
`NOT_SUPPORTED`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 2.47s.

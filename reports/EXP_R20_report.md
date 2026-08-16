# EXP_R20 report — scientific reboot

## Scientific question
Bootstrapped transition ensembles expose useful epistemic uncertainty for controller selection.

## New scientific element
This EXP introduces the `ensemble` formulation and compares ensemble_mean, ensemble_lowvar, ensemble_worstcase, nearest. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `ensemble_worstcase` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| ensemble_mean | 0.8048 | 1.0000 | 0.76755 | 0.9320 | 0.2274 |
| ensemble_lowvar | 0.5759 | 1.0000 | 1.73182 | 1.3804 | 0.1860 |
| ensemble_worstcase | 0.8638 | 1.0000 | 0.52038 | 0.8539 | 0.2572 |
| nearest | 0.8048 | 1.0000 | 0.76755 | 0.9320 | 0.2274 |


## Decision
`SUPPORTED_STAGE`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 7.31s.

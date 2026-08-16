# EXP_R62 report — scientific reboot

## Scientific question
A causal benchmark with action-conditioned synthetic feedback can rank planners before robot collection.

## New scientific element
This EXP introduces the `causal_benchmark` formulation and compares compliance_plant, history_plant, shock_plant, teacher_forced. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `shock_plant` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| compliance_plant | 0.8860 | 1.0000 | 0.12000 | 1.9636 | 0.3224 |
| history_plant | -0.8250 | 0.8008 | 2.66213 | 17.3210 | 11.9547 |
| shock_plant | 0.9013 | 1.0000 | 0.06635 | 1.8238 | 0.2678 |
| teacher_forced | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |


## Decision
`SUPPORTED_STAGE`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 23.29s.

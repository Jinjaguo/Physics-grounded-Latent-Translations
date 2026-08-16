# EXP_R48 report — scientific reboot

## Scientific question
Learned F3 can replace oracle boundaries when F2 target arrival is already reliable.

## New scientific element
This EXP introduces the `learned_f3` formulation and compares hazard_switch, distance_switch, oracle_switch, fixed. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `oracle_switch` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| hazard_switch | 0.6047 | 1.0000 | 1.72891 | 1.3208 | 0.2063 |
| distance_switch | 0.8860 | 1.0000 | 0.12000 | 1.9636 | 0.3224 |
| oracle_switch | 0.9045 | 1.0000 | 0.08039 | 1.8250 | 0.0000 |
| fixed | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |


## Decision
`SUPPORTED_STAGE`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 0.53s.

# EXP_R46 report — scientific reboot

## Scientific question
Calibrated F3 confidence should gate target authority continuously rather than hard switch.

## New scientific element
This EXP introduces the `f3_control` formulation and compares confidence_gate, linear_gate, hard_gate, fixed. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `confidence_gate` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| confidence_gate | 0.9032 | 1.0000 | 0.05409 | 1.8341 | 0.2594 |
| linear_gate | 0.8900 | 1.0000 | 0.10434 | 1.9283 | 0.3056 |
| hard_gate | 0.8757 | 1.0000 | 0.16277 | 2.0179 | 0.3423 |
| fixed | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |


## Decision
`SUPPORTED_STAGE`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 0.40s.

# EXP_R26 report — scientific reboot

## Scientific question
MPPI-style cost-weighted averaging is less brittle than elite-only CEM.

## New scientific element
This EXP introduces the `mppi` formulation and compares mppi, mppi_low_temp, mppi_support, cem. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `mppi_support` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| mppi | 0.9009 | 1.0000 | 0.06636 | 1.8380 | 0.2527 |
| mppi_low_temp | 0.9010 | 1.0000 | 0.06632 | 1.8385 | 0.2530 |
| mppi_support | 0.9011 | 1.0000 | 0.06651 | 1.8391 | 0.2527 |
| cem | 0.9008 | 1.0000 | 0.06629 | 1.8383 | 0.2526 |


## Decision
`NOT_SUPPORTED`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 7.96s.

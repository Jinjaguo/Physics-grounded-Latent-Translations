# EXP_R37 report — scientific reboot

## Scientific question
Tube-style robust MPC improves worst-case arrival under latent perturbations.

## New scientific element
This EXP introduces the `tube` formulation and compares tube, tube_tight, tube_loose, fixed. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `tube_tight` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| tube | 0.9013 | 1.0000 | 0.06639 | 1.8226 | 0.2672 |
| tube_tight | 0.9013 | 1.0000 | 0.06648 | 1.8233 | 0.2676 |
| tube_loose | 0.9013 | 1.0000 | 0.06650 | 1.8240 | 0.2673 |
| fixed | 0.9012 | 1.0000 | 0.06633 | 1.8238 | 0.2676 |


## Decision
`NOT_SUPPORTED`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 14.78s.

# EXP_R66 report — scientific reboot

## Scientific question
A prospective collection protocol with complete checkpoints is the shortest path to physical F2-MPC validation.

## New scientific element
This EXP introduces the `prospective` formulation and compares full_state_protocol, minimal_state_protocol, branch_protocol, current_archive. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `full_state_protocol` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| full_state_protocol | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |
| minimal_state_protocol | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |
| branch_protocol | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |
| current_archive | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |


## Decision
`NOT_SUPPORTED`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 0.42s.

# EXP_R42 report — scientific reboot

## Scientific question
Temporal history improves F3 completion classification beyond a single latent pair.

## New scientific element
This EXP introduces the `f3` formulation and compares history_linear, hazard, change_point. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `history_linear` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| history_linear | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |
| hazard | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |
| change_point | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |


## F3/stage metrics
{
  "heldout_count": 2862.0,
  "balanced_accuracy": 0.6495457721872816,
  "early_switch_rate": 0.4947589098532495,
  "late_miss_rate": 0.3473095737246681,
  "auroc_proxy": 0.7092829680807478
}

## Decision
`NOT_SUPPORTED`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 0.33s.

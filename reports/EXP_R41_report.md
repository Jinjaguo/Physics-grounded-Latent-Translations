# EXP_R41 report — scientific reboot

## Scientific question
Completion can be detected from a simple explicit progress signal before learned sequence models.

## New scientific element
This EXP introduces the `f3` formulation and compares distance, hazard, change_point. It is not an interface audit and does not reuse the previous gate as an experiment.

## Data and frozen components
The benchmark uses 864 episode-disjoint latent windows (train=206, development=181, held-out=477). Representation, decoder, F1, historical F2, and R8 are frozen; train target regions are never built from held-out futures.

## Development selection
Selected `distance` before opening held-out.

## Held-out results
| method | dev score | heldout arrival | heldout continuity | hidden MSE | support |
|---|---:|---:|---:|---:|---:|
| distance | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |
| hazard | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |
| change_point | 0.8772 | 1.0000 | 0.15495 | 2.0147 | 0.3492 |


## F3/stage metrics
{
  "heldout_count": 2862.0,
  "balanced_accuracy": 0.4416491963661775,
  "early_switch_rate": 0.283717679944095,
  "late_miss_rate": 0.7833682739343116,
  "auroc_proxy": 0.41614036012991745
}

## Decision
`NOT_SUPPORTED`. This is a stage result only; overall hierarchical physical closed-loop success remains false. Remaining bottleneck: physical causal feedback, learned F3 integration, and recoverable controller checkpoints. Runtime: 0.30s.

# EXP_G29 Report: Conservative Action-Effect Coordinates

## Conclusion

**NOT SUPPORTED.** A compact bottleneck supervised by realized intervention effects improved held-out ranking, and it made 650 genuine nonzero-candidate interventions, but it did not improve pooled closed-loop control. The conservative action-effect coordinate completed 23/30 ordered tasks, compared with 24/30 for the equal-structure raw-effect model, 24/30 for the direct non-bottleneck model, 25/30 for the G28 aligned point ranker, and 25/30 for physical shooting. It beat the unusually weak single-proposal realization (22/30), but failed every stronger part of the declared decision rule.

## Models and frozen gate selection

G29 trained 90 models: three-member ensembles for effect-coordinate, raw-effect, and direct families in each of ten leave-one-attempt-out folds. The effect coordinate was an explicit eight-dimensional prediction of realized end-effector displacement, object displacement, utility and success. Its loss combined effect regression, groupwise distance preservation, utility regression, and best-candidate classification. The raw-effect family used the same bottleneck and losses with raw action features; the direct family removed effect supervision and used a 64-D hidden scorer.

Held-out ranking accuracy was 0.7526 for the effect coordinate, 0.7349 for raw effect, and 0.7353 for direct scoring. Cross-fitted gate sweeps selected `(beta=1.0, threshold=0.0)` for both effect families and `(beta=0.0, threshold=0.05)` for direct scoring. These choices and all checkpoints were frozen before prospective rollouts.

## Prospective results

G29 executed 240 new simulator rollouts under a new method-neutral explicit-noise schedule: three repeats, ten canonical starts, and eight methods. All learned choices committed exact native pi0.5 action bytes and replanned from realized simulator observations.

| Method | Ordered success | Per-repeat | Endpoint error | Jerk | Nonzero interventions | Candidate steps |
|---|---:|---:|---:|---:|---:|---:|
| conservative effect coordinate | 23/30 | 7, 7, 9 | 0.07387 | **0.07979** | 650/1,249 | 0 |
| always-rank effect coordinate | 24/30 | 8, 8, 8 | 0.06099 | 0.08026 | 802/1,210 | 0 |
| conservative raw effect | 24/30 | 8, 8, 8 | 0.06452 | 0.08437 | 587/1,215 | 0 |
| conservative direct scorer | 24/30 | 9, 7, 8 | 0.07853 | 0.08589 | 253/1,260 | 0 |
| G28 aligned point | **25/30** | 9, 8, 8 | 0.08365 | 0.08355 | n/a | 0 |
| physical shooting | **25/30** | 9, 9, 7 | **0.05413** | 0.08399 | n/a | 18,812 |
| single raw pi0.5 | 22/30 | 8, 7, 7 | 0.08163 | 0.08362 | n/a | 0 |
| initial-observation open loop | 0/30 | 0, 0, 0 | 1.06108 | 0.11294 | n/a | 0 |

The effect method had 2 paired wins and 1 loss against single pi0.5, showing that learned intervention sometimes repaired a baseline failure. However, it had 1 win/2 losses against raw effect, 2/3 against direct, 3/5 against G28 point, and 2/4 against physical shooting. Conservative gating itself was not beneficial: always ranking produced one more pooled success. The effect bottleneck's better offline ranking therefore remains a diagnostic result, not a causal control advantage.

## Audit and artifacts

The independent audit passed with zero failures. It checked all 90 ensemble members and their 50-epoch logs, recomputed 1,242 held-out predictions and all three gate sweeps, rebuilt 4,934 live effect decisions plus 1,210 G28 point decisions, regenerated 28,428 explicit noise seeds, verified 8,711 native commits and all 240 rollout chains, and reconstructed pooled, per-repeat, paired, intervention, and acceptance metrics.

- `experiments/EXP_G29/fold_models/`, `model_selection.json`, and `common_noise_manifest.json`
- `experiments/EXP_G29/rollouts/`, `case_metrics.jsonl`, and `metrics.json`
- `experiments/EXP_G29/audit.json`
- `scripts/experiments/run_exp_g29_conservative_effect_coordinates.py`
- `scripts/experiments/audit_exp_g29.py`

The post-EXP disk check left 849 GB free.

## Consequence for the system

G24-G29 repeatedly show that using a latent only to score a finite native proposal set does not establish control-relevant value. G30 will change the intervention itself: learn an action autoencoder with an effect-conditioned coordinate, optimize locally in that coordinate, decode a new executable action prefix, and compare it against matched raw-action optimization. This tests whether the latent can serve as a genuine continuous control coordinate rather than another selector feature.

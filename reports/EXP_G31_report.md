# EXP_G31 Report: Latent-Verified Checkpoint Recovery

## Conclusion

**NOT SUPPORTED for the latent hypothesis; SUPPORTED for adaptive checkpoint recovery as a non-latent mechanism.** The latent verifier completed 22/30 ordered tasks. Raw-visual, physical-state and shuffled-latent adaptive verifiers each completed 25/30, while exhaustive physical shooting completed 26/30. The exact action/effect correspondence was therefore unnecessary and harmful in this formulation. However, sequential trial, exact restore and retry often matched strong success with substantially fewer rejected simulator steps than exhaustive shooting.

## Verifiers and frozen thresholds

G31 trained forty equal-capacity leave-one-attempt-out verifiers from all 1,242 G24 causal branches. Latent features contained current physical state, native-action latent, realized aligned visual effect, their residual, and realized physical displacement. Raw-visual used raw embedding difference plus native actions; physical used physical displacement plus native actions; shuffled latent rotated action latents within each three-candidate group.

Held-out ranking accuracy was 0.6524 for latent, 0.6485 for raw visual, 0.6825 for physical, and **0.6925 for shuffled latent**. Cross-fitted sequential retry selected thresholds 0.5, 0.5, 0.0 and 0.0 respectively. Shuffled latent also had the best held-out sequential utility. These facts already argued against correspondence before the prospective test.

## Adaptive causal execution

At every decision the controller captured the full MuJoCo, environment, controller and observable state, requested three common-noise native proposals, and tried them in order. An accepted trial remained the actual current simulator state and was not re-executed. A rejected trial triggered exact restoration before the next candidate. Candidate 2 was a guaranteed fallback. Every trial stored before/after pixels, embeddings, physical changes, verifier features/scores, accepted flags, checkpoint integration state, controller state and restore error.

The first `EXP_G31` directory contains a preserved pre-rollout implementation failure caused by an invalid initializer unpack. `EXP_G31_retry1` is the formal artifact; it retrained deterministically and completed all causal trials.

| Method | Ordered success | Per-repeat | Rejected candidate steps | Endpoint error | Jerk |
|---|---:|---:|---:|---:|---:|
| latent verified recovery | 22/30 | 8, 7, 7 | 10,035 | 0.08480 | 0.08547 |
| raw-visual verified recovery | 25/30 | 8, 9, 8 | 10,385 | 0.07848 | 0.08341 |
| physical verified recovery | 25/30 | 9, 8, 8 | 6,440 | 0.06667 | 0.08619 |
| shuffled-latent recovery | 25/30 | 9, 7, 9 | **6,155** | 0.06636 | 0.08300 |
| exhaustive physical shooting | **26/30** | 10, 8, 8 | 18,013 | 0.06519 | 0.07912 |
| G28 aligned point | 24/30 | 9, 8, 7 | 0 | 0.06392 | **0.07908** |
| single raw pi0.5 | 24/30 | 8, 8, 8 | 0 | **0.05987** | 0.08087 |
| initial-observation open loop | 0/30 | 0, 0, 0 | 0 | 1.08639 | 0.11306 |

The latent verifier had zero paired wins and three losses against each of raw and physical recovery; it had one win and four losses against shuffled latent, and zero wins/four losses against exhaustive shooting. By contrast, physical recovery retained 25/26 of exhaustive shooting's successes while using only 35.8% as many rejected branch steps. On repeat 2, raw recovery uniquely solved attempt 44 while exhaustive shooting, point and single all failed. The recovery mechanism is therefore useful, but the learned action latent is not its cause.

## Audit and artifacts

The independent audit passed after correcting an audit-only prefix-length assumption for terminal success. It checked 40 verifier checkpoints, recomputed 1,656 held-out family-group decisions and four threshold sweeps, re-encoded and rescored 11,755 saved trial image pairs, verified 11,755 exact restores with maximum integration-state error 0, regenerated 29,095 explicit-noise seeds, verified 8,923 committed native prefixes and all 240 rollout chains, and rebuilt all aggregate/recovery/paired metrics.

- `experiments/EXP_G31/` — preserved failed pre-rollout attempt
- `experiments/EXP_G31_retry1/fold_models/`, `model_selection.json`, and `common_noise_manifest.json`
- `experiments/EXP_G31_retry1/rollouts/`, `case_metrics.jsonl`, `metrics.json`, and `audit.json`
- `scripts/experiments/run_exp_g31_latent_verified_recovery.py`
- `scripts/experiments/audit_exp_g31.py`
- adaptive extension in `scripts/experiments/run_exp_g14_outcome_latent.py`

The post-EXP disk check left 848 GB free.

## Consequence for the system

G31 establishes a useful causal primitive—efficient exact rollback after observed bad progress—but again falsifies the latent contribution. G32 will simplify this primitive by replacing learned verifier representations with directly measured realized task utility and will compare sequential stopping against exhaustive best-of-three. This determines the strongest defensible F2 skeleton before testing any separate latent memory role under perturbation.

# Next Experiment from EXP_G1: EXP_G2 Causal F2 Isolation with Oracle Completion

## Hypothesis

A receding-horizon F2 controller that evaluates executable proposals using the realized LIBERO state will improve physical progress and continuation quality over open-loop decoded F2, F1-only, copy/hold, and direct state/action baselines. The hypothesis is about executed control under oracle completion boundaries, not offline latent prediction.

## Required implementation

Implement EXP_G2 as a new script and new `experiments/EXP_G2/` directory. Reuse the exact `LiberoSnapshot` restore interface validated by EXP_G1. Load the frozen independent LIBERO representation, semantic predictor, F1, and F2 checkpoints recorded in `results/dynamics/twentieth_wave/2026-08-14_dynamics_8/wave20_frozen_model_manifest.json`; inspect the real checkpoint payloads and `train_wave20_dynamics.py` interfaces rather than guessing fields.

Use only train episodes for any fitting/calibration and development episodes for method selection. Do not open the frozen test split in this EXP. Select at least one certified checkpoint per LIBERO-10 task and retain the episode/task stratification.

At each decision boundary, form executable 16-step action chunks from genuinely different control families: frozen F1 decode, frozen F2-refined decode, current-latent copy decode, direct action continuation, hold/damped controls, and a norm-matched non-learned proposal. Include open-loop F1/F2 variants that commit to their decoded chunk and closed-loop variants that execute only a short prefix before replanning.

For the causal F2 variant, execute candidate prefixes from identical restored current checkpoints, observe their realized physical states, and rank them with an explicitly declared development-only objective based on current-action physical progress, continuity, support, and constraint satisfaction. After selection, restore the current checkpoint, execute the selected prefix as the committed intervention, capture the resulting observation/state, update the rolling executed-action latent, create a new full checkpoint, and replan. The committed trajectory must therefore follow `proposal -> candidate execution -> realized feedback -> selection -> committed execution -> observation -> re-encoding -> replanning`.

Use the recorded source continuation and terminal physical trajectory only to define an oracle current-action completion boundary and development target/progress measurements. Keep source controls as an upper/reference baseline; do not place future source controls inside the deployable learned candidate set. If sparse task success makes the ranking objective unusable at early branches, derive target object/eef state from train/development source continuations and state exactly which fields are oracle. This is allowed only for isolating F2; it cannot support autonomous F3 claims.

## Evaluation

Report, per method and per task:

- realized physical progress toward the current-action target;
- official task predicate success when reachable within the rollout horizon;
- endpoint object/eef error to the oracle source completion state;
- continuity and action jerk;
- constraint violations and nonfinite/unstable rollouts;
- latent support distance and representation-dependent selection behavior;
- intervention count, executed steps, and runtime.

Compare open-loop versus receding-horizon use of the same F1/F2 proposal model, F2 versus F1, causal candidate evaluation versus selection without realized feedback, and the best latent controller versus direct state/action selection. Use matched checkpoints and report episode/task-stratified aggregates. Development tuning must remain inside this EXP and must not consume another ID.

## Required artifacts

Save exact run metadata and environment versions, selected checkpoint manifest, train/development definitions, calibration records, model-loading records, every candidate intervention log, every committed rollout with actions and realized feedback, rolling latents, per-case metrics, aggregate metrics, runtime, and an independent audit. Every winning comparison in the report must be recomputable from these files.

## Decision rule

`SUPPORTED` requires the causal receding F2 controller to outperform both its open-loop F2 counterpart and F1-only on the preregistered primary realized physical-progress metric without increasing instability, and to beat or add measurable value beyond the strongest direct state/action baseline. If F2 loses, report `NOT_SUPPORTED` and preserve the negative result. EXP_G3 must then pivot to the empirically strongest planner family or simplify/remove F2; it must not spend a new ID on scalar retuning.

Regardless of outcome, EXP_G2 is complete only after actual simulator/controller interventions produce realized feedback and audited metrics. Interface or model-loading failures remain debugging work inside EXP_G2.

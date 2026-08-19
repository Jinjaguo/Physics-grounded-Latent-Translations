# Next Experiment from EXP_G25: EXP_G26 Uncertainty-Supported Transition Coordinates

## Hypothesis

G24 showed that visual prediction can rank native proposals, while G25 showed that recurrent latent history is not useful. Both experiments use point predictors and can select actions in regions where a model is confidently wrong. The new hypothesis is that an aligned action/visual-transition coordinate becomes control-relevant when it represents epistemic support across independently bootstrapped outcome models: penalizing disagreement in the learned transition coordinate should reject model-exploiting native proposals and improve ordered success beyond a capacity-matched uncertainty ensemble in direct state/action space.

This changes both model family and control objective. It is not another recurrent setting. Native pi0.5 actions remain untouched; the coordinate affects control only through ensemble support and risk-sensitive candidate selection.

## Bootstrap models and protected tuning

Use the 1,242 audited G24 branches and ten leave-one-attempt-out splits. Within each training fold, fit at least five bootstrap members for each of two equal-budget families:

1. aligned transition-coordinate models predicting visual-coordinate change, physical displacement, progress and success;
2. direct visual/state/action models predicting the same physical progress and success targets without an action bottleneck.

Each bootstrap member must receive a group-level resample so all candidates from a matched decision stay together. Compute ensemble mean progress, epistemic variance in progress and physical outcome, and latent visual-transition disagreement where applicable. Choose uncertainty weights and any adaptive proposal-budget rule only from nested training-fold validation through a systematic sweep inside G26. Freeze one rule before prospective outcomes. Save bootstrap membership, checkpoints, calibration curves, coverage versus error, held-out risk-ranking accuracy and regret.

## Mechanism comparisons

The prospective controllers must compare:

1. aligned latent ensemble with the frozen uncertainty/support penalty;
2. equal-member direct visual/action uncertainty ensemble;
3. aligned ensemble mean with uncertainty penalty removed;
4. aligned ensemble with disagreement shuffled across candidates;
5. the best G25 raw-history GRU;
6. the G24/G25 memoryless aligned point model;
7. physical matched-branch shooting;
8. single raw pi0.5;
9. initial-observation open loop.

If nested validation supports it, both uncertainty ensembles may use the same adaptive native-proposal budget, for example requesting more proposals only when all initial candidates have high epistemic risk. Proposal cost and inference time must be reported. No ensemble may decode, perturb, or interpolate action bytes.

## Prospective causal protocol

Run fresh full `lift -> place` rollouts from all ten canonical perturbed snapshots with autonomous state-window F3. At every decision save current dual-view observations, all exact native proposals, every ensemble member's predicted transition/progress/success, mean and disagreement terms, support penalty, selected index, committed action bytes, realized next observation/state, prediction error, and F3 state. The next decision must consume the physically realized feedback.

## Decision rule

The uncertainty-supported transition latent contributes only if it strictly exceeds the direct uncertainty ensemble, aligned mean-only and shuffled-disagreement ablations, raw-history GRU, memoryless aligned point model, and single raw pi0.5 in ordered success. It must match or beat physical shooting with fewer executed candidate steps, or strictly beat it. Ties do not support the latent claim. Calibration or lower prediction error cannot substitute for closed-loop success. If the direct ensemble wins or ties, keep the non-latent controller and move G27 to a different representation or control role rather than retuning uncertainty coefficients as a new EXP.

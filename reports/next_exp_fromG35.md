# Next Experiment from EXP_G35: EXP_G36 Causal Physical-Effect Action Coordinate

## Hypothesis

G35 showed robust three-stage execution but falsified the shared phase-coordinate F3: state F3 was stronger, while removing physical-feedback F2 caused a 25/30 to 7/30 collapse. G36 therefore moves the action latent to the empirically causal bottleneck. A compact physical-effect coordinate jointly encoding a proposed native action and its realized end-effector/object displacement can detect intervention-induced model violations and select recoverable branches at least as reliably as the direct six-dimensional physical residual, while improving strict three-stage success or recovery cost.

## New representation and training

Use only train/development causal branches from the audited G24 dataset. For each checkpoint, pair the current physical state and 35-D native five-action prefix with the realized six-dimensional end-effector/book displacement. Train and save several genuinely distinct mechanisms within G36:

1. a nonlinear action-effect encoder producing a six-dimensional coordinate, plus a decoder back to physical displacement;
2. a realized-effect encoder mapping observed displacement into the same coordinate, trained with alignment, decoded displacement, and contrastive negative-pair losses;
3. a predictive bottleneck without contrastive alignment;
4. a shuffled action/outcome pairing ablation;
5. G33's direct physical displacement predictor as the fixed non-latent baseline.

Use leave-one-training-attempt-out fitting and residual calibration. Select latent dimension, loss weights, and residual threshold within this EXP from train/development evidence; do not assign new EXP IDs to scalar sweeps. Freeze all models and thresholds before prospective test starts.

## Causal three-stage evaluation

Fix G35's state-binary autonomous F3, frozen prompts, two-switch executor, canonical test starts, common explicit π0.5 noise, and two real 3 cm book interventions. Compare at least 210 prospective rollouts across three repeats:

1. aligned physical-effect latent recovery;
2. predictive-bottleneck latent recovery;
3. shuffled-pair latent ablation;
4. direct physical residual recovery;
5. oracle disturbance rollback;
6. always rollback;
7. no recovery;
8. a clean no-intervention control if resources permit, yielding 240 rollouts.

Every verifier must execute the candidate from an exact checkpoint, observe the physically realized next state, encode the realized effect, accept or restore/retry, and then continue from the accepted state. Save action-coordinate inputs, predicted/realized coordinates, decoded displacement, residual, threshold, perturbation labels, exact restore errors, committed actions, state-F3 switches, grasp/lift retention, and final task metrics.

## Decision rule

The physical-effect action latent contributes meaningfully only if it has 60/60 disturbance recall, a false-trigger rate no worse than direct physical residual, strictly exceeds direct physical recovery in pooled strict three-stage success with positive paired balance, uses no more rejected candidate steps, and strictly exceeds its shuffled and no-recovery ablations. A tie with direct physical recovery is not support.

If the direct six-dimensional baseline matches or wins, the action-latent requirement remains unsupported and the final architecture should retain direct physical F2. G37 must then test a different source of latent value—multi-task transfer or task-agnostic compression across genuinely different object/action families—rather than another task-5 residual variant. If the effect latent wins, G37 will integrate it with state F3 and run open-loop, restart, feedback, language-gating, and coordinate-decoder ablations toward the final acceptance test.

## Required evidence

Save dataset and split manifests, exact source branch membership, model checkpoints and logs, cross-fitted predictions, threshold sweeps, frozen protocol, all prospective causal rollouts, predicted/realized latent and decoded physical effects, interventions, exact restores, explicit noise seeds, native commits, strict three-stage and detection metrics, paired comparisons, runtime metadata, and an independent audit that recomputes every representation output, verifier decision, action chain, and conclusion.

# Next Experiment from EXP_G3: EXP_G4 State-Conditioned Residual Imitation

## Hypothesis

EXP_G3 failed because the action-only latent and myopic physical score did not identify contact/manipulation phase. A policy conditioned on the realized robot/object state can recover the demonstrated control sequence in closed loop; augmenting it with the frozen action latent will further improve generalization beyond a state-only policy.

## Training data and model families

Use only the frozen Wave19 train episodes for fitting. For each task, load the saved per-step `robot_states.npz`, `object_states.npz`, and `actions.npy`. Build a causal feature from current robot joints/velocities, end-effector pose, gripper state, task-local body positions/velocities, previous executed action, and normalized episode progress. Fit and compare inside one EXP:

1. nearest-state retrieval with the same physical feature and train action labels;
2. a state-only residual MLP predicting the next executable action;
3. the same residual MLP augmented with the frozen rolling 16-step action latent;
4. a phase-conditioned retrieval/MLP variant if train-only development calibration shows state aliasing;
5. the EXP_G3 receding F1 and causal-state winners as frozen baselines.

Use train episodes for fitting and a protected subset of development episodes for model/hyperparameter selection. Treat coefficient/architecture choices as one internal sweep. Do not open the test split. Save checkpoints, normalization, training logs, split membership, and development selection records.

## Causal rollout

Evaluate from the same latest certified development checkpoints and complete remaining horizons used in EXP_G3. At every controller step or short prefix, use the newly realized physical state to recompute the policy action. Do not teacher-force state, phase, latent, or actions after rollout begins. Stop only at the official oracle completion predicate, environment termination, or the registered horizon.

All methods must execute through LIBERO and save actions, physical states, observations, rolling latents, completion state, and runtime. Use the recorded source only as an upper/reference baseline and for oracle F3, never as online policy input.

## Primary decision

Primary metric is official current-action success rate, with failure target error, progress, completion time, stability, jerk, and task breakdown secondary. The state-conditioned family is supported only if it improves success above the 0/10 EXP_G3 deployable result. The latent contribution is supported only if state-plus-latent beats the matched state-only architecture under the same training/evaluation protocol. If state-only wins, simplify the system and treat the latent result as negative. If all fitted policies fail, EXP_G5 must move to a stronger sequence/vision-conditioned policy or online data aggregation rather than retuning this MLP.

## Required artifacts

Create `experiments/EXP_G4/` with exact data membership, feature definitions, normalization, train/development records, fitted checkpoints, training logs, every causal rollout, per-case/aggregate metrics, and an independent audit. Then create the detailed G4 report and executable next-experiment prompt. Missing state fields or dimensional differences are implementation work inside G4, not a stopping condition.

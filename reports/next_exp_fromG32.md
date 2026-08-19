# Next Experiment from EXP_G32: EXP_G33 Action-Coordinate Disturbance Memory

## Hypothesis

G32 shows that transparent rollback can recover from a bad executed branch, but on clean starts the zero-trial controller is equally successful and cheaper. The next bottleneck is therefore deciding **when** rollback is warranted. The hypothesis is that an action-conditioned aligned latent transition provides a useful execution-memory residual: after an external object displacement, the realized visual transition will disagree with the transition predicted from the current action coordinate strongly enough to trigger rollback more selectively and successfully than equal-protocol raw-action visual or physical-transition residuals.

## New mechanism and calibration

Add an optional pre-candidate simulator intervention hook to the audited G14 causal rollout path. It must run only after exact restoration and before execution of a candidate, so a rejected disturbed trial can be removed by the next exact restore. The intervention shifts the actual black-book free joint in XY, zeros its free-joint velocity, advances MuJoCo kinematics without an unlogged policy action, regenerates camera observations, and logs pre/post integration state, object pose, pixels, requested offset, decision, candidate, and active stage.

Use the audited G24 action-conditioned transition dataset to calibrate leave-one-attempt-out clean residual thresholds for three genuinely different detectors:

1. aligned action-latent visual residual: realized dual-view latent change minus the aligned predictor's action-coordinate prediction;
2. raw-action visual residual: the same visual transition predicted directly from the native action prefix without the action bottleneck;
3. raw-action physical residual: realized end-effector/book displacement minus a direct state/action predictor.

Perform the entire clean quantile sweep inside G33 and freeze one threshold per held-out attempt and detector before perturbed rollouts. Preserve the learned action coordinates and all predicted/realized residual components so the latent intervention can be audited rather than inferred from a label.

## Prospective causal intervention

Use three explicit-noise repeats over the ten canonical starts, autonomous F3, raw π0.5 proposal order, three candidates, and five-step replanning. Inject two deterministic, attempt-balanced XY disturbances into candidate zero at predeclared decision indices during early and later execution. Compare 240 paired rollouts:

1. aligned action-latent residual-triggered recovery;
2. raw-action visual residual-triggered recovery;
3. raw-action physical residual-triggered recovery;
4. oracle disturbance-triggered recovery;
5. always reject the first candidate;
6. perturbed execution with no recovery;
7. clean execution with the same adaptive machinery but no intervention;
8. initial-observation open loop under the same perturbation schedule if the fixed-action hook can preserve complete intervention logging; otherwise use a native single-proposal perturbed control and retain the already established open-loop failure as context, not as new evidence.

For every adaptive decision, execute the candidate in the real simulator, observe the resulting pixels and physical state, compute the detector score, accept the already reached state or restore the exact full checkpoint, and replan from the realized state. A simulator perturbation, detector score, accepted flag, restoration, committed native action, and final metric must all be linked in one rollout artifact.

## Evaluation and decision rule

Report detector false-trigger rate on nonintervened decisions, perturbation recall, rollback precision, exact restore error, recovered ordered success, per-repeat success, endpoint error, rejected steps, intervention-to-recovery latency, switching error, and paired wins/losses. Recompute thresholds, features, scores, triggers, noise seeds, action commits, restores, and aggregates in an independent audit.

The latent-memory hypothesis is supported only if the aligned detector has higher perturbation recall at no worse nonintervened false-trigger rate than both raw-action detectors, strictly more pooled ordered successes than both, and a positive paired win/loss balance against each. It must also recover more perturbed cases than no recovery while using fewer rejected steps than always rollback. If it fails, remove latent disturbance detection and pivot G34 to a different control-relevant latent role rather than tuning another anomaly threshold. If it succeeds, G34 will integrate the frozen detector with the G32 clean controller and test longer or stronger disturbances plus current-action/F3/memory ablations.

## Required artifacts

Save the new rollout-hook code, calibration inputs/sweeps/thresholds, exact intervention manifest, all 240 rollout NPZ files, raw JSONL metrics, explicit-noise manifest, model/source references, runtime metadata, aggregate and paired metrics, and an independent `audit.json`. Infrastructure errors do not consume G33; preserve failed directories and repair them within this EXP before any conclusion.

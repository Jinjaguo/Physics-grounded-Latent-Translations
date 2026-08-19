# Next Experiment from EXP_G30: EXP_G31 Latent-Verified Checkpoint Recovery

## Hypothesis

G30 shows that optimizing a learned value before intervention catastrophically exploits model error. The hypothesis is that an action latent can still contribute **after** a real intervention: a realized action/visual-effect coordinate can verify whether a native pi0.5 prefix made acceptable current-action progress, and a full simulator/controller checkpoint can recover the pre-intervention state before trying the next native candidate. This preserves policy support while using the latent as a causal feedback and recovery coordinate.

## Realized-effect verifiers

From audited G24 matched-state branches, build leave-one-attempt-out realized-transition datasets. Train matched verifier families that predict realized utility or acceptance from:

1. realized visual-effect coordinate plus the action latent and their prediction residual;
2. raw realized visual embedding difference plus the native action prefix;
3. physical end-effector/object displacement plus native action;
4. a shuffled latent-residual ablation that breaks action/effect correspondence.

Choose verifier family thresholds only on cross-fitted sequential-retry simulations over held-out three-candidate groups: try candidate 0, accept if verified, otherwise restore and try candidate 1, then candidate 2. Save every fold checkpoint, prediction, threshold sweep, sequential choice and realized training utility.

## Adaptive causal controller

Implement a closed-loop rollout that, at every decision, captures the complete validated simulator/controller checkpoint, requests three method-neutral native pi0.5 proposals, and executes candidates sequentially from that exact checkpoint. After each candidate prefix, capture the realized observation/state and compute the verifier. If accepted, keep that already-reached state and continue replanning. If rejected, restore the complete checkpoint before trying the next candidate. Candidate 2 is accepted by default so the controller always advances.

Compare:

1. latent-verified adaptive recovery;
2. raw-visual verified recovery;
3. physical-state verified recovery;
4. shuffled latent-effect verifier;
5. exhaustive physical shooting;
6. G28 native point ranking without trial execution;
7. single pi0.5 without recovery;
8. initial-observation open loop.

Save every pre-decision checkpoint identifier/state summary, candidate execution, realized image/state, verifier score, accept/reject event, restore event, finally committed native action, subsequent observation, and autonomous F3 state. Rejected trial steps must be counted separately from committed steps. A restore must reproduce the recorded checkpoint before the next trial.

## Prospective protocol and decision rule

Freeze all verifiers and thresholds before three paired repeats over ten canonical starts. Use a new explicit-noise schedule shared by methods. The latent verifier must actually reject and recover at least once.

The hypothesis is supported only if latent-verified recovery strictly exceeds raw-visual, physical-state, shuffled-latent, point-ranking and single-proposal controls in pooled ordered success, has positive paired win-minus-loss balance against raw/physical/single controls, and matches or beats exhaustive physical shooting while executing fewer rejected candidate steps. Otherwise G32 must not return to latent proposal scoring or latent action synthesis; it must simplify the integrated controller or test a different latent role such as recovery-state memory under external perturbations.

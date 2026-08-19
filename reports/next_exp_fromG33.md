# Next Experiment from EXP_G33: EXP_G34 Monotonic Execution-Phase Coordinate

## Hypothesis

Action latents have failed as proposal scorers, generators, trial verifiers, and disturbance detectors. G12's generic action-history latent also hurt F3, while G14's outcome latent only won a switch-error tie-break. G34 tests a different representation hypothesis: a compact execution coordinate trained explicitly for within-action temporal order and grounded jointly in current dual-view observation, robot state, active language action, and recent executed controls can provide a more reliable autonomous lift-completion signal under perturbation than state-only F3 or a matched binary visual classifier.

## New representation and training

Construct a train-only dataset from all task-5 training episodes at actual policy-issue steps. Each row must contain raw dual-view images, policy state, the previous 16 executed native actions, active atomic-language identity, certified lift boundary, normalized within-lift progress, and completion label. Preserve episode membership and action-step alignment.

Train and save three matched-capacity families inside this EXP:

1. monotonic phase coordinate: dual-view CNN plus recent-action encoder and physical state, producing a 16-D action-state latent and scalar coordinate; train with completion BCE, progress regression, and within-episode temporal ranking loss;
2. direct binary visual/action completion model: identical encoder/capacity but completion BCE only, with no coordinate/ranking constraint;
3. shuffled-order coordinate ablation: identical losses and capacity after permuting progress/order targets within each training episode while preserving class labels.

Use leave-one-training-episode-out predictions to select coordinate/classification thresholds and persistence jointly within G34. Threshold sweeps are part of this EXP, not new EXP IDs. Freeze model checkpoints, normalization, thresholds, and persistence before prospective execution.

## Fixed causal control and perturbation protocol

Fix G33's raw-action physical-residual recovery as F2/memory for every method. Use the same two 3 cm pre-candidate object interventions, three explicit-noise repeats, ten canonical starts, exact checkpoint rollback, current-action protection, current-state retargeting, and native π0.5 proposals. Compare 240 prospective rollouts:

1. monotonic phase-coordinate autonomous F3;
2. direct binary visual/action F3;
3. shuffled-order coordinate F3;
4. G12 state-window F3;
5. G14 outcome-latent F3;
6. train-median fixed switch;
7. oracle physical completion switch;
8. future-visible/no-protection control, while keeping the physical recovery mechanism matched wherever the rollout interface permits.

Every F3 decision must use the observation reached after the previous real controller intervention. The place prompt must begin from the physically reached lift state, never from the initial snapshot. Save online image/state/action inputs, 16-D coordinate, scalar progress/logit, threshold, persistence state, switch event, retrospective oracle boundary, disturbance/recovery evidence, executed actions, and final task metrics.

## Decision rule

Report held-out training-episode ranking/completion metrics, prospective ordered success, premature and delayed switches, absolute switch error, lift retention, endpoint error, action jerk, perturbation recovery, and paired wins/losses. The monotonic latent is supported only if it strictly exceeds state-window, outcome-latent, direct binary, and shuffled-order F3 in pooled ordered success; has a positive paired balance against state and direct binary; has no more premature switches; and retains at least the physical-recovery backbone's 24/30 reference performance. If success ties, the latent claim is not supported even if switch timing improves.

If the hypothesis fails, G35 must stop searching for another small latent head on the same lift/place data. It should instead build a multi-task/action representation dataset or test a representation intervention on longer ordered composition, because the single-task latent has then failed every available control role. If it succeeds, G35 will run integrated ablations of phase latent, physical recovery, causal feedback, current-action protection, and autonomous F3.

## Required evidence

Save dataset/split manifests, aligned issue-step rows, three checkpoints and full training logs, leave-one-episode-out predictions and threshold sweeps, frozen protocol, all 240 causal rollouts, perturbation/restore/action/F3 traces, raw and aggregate metrics, runtime metadata, and an independent audit that rebuilds training membership, predictions, online F3 decisions, recovery decisions, noise seeds, action commits, and final conclusions.

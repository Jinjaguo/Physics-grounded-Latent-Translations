# Next Experiment from EXP_G18: EXP_G19 Executed Checkpoint Recovery

## Hypothesis

G18 rejects transition latents as better proposal scorers, but a compressed execution representation may still be useful for state memory. After a controlled mid-episode disturbance, a learned action-progress latent may select a recoverable checkpoint whose executed return and continuation succeed more often than direct nearest-state or latest-checkpoint recovery. This tests a distinct control role required by the final goal rather than another action-selection variant.

## Checkpoint dataset and representation

Collect fresh successful closed-loop lift-to-place trajectories before opening the held-out disturbance cohort. Save multiple complete checkpoints per trajectory, including simulator state, controller state, observation, active action/F3 state, EEF/object state, recent action history, and continuation metadata. Train and compare two memory encoders on trajectory-local temporal/progress supervision:

1. a sequence contrastive action-progress encoder over observation/state/action history;
2. a deterministic sequence autoencoder with progress and phase heads;
3. a direct state-progress predictor without a latent bottleneck.

Split by trajectory and choose the latent family using held-out checkpoint-ordering and phase accuracy. Save exact dataset membership and all model checkpoints.

## Causal recovery test

Freeze a matched held-out cohort. Start each task with the same closed-loop controller, save checkpoints during successful execution, then apply a controlled, action-executed EEF displacement after lift completion while preserving the grasp. From the disturbed physical state compare:

1. latent-selected checkpoint plus executed proportional OSC return;
2. direct-state-selected checkpoint plus the same executed return;
3. latest eligible checkpoint plus the same executed return;
4. no return, immediate continuation;
5. restart-from-initial-state execution;
6. direct simulator restoration to the selected checkpoint as a non-physical upper bound.

For methods 1--3, recovery is credited only after controller actions physically reduce both EEF and object distance to the stored state; direct simulator restoration cannot support the recovery claim. After return, all methods continue closed-loop from their physically realized state with autonomous F3/place execution. Save the disturbed state, candidate checkpoint IDs, selection features/latents, every return action and resulting state, continuation rollout, and final metrics.

## Metrics and decision

Primary metrics are disturbance-to-return physical error reduction, stable grasp retention, recovery completion, ordered task success, continuation success, recovery action count, and endpoint error. The latent contributes only if it strictly improves ordered recovery success over direct-state and latest-checkpoint recovery on matched cases; lower latent distance alone is irrelevant. Executed recovery is supported if one physical-return method strictly improves over no recovery and restart. If latent selection again loses, G20 must retain the strongest conventional recovery mechanism and investigate a latent temporal completion/switching role or simplify the final architecture rather than tuning this selector.

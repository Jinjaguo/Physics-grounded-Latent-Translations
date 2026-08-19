# Next Experiment from EXP_G11: EXP_G12 Learned Temporal F3 Integration

## Hypothesis

The 4/5 contextual post-lift controller is strong enough for end-to-end composition. A temporal F3 using recent robot state and action history should approximate the lift boundary and trigger contextual place without simulator-ground-truth object height. Adding the learned action latent should improve switch timing and downstream composition relative to a matched state-only temporal model.

## Train-only completion dataset

Use task-5 Wave19 training episodes only. At every saved π0.5 policy-issue time, align the 8-D policy robot state and preceding executed action history with the already validated ground-truth lift boundary. Build fixed recent-issue windows with explicit left padding. Train matched temporal completion classifiers:

1. `state_window_f3`: recent 8-D robot-state window.
2. `state_latent_window_f3`: the same state window plus the frozen rolling 32-D action latent at each issue.

Use a fixed 0.5 probability threshold and two consecutive positive decisions, selected before development rollout. Save sequence membership, issue-step alignment, class counts, normalization, checkpoints, training logs, and train-only classification metrics. Do not use book body height online for learned F3.

## End-to-end methods

From the same five pre-policy development snapshots, run complete causal control with the G9 two-proposal state-value F2. Before switching, expose only `pick up the black book`; after switching, use G11’s winning completed-lift contextual place prompt. Compare:

1. `oracle_f3`: 4 cm×3-step book-height oracle; upper-bound, not deployable.
2. `fixed_train_median_f3`: switch at the median train lift-boundary step.
3. `state_window_f3`: autonomous temporal state classifier with persistence.
4. `state_latent_window_f3`: matched autonomous state+action-latent classifier.
5. `future_visible_full_prompt`: original composite prompt throughout, with no explicit switch.

For learned methods, book height may be logged only for retrospective boundary error and must not affect switching or actions. Every trace must log current protected action, F3 probability/state, switch decision, both causal F2 candidate executions, committed action, realized observation/state, latent update, lift retention, and official success.

## Metrics and acceptance for this stage

Primary metric is end-to-end official success after an actual switch. Report lift completion, premature/late switch error relative to retrospective oracle, place success, official success, and current-action failures. Learned F3 is supported only if it reaches at least 3/5 official success and beats fixed-time switching in success or the declared endpoint-error tie-break. Action-latent contribution requires the matched latent F3 to beat state-only in official success, with switch error then endpoint error as tie-breakers.

Even if learned F3 succeeds on development, do not declare final acceptance yet. EXP_G13 must freeze the integrated architecture and run protected held-out/test confirmation plus required open-loop, restart, F2-disabled, F3-disabled, and latent-disabled ablations. If both learned F3 variants fail, EXP_G13 must use their saved temporal errors to change the completion model or observation family rather than returning to oracle switching.

## Required artifacts

Save the completion dataset, two temporal checkpoints, all end-to-end proposals/candidate/commit traces, active-action and switch logs, retrospective oracle comparisons, aggregates, exact command/environment, and independent audit under `experiments/EXP_G12/`, followed by the report and next executable prompt.

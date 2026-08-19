# Next Experiment from EXP_G13: EXP_G14 Outcome-Grounded Latent on Prospective Starts

## Hypothesis

The old frozen latent overfits because it was not trained to preserve control-relevant outcomes. A compact action-window encoder jointly supervised to predict lift/place phase and future realized book/eef displacement should generalize phase completion better than the old latent and contribute positively to autonomous switching.

## New latent and matched F3 training

From task-5 training episodes, construct fixed 16×7 action-history windows with left padding. For every policy-issue step, supervise an encoder bottleneck using:

- lift/place phase at the validated ground-truth boundary;
- five-step future end-effector displacement;
- five-step future `black_book_1_main` displacement;
- next-action reconstruction or prediction.

Train a compact outcome-grounded latent without development/test data. Then train matched temporal F3 models on identical recent policy-state windows: state-only, state+old-latent, and state+new-outcome-latent. Select architecture/loss settings within training data or the already-open development split as one systematic G14 comparison; do not use G13 test episodes for model choice.

## Prospective untouched starts

Identify official task-5 init-state indices not present in any Wave19 train/development/test episode. Save this exclusion proof before rollout. For each available untouched index, create the real LIBERO environment, set the official initial state, execute the recorded ten stabilization dummy controls, and capture a complete pre-policy snapshot. These are newly constructed prospective starts, not reassigned Wave19 test cases.

Run the same causal π0.5 state-value F2 and completed-lift contextual place controller with:

1. `state_f3`;
2. `old_latent_f3`;
3. `outcome_latent_f3`;
4. `fixed_time_f3`;
5. `future_visible_full_prompt`.

All methods must execute full rollouts; learned F3 variants may not use book height online. Save active action, F3 probability/switch, candidate execution, committed action, realized feedback, latent, retrospective oracle, and official success.

## Metrics and decisions

Primary metric is prospective official success; secondary metrics are premature switches, switch error, endpoint error, lift retention, and causal steps. Outcome latent contribution requires `outcome_latent_f3` to beat both state-only and old-latent F3 in official success, with switch error and endpoint error as ordered tie-breakers.

If the new latent contributes and the integrated system is robust, EXP_G15 should rerun required ablations on a second prospective seed cohort before considering final acceptance. If it fails, preserve the result and change the latent’s causal role or representation family in EXP_G15; do not return to the already falsified nearest-support penalty.

## Required artifacts

Save outcome-latent dataset membership/targets, checkpoints/logs, untouched-index exclusion manifest, new exact snapshots, all prospective rollouts, metrics, exact command/environment, and an independent audit under `experiments/EXP_G14/`, followed by the report and next prompt.

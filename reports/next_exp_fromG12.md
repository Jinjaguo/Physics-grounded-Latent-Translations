# Next Experiment from EXP_G12: EXP_G13 Latent-Support F2 and Frozen Test Ablations

## Hypothesis

Action latent harmed temporal F3 because train-perfect completion features did not generalize online. Its better-supported role is constraining low-level proposals to action trajectories observed during the active phase. Penalizing candidate continuation latents that are far from train-only lift/place latent support should improve the frozen state-value F2 without changing the successful state-window F3.

## Train-only latent support and development selection

Using task-5 training episodes and the existing lift boundaries, build normalized latent support sets for pre-lift and post-lift phases. At each causal F2 decision, append each physically executed candidate prefix to the actual history, encode its resulting 32-D latent, compute nearest-support distance for the current protected action, and rank by:

`realized state value - lambda * normalized latent support distance`.

Run one systematic development sweep over `lambda = {0, 0.05, 0.1, 0.2}` inside EXP_G13 with the frozen state-window F3 and contextual place prompt. Select by official success, then endpoint error. This sweep is one mechanism experiment, not separate EXP IDs. Freeze lambda and every component before opening test.

## One-time task-5 test evaluation

Use all task-5 test episodes from the Wave19 split, starting from their exact pre-policy snapshots. Evaluate the selected full system and these actual executed baselines:

1. `full_latent_support`: causal state-value F2 plus selected latent support, state-window learned F3, contextual place.
2. `latent_disabled`: identical system with lambda zero.
3. `f2_disabled_single`: one π0.5 proposal per decision, same learned F3.
4. `f3_disabled_future_visible`: composite prompt throughout, same causal F2.
5. `restart_after_switch`: learned F3 triggers, then restore the initial snapshot before contextual place.
6. `open_loop_initial_observation`: pre-generate the action sequence from repeated initial-observation π0.5 requests, then execute without feedback.
7. `teacher_replay`: actually replay the archived successful test controls as an upper-bound baseline, clearly labeled non-deployable.

All methods must execute in LIBERO and save realized feedback and official success; config generation or replay-only analysis does not count. Learned F3 must never use book height online.

## Metrics and decisions

Primary metric is official test success. Report composition success, lift retention, switch behavior, endpoint error, candidate steps, and action continuity. The full system must outperform open-loop, restart, F2-disabled, F3-disabled, and latent-disabled baselines for a final success claim; teacher replay is an upper bound, not a deployable competitor.

Action-latent contribution is supported only if selected nonzero lambda beats lambda zero on development and the frozen full system beats latent-disabled on test in official success, with endpoint error only as a predeclared tie-break. If development selects lambda zero or test does not favor latent support, report that the final latent acceptance condition remains unmet and use EXP_G14 to change the latent mechanism rather than claiming success.

## Required artifacts

Save train-only support membership/statistics, development sweep rollouts and selection, a timestamped frozen-system manifest before test access, all test proposals/candidate/commit traces and baselines, exact commands/environment, aggregate metrics, and an independent audit under `experiments/EXP_G13/`, followed by the report and next executable prompt.

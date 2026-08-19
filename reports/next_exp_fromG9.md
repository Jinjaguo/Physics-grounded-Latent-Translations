# Next Experiment from EXP_G9: EXP_G10 Oracle-Switched Lift-to-Place Composition

## Hypothesis

The G9 causal state-value π0.5 controller is reliable enough to expose the next architectural question: whether future-goal information destabilizes the current lift and whether place succeeds only when retargeted from the physically realized lifted state. Hiding the place prompt until a ground-truth lift boundary should improve protected lift completion, and continuing from that realized state should outperform restarting before place.

## Dataset construction and object mapping

Use all five Wave19 development episodes for task 5, `pick up the book and place it in the back compartment of the caddy`; do not open test episodes. Restore the exact snapshot at the end of the ten-step stabilization period (`exact_snapshots.pkl[wait_steps]`) so every method begins before policy control. The actual MuJoCo model identifies body 23 as `black_book_1_main`; save and audit this name/id mapping rather than hard-coding an unverified guessed object.

Define oracle lift completion from simulator ground truth as the book remaining at least 4 cm above its initial body height for a short persistence window while not yet satisfying the final task predicate. Validate on the five successful archived development trajectories that the predicate fires before official task success and save those boundary indices. This is an oracle F3 isolation experiment; it must not be reported as learned/autonomous switching.

## Methods and execution

Use the G9 two-proposal causal state-value mechanism at every decision: issue fresh π0.5 proposals from the current observation, execute both five-step prefixes from the same decision snapshot, score their realized states with the frozen task-5 train-only state value, restore, commit the winner, and update observation/history.

Compare at least:

1. `future_visible_full_prompt`: expose the original composite prompt for the entire rollout; measure whether lift completes and remains stable before placement.
2. `hard_gate_realized_retarget`: issue only `pick up the black book` until the oracle lift predicate, then switch to `place the black book in the back compartment of the caddy` and continue from the actually reached state.
3. `hard_gate_restart_place`: execute the same protected lift, but at the oracle boundary restore the initial snapshot before issuing the place prompt. This is the required restart baseline and must not masquerade as a deployable controller.
4. `future_only_place`: issue the place prompt from the initial state without the lift stage, isolating whether π0.5 can silently perform both actions despite the decomposition.

Log the active prompt/action ID at every decision, both proposals, candidate executions, committed actions, realized robot/object state, book height, lift-predicate persistence, switch event/state, post-switch starting state, final official success, and any lift loss/drop after completion.

## Metrics and decisions

Primary composition metric is official place success after an observed lift boundary. Current-action protection is supported only if hard gating improves lift-completion or lift-retention over the future-visible method across the five matched starts. Realized-state retargeting is supported only if `hard_gate_realized_retarget` beats `hard_gate_restart_place` in official success; endpoint distance is a declared tie-breaker.

Report lift completion, lift-to-place success, official final success, switch latency, book-height retention, causal candidate steps, and action continuity. G10 advances the final goal only at the oracle-F3 stage; even perfect results do not satisfy autonomous switching.

If protected realized-state execution works, EXP_G11 must replace oracle F3 with learned/temporal completion models and compare their actual switch behavior. If composition fails, EXP_G11 must use the saved failure states to change the place controller or atomic prompt/control formulation, not repeat the same gate.

## Required artifacts

Save the development episode/snapshot manifest, verified body map, archived oracle-boundary validation, every candidate/committed transition, prompt/action-state trace, switch states, aggregate metrics, exact commands/environment, and an independent audit under `experiments/EXP_G10/`, followed by the report and next executable prompt.

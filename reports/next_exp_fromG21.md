# Next Experiment from EXP_G21: EXP_G22 Counterfactual-Regret F3

## Hypothesis

G21 rejects latent gating of F2, while G12--G18 show that autonomous switching remains a control-relevant role and outcome-latent timing sometimes improves without improving success. A representation trained on the downstream causal regret of `switch now` versus `continue current action` may provide the missing control-aligned latent signal. Unlike prior F3 models trained on height labels, G22 will supervise the representation with realized continuation outcomes.

## Matched switch-intervention dataset

Freeze single-proposal receding pi0.5 as F2. From each G18 perturbed snapshot, execute lift until several checkpoints spanning the autonomous boundary. At every checkpoint restore the exact simulator/controller state and execute at least three mechanisms:

1. switch immediately to contextual place;
2. continue protected lift for one proposal chunk, then switch;
3. continue protected lift for two proposal chunks, then switch.

Every branch must continue through the simulator to success or horizon. Save checkpoint state, observation, recent action history, old/outcome latents, switch delay, complete continuation, terminal success, endpoint error, lift retention, and candidate-independent controller cost. Label switch-now regret by success first and endpoint error second. These are new matched causal interventions, not retrospective height labels.

## Models

Fit by held-out attempt:

1. a sequence contrastive action latent with switch-regret and phase heads;
2. a deterministic sequence autoencoder with the same heads;
3. a direct state/history MLP without a latent bottleneck.

Select the latent family using held-out pairwise mechanism ranking and regret MAE. Freeze threshold/persistence without using prospective end-to-end outcomes.

## End-to-end causal test

Execute new matched rollouts from the ten perturbed snapshots with single-proposal receding feedback:

1. counterfactual-regret latent F3;
2. direct regret F3;
3. existing state-window temporal F3;
4. oracle height F3;
5. fixed-time F3;
6. future-visible/no-protection control.

All learned methods must autonomously decide online from realized observations and action history. Place must start from the physical state reached by lift. Report success, premature/late switch, switch error, lift retention, endpoint error, and committed steps.

## Decision

The latent contributes only if it strictly beats the direct regret F3 and existing state F3 in ordered success, with switch error as a tie-break only after equal success, and also beats fixed/future-visible controls. If it loses again, G23 will accept that the tested action-history latents do not provide a control advantage, retain the simplest state-F3 plus single-feedback architecture, and seek a different representation family trained jointly with the policy rather than adding another post-hoc latent head.

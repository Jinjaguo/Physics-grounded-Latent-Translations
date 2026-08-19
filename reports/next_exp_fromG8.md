# Next Experiment from EXP_G8: EXP_G9 π0.5 Proposal Control with Causal Feedback

## Hypothesis

EXP_G8 failed because a small behavior-cloned CNN discarded the pretrained vision-language-action prior and compounded error off the source distribution. The official local π0.5 policy should be a stronger atomic controller. Among several stochastic π0.5 chunks proposed from the same current observation, an action-latent phase criterion and an executed-feedback value criterion should select controls more reliably than a single π0.5 sample.

## Infrastructure and data

Use the existing official `pi05_libero` PyTorch checkpoint, normalization assets, websocket server, and exact LIBERO snapshots. If live checkpoint inference is not currently connected to the experiment harness, implement that client/adapter inside EXP_G9 before consuming the EXP ID. Use the real instruction from each episode and the exact image/state transform in `collect_wave19_libero.py`.

Fit any proposal value or phase target only from Wave19 training episodes. Development tasks 0–4 may choose proposal count, value formulation, or execution prefix as a systematic within-EXP sweep. Freeze that choice before reporting tasks 5–9. Do not open the test split and do not use archived future controls as online commands.

## Methods and intervention

From each unchanged late EXP_G3 checkpoint, compare at least these mechanisms under the same remaining horizon:

1. `pi05_single`: query one fresh π0.5 chunk from the current dual-camera observation, execute five controls, observe the realized state, and replan.
2. `pi05_latent_rank`: query a batch of genuinely stochastic π0.5 chunks, encode each candidate continuation with the frozen action representation, rank it against a task/phase target retrieved from training demonstrations, execute the winner, observe, update history, and replan.
3. `pi05_causal_value`: restore the same full simulator checkpoint for each candidate, physically execute the candidate prefix, score the realized next state with a train-only completion/progress value model, restore the decision checkpoint, commit the winning prefix, and continue from its realized feedback. Log both candidate rollouts and the committed rollout.

Include a matched non-latent value/ranking ablation so any benefit assigned to the action latent is identifiable. Gripper values must be the actual sign commands before execution and continuity metrics.

## Metrics and validity

Primary metric is official LIBERO success over the ten atomic cases, with tasks 5–9 reported as confirmation. Secondary metrics are endpoint target error, selection regret relative to the evaluated candidate set, action jerk, number of causal candidate steps, and whether the selected candidate differs from the first proposal. Compare against G4’s 2/10 state+latent result and G3’s 5/10 source-reference upper-bound trajectory.

EXP_G9 is valid only after new live π0.5 proposals have been issued, selected actions have actually executed from restored states, realized observations/states have updated the next decision, and auditable metrics exist. A server/import/checkpoint problem remains an internal EXP_G9 gate and cannot end the experiment.

The π0.5 low-level family is supported only if it exceeds 2/10 overall and produces at least one confirmation-half success. Latent contribution requires the matched latent selector to beat its non-latent counterpart in official success, with endpoint error as a declared tie-breaker. If π0.5 restores reliable atomic execution, EXP_G10 should begin explicit current-action protection and oracle-switched `lift -> place` construction. If not, EXP_G10 must change control formulation using the causal candidate data collected here, such as a learned residual/value-guided controller rather than another image-regression variant.

## Required artifacts

Save server/checkpoint identity, commands, split and training-value manifests, every raw proposal, latent/value score, pre-candidate snapshot identifier, candidate executed actions and realized states, committed actions and observations, checkpoints for fitted models, case-level metrics, aggregates, and an independent audit under `experiments/EXP_G9/`. Then write the report and the next executable prompt.

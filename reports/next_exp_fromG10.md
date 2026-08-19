# Next Experiment from EXP_G10: EXP_G11 Post-Lift Place Controller Isolation

## Hypothesis

G10’s 5/5 lift and 0/5 hard-switched place indicate that the place prompt and the full-trajectory G9 progress value are mismatched to post-lift states. Preserving completed-lift context in the prompt and fitting a value only on post-lift training suffixes should improve place completion. The action latent may become useful specifically for discriminating place-phase proposals even though it harmed whole-task G9 ranking.

## Train-only post-lift dataset

For every task-5 Wave19 training episode, use the verified `black_book_1_main` body 23 and the same 4 cm×3-step ground-truth boundary to select the successful post-lift suffix. Build matched task-5 state and state+latent value datasets whose target is normalized progress from that boundary to official success. Save episode membership, boundary indices, feature definitions, checkpoints, training logs, and prove that no development/test episode enters fitting.

## Matched realized switch states

For each of the five G10 development starts, execute the protected lift exactly once using G9 causal state-value F2 and the lift-only prompt. At the oracle boundary, save the complete MuJoCo/controller snapshot, current observation, physical state, committed lift history, and latent. All post-lift methods must branch from this identical exact snapshot and identical history; do not rerun a different stochastic lift per method.

## Place-controller branches

From each saved lifted state, compare at least:

1. `place_only_full_value`: place-only prompt with the original G9 full-trajectory state value.
2. `contextual_place_full_value`: prompt stating that the book is already lifted and now must be placed, with the same full value.
3. `full_prompt_full_value`: original composite prompt after the lift boundary, testing whether π0.5 needs its training-style task context.
4. `place_only_suffix_state_value`: place-only prompt with the new post-lift state value.
5. `place_only_suffix_state_latent_value`: matched post-lift state+latent value.

Each controller must query multiple fresh π0.5 proposals, physically execute candidates from a recoverable decision snapshot, rank realized feedback, restore, commit the winner, and replan. Log prompt, candidate/committed actions, realized state/observation, action latent, book height/retention, and official success.

## Metrics and decision

Primary metric is official place success from the same five realized lift states. Secondary metrics are final target error, lift retention/drop, action continuity, candidate selection changes, and suffix-value ranking behavior. Context preservation is supported if contextual or full prompt beats place-only under the same full value. Suffix value is supported if it beats the full value under the same place prompt. Latent contribution requires suffix state+latent to beat suffix state in official success, with error as tie-breaker.

If a post-lift controller reaches at least 3/5 official success, EXP_G12 should integrate it with oracle-switched lift and then proceed to learned F3. If all variants remain weak, EXP_G12 must change the place action/control representation using the exact failure snapshots, such as target-conditioned spatial value or object-relative controller, rather than tuning prompt text alone.

## Required artifacts

Save post-lift dataset/checkpoints, exact realized switch snapshots, a branch manifest proving matched starts, every proposal/candidate/committed transition, metrics, exact command/environment, and an independent audit under `experiments/EXP_G11/`, followed by the report and next executable prompt.

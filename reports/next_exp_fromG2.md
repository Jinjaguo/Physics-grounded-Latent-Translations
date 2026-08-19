# Next Experiment from EXP_G2: EXP_G3 Long-Horizon Atomic Completion

## Hypothesis

The EXP_G2 causal latent proposal portfolio will preserve its local physical-feedback advantage over a complete remaining manipulation phase and improve official current-action completion relative to receding F1, receding F2, and direct state/action shooting under oracle F3 boundaries.

## Exact intervention

Use only development episodes and select one source episode per LIBERO-10 task. Within each selected episode choose the latest certified branch (`branch_075` when available, otherwise the latest eligible branch) so that the remaining source continuation is at least 128 control steps. Execute each method for the full number of remaining recorded source steps or until the official LIBERO success predicate / environment termination fires.

Compare five methods from matched full-state checkpoints:

1. `source_oracle`, as an upper/reference execution only;
2. `receding_F1`, replanned from the latest executed-action history;
3. `receding_F2`, using the registered refinement;
4. `causal_state`, with direct/damped/hold/norm-matched-random proposals evaluated through realized feedback;
5. `causal_latent`, with the same direct candidates plus F1/F2/copy latent proposals and support-aware selection.

Use four-step committed prefixes. At every causal decision, actually execute all candidate prefixes from the same current checkpoint, score their realized state against the oracle current-action physical trajectory, restore, commit the winner, observe, re-encode, checkpoint, and repeat. Future source actions must remain excluded from deployable candidates. The saved source trajectory supplies only the oracle current-action path/completion boundary, which must be clearly marked as unavailable to the final autonomous system.

## Evaluation and decision

The primary metric is official current-action success rate across the ten task-stratified episodes. Secondary metrics are completion step, final physical target error for failures, trajectory progress, jerk, instability, task breakdown, proposal-selection composition, intervention count, and runtime. Report both success and failure cases; do not replace zero success with a surrogate claim.

`SUPPORTED` requires `causal_latent` to improve official success over receding F1, receding F2, and causal state without instability. If success rates tie, it may be considered supported only if all tied successful cases complete and causal latent has lower paired failure endpoint error and no worse completion efficiency; otherwise write `NOT_SUPPORTED`.

If all learned methods fail while source succeeds, EXP_G4 must pivot to a new long-horizon controller/model family trained on the executed transition evidence, rather than tuning the same candidate-score coefficients. If causal state matches or beats causal latent, simplify F2 and test the state-space winner. If causal latent succeeds, EXP_G4 should move to two-action composition with oracle switching and current-state retargeting.

## Required artifacts

Write a new `experiments/EXP_G3/` directory with exact command/revision/environment records, selected branch and per-case horizon manifest, every candidate intervention and committed rollout, per-case and aggregate success/physical metrics, runtime, and an independent audit. Then write `reports/EXP_G3_report.md` and `reports/next_exp_fromG3.md`. Dependency or interface failures remain debugging inside EXP_G3 and cannot consume its ID.

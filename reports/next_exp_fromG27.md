# Next Experiment from EXP_G27: EXP_G28 Common-Noise Matched Representation Trial

## Hypothesis

Across G24-G27, nominally identical baselines vary by two successes because each method receives independent stochastic pi0.5 proposals. This can hide or fabricate a one-case representation gain. The hypothesis is that, under common diffusion-noise schedules and repeated closed-loop evaluation, an outcome-aligned transition coordinate provides a reproducible advantage over equal-capacity unaligned-latent and raw-action point rankers.

## Deterministic proposal primitive

Extend the official local pi0.5 batch server to accept an explicit initial flow-matching noise tensor for every observation in a batch. The client generates noise from a recorded seed keyed only by `(repeat, attempt, decision, candidate)`, never by method. Validate that identical observation plus identical noise produces bitwise-identical raw and postprocessed action chunks, while changing noise produces distinct proposals. Save seeds and exact noise arrays or a regenerable seed/shape manifest with every rollout.

This is common-random-number control, not deterministic replay of outcomes. Methods may reach different observations, so their actions can diverge, but the stochastic innovation at corresponding decisions/candidates is paired.

## Matched point rankers

Using the audited G24 groups and fold-specific encoders, train three exactly matched candidate-independent point rankers:

1. aligned action/visual-transition coordinates;
2. unaligned action latents from the G24 no-alignment model;
3. normalized raw native-action prefixes with direct visual predictions.

Use the same 128-D padded feature width, architecture, loss, training schedule and leave-one-attempt-out folds. Save held-out ranking/regret and exact parameter counts. No set pooling, recurrence or uncertainty penalty is used.

## Repeated prospective protocol

Run three full repeats over all ten canonical snapshots with autonomous state-window F3. Compare:

1. aligned latent point ranker;
2. unaligned latent point ranker;
3. raw-action point ranker;
4. G27 aligned set ranker;
5. G27 raw set ranker;
6. physical matched-branch shooting;
7. single raw pi0.5;
8. initial-observation open loop.

Every method/repeat uses the same recorded proposal-noise schedule. Save per-repeat and pooled outcomes, paired attempt-level wins/losses, exact proposal noise identifiers, raw actions, committed actions, realized feedback and F3 states. Physical shooting receives the same candidate noise as learned three-proposal methods; single proposal receives candidate zero.

## Decision rule

The aligned coordinate contributes only if its pooled ordered-success count strictly exceeds unaligned and raw matched point rankers, both set rankers, and single pi0.5, with a positive paired win-minus-loss balance against each representation baseline. It must also match or beat physical shooting using fewer candidate-executed steps, or strictly beat it. Per-repeat cherry-picking is forbidden. If aligned ties or loses on pooled success, G29 must accept that this representation is unnecessary and test a genuinely different action representation or system role.

# Next experiment after Wave 19

Wave 20 should adjudicate the one failed prerequisite—joint semantic and motor fidelity—without opening the current
50-episode final test split and without training F1/F2 as a workaround.

## Frozen question

Can a prospectively motor-weighted LIBERO representation retain Wave-19's bidirectional language addressability
while clearing the motor gate with a real safety margin rather than another threshold-edge result?

## Data

- Keep the Wave-19 140-episode train split and 50-episode final test split unchanged.
- Treat the already-read Wave-19 development split as development-only evidence, not fresh confirmation.
- Prospectively collect 5 new certified episodes/task (50 total) with a new registered π0.5 seed and the same exact
  snapshot/future-support protocol. Freeze these as a new confirmation-development set; do not merge them into the
  current final test.

## Model comparison

Run six new registered seeds for exactly two paired conditions:

1. the Wave-19 reconstruction-only anchor;
2. correct language with `L = 2 × L_reconstruction + L_semantic`, keeping the same 32=16+16 architecture,
   action-only input, OpenCLIP encoder, gradient isolation, 40 epochs, and EMA 0.999.

The factor `2` and all seeds must be registered before the fresh confirmation episodes are evaluated. Do not sweep
loss weights on the confirmation set.

## Gate

Require on the fresh confirmation set:

- positive mean semantic delta in both directions;
- source-episode-clustered lower 95% > 0 in both directions;
- correct-language continuous MSE ≤ `1.15 ×` paired reconstruction-only MSE;
- gripper sign-accuracy drop ≤ `0.02`;
- all six seeds complete and all outputs finite.

The stricter 1.15/0.02 margins distinguish a genuinely motor-preserving representation from Wave 19's
`1.200444` threshold-edge outcome.

If this gate fails, stop and conclude that the current factorized representation family does not yet support the
cross-domain claim; do not try more seeds or rescue F2. If it passes, freeze the selected checkpoint using the same
seed-selection rule, train the preregistered LIBERO F1/F2 once, run O1–O8, and open the untouched Wave-19 final test
only if the offline gate authorizes B0–B5 and proposal recovery.

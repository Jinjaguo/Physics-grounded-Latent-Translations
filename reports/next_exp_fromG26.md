# Next Experiment from EXP_G26: EXP_G27 Permutation-Equivariant Proposal-Set Coordinates

## Hypothesis

G26's correctly assigned latent uncertainty reaches 8/10 while a cyclically misassigned risk reaches 10/10. This falsifies the proposed uncertainty mechanism and suggests that independently scoring candidates misses useful relative structure in each proposal set. The new hypothesis is that an aligned transition coordinate is useful when proposals are ranked jointly: a permutation-equivariant set model can compare relative direction, support, and outcome among candidates without attaching meaning to their array index.

## Set-wise model families

Use all 414 audited G24 three-candidate groups with leave-one-attempt-out evaluation. Construct candidate features using the protected fold-specific G24 image encoder and predictors, then train equal-parameter set rankers:

1. an aligned-coordinate DeepSets ranker using current visual/state context, action-transition coordinates, and predicted visual/physical outcome;
2. a raw-action DeepSets ranker using the exact normalized native action prefix and direct visual/physical outcome;
3. a candidate-independent point MLP with matched feature width, isolating the set-context contribution.

Each DeepSets model embeds candidates, pools their mean, and scores each candidate from its local embedding plus pooled context. Train with whole-group cross-entropy and realized-progress regression. Save exact group membership, fold models, held-out ranking/regret, and predictions for all six permutations of every held-out group. A valid implementation must be permutation equivariant to numerical tolerance.

## Prospective methods

Run fresh autonomous `lift -> place` rollouts from all ten canonical snapshots, comparing:

1. aligned-coordinate set ranker;
2. equal-capacity raw-action set ranker;
3. aligned point-ranker ablation;
4. G26 aligned uncertainty ensemble;
5. G26 shuffled-disagreement controller;
6. G25 raw-history GRU;
7. memoryless aligned visual scorer;
8. physical matched-branch shooting;
9. single raw pi0.5;
10. initial-observation open loop.

At every set-ranked decision, additionally score all six candidate permutations and save the unpermuted results. These checks must agree before the selected native action is committed. Save all proposal features, pooled set context, scores, permutation errors, selected index, exact executed action, realized feedback, F3 state, and terminal outcome.

## Decision rule

The set-coordinate hypothesis is supported only if the aligned set ranker is permutation equivariant, strictly exceeds the raw set ranker, aligned point model, G26 uncertainty and shuffled controllers, raw-history GRU, memoryless aligned, and single pi0.5 in ordered success, and matches or beats physical shooting with fewer candidate-executed steps. Ties do not support a latent claim. If the raw set model wins or ties, set-relative reasoning may be retained but the action-coordinate claim fails; G28 must then move to a different mechanism rather than tune set-network width.

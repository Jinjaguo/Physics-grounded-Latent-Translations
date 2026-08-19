# Next Experiment from EXP_G29: EXP_G30 Generative Latent Action Refinement

## Hypothesis

Selection-only experiments G24-G29 never let the latent representation create a control intervention; they merely ranked complete actions already proposed by pi0.5. The hypothesis is that a low-dimensional action coordinate supervised by realized effects can support local optimization and decoding of a new executable action prefix, yielding better ordered success than equal-budget optimization in raw 35-D action space and than the original native proposal.

## Models and causal training data

Use all audited G24 matched-state interventions. For every leave-one-attempt-out fold, train a three-member action-coordinate ensemble with:

1. a 35-D normalized native-prefix encoder into an 8-D coordinate;
2. a decoder back to the exact five-step, seven-dimensional native action prefix;
3. a state-conditioned realized-effect head;
4. a state/coordinate utility head.

Train with reconstruction, realized physical-effect, utility, groupwise ranking and local-distance losses. Train equal-split raw-action utility ensembles without the encoder/decoder. Save target normalization, train latent/action support banks, reconstruction error, held-out ranking, ensemble predictions, and exact parameter counts. Select search scale, support penalty, pessimism and refinement gate only from cross-fitted training/held-out branch evidence before prospective testing.

## New control interventions

At each realized closed-loop state, request candidate zero from pi0.5, encode its native prefix, run a bounded local CEM search in the 8-D coordinate, decode the best supported coordinate into a new native action prefix, and choose between base and refined actions with a frozen pessimistic advantage gate. The decoded action must actually be executed in LIBERO and saved byte-for-byte.

Compare:

1. latent-coordinate CEM refinement;
2. single-pass latent random search with the same sample budget, isolating iterative optimization;
3. equal-budget raw 35-D CEM refinement;
4. latent encode/decode reconstruction without optimization;
5. G28 aligned native-proposal point ranking;
6. physical matched-branch shooting;
7. unmodified single pi0.5;
8. initial-observation open loop.

Save every search seed, sampled coordinate/action, predicted member utility, support penalty, elite update, decoded prefix, selected action, realized next feedback, and F3 state. The audit must prove that generated committed prefixes differ from the base policy and exactly match the saved decoded interventions.

## Prospective protocol and decision rule

Freeze all models and search settings before opening three paired repeats over the ten canonical starts. Pair pi0.5 noise by `(repeat, attempt, decision, candidate)` and pair local-search randomness between latent and raw formulations as far as their dimensionalities permit. Run 240 fresh rollouts with autonomous F3 and realized-state replanning.

The latent coordinate is supported only if latent CEM strictly exceeds raw CEM, latent random search, latent reconstruction, G28 point ranking, and single pi0.5 in pooled ordered successes, with more paired wins than losses against raw CEM and single pi0.5. It must match or beat physical shooting with fewer simulator branch steps, or strictly beat it. At least one executed prefix must be a genuinely decoded non-base intervention. If raw CEM or the native proposal matches or wins, G31 must simplify F2 or move latent contribution out of proposal refinement.

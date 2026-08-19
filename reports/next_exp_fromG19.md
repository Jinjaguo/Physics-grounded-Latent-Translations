# Next Experiment from EXP_G19: EXP_G20 Recovery-Trigger Gating

## Hypothesis

G19 shows that return is physically reliable but usually unnecessary and sometimes harmful. A useful recovery system should decide whether to return, not always return. A learned action-progress latent may expose off-trajectory risk that is not captured by the current EEF/object coordinates and therefore select between direct continuation and direct-state checkpoint return more effectively than a direct-state risk gate.

## Training and frozen evaluation

Use G19's audited matched outcomes to construct attempt-level intervention pairs containing the exact disturbed sequence/state, no-return outcome, direct-return outcome, checkpoint selection, physical return cost, and endpoint error. Fit and compare leave-one-attempt-out gates:

1. logistic risk/difference model on the frozen G19 action-progress latent;
2. logistic risk/difference model on direct physical state and disturbance displacement;
3. a joint latent-plus-state gate as a diagnostic family.

For each held-out attempt, train only on the other attempts and freeze the trigger before executing any new rollout. The label is whether direct-state return strictly dominates no-return by success first and endpoint error second. Save fold membership, normalization, weights, probabilities, and selected intervention.

## New causal rollouts

From every exact G19 disturbed snapshot, execute fresh matched closed-loop continuations under:

1. latent trigger: return only when the latent gate requests it;
2. direct-state trigger;
3. joint trigger;
4. never return;
5. always direct-state return;
6. oracle trigger using the saved G19 paired outcome only as a non-deployable upper bound.

Whenever a gate triggers, execute the same audited OSC return controller toward the direct-state-selected checkpoint; never use simulator restore as recovery. Every method then replans from its physically realized state. Save gate input/probability, selected intervention, return actions/states, all continuation candidate branches, and final task outcome.

## Decision

Latent gating contributes only if it strictly beats direct-state and joint-with-latent ablations is not sufficient by itself; the latent-only method must also beat always-return and never-return in task success, with endpoint error as a declared tie-break only after equal success. Adaptive recovery is supported only if one learned deployable gate strictly beats both fixed intervention policies. If no learned gate improves on never-return, G21 will remove checkpoint return from the primary system and test latent contribution through a counterfactual intervention-aware F3 or representation trained directly on realized control regret rather than continuing to tune recovery.

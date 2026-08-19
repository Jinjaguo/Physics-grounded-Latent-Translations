# Next Experiment from EXP_G34: EXP_G35 Multi-Action Coordinate on Three-Stage Composition

## Hypothesis

G34 falsified the claim that accurate phase coordinates or strict future-action gating improve the forgiving two-stage `lift -> place` task. G35 tests a different representation and a harder causal question: a shared language-conditioned coordinate trained across distinct atomic actions can identify both `grasp -> lift` and `lift -> place` boundaries and preserve unfinished physical predicates in a three-stage program better than state-only switching, fixed timing, or full future-prompt exposure.

This is not another scalar threshold variant of G34. It introduces a multi-action dataset, a shared action-identity/phase model, two online switches, explicit stage-retention metrics, and a longer control composition.

## Interface and dataset construction

First inspect the exact task-5 BDDL, π0.5 prompt interface, saved demonstrations, gripper/contact observations, and existing rollout code. Implement a three-stage executor using real supported prompts for `grasp the black book`, `lift the black book`, and `place the black book on the right side of the shelf`. If the policy does not respond reliably to the shortest phrase, test a small prompt-family sweep on training starts inside G35 and freeze one prompt per atomic action before held-out evaluation. A prompt/interface failure is an implementation problem within G35 and consumes no new EXP ID.

Collect new train-only causal traces for all three atomic prompts from restorable training starts. Every policy issue row must save dual-view observations, robot/gripper state, object pose, recent native actions, active action identity, physically certified predicates, and the realized next state. Label grasp completion using actual object/gripper contact and closed-gripper evidence, lift completion using sustained object-height gain while grasp is retained, and place completion using the task predicate. Preserve exact checkpoint/controller state and prompt provenance.

Train and cross-fit a shared multi-action coordinate with a language/action embedding, 16-D execution representation, per-action progress coordinate, and completion head. Compare it against:

1. matched per-action direct binary classifiers without a coordinate/ranking loss;
2. explicit physical state predicates using gripper/contact/object height;
3. train-median fixed boundaries;
4. full future prompt visible from the initial state;
5. an oracle-predicate executive as an upper-bound switching comparator;
6. no-F2 and no-feedback ablations where executable through the same causal interface.

Thresholds, persistence, prompts, and any calibration are selected only from train episodes and frozen before prospective evaluation.

## Prospective causal execution

Run at least three explicit-noise repeats over the ten canonical held-out starts for every method. Execute one continuous `grasp -> lift -> place` episode: the second and third actions must begin from their physically reached states, and both switches must be autonomous except for the oracle comparator. Use G33's physical-residual recovery as the fixed F2 backbone for primary methods. Inject the same pair of real 3 cm book disturbances, saving executed disturbed candidates, realized observations, detector decisions, restores/retries, and final metrics.

To make action protection identifiable, record whether grasp contact is achieved before lift authority, whether grasp remains retained through the lift boundary, object height at both switches, premature-stage transitions, lost-grasp events, and final placement. Full-prompt exposure must truly supply all three action semantics from the start; gated methods must supply only the active action to low-level policy control while the executive retains the ordered program.

## Decision rule

The shared action coordinate is supported only if it strictly exceeds both direct binary and physical-state autonomous executives in pooled three-stage success, has positive paired win/loss balance against both, reduces premature transitions or lost-grasp events versus full-prompt exposure, and reaches at least 24/30 successful compositions under perturbation recovery. A tie is not support.

If physical predicates or direct binary switching match or beat it, simplify the final system and treat the learned-coordinate requirement as still unmet. If full-prompt exposure remains best without damaging grasp/lift retention, the current-action-protection premise is falsified for this policy/task family and G36 must move to a different multi-action task whose subgoals have genuinely conflicting controls rather than tuning the same program. If the shared coordinate wins, G36 will ablate language conditioning, temporal ordering, F2 recovery, feedback, and action gating in the integrated three-stage stack.

## Required machine-verifiable evidence

Save the prompt sweep and frozen selection, dataset/split manifest, raw aligned rows, predicate labels, model checkpoints and training logs, cross-fitted predictions and rule sweeps, full simulator/controller checkpoints, all prospective three-stage rollout arrays, two-switch traces, perturbation/restore evidence, explicit seeds, per-case and aggregate metrics, runtime/environment metadata, and an independent audit. The audit must rebuild dataset membership and labels, recompute held-out and online model outputs, verify prompt/stage routing, reconstruct both switch decisions, verify physical interventions, exact restores and native commits, and recompute the declared winner and support status.

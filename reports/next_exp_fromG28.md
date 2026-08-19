# Next Experiment from EXP_G28: EXP_G29 Conservative Action-Effect Coordinates

## Hypothesis

G28 shows that indiscriminately reordering every proposal set harms a strong native pi0.5 baseline: only 12/414 training groups contain more than 0.01 utility improvement over candidate zero. The hypothesis is that an action coordinate supervised by the **realized physical effect of an intervention**, combined with conservative baseline bootstrapping, can exploit the few high-confidence improvements while preserving candidate zero elsewhere. This should outperform an equally conservative raw-action model and the always-rank aligned model under common proposal noise.

## New representation and data construction

Use all audited G24 matched-state branches. For each branch, construct a realized effect target from normalized end-effector displacement, object displacement, realized utility, and success. Train leave-one-attempt-out effect encoders that map current visual/state context plus a native action prefix into a compact bottleneck. Supervise the bottleneck with effect regression and a groupwise contrastive/ranking objective: branches with similar realized effects should be near each other, and the realized best branch in a matched group should have positive advantage over candidate zero.

Train an equal-capacity raw-action advantage model and a direct non-bottleneck advantage regressor as controls. All preprocessing, architecture counts, training logs, fold membership and held-out predictions must be saved. Within G29, use cross-fitted held-out groups to select a finite intervention rule from a declared grid over confidence/advantage gates; this sweep is part of G29 and cannot consume another EXP ID. The selection objective is conservative realized utility improvement over candidate zero, with ranking accuracy as a secondary diagnostic.

## Controller mechanisms

At a prospective decision, request three native pi0.5 proposals under explicit common noise. Compare:

1. conservative action-effect coordinate: select a nonzero candidate only when predicted advantage over candidate zero exceeds the cross-fitted gate and coordinate support/confidence is sufficient;
2. always-rank action-effect coordinate, isolating conservative bootstrapping;
3. conservative equal-capacity raw-action advantage model;
4. conservative direct non-bottleneck effect/advantage regressor;
5. G28 aligned point ranker;
6. physical shooting;
7. single raw pi0.5 candidate zero;
8. initial-observation open loop.

The effect coordinate must affect actual candidate selection in at least some decisions; a controller that always falls back is an invalid mechanism result. Save every predicted coordinate, support/confidence value, candidate-zero advantage, intervention decision, selected native action, realized feedback, F3 state and common-noise seed.

## Prospective protocol

Freeze all folds and the gate before opening new test rollouts. Run three paired repeats on all ten canonical snapshots with autonomous F3 and current-state replanning. Use a new method-neutral explicit-noise seed base and the same `(repeat, attempt, decision, candidate)` schedule for all methods. Save 240 new rollout artifacts plus pooled, per-repeat and paired outcomes. Report intervention frequency and conditional outcomes in addition to ordered success, lift completion, endpoint error, jerk and candidate-executed steps.

## Decision rule

The hypothesis is supported only if the conservative action-effect coordinate strictly exceeds both the conservative raw model and single pi0.5 in pooled ordered successes, has more paired wins than losses against each, and performs at least one genuine nonzero-candidate intervention. It must match or exceed physical shooting with fewer candidate-executed steps, or strictly beat it. The always-rank ablation must be reported even if it wins. If the bottleneck ties or loses to a non-bottleneck/raw control, G30 must not preserve it for narrative reasons; it must move the learned latent to a different system role or simplify F2.

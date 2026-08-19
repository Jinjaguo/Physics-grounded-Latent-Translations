# Next Experiment from EXP_G36: EXP_G37 Cross-Task Action-Latent Transfer

## Hypothesis

G36 showed that a compact latent is unnecessary when a direct physical predictor has dense matched-task supervision. A learned action representation may instead be useful as a shared transfer interface. The G37 hypothesis is that a language-conditioned action/outcome latent trained across distinct LIBERO tasks can adapt to a held-out task with fewer target-task causal branches than task-specific direct regression, while preserving disturbance detection and closed-loop recovery.

## New data and representation

Construct matched-state causal branch datasets for at least three genuinely different supported manipulation tasks, not merely different starts of task 5. Each row must contain task/action language, current robot and relevant-object state, native action prefix, realized end-effector/object displacement, checkpoint identity, and split membership. Branches must be executed in the simulator from restorable physical/controller snapshots. Preserve train-task, adaptation, calibration, and untouched prospective-test membership.

Train and compare within G37:

1. a shared language-conditioned action/effect latent with task-agnostic encoders and a task-conditioned physical decoder;
2. the same shared latent with language/action identity removed;
3. a task-specific direct physical predictor;
4. a pooled direct predictor with task identity;
5. a shuffled-language or shuffled-action/outcome ablation.

Use leave-one-task-out evaluation and a target-task adaptation curve that includes zero-shot and several nontrivial data budgets. Budget changes are a single within-EXP sweep, not separate EXP IDs. Select the operating point using source-task and target adaptation/calibration data only, then freeze it.

## Prospective causal evaluation

On at least two held-out task/action families and multiple untouched starts, execute real native proposals from exact checkpoints, impose controlled object perturbations, observe realized feedback, accept or restore/retry, and continue the actual controller. Compare the selected shared latent, direct task-specific, pooled direct, language-free, shuffled, oracle, and no-recovery mechanisms under matched explicit policy noise. Save every intervention, tried action, realized effect, representation output, decision, exact restore error, and downstream completion metric.

## Decision rule

The cross-task latent is supported only if it strictly improves the prespecified low-data target-task detection/recovery metric over both task-specific and pooled direct predictors on each held-out action family, has positive paired end-to-end balance, and loses its advantage when language or causal pairing is removed. A gain only in offline displacement error is insufficient.

If no supported multi-task branch interface exists, G37 must implement the missing task-generic relevant-object state/snapshot collector before consuming the EXP ID. If transfer still fails after valid causal evaluation, retain direct F2 and make G38 test a different latent intervention role, such as proposal generation constrained by learned action support rather than residual verification.

## Required evidence

Save task manifests, raw branch rollouts, snapshots, split/adaptation membership, checkpoints and logs, adaptation curves, frozen selection, prospective causal rollouts, explicit noise, intervention/restore/action chains, per-task and pooled metrics, paired comparisons, runtime metadata, and an independent audit that reconstructs the datasets and recomputes the conclusion.

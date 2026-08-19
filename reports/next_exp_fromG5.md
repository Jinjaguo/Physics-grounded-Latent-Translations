# Next Experiment from EXP_G5: EXP_G6 Closed-Loop Data Aggregation

## Hypothesis

The G4/G5 policies fail because small errors move them outside the expert-state training distribution. Adding corrective state-action pairs collected on the policy's own train-branch rollouts will improve official development completion beyond 2/10.

## Collection and training

Use only Wave19 train episodes. Select at least two certified train branches per task. Restore each exact checkpoint and roll out the frozen G4 state-plus-latent MLP for the available source horizon. At every learner-visited state, save the complete realized physical feature and rolling latent, and assign the demonstrated source action at the same causal step as the correction label. The label is allowed during train-only aggregation; it must never be used in development deployment.

Compare three training constructions inside G6:

1. original expert-only state-plus-latent MLP (frozen G4 baseline);
2. equal-weight expert plus learner-visited corrective samples;
3. corrective-sample upweighting selected only on a protected development subset.

Refit matched models, preserve collection rollouts and labels, and record how far learner states lie from the original expert feature distribution.

## Evaluation

Execute the selected aggregated policy and the frozen G4 baseline from the unchanged ten late development checkpoints over complete horizons. Both must choose actions only from current realized state, previous executed action, elapsed time, and rolling latent. Source futures remain unavailable online; oracle F3 is retained solely to isolate atomic control.

Primary metric is official success rate. Secondary metrics are completion time, failure target error, jerk, stability, and per-task gains/regressions. G6 is supported only if aggregation exceeds the G4 2/10 result without nonfinite or unstable execution. If it fails, G7 must move to chunked diffusion/transformer or vision-conditioned policy learning, not another weighting sweep.

## Required artifacts

Create `experiments/EXP_G6/` with selected train branches, raw aggregation rollouts, correction labels, feature-distance diagnostics, fitted checkpoints/logs, development rollouts, metrics, exact run/environment records, and an independent audit. Then write the detailed report and executable G7 prompt.

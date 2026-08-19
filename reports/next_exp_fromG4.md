# Next Experiment from EXP_G4: EXP_G5 Recurrent Closed-Loop Sequence Policy

## Hypothesis

The G4 MLP sometimes completes an action but suffers from state aliasing and high-frequency correction because every step is independent. A recurrent policy trained over complete trajectories will preserve manipulation phase and improve long-horizon success and smoothness. A state-plus-latent recurrent policy will outperform an otherwise identical state-only recurrent policy if the action latent supplies additional execution-coordinate information.

## Implementation and training

Use only Wave19 train episodes. For each LIBERO-10 task, construct complete causal sequences from state `t`, previous action, normalized elapsed time, and action `t`. Train matched state-only and state-plus-latent GRU policies with the same hidden size, optimizer, epochs, and output head. The latent variant additionally receives the frozen rolling executed-action latent. Perform any hidden-size/regularization choice inside G5 using train/development only; do not consume a new EXP for scalar tuning.

Save all 20 checkpoints, normalization, training curves, exact episode membership, architecture definitions, and selection records. Do not open the test split.

## Closed-loop intervention

Evaluate on the exact G3/G4 late development checkpoints and complete remaining horizons. Initialize each GRU hidden state by replaying the saved causal pre-branch physical/action history from the start of that same episode up to the branch. This history is available before the intervention and must not include future information. After the branch, update the recurrent state only with the actually realized simulator state, previous executed control, time, and—only for the latent condition—the rolling latent of executed controls.

Compare source oracle, G4 state-only MLP, G4 state-plus-latent MLP, state-only GRU, and state-plus-latent GRU. Execute every proposed control through LIBERO, log realized feedback, and stop only at oracle current-action completion, environment termination, or the fixed per-case horizon.

## Decision rule

Primary metric is official success rate; secondary metrics are completion steps, failure target error, jerk, stability, and per-task outcome. Recurrent memory is supported only if a GRU exceeds its matched MLP success without instability. Latent contribution is supported only if the latent GRU beats the state-only GRU. If neither improves beyond 2/10, G6 must use online recovery/data aggregation or a stronger chunked/vision-conditioned policy rather than retuning the same GRU.

## Required evidence

Create `experiments/EXP_G5/` containing run/environment records, train/development membership, normalization, checkpoints/logs, all closed-loop rollouts, per-case and aggregate metrics, and an independent audit. Then write the detailed G5 report and an executable next-experiment prompt derived from the actual winner.

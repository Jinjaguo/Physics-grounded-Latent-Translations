# Next Experiment from EXP_G15: EXP_G16 Decodable Latent-Chunk Shooting

## Evidence-driven pivot

G15 supports feedback and gives causal F2 a 10/10 versus 9/10 advantage over single sampling, but the outcome-history latent adds nothing beyond state F3. The latent currently describes past execution and only enters completion detection; it is not a coordinate through which control is proposed. EXP_G16 changes that mechanism and the F1/F2/F3 responsibilities.

- F1 becomes the current-action π0.5 base chunk generator.
- F2 searches a learned, decodable action-chunk coordinate around that base, executes each decoded candidate from the exact current snapshot, scores realized progress, and commits the winner.
- F3 uses the simpler frozen state-window temporal switcher because G15 rejected the need for a history latent in F3.

This design is influenced by the state-conditional latent action planning idea in [TAP](https://arxiv.org/abs/2208.10291) and by recent temporally grounded continuous latent-action work such as [LatentVLA](https://doi.org/10.1609/aaai.v40i22.38926). The experiment must implement and execute the local mechanism here; literature is not evidence.

## Representation training and development selection

From only the 14 task-5 training demonstrations, construct overlapping 10×7 action chunks separately labeled by lift/place phase. Fit at least three genuinely different action-coordinate families in this one EXP:

1. phase-conditioned PCA with a low-dimensional continuous code and exact linear decoder;
2. deterministic phase-conditioned autoencoder with reconstruction and temporal smoothness losses;
3. phase-conditioned beta-VAE with a continuous stochastic code, action reconstruction, and KL regularization.

Preserve continuous robot commands, but map decoded gripper commands back to the supported ±1 control values and clip the six arm controls to the real controller bounds. Save dataset membership, normalizers, checkpoints, train curves, latent dimension, and reconstruction/rollout diagnostics.

Use only the already-open G15 cardinal cohort for development. At each decision, query one π0.5 base chunk, encode it, and create a fixed-budget latent candidate set containing the decoded base plus symmetric coordinate perturbations. Physically branch-execute each candidate from the same complete simulator snapshot. Select the representation family and search scale by development official success first, then endpoint error, using a systematic within-EXP sweep. Do not use the harder test perturbations during this selection.

## Frozen harder perturbation test

Before opening test rollouts, freeze the selected representation, F2 search parameters, state-F3 checkpoint, value objective, candidate budget, and a new ten-case perturbation manifest derived from G14 snapshots. Use predeclared 6 cm cardinal shifts with directions disjoint from the same-attempt G15 direction where possible. Construct all exact snapshots before any evaluated rollout and apply the same 5 mm XY/Z physical-validity rule used by repaired G15. If 6 cm is physically invalid for a case, replace the entire test magnitude before rollout using a predeclared fallback such as 5 cm; do not adapt magnitude based on controller outcomes.

Execute these matched methods on every frozen valid test snapshot:

1. selected latent-coordinate shooting with state F3;
2. equal-candidate-budget independent raw π0.5 proposal shooting with state F3;
3. equal-candidate-budget raw-action Gaussian perturbation shooting around one π0.5 chunk;
4. decoded-base-only latent projection, with no shooting;
5. single raw π0.5 proposal with state F3;
6. initial-observation open loop;
7. same-attempt unperturbed action replay.

All candidate evaluations must be actual branch executions. Save the base chunk, encoded coordinate, each intervened coordinate or raw perturbation, decoded/executed actions, realized candidate endpoint and score, selected index, committed prefix, next observation/state/latent, switch state, and final metrics.

## Metrics and decisions

Primary metric is official ordered lift→place success on the frozen harder perturbations. Secondary metrics are lift completion/retention, premature/delayed switch, endpoint error, action jerk, candidate/committed simulator steps, wall time, and per-direction behavior.

The learned action latent contributes only if latent-coordinate shooting strictly beats both equal-budget raw π0.5 shooting and equal-budget raw-action perturbation in success. It must also beat decoded-base-only projection, showing that intervention/search in the coordinate matters rather than mere denoising. Ties do not support the latent claim. Feedback and retargeting claims additionally require gains over open loop and action replay.

If a raw planner wins, simplify around raw action-space causal MPC and reject the latent-coordinate thesis for this formulation. If a latent family wins development but not frozen test, record overfitting and in EXP_G17 change the representation training target to realized action-conditioned transitions rather than retuning search scale on test.

## Required artifacts

Write new artifacts under `experiments/EXP_G16/`: chunk dataset/split manifests, all fitted representation checkpoints and logs, development intervention traces and selection manifest, frozen-before-test manifest, physically validated test snapshots, all matched test rollouts, exact command/environment metadata, aggregate metrics, and an independent audit that reconstructs latent decode/proposal/commit chains and all baselines. Then create the G16 report and next executable experiment.

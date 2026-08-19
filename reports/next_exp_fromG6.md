# Next Experiment from EXP_G6: EXP_G7 Multimodal Action-Chunk Control

## Hypothesis

One-step mean regression averages incompatible manipulation modes and cannot recover contact sequences. Predicting a coherent multi-step action chunk, with an explicit mixture over modes, will improve long-horizon completion while receding execution preserves physical feedback.

## Models and training

Use Wave19 train episodes only. For each task, pair the current realized state feature with the next 16 demonstrated controls. Train and compare within G7:

1. deterministic state-plus-latent chunk MLP;
2. a capacity-matched state-only chunk MLP;
3. a state-plus-latent mixture-density network with at least four action-chunk modes and learned mixture weights/diagonal variance.

Select architecture and mode count within the same EXP using train/development tasks 0–4. Save exact window construction, normalizations, checkpoints, and training likelihood/MSE curves. Do not open the test split.

## Closed-loop execution

From the unchanged late development checkpoints, predict a 16-step chunk, execute only its first four controls through LIBERO, observe the realized state, update the rolling executed-action latent, and predict again. The mixture controller must choose its highest-probability mode without oracle future state or action. Stop at oracle current-action completion, termination, or the fixed case horizon.

Compare the three new controllers with the G4 2/10 state-plus-latent MLP and source reference. Primary metric is official success; secondary metrics are completion time, error, jerk, stability, mode usage, and per-task outcomes. A latent contribution requires the latent deterministic model to beat its state-only match. Multimodality is supported only if the mixture beats deterministic latent chunk control.

If G7 does not exceed 2/10, G8 must add visual observation features or train a modern chunked policy from the available image/state/action data rather than another deterministic state regressor.

## Evidence

Create `experiments/EXP_G7/` with exact splits, window manifests, checkpoints/logs, all actual rollouts and mode choices, metrics, environment/run records, and an independent audit. Produce the G7 report and evidence-derived next experiment afterward.

# Next Experiment from EXP_G7: EXP_G8 Dual-Camera Visual Chunk Policy

## Hypothesis

Failures persist because body-state vectors do not encode object identity, occlusion, or task-relevant contact appearance. A dual-camera visual policy trained at the source policy's native issue timebase will exceed the 2/10 state-policy ceiling; adding the learned action latent will further improve execution phase control.

## Training

Use only Wave19 train episodes and their saved `policy_observations.npz` plus `postprocessed_policy_action_chunks.npy`. Train a shared task-conditioned lightweight CNN chunk policy using processed agent-view RGB, wrist RGB, saved 8-D robot state, and task identity. Train a matched variant adding the rolling frozen action latent at each issue step. Emit 10×7 controls and convert gripper output to the real sign command before execution and jerk evaluation.

Downsample images consistently for tractable training, recording the exact transformation. Use development tasks 0–4 for any architecture choice and retain tasks 5–9 as confirmation. Do not open the test split.

## Execution and decision

From the unchanged late development checkpoints, capture current dual-camera images and robot state, apply the same orientation/resize convention as training, predict ten controls, execute five through LIBERO, observe, update history/latent, and replan. No future source state/action is allowed online; oracle F3 remains only for atomic completion isolation.

Compare visual state, visual state+latent, G4 state+latent MLP, and source reference. Primary metric is official success; secondary metrics include confirmation success, error, jerk, stability, and visual/latent ablations. Visual control is supported only if a visual model exceeds 2/10. Latent contribution requires the matched latent model to beat visual state alone.

If G8 succeeds robustly, G9 should construct explicit atomic `lift -> place` composition with oracle switching. If it fails, G9 must test a stronger pretrained visual policy or the available π0.5 source policy as the low-level controller while retaining latent intervention/switching ablations.

Save all training windows, checkpoints/logs, actual rollouts, commands, metrics, and an independent audit under `experiments/EXP_G8/`, followed by the report and next prompt.

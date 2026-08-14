# Next experiment after wave 19

Do not run LIBERO F1/F2 from the failed Wave-19 representation and do not round the `1.200444393` motor-MSE ratio down to the frozen `1.2` threshold. The final 50-episode LIBERO test split remains unopened.

Wave 20 should prospectively test one motor-margin change: use the same 32=16+16 action-only representation and six new seeds, but train correct-language models with `L = 2 × L_reconstruction + L_semantic` against paired reconstruction-only anchors. Collect 5 new certified official-LIBERO-10 episodes per task under a new registered π0.5 seed as a fresh confirmation-development set; do not merge them into the current test split or sweep loss weights on them.

Require positive clustered semantic deltas in both directions, continuous MSE ratio ≤1.15, gripper accuracy drop ≤0.02, six complete seeds, and finite outputs. If this gate fails, stop the representation family. If it passes, freeze one checkpoint, run the already-specified F1/F2 O1–O8 gate once, and open the untouched Wave-19 final test only if offline replication passes. Full protocol: `results/dynamics/nineteenth_wave/2026-08-14_dynamics_7/nineteenth_wave_next_experiment.md`.

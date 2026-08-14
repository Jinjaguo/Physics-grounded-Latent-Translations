# Next experiment after wave 20

Wave 20 passed the independent LIBERO representation gate but rejected the offline O1–O8 dynamics gate. Do not retrain the representation, add seeds, round the O1 confidence interval, or open the untouched final test.

Wave 21 should test one preregistered mechanism change without optimizer updates: project each frozen F2 correction onto the local execution-latent tangent space defined by 20 nearest Wave-19 train latents and the smallest PCA basis explaining 90% variance. Keep seed-202820 representation, semantic/F1/F2 checkpoints, four iterations, and step size 0.01 frozen. Do not sweep neighbors, variance threshold, or projection strength.

Evaluate F1, original F2, and tangent-projected F2 on the 50 Wave-20 confirmation episodes, which have not been used for dynamics training or dynamics evaluation. Require unchanged O1–O8 plus H8 normal distance no worse than original F2. If the gate fails, stop the current refinement family. If it passes, freeze hashes and open the untouched final test once for B0–B5 and proposal recovery. Full protocol: `results/dynamics/twentieth_wave/2026-08-14_dynamics_8/twentieth_wave_next_experiment.md`.

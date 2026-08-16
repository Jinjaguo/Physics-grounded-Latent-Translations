# Next experiment from EXP_R1

EXP_R1 decision: **NOT_SUPPORTED**. Development selected `traj_full`; the held-out result is preserved in `results/EXP_R1/heldout_metrics.json`.

If the path-planning gate failed, the likely bottleneck is the connection geometry or the target-region construction, not language representation itself. EXP_R2 should keep the representation, decoder, and F1/F2 checkpoints frozen, then test a broader goal-conditioned graph/trajectory formulation with (1) source-conditioned target regions, (2) explicit graph route plus continuous smoothing, (3) horizon H=2/4/8, and (4) target-set rather than centroid costs. It must retain interpolation, F1, old F2, graph, and full trajectory-optimization baselines and open a new held-out evaluation only after development selection. Do not return to a pointwise force-field sweep.

If the gate passed, the next experiment should remove exact endpoint knowledge, keep oracle switching, and test language-derived target regions before introducing learned F3.

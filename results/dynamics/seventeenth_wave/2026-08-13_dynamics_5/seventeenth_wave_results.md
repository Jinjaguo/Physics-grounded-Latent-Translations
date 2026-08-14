# PGLT wave-17 continuous-play long-horizon experiment

## Outcome

Wave 17 reconstructed **11** physically continuous VyoJ CALVIN play sessions and retained **61** non-overlapping 160-frame blocks. The frozen causal Protocol A had H1/H2/H4/H8 support of **303/300/290/61** starts.

The source-session-clustered primary AUC comparison was F1 **8.438474** versus F2 **7.422854**, Delta(F2-F1) **-1.015620**, 95% CI **[-1.211956, -0.800167]**. Therefore C3c-long is **SUPPORTED**, C3d is **SUPPORTED**, and context dependency is **ROBUST_TO_BOUNDARIES**.

## Frozen Protocol-A metrics

| metric | F1 H1 | F2 H1 | F1 H2 | F2 H2 | F1 H4 | F2 H4 | F1 H8 | F2 H8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| execution MSE | 0.963563 | 0.878906 | 1.223159 | 1.116637 | 1.257765 | 1.137231 | 1.420660 | 1.140924 |
| decoded continuous MSE | 0.041035 | 0.038050 | 0.052049 | 0.048752 | 0.054265 | 0.050858 | 0.052766 | 0.048044 |
| execution kNN radius | 2.287009 | 2.149729 | 2.060329 | 1.826553 | 2.188102 | 1.849043 | 2.859241 | 2.195206 |

## Required questions

1. Physically continuous sessions reconstructed: **11**.
2. Eligible >=10-window blocks: **61**.
3. Blocks with 0/1/2+ annotation boundaries: **0/0/61**.
4. Protocol-A H1/H2/H4/H8 starts: **303/300/290/61**.
5. Primary blocks crossing a reset/discontinuity: **none**; every block stayed inside one authoritative session row and had contiguous source frames.
6. Block construction frozen before H4/H8 inference: **yes**; the manifest and prospective preregistration hashes were written first.
7. F1/F2 completely frozen: **yes**; representation, semantic predictor, F1, F2, and EMA all had zero updates and before/after tensor hashes matched.
8. Protocol-A session AUC upper CI below zero: **yes**, CI upper bound -0.800167.
9. F2 beats F1 at H4 execution MSE: **yes**.
10. F2 beats F1 at H8 execution MSE: **yes**.
11. F2 reduces H8 decoded-action error: **yes**.
12. F2 reduces H8 execution kNN radius: **yes**.
13. F2 advantage across annotation boundaries: **ROBUST_TO_BOUNDARIES**; boundary H8 starts=61 across 11 sessions, clustered upper CI=-0.978302.
14. Benefit without future task labels: **yes** under causal held context.
15. Protocol-B context effect: On 3 common H8 starts, exogenous-minus-causal normalized error was -0.034167 for F1 and -0.041979 for F2.
16. H4/H8 correction-target cosine: mean **0.143734**, positive fraction **0.664**.
17. Iteration behavior at H4/H8: execution error, decoded error, and kNN radius decrease from iteration 0 to 4 at both horizons. Step-local empirical normal distance decreases at H4 but rises slightly at H8; nevertheless the complete F2 rollout lowers H4/H8 normal distance relative to F1 by mean **0.329416**.
18. C3c-long: **SUPPORTED**.
19. C3d: **SUPPORTED**.
20. Robust to semantic task boundaries: **ROBUST_TO_BOUNDARIES**.
21. DEL remains a permanent negative baseline: **yes**; it was not run, tuned, or rescued.
22. Defensible paper story: **Language anchors action meaning; refinement stabilizes continuous latent evolution across continuous robot motion and semantic task boundaries.**
23. Additional data needed: **not for the preregistered wave-17 gate**.

## Interpretation constraints

Protocol A is the primary autonomous comparison. Protocol B is an exogenous-context diagnostic and is not autonomous task planning. Wave-17/wave-16 source-overlap pairs: **27**; therefore wave-17 H1/H2 are not described as another independent replication, and the novel confirmatory evidence is H4/H8. Public annotations were never concatenated; language boundaries were used only as metadata after physical continuity was established. No future action or robot state was read by either rollout.

# Twenty-fourth wave results: transition displacement families

Run date: 2026-08-14T18:45:18.822736-04:00

## Outcome

- M2 state/horizon-conditioned displacement family: **REJECTED**
- C13/C14: **NOT_TESTED**
- LCT-TD optimizer steps: **0**
- Held-out arrays materialized: **false**

A2/A3 passed: state-conditioned paired displacements predict direction and beat a goal+horizon mean. A1/A4/A5/A6 failed: horizon cores were not better, D2 lost to prototype/B1 on required errors, endpoint identity degraded, and continuity worsened. The characteristic norm shrinkage indicates that weighted averaging is canceling heterogeneous displacement modes.

## Required questions

1. 560 metadata rows were reconstructed for H1/H2/H4; 396 train/development records contain paired arrays and 164 test rows remain masked.
2. 31 source sessions in metadata; Phase A materialized 25 train/development sessions.
3. All 18 goal/horizon cells are adequate; train counts are at least 25 and K=20 is available.
4. No. Aggregate HorizonCoreGain=-0.698418; static Wave23 cores were closer.
5. No. Its clustered lower95=-0.810751.
6. Yes directionally. Both full and execution cosine are positive with lower bounds above zero.
7. Full cosine=0.627467 [0.599828, 0.655551].
8. Execution cosine=0.647801 [0.619503, 0.674799].
9. Yes. Goal-mean minus D2 full MSE=0.486718, lower95=0.396073.
10. Yes. Execution improvement=0.682711, lower95=0.538168.
11. Yes for predicting displacement relative to a goal+horizon mean, but not strongly enough to authorize the complete executable-transition mechanism.
12. No. D2 H2 full MSE=1.208116 exceeded prototype=0.677964; H4 decoded MSE also exceeded prototype.
13. No jointly. Endpoint macro 0.478711 < B1 0.522248, although decode/reencode improved.
14. No. D2 continuity=0.267568 > B1=0.185518.
15. No. M2 is REJECTED (A1/A4/A5/A6 failed).
16. None; LCT-TD training was forbidden.
17. Not tested; no LCT-TD exists.
18. Not tested; no LCT-TD exists.
19. Not tested; no LCT-TD exists.
20. Not tested; no LCT-TD exists.
21. Development D2 family margins are reported, but held-out lower95 was not opened.
22. Not tested held-out.
23. Not tested held-out.
24. Direction remained positive at H1/H2/H4, but the full mechanism failed jointly.
25. Source-state directional benefit is broad descriptively; C13 breadth was not tested.
26. Held-out was masked. 6 development cases were analyzed at all horizons.
27. C13 is NOT_TESTED.
28. C14 is NOT_TESTED.
29. Waves21–23 are best explained by a multimodal, state-dependent transition distribution: language changes direction and local source states inform it, but deterministic neighborhood averaging cancels modes and underestimates magnitude; static/horizon endpoint sets do not repair this.
30. Defensible claim: source-conditioned paired train transitions predict development displacement direction and outperform a goal+horizon mean, but deterministic averaged displacement is insufficient for executable identity and continuity.
31. If C13 had passed, the next step would be matched-state closed-loop CALVIN execution with receding-horizon decoding, frozen B1/LCT-TD controls, and no rescue.
32. The causal language-vector-field effect and execution redirection remain supported; Wave24 adds that current-state neighbors carry predictive directional information, while an executable state/horizon selector remains unsupported.

## Scientific decision

Do not train LCT-TD with a softmin-to-mean displacement target. The surviving signal is distributional: current state narrows the transition family, but a single averaged vector is not an adequate estimator of its executable mode.

## Discipline disclosure

Train built S1/S2/S3 and tau; development alone decided M2. Test rows remained null in Parquet. No new K/tau, lambda sweep, loss, seed, closed-loop rollout, F2, DEL, cycle rescue, or endpoint attraction was introduced.

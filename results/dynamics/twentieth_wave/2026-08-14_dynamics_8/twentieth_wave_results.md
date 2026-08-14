# Twentieth-wave results: prospective LIBERO motor-margin adjudication

## Outcome

Wave 20 passed the preregistered independent LIBERO representation gate and then failed the frozen offline O1–O8
dynamics authorization. The experiment stopped before opening the old 50-episode final test, so no closed-loop
claim—positive or negative—is available.

## Frozen data and fresh collection

The Wave-19 140 train / 50 development / 50 final-test membership and manifest hashes were preserved exactly.
Wave 20 collected a genuinely new confirmation-development set using the same official `libero_10` suite and
fixed official π0.5 checkpoint, with policy seed `200820`, new environment seeds, formal Numba JIT, immutable action
copies, and corrected `mj_setState -> mj_forward -> mj_setState` restoration.

| task | attempts | official successes | certified confirmation episodes |
|---:|---:|---:|---:|
| 0 | 5 | 5 | 5 |
| 1 | 5 | 5 | 5 |
| 2 | 5 | 5 | 5 |
| 3 | 6 | 5 | 5 |
| 4 | 5 | 5 | 5 |
| 5 | 7 | 7 | 5 |
| 6 | 6 | 5 | 5 |
| 7 | 5 | 5 | 5 |
| 8 | 8 | 5 | 5 |
| 9 | 5 | 5 | 5 |
| **total** | **57** | **52** | **50** |

Two short official task-5 successes lacked the frozen future support and remained raw but uncertified. All 72
eligible admitted branches reproduced integration state, controller state, object state, predicates, and terminal
success exactly; maximum discrepancies were zero.

## Representation adjudication

Six new preregistered seeds trained R0 and R1 from scratch on only the unchanged 140-episode train split. The only
scientific change was R1 objective `2.0*L_reconstruction + L_semantic`; an identically weighted shuffled-language
model was used only for the frozen semantic control. Every model used the unchanged 32=16+16 action-only
architecture, 40 epochs, EMA 0.999, frozen OpenCLIP text features, and the Wave-19 normalization.

| gate metric | result | threshold |
|---|---:|---:|
| action-to-text mean delta | 0.910000000 | >0 |
| action-to-text lower 95% | 0.896666667 | >0 |
| text-to-action mean delta | 0.916666667 | >0 |
| text-to-action lower 95% | 0.866666667 | >0 |
| R1 continuous MSE | 0.00210118968 | — |
| R0 continuous MSE | 0.00186076100 | — |
| motor ratio | 1.129209867 | ≤1.15 |
| gripper accuracy drop | 0.000441054 | ≤0.02 |

The representation gate passed. Translation and rotation ratios were `1.129332969` and `1.126310067`. The largest
per-dimension penalty was action dimension 1 at `1.186737853`; dimension 5 was next at `1.153354534`. EMA improved
the motor ratio for five of six seeds, not all six. Five of six new seed ratios were below the Wave-19 aggregate
ratio `1.200444393`; stronger reconstruction pressure therefore reduced the penalty consistently in the aggregate
and in five seeds, but not universally. The frozen median-ratio selection rule chose seed `202820` (ratio
`1.106211902`).

## Offline dynamics O1–O8

The selected representation was frozen. Semantic, F1, and exact-F1-initialized four-iteration F2 models were
trained exactly once on the old 140-episode train split and evaluated on the old 50-episode development split.
Fresh confirmation episodes did not enter dynamics training or this gate.

| condition | result | evidence |
|---|:---:|---|
| O1 clustered ΔAUC upper95 < 0 | FAIL | mean `-0.131295617`, CI `[-0.271117622, 0.001894005]` |
| O2 F2 H4 execution MSE < F1 | PASS | `1.468828646 < 1.523426126` |
| O3 F2 H8 execution MSE < F1 | FAIL | `2.153231758 > 2.152713848` |
| O4 F2 H8 decoded MSE < F1 | PASS | `0.061540555 < 0.063421833` |
| O5 F2 H8 kNN radius < F1 | FAIL | `2.406725152 > 2.288421908` |
| O6 mean correction-target cosine > 0 | PASS | `0.125322898` |
| O7 positive correction fraction > 0.5 | PASS | `0.633068081` |
| O8 F2 normal distance < F1 | FAIL | `1.223320502 > 1.165033846` |

The mixed result is informative: F2 corrections were directionally aligned and improved decoded error, but did not
remain on the empirical execution manifold at long horizons. The frozen gate was rejected. No final-test opening
manifest was created, and B0–B5, direction controls, proposal recovery, and official closed-loop success were not
run.

## Required final questions

1. Yes. The Wave-19 140/50/50 memberships and manifest hashes were preserved exactly.
2. Yes. The old final 50 remained unopened; offline authorization failed.
3. Yes. Exactly 50 new certified episodes, five per official task, were collected.
4. Yes. All 72 fresh admitted branches reproduced source continuations with zero discrepancy.
5. Yes. Six new seeds were frozen before fresh confirmation evaluation.
6. Yes. The sole scientific representation change was `2*L_rec + L_sem`.
7. Yes. A2T/T2A deltas were `0.91`/`0.916667`.
8. Yes. Lower bounds were `0.896667`/`0.866667`.
9. The continuous MSE ratio was `1.129209867202268`.
10. Yes. It cleared 1.15 by `0.020790132798` ratio units.
11. The gripper accuracy drop was `0.000441054379411`.
12. Yes, all six seeds and all registered conditions completed with finite outputs.
13. Representation gate: **PASS**.
14. Dimension 1 contributed the largest relative penalty (`1.186738`), followed by dimension 5 (`1.153355`).
15. No. EMA improved the ratio for five of six seeds; seed `202820` worsened from raw `1.026062` to EMA `1.106212`.
16. Aggregate yes, and five of six seeds beat the Wave-19 ratio; not universally across seeds.
17. EMA epoch-40 seed `202820` was frozen by the preregistered lower-central median motor-ratio rule among seeds
    with finite outputs and positive bidirectional semantic deltas.
18. No. F2 passed O2/O4/O6/O7 but failed the full O1–O8 gate.
19. ΔAUC(F2−F1) was `-0.131295617`, CI `[-0.271117622, 0.001894005]`.
20. Not tested; final closed loop was not authorized.
21. Not tested; B3–B5 controls require final-test authorization.
22. Not tested; proposal recovery requires final-test authorization.
23. Independent semantic addressability and motor fidelity replicated. The long-horizon refinement advantage did
    not pass the frozen LIBERO offline gate.
24. Defensible story: the semantic/executable action-coordinate representation replicates on LIBERO with a real
    motor margin, but CALVIN-style iterative refinement has not replicated under the frozen LIBERO dynamics gate.
25. Before submission, a separately preregistered mechanism test must resolve the off-manifold refinement pattern;
    final-test closed loop must remain sealed until a fresh offline gate passes.

## Bottom line

Wave 20 answered its primary representation question positively and its conditional dynamics question negatively.
It strengthens the cross-domain representation claim, but does not authorize a cross-domain refinement or
closed-loop claim.

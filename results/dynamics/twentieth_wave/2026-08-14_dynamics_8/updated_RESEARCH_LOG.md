# RESEARCH_LOG

## 2026-08-12T05:40:01-04:00 — dynamics_1

Completed the full thirteenth-wave frozen-latent dynamics protocol. Block-A winner: matched_refinement; Block-B winner: history_mlp; variational evidence: not supported. Official validation was read once after confirmation-manifest freeze. See `reports/dynamics_1_results.md`.

## 2026-08-12T17:54:08-04:00 — dynamics_2

Completed frozen DEL failure adjudication. Diagnosis: variational_model_mismatch. No learned model was retrained; validation was descriptive only. See `reports/dynamics_2_results.md`.

## 2026-08-12T22:23:37-04:00 — dynamics_3 / wave 15

Completed the full factorized executable-subspace experiment. Development hard gate: FAIL; C3b: REJECTED; C3c: SUPPORTED. Official validation was read exactly once after manifest freeze. See `reports/dynamics_3_results.md`.

## 2026-08-12T23:02:40-04:00 — dynamics_4 / wave 16

Exhaustive long-trajectory audit found 0 eligible >=10-window annotation-consistent segments. The >=60 data gate failed; the supported CALVIN collector requires unavailable VR infrastructure, so all prospective F1/F2 metrics remained unread. C3c-local stays supported; C3c-long and C3d are not evaluated due to insufficient data. See `reports/dynamics_4_results.md`.

## 2026-08-13T14:30:11-04:00 — dynamics_4 / wave 16

Resumed the open-data audit at Tier 1B. Audited 4,836 RoboVerse trajectories and the source-wide 22,966-record VyoJ ABCD annotation table from staged subset_training_023; every direct six-task annotation is shorter than 160 frames. The open-data count is 0/task and 0 total, so the adequacy gate blocked all primary F1/F2 inference. C3c-long is NOT_TESTED_INSUFFICIENT_DATA and C3d is NOT_TESTED; the exact fallback is 10 new CALVIN-compatible segments for each of six tasks. See `reports/dynamics_4_results.md`.

## 2026-08-13T15:09:29-04:00 — dynamics_4 amended public external replication

Completed the amended public-data H1/H2 external replication on 60 VyoJ CALVIN segments (10/task, four non-overlapping H16 windows each). This experiment evaluated **H1 and H2 only**; **H4 and H8 were not run**. F1 mean AUC=0.728379, F2 mean AUC=0.632797, paired Delta=F2-F1 -0.095582 with 95% CI [-0.113417, -0.079462]; gate=PASS. C3c-local=STRENGTHENED_BY_INDEPENDENT_PUBLIC_EXTERNAL_REPLICATION; C3c-long remains NOT_TESTED.

## 2026-08-13T21:51:06-04:00 — dynamics_5 continuous-play H1/H2/H4/H8

Reconstructed 11 public VyoJ CALVIN source sessions and evaluated 61 non-overlapping 160-frame continuous blocks. This wave ran H1, H2, H4, and H8 under Protocol A causal held context, with Protocol B explicitly secondary/exogenous. Session-clustered Protocol-A Delta AUC=-1.015620, 95% CI [-1.211956, -0.800167]. C3c-long=SUPPORTED; C3d=SUPPORTED; context_dependency=ROBUST_TO_BOUNDARIES. Representation/F1/F2/DEL were not trained; DEL was not run.

## 2026-08-14T03:52:04-04:00 — dynamics_6 / wave 18 reconstruction gate

The planned closed-loop causal continuation study could not be executed because the retained public CALVIN artifacts do not permit exact reconstruction of source branch states. The mandatory Phase-0 audit replayed 336 recorded transitions across one held-out validation annotation for each of the six canonical tasks. Continuous state and terminal predicates matched, but every pair differed by one exposed contact point; more fundamentally, the retained sources contain 0 exactly reconstructable independent source episodes versus the required 180. The technical reconstruction gate is `FAIL`, but **closed-loop refinement did not fail—it was not evaluated**: representation/F1/F2/DEL were never loaded, and C4/C5/C6 are `NOT_TESTED_RECONSTRUCTION_GATE_FAILURE`. The first diagnostic attempt was preserved and invalidated because a mutable action array was accidentally scaled twice for twin B; the corrected run used independent action copies without changing preregistration or sample selection. See `reports/dynamics_6_results.md`.

## 2026-08-14T10:30:14-04:00 — dynamics_7 / wave 19 official LIBERO-10

Resolved internal `LIBERO-long` to official `libero_10` and prospectively collected 335 fixed-π0.5 attempts: 306 official successes, 29 failures, 297 certified episodes, and 415 exact branches. All admitted twin/source replays had zero integration/controller/object discrepancy. Task 8 reached 24 certified episodes at the frozen 50-attempt cap, yielding a balanced 240-episode 140/50/50 split. The six-seed representation showed strong semantic deltas (0.94 action-to-text, 0.90 text-to-action) and passed gripper fidelity, but continuous MSE ratio `1.200444393` exceeded the frozen `1.2` maximum. The R-gate failed; F1/F2, offline O1–O8, B0–B5, and perturbation recovery were not run, and the final test split remained unopened. See `reports/dynamics_7_results.md`.

## 2026-08-14 — dynamics_8 / wave 20 motor-margin adjudication

Preserved the Wave-19 140/50/50 split and collected 50 fresh official-LIBERO-10 confirmation episodes (5/task) in 57 attempts. All 72 eligible fresh branches reproduced integration/controller/object state and predicates exactly. Six new paired seeds with `2*L_rec + L_sem` passed the stricter representation gate: A2T/T2A deltas `0.91`/`0.916667`, motor ratio `1.129209867 <= 1.15`, and gripper drop `0.000441054`. The selected seed `202820` then authorized one frozen F1/F2 run. Offline ΔAUC(F2−F1) was `-0.131295617`, 95% CI `[-0.271117622, 0.001894005]`; O1, O3, O5, and O8 failed, so O1–O8 was rejected. The final test remained unopened; closed-loop B0–B5 and proposal recovery were not tested. See `reports/dynamics_8_results.md`.

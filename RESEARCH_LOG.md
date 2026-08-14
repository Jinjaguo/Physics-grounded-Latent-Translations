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

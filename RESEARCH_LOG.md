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

## Wave 21 — Language-conditioned latent transition (2026-08-14)

- Executed `prompts/dynamics_9.md` on official CALVIN continuous play.
- Audited 560 annotation-onset transitions across 31 physically continuous sessions; official labels are sparse and annotation gaps are retained.
- Frozen seed-810 CALVIN representation/decoder/text projection; trained B0/B1/B2 with six preregistered seeds and no target-region loss.
- C7: **REJECTED**; C8: **REJECTED**.
- RedirectGain=0.250126 [0.136495, 0.370798]; execution=0.183855 [0.100917, 0.263777].
- Full artifacts: `results/dynamics/twenty_first_wave/2026-08-14_dynamics_9`.

## Wave 22 — Executable coordinate consistency (2026-08-14)

- Executed `prompts/dynamics_10.md` with frozen Wave21 B1/representation/decoder/text projection.
- Phase A: A1-A4 passed; A5 failed. M0 **REJECTED**; C9/C10 **NOT_TESTED**.
- Frozen K=4 cycle projection reduced residual 2.939692→0.272356, but full RedirectGain became 0.094419 [-0.000694, 0.171879] and endpoint accuracy fell.
- No Wave22 optimizer, lambda sweep, checkpoint, or held-out LCT-CC prediction was created.
- Full local artifacts: `results/dynamics/twenty_second_wave/2026-08-14_dynamics_10`.

## Wave 23 — Goal-specific executable alignment (2026-08-14)

- Built exact train-only 75% goal cores (118–243 points/class), K=20.
- M1 **SUPPORTED_FOR_INTERVENTION**: all five development-only gates passed.
- Trained 18 LCT-GA models (3 lambdas × 6 paired seeds); no candidate passed both +0.05 identity improvements.
- Development λ=0.03 preserved redirects and continuity but endpoint/decode-reencode accuracy worsened.
- C11/C12 **NOT_TESTED**; Wave23 held-out test remained unopened.
- Full artifacts: `results/dynamics/twenty_third_wave/2026-08-14_dynamics_11`.

## Wave 24 — State/horizon-conditioned displacement families (2026-08-14)

- Reconstructed 560 transition metadata rows; materialized 396 train/dev paired records and kept 164 test rows masked.
- All 18 goal/horizon train cells adequate; K=20, tau train-only.
- M2 **REJECTED**: A2/A3 passed, A1/A4/A5/A6 failed.
- D2 cosine full/execution=0.627467/0.647801; it beat goal+horizon means but shrank displacement magnitude and failed identity/continuity.
- C13/C14 **NOT_TESTED**; no optimizer or held-out materialization.
- Execution note: the first direct `report` command stopped before experiment code at `ModuleNotFoundError` because `PYTHONPATH` was absent; rerunning with `PYTHONPATH=.:src` completed the report.
- Full artifacts: `results/dynamics/twenty_fourth_wave/2026-08-14_dynamics_12`.

## Wave 25 — Broad transition implementation sweep (2026-08-14)

- Compared 66 causal candidates across deterministic, mode, mixture, retrieval, cVAE, flow, diffusion, and phase-aware families.
- Development eligible=0; selected=[].
- C15=NOT_TESTED_NO_DEVELOPMENT_CANDIDATE; C16=NOT_TESTED; C17=NOT_TESTED.
- Held-out opened after preregistration=False.
- Execution issues and discarded partial runs are documented in `wave25_execution_log.md`; the final sweep used corrected H4 indexing and seed-before-reset initialization.
- Full artifacts: `results/dynamics/twenty_fifth_wave/2026-08-14_dynamics_13`.

## Wave 26 — Rich causal state × structured continuous flow (2026-08-14)

- Ran 79 development entries across S0–S7 audit, eight flow families, objectives, non-flow controls, and D0/D1/D2.
- D3 and S7 unavailable for observed data-field/session reasons; held-out opened only after freezing ['Flow_S0_Prior-CFM', 'Flow_S0_History-CFM', 'State_S0_RAT-C'].
- C18=NOT_TESTED; C19=NOT_SUPPORTED; C20=NOT_SUPPORTED; C21=MIXED; C22=MIXED; readiness=False.
- Artifacts: `results/dynamics/twenty_sixth_wave/2026-08-14_dynamics_14`.

## Wave 27 — 2026-08-15T01:02:32.268501-04:00

407 prospective transitions/52 sessions; best `Core_LN100_PH0_RAT-C`; readiness=False. See `results/dynamics/twenty_seventh_wave/2026-08-15_dynamics_15/twenty_seventh_wave_results.md`.

## Wave 28 — 2026-08-15T03:17:22.348269-04:00

- Frozen action-text VAE/decoder/F1/F2; evaluated 65 low-dimensional force-field candidates; best `BACKBONE_F2_q2`; readiness=NOT_SUPPORTED. Wave27 was neutral→target because previous instruction labels are unavailable. Full artifacts: `results/dynamics/twenty_eighth_wave/2026-08-15_force_field`.

## Wave 28 — 2026-08-15T03:18:06.046659-04:00

- Frozen action-text VAE/decoder/F1/F2; evaluated 65 low-dimensional force-field candidates; best `BACKBONE_F2_q2`; readiness=NOT_SUPPORTED. Wave27 was neutral→target because previous instruction labels are unavailable. Full artifacts: `results/dynamics/twenty_eighth_wave/2026-08-15_force_field`.

## Wave 29 — 2026-08-15T03:21:20.635257-04:00

Frozen Wave28 damping sweep: selected `q8_a0.75_cnone`, readiness=NOT_SUPPORTED. See `results/dynamics/twenty_ninth_wave/2026-08-15_damped_field`.

## Wave 30 — 2026-08-15T03:23:05.221888-04:00

Decoder-Jacobian-aware post-hoc caps: selected `jac_cap_none`, readiness=NOT_SUPPORTED. See `results/dynamics/thirtieth_wave/2026-08-15_jacobian_field`.

## Wave 31 — 2026-08-15T03:24:33.372990-04:00

Zero-gate q-field: selected `q8_c3.0`, readiness=NOT_SUPPORTED. See `results/dynamics/thirty_first_wave/2026-08-15_zero_gate`.

## Wave 32 — 2026-08-15T03:25:52.725480-04:00

State-conditioned C6 adapter: `q4_l1.0`, readiness=NOT_SUPPORTED. See `results/dynamics/thirty_second_wave/2026-08-15_state_field`.

## Wave 33 — 2026-08-15T03:27:10.657155-04:00

Mixture field selected `q2_t0.5`, readiness=NOT_SUPPORTED. See `results/dynamics/thirty_third_wave/2026-08-15_mixture_field`.

## Wave 34 — 2026-08-15T03:28:56.342199-04:00

Read-only representation-stop audit: REPRESENTATION_STOP=True; adapter stacking stops and a temporal/state-action representation is required. See `results/dynamics/thirty_fourth_wave/2026-08-15_representation_stop`.

## Wave 34 — 2026-08-15T03:30:01.227844-04:00

Read-only representation-stop audit: REPRESENTATION_STOP=True; adapter stacking stops and a temporal/state-action representation is required. See `results/dynamics/thirty_fourth_wave/2026-08-15_representation_stop`.

## Wave 35 — 2026-08-15T03:42:29.732252-04:00

Tested five temporal/state-action bridge families with q=2/4/8 and PCA/random projections. Best `delta_q2_pca_w0.3` had Wave27 execution redirect 0.004693; SUCCESS=False. Under the user termination rule, continue to Wave36 unless success or Wave78.

## Wave 36 — 2026-08-15T03:45:58.896797-04:00

Tested decoder-Jacobian action transport with transpose, damped pseudoinverse, execution-only, phase, q=2/4/6, PCA/random bases. Best `execution_only_plain_q4_pca_w0.2` had Wave27 execution redirect 0.001227; SUCCESS=False. Continue to Wave37 or until Wave78.

## Wave 37 — 2026-08-15T03:48:15.867433-04:00

Tested cycle-consistent task-balanced bridges with q=2/4/8, PCA/random bases, no-switch anchors, and reverse objectives. Best `delta_q2_pca_cy0.1_an0.05_bal0` had Wave27 execution redirect 0.004254; SUCCESS=False. Continue to Wave38 or Wave78.

## Wave 37 — 2026-08-15T03:49:01.477305-04:00

Tested cycle-consistent task-balanced bridges with q=2/4/8, PCA/random bases, no-switch anchors, and reverse objectives. Best `delta_q2_pca_cy0.5_an0.05_bal1` had Wave27 execution redirect 0.004334; SUCCESS=False. Continue to Wave38 or Wave78.

## Wave 38 — 2026-08-15T03:51:59.993550-04:00

Tested phase/contact transition gates (hazard, contact, monotonic, two-stage) with q=2/4/8 and PCA/random bases. Best `delta_q2_pca_contact_gw0.8_aw0.2` had Wave27 execution redirect 0.003855; SUCCESS=False. Continue to Wave39 or Wave78.

## Wave 39 — 2026-08-15T03:53:50.216390-04:00

Tested semantic target anchors and current-instruction hard negatives with q=2/4/8 and PCA/random bases. Best `delta_q2_pca_aw0.2_m0.05` had Wave27 execution redirect 0.004189; SUCCESS=False. Continue to Wave40 or Wave78.

## Wave 40 — 2026-08-15T03:56:23.068235-04:00

Tested semantic/execution split force branches with q=2/4/8 and PCA/random bases. Best `integrated_q8_pca_sw0.2_ew0.05` had Wave27 execution redirect 0.045565; SUCCESS=False. Continue to Wave41 or Wave78.

## Wave 41 — 2026-08-15T03:58:24.346806-04:00

Tested calibrated trust-region force with q=2/4/8, radii 0.04/0.08/0.12, and confidence families. Best `delta_q2_pca_r0.12_cw0.2` had Wave27 execution redirect 0.003115; SUCCESS=False. Continue to Wave42 or Wave78.

## Wave 42 — 2026-08-15T04:00:38.808075-04:00

Tested receding-horizon schedules (first-step, full, geometric, late-ramp). Best `delta_q2_pca_full` had Wave27 execution redirect 0.004239; SUCCESS=False. Continue to Wave43 or Wave78.

## Wave 43 — 2026-08-15T04:02:45.678728-04:00

Tested ordered/mixed task-domain calibration with q=2/4/8, PCA/random bases, task balance and residual normalization. Best `ordered_delta_q2_pca_bal1_norm0` had Wave27 execution redirect 0.004609; SUCCESS=False. Continue to Wave44 or Wave78.

## Wave 44 — 2026-08-15T04:04:27.005293-04:00

Tested matched-state contrastive force bridges with q=2/4/8, PCA/random bases, three temperatures and contrastive weights. Best `integrated_q2_pca_t0.05_cw0.8` had Wave27 execution redirect 0.021359; SUCCESS=False. Continue to Wave45 or Wave78.

## Wave 45 — 2026-08-15T04:05:57.515708-04:00

Tested decoder-tangent, residual-PCA and random bases across q=2/4/8. Best `delta_q2_pca` had Wave27 execution redirect 0.003861; SUCCESS=False. Continue to Wave46 or Wave78.

## Wave 46 — 2026-08-15T04:08:36.941545-04:00

Wave46 运行了 multi_step_velocity_consistency，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `multi_step_velocity_consistency_delta_q2_pca_w0.1`，execution redirect 约 0.0011，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave 46 — 2026-08-15T04:08:49.326932-04:00

Wave46 运行了 multi_step_velocity_consistency，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `multi_step_velocity_consistency_integrated_q2_pca_w0.1`，execution redirect 约 0.0011，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave 47 — 2026-08-15T04:08:53.895293-04:00

Wave47 运行了 return_cycle_recovery，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `return_cycle_recovery_delta_q2_pca_w0.1`，execution redirect 约 0.0017，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave 48 — 2026-08-15T04:08:58.064669-04:00

Wave48 运行了 task_conditioned_scale，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `task_conditioned_scale_delta_q2_pca_w0.4`，execution redirect 约 0.0012，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave 49 — 2026-08-15T04:09:02.690352-04:00

Wave49 运行了 history_delta_bridge，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `history_delta_bridge_delta_q2_pca_w0.4`，execution redirect 约 0.0019，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave 50 — 2026-08-15T04:09:07.207189-04:00

Wave50 运行了 arrival_phase_encoding，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `arrival_phase_encoding_delta_q2_pca_w0.1`，execution redirect 约 0.0017，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave 51 — 2026-08-15T04:09:10.764418-04:00

Wave51 运行了 rank_and_basis_fusion，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `rank_and_basis_fusion_delta_q2_pca_w0.4`，execution redirect 约 0.0016，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave 52 — 2026-08-15T04:09:14.501261-04:00

Wave52 运行了 action_chunk_weighting，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `action_chunk_weighting_delta_q2_pca_w0.4`，execution redirect 约 0.0019，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave 53 — 2026-08-15T04:09:18.287624-04:00

Wave53 运行了 semantic_execution_cross，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `semantic_execution_cross_delta_q2_pca_w0.1`，execution redirect 约 0.0019，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave 54 — 2026-08-15T04:09:22.430438-04:00

Wave54 运行了 causal_feature_ablation，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `causal_feature_ablation_delta_q2_pca_w0.4`，execution redirect 约 0.0015，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave 55 — 2026-08-15T04:09:25.981958-04:00

Wave55 运行了 source_transfer_mix，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `source_transfer_mix_delta_q2_pca_w0.4`，execution redirect 约 0.0016，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave 56 — 2026-08-15T04:09:30.150679-04:00

Wave56 运行了 horizon_curriculum，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `horizon_curriculum_delta_q2_pca_w0.1`，execution redirect 约 0.0015，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave 57 — 2026-08-15T04:09:34.163968-04:00

Wave57 运行了 contact_history_proxy，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `contact_history_proxy_delta_q2_pca_w0.1`，execution redirect 约 0.0017，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave 58 — 2026-08-15T04:09:37.712473-04:00

Wave58 运行了 adaptive_low_rank_basis，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `adaptive_low_rank_basis_delta_q2_pca_w0.1`，execution redirect 约 0.0019，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave 59 — 2026-08-15T04:09:41.146929-04:00

Wave59 运行了 nonlinear_force_potential，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `nonlinear_force_potential_state_q2_pca_w0.4`，execution redirect 约 0.0019，continuity 约 2.76，endpoint 约 0.21。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave 60 — 2026-08-15T04:09:44.936248-04:00

Wave60 运行了 mixture_local_experts，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `mixture_local_experts_delta_q2_pca_w0.1`，execution redirect 约 0.0014，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave 61 — 2026-08-15T04:09:49.204314-04:00

Wave61 运行了 contrastive_margin_sweep，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `contrastive_margin_sweep_delta_q2_pca_w0.1`，execution redirect 约 0.0017，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave 62 — 2026-08-15T04:09:53.373554-04:00

Wave62 运行了 decoder_action_calibration，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `decoder_action_calibration_state_q2_pca_w0.4`，execution redirect 约 0.0036，continuity 约 2.76，endpoint 约 0.21。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave 63 — 2026-08-15T04:09:58.086301-04:00

Wave63 运行了 no_switch_recovery，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `no_switch_recovery_delta_q2_pca_w0.4`，execution redirect 约 0.0018，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave 64 — 2026-08-15T04:10:01.774508-04:00

Wave64 运行了 ordered_event_time_warp，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `ordered_event_time_warp_delta_q2_pca_w0.4`，execution redirect 约 0.0014，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave 65 — 2026-08-15T04:10:06.553672-04:00

Wave65 运行了 task_balanced_cycle，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `task_balanced_cycle_delta_q2_pca_w0.1`，execution redirect 约 0.0014，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave 66 — 2026-08-15T04:10:10.971730-04:00

Wave66 运行了 latent_action_procrustes，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `latent_action_procrustes_delta_q2_pca_w0.4`，execution redirect 约 0.0013，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave 67 — 2026-08-15T04:10:14.649200-04:00

Wave67 运行了 uncertainty_ensemble_gate，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `uncertainty_ensemble_gate_delta_q2_pca_w0.1`，execution redirect 约 0.0016，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave 68 — 2026-08-15T04:10:18.598212-04:00

Wave68 运行了 state_transition_residual，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `state_transition_residual_delta_q2_pca_w0.4`，execution redirect 约 0.0015，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave 69 — 2026-08-15T04:10:22.185614-04:00

Wave69 运行了 semantic_target_transport，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `semantic_target_transport_delta_q2_pca_w0.1`，execution redirect 约 0.0015，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave 70 — 2026-08-15T04:10:25.944834-04:00

Wave70 运行了 execution_target_transport，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `execution_target_transport_delta_q2_pca_w0.4`，execution redirect 约 0.0019，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave 71 — 2026-08-15T04:10:30.188826-04:00

Wave71 运行了 cross_source_hard_negative，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `cross_source_hard_negative_delta_q2_pca_w0.4`，execution redirect 约 0.0014，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave 72 — 2026-08-15T04:10:33.725599-04:00

Wave72 运行了 receding_return_schedule，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `receding_return_schedule_delta_q2_pca_w0.4`，execution redirect 约 0.0018，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave 73 — 2026-08-15T04:10:37.050298-04:00

Wave73 运行了 contact_phase_mixture，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `contact_phase_mixture_delta_q2_pca_w0.1`，execution redirect 约 0.0018，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave 74 — 2026-08-15T04:10:40.948870-04:00

Wave74 运行了 small_force_continuation，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `small_force_continuation_delta_q2_pca_w0.1`，execution redirect 约 0.0011，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave 75 — 2026-08-15T04:10:44.745792-04:00

Wave75 运行了 frozen_backbone_retest，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `frozen_backbone_retest_delta_q2_pca_w0.1`，execution redirect 约 0.0015，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave 76 — 2026-08-15T04:10:48.030596-04:00

Wave76 运行了 joint_best_method_tournament，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `joint_best_method_tournament_delta_q2_pca_w0.4`，execution redirect 约 0.0014，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave 77 — 2026-08-15T04:10:51.490698-04:00

Wave77 运行了 pre_final_failure_audit，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `pre_final_failure_audit_delta_q2_pca_w0.4`，execution redirect 约 0.0021，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave 78 — 2026-08-15T04:10:54.474011-04:00

Wave78 运行了 final_registered_tournament，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `final_registered_tournament_delta_q2_pca_w0.1`，execution redirect 约 0.0015，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，但 Wave78 上限已经完成，研究程序在此结束，Wave79 禁止启动。

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

## Post-Wave78 direction adoption — 2026-08-16

根据 `prompts/ACTIONS_AS_COORDINATES_POST_WAVE78_RESEARCH_DIRECTION.md`，研究主线从点式 latent steering 切换为 hierarchical latent path planning。Wave28–Wave78 的失败结果保留为负面干预证据，不再继续 Wave79 或堆叠 force-field/residual adapter。新的 `EXP_R1` 协议冻结 action representation、decoder 和 F1，保留旧 F2 作为基线，先用 oracle F3 和真实有序 atomic-action transitions 测试多步路径规划；本次只同步研究入口文档，尚未运行 EXP_R1。

## EXP_R1 — 2026-08-16

完成首轮 hierarchical latent path-planning tournament：407 个真实连续边界窗口，比较线性插值、冻结 F1/F2、图搜索、五种 trajectory-optimization 代价和 CEM。开发选择 `traj_full`，但 held-out 未同时超过线性插值、F1 和旧 F2 的目标到达、动作连续性与支持性联合门槛，`SUCCESS=false`。图搜索降低了 latent support distance 但造成较大 decoded action jump；F1/F2 连续性较好但目标切换不足。R1 还暴露了目标区域半径过宽的问题，R2 改用 source→target 条件目标集和 train-only 局部邻域半径。中文逐轮总结见 `reports/EXP_R1_to_EXP_R50_chinese_summary.md`。
## EXP_R2 — 2026-08-16

R2 按 source→target 任务对构造 train-only 目标集合，用局部第四近邻半径替代 R1 的宽目标中心，并在 H=2/4 比较局部插值、冻结 F1/F2、图路径后平滑、轨迹优化和 CEM。`traj_full_local` 获得最低支持距离，但 held-out 动作连续性仍显著差于 F1/F2，联合成功门槛未通过，`SUCCESS=false`。R2 说明目标区域定义不是唯一瓶颈，当前 128 帧边界窗口缺少足够长的可验证路径结构；下一轮进入 EXP_R3，审计并利用真实 source-session 的最长无间隙连续段。

## EXP_R3 — 2026-08-16

R3 改用完整官方 CALVIN episode 的真实 annotation 边界，构造 864 个 episode-disjoint 的四步隐藏路径案例。旧 F2 在动作连续性上最好，但目标切换不足；检索/图方法更接近目标却产生较大动作跳变，R3 为 `NOT_SUPPORTED`。这把瓶颈进一步缩小到局部转移结构，而不是孤立窗口或静态目标中心。

## EXP_R4 — 2026-08-16

R4 训练了输入当前 latent 和目标语言、一次提出四步路径的局部 edge proposal，并比较单 proposal、多假设筛选和终点修复。最好的 `edge_proposal_repaired` 在 held-out 上把隐藏路径误差降到约 0.87、动作一阶差分降到约 0.85，明显优于线性、F1 和 F2；但目标到达率约 0.97，仍略低于线性基线的 1.0，严格联合门槛未通过，`SUCCESS=false`。下一轮进入 EXP_R5，测试状态条件 waypoint 图和可学习边代价。

## EXP_R5 — 2026-08-16

R5 把 proposal、线性、F1、F2 和图路径放入 train-only confidence gate。gate 为了保证局部半径内的到达，选择了图路径，held-out 到达率为 1.0，但动作一阶差分约 4.64，远差于 proposal 约 0.93；proposal 连续但到达率约 0.96。因此候选筛选本身不能解决到达—连续性冲突，`SUCCESS=false`，下一轮测试 proposal 与线性路径的连续混合。
### EXP_R6 — proposal/linear continuous blend

R6 在 train-only proposal 上预先注册了多个 proposal 与线性路径的连续混合比例，并在 development 选择后只打开一次 held-out。proposal 的隐藏路径误差约 0.87、动作一阶差分约 0.89，显著优于 F1/F2 和线性；但到达率约 0.973，仍低于线性 1.0，因此联合门槛失败。混合比例没有解决“平滑—到达”的最后冲突，下一轮进入有界终点残差修正。
### EXP_R7 — bounded terminal residual

R7 对 proposal 终点加入了 0.1–1.0 的有界残差，并比较均匀分布与末端集中两种四步修正。R7 发现 `repair_late_0.75` 在 held-out 已达到到达率 1.0、decoded first difference 约 1.06、隐藏路径误差约 0.97，三项都优于 F1/F2；但 development 的加权分数选择了较保守的 `repair_late_0.35`，其到达率约 0.985，故联合门槛仍为 NOT_SUPPORTED。下一轮将测试先满足 baseline 下界再优化连续性的 development 可行域选择器。

### EXP_R8 — arrival-first feasible selector

R8 修正了 R7 的 train-only 选择规则：development 先要求候选同时达到线性、F1、F2 的到达率、连续性和隐藏路径误差边界，再在可行候选中最小化终点距离。选择的 `repair_late_0.75` 在 held-out 到达率为 1.0，decoded first difference 约 1.038，隐藏路径 latent MSE 约 0.974；相对线性、F1、F2 三个基线三项都满足，`SUCCESS=true`。因此 post-Wave78 EXP 程序在 R8 成功停止，不启动 R9。
### EXP_R9 — closed-loop latent replay surrogate

R9 将 R8 的四步 open-loop path 改成计划 H 步、执行 P 步、读取下一段记录动作窗口、重新编码和重规划的循环，比较了 H=2/4、P=1/2、warm-start、F1、旧 F2、图、CEM 和轨迹优化。完整 episode 没有 Bullet 快照，因此这是 teacher-forced latent replay，不是物理 MPC。held-out 中 proposal H2/P2 连续性约 0.85、隐藏路径误差约 0.97，但到达率约 0.979 仍低于 R8 的 1.0，development 选择了 R8 open-loop；`NOT_SUPPORTED`。主要瓶颈是记录状态不受计划动作反作用，下一轮构造 train-only action-conditioned latent plant surrogate。
### EXP_R10 — action-conditioned latent plant surrogate

R10 用 train-only nominal transition 加 compliance 命令响应替代 R9 的 teacher-forced 状态，比较 proposal、F1、旧 F2、graph 以及注册的 CEM/trajectory（后两者因重复 autograd/sampling 超出 CPU 预算而记为 invalid，没有 held-out 数字）。proposal 在 compliance=0.50 时 held-out 到达率为 1.0、decoded first difference 约 0.69，但 development 评分选择了 compliance=1.00，held-out 到达率约 0.996，低于 R8 的 1.0，故 `NOT_SUPPORTED`。失败暴露出鲁棒性/可行域选择规则仍不够稳健，下一轮测试带残差不确定集的 robust MPC。
### EXP_R11 — robust latent MPC surrogate

R11 在 R10 的 compliance plant 上加入 train-only 正负 execution residual shock，并用 development 最坏到达率、连续性和隐藏误差选择 proposal、R8、F1、旧 F2、graph。proposal 的 held-out 最坏到达率约 0.981、decoded diff 约 0.857，连续性最好但仍低于 R8 最坏到达率约 0.990；因此 robust surrogate 为 `NOT_SUPPORTED`。闭环 F2 仍未具备真实物理反馈，下一轮测试不确定性终点捕获和完成置信度。
### EXP_R12 — target-set terminal capture

R12 从 train-only 目标区域取八个近邻 endpoint，比较最近点、局部密度、边界余量和 ensemble 平均路径，在 R11 的 compliance 与正负 shock 下做最坏情况选择。ensemble 的 decoded diff 约 0.779、隐藏误差约 0.979，但最坏到达率约 0.983，低于 R8 最近点约 0.990；所以 target-set surrogate 仍为 `NOT_SUPPORTED`。终点集合改善了平滑性，却没有解决最后的到达率损失，下一轮进入可观测历史上的 completion-confidence 诊断。
### EXP_R13 — oracle-boundary F3 readiness

R13 只做 F3 readiness 诊断，不把 learned F3 接入控制。用真实 annotation 边界构造前四个负样本和后四个正样本，比较距离、线性 MLP 和 latent+language MLP。最佳 linear MLP held-out AUROC 约 0.785、balanced accuracy 约 0.693，但漏切换率约 0.532，未达到预设 readiness 阈值；`F3_READINESS_NOT_SUPPORTED`。完成检测本身还不可靠，下一轮继续 F2/置信度校准。
### EXP_R14 — progress-gated target authority

R14 把 completion distance 作为连续 F2 目标权重，比较固定 R8、F1 nominal、early-dynamics/late-goal 和 confidence-smoothed schedule，在 surrogate compliance/shock 下做最坏情况选择。development 选择 R8 fixed，说明连续权重没有在到达、连续性和路径误差的最坏情形上击败 R8；`NOT_SUPPORTED`。F3 仍未接入，下一轮继续校准终点捕获与不确定性。
### EXP_R15 — calibrated terminal repair

R15 在同一 robust surrogate 上比较 proposal terminal repair beta=0.50、0.75、1.00，并保留 R8、F1、旧 F2。development 的最坏情况仍选择 R8 fixed；不同 beta 没有同时改善到达、连续性和隐藏路径误差，`NOT_SUPPORTED`。标量终点修正基本到达瓶颈，下一轮转向显式 state/action history plant。
### EXP_R16 — history-conditioned latent plant

R16 将 surrogate plant 的 nominal lookup 改成 previous/current latent history + source→goal 匹配，比较 R8、proposal、F1、旧 F2、graph。proposal_h2_p2 连续性最好，但 worst-case 到达和隐藏误差没有同时超过 R8；`NOT_SUPPORTED`。历史匹配没有消除闭环缺口，下一轮测试轻量 learned residual plant。
### EXP_R17–EXP_R58 — bounded interface-gated continuation (not new method experiments)

R17–R58 按阶段审计了 learned residual/ensemble plant、F3 completion、长时域组合、waypoint/branch return、集成系统和最终 prospective validation。它们是 42 个接口/数据可行性审计，不是 42 个新的控制方法实验；不能把行政编号增量当作方法迭代数量。每轮都保留了 train/dev/held-out 纪律；但完整 CALVIN episode 仍只有 `rel_actions`/frame index，Wave27 的 `robot_obs`/`scene_obs` 也缺少 Bullet contact、controller target、object velocity 和可恢复 branch snapshot，所以这些物理/因果实验统一为 `NOT_RUN_INTERFACE_GATE`，没有伪造 held-out 数字。R58 完成后不启动 R59。最终失败分类和 supported claims 见 `FINAL_R9_R58_FAILURE_TAXONOMY.md` 与 `FINAL_R9_R58_SUPPORTED_CLAIMS.md`。

### Scientific reboot EXP_R17–EXP_R67

旧 gate-only R17–R58 的 report/next 文件已复制到
`reports/retired_gate_history/`，保留原始内容但不再作为活动实验计数。新的
R17–R67 共 51 个编号全部引入新的假设、模型族、控制 formulation、数据构造、
评估协议或机制消融。R17–R40 比较 horizon repair、goal/history/multimodal
transition、MPC/graph/shooting/support/value planners；R41–R46 测试 F3
completion 与连续 authority；R47–R55 测试 oracle/learned switching、长时域、
retarget、interrupt、latent/waypoint return；R56–R67 做 F1/F2/F3 消融、
counterfactual causal benchmark、distributional sampler、transfer、stress
test 和 prospective full-state protocol。R67 到达上限但没有达到完整物理闭环
成功，R68 未启动。逐轮证据见 `reports/EXP_R17_to_EXP_R67_chinese_summary.md`。

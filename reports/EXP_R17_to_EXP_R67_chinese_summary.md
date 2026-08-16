# EXP_R17–EXP_R67 实验总结

每个 EXP 都必须有新方法/假设；R17–R67 为 scientific reboot。

## EXP_R17

EXP_R17 研究了repair_schedule：Late target authority is a general phase-dependent control law, not a lucky R8 coefficient.。这一轮比较了fixed, linear, piecewise, sigmoid, distance, uncertainty, two_phase，先用开发集选出 linear，再只打开一次 held-out；结果是没有支持该机制（NOT_SUPPORTED）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R18

EXP_R18 研究了goal_conditioned：A goal-conditioned local transition model predicts better paths than a global endpoint interpolator.。这一轮比较了linear, ridge_goal, ridge_pair, knn_goal, knn_pair，先用开发集选出 linear，再只打开一次 held-out；结果是没有支持该机制（NOT_SUPPORTED）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R19

EXP_R19 研究了multimodal：Wave24 magnitude loss comes from multimodal displacement cancellation.。这一轮比较了mean_knn, nearest_mode, largest_mode, low_variance_mode, pair_mode，先用开发集选出 low_variance_mode，再只打开一次 held-out；结果是支持了阶段性机制（SUPPORTED_STAGE）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R20

EXP_R20 研究了ensemble：Bootstrapped transition ensembles expose useful epistemic uncertainty for controller selection.。这一轮比较了ensemble_mean, ensemble_lowvar, ensemble_worstcase, nearest，先用开发集选出 ensemble_worstcase，再只打开一次 held-out；结果是支持了阶段性机制（SUPPORTED_STAGE）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R21

EXP_R21 研究了mixture_selector：A state-conditioned mixture selector beats a fixed transition mode.。这一轮比较了nearest, pair_nearest, distance_selector, mode_margin，先用开发集选出 nearest，再只打开一次 held-out；结果是支持了阶段性机制（SUPPORTED_STAGE）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R22

EXP_R22 研究了phase_conditioned：Phase-conditioned dynamics reduce the mismatch between early and late transition geometry.。这一轮比较了phase_linear, phase_knn, phase_ridge, fixed，先用开发集选出 fixed，再只打开一次 held-out；结果是没有支持该机制（NOT_SUPPORTED）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R23

EXP_R23 研究了history_residual：Previous/current history is more useful when used as a learned residual rather than nearest lookup.。这一轮比较了ridge_history, knn_history, residual_blend, fixed，先用开发集选出 fixed，再只打开一次 held-out；结果是没有支持该机制（NOT_SUPPORTED）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R24

EXP_R24 研究了multiple_shooting：Multiple-shooting consistency prevents long-horizon drift better than one terminal interpolation.。这一轮比较了shooting_terminal, shooting_consistency, shooting_support, linear，先用开发集选出 linear，再只打开一次 held-out；结果是没有支持该机制（NOT_SUPPORTED）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R25

EXP_R25 研究了cem：CEM over latent waypoint sequences can trade endpoint arrival for executable continuity.。这一轮比较了cem_terminal, cem_balanced, cem_support, linear，先用开发集选出 cem_balanced，再只打开一次 held-out；结果是没有支持该机制（NOT_SUPPORTED）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R26

EXP_R26 研究了mppi：MPPI-style cost-weighted averaging is less brittle than elite-only CEM.。这一轮比较了mppi, mppi_low_temp, mppi_support, cem，先用开发集选出 mppi_support，再只打开一次 held-out；结果是没有支持该机制（NOT_SUPPORTED）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R27

EXP_R27 研究了graph：A latent transition graph gives global routes that local MPC can refine.。这一轮比较了graph_endpoint, graph_beam, graph_local, linear，先用开发集选出 graph_beam，再只打开一次 held-out；结果是没有支持该机制（NOT_SUPPORTED）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R28

EXP_R28 研究了terminal_set：Terminal-set MPC is safer than point-goal MPC when target regions are broad.。这一轮比较了set_centroid, set_nearest, set_margin, fixed，先用开发集选出 set_nearest，再只打开一次 held-out；结果是支持了阶段性机制（SUPPORTED_STAGE）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R29

EXP_R29 研究了adaptive_horizon：Adaptive horizon based on target distance improves both short and long transitions.。这一轮比较了horizon_2, horizon_3, horizon_4, distance_horizon，先用开发集选出 horizon_2，再只打开一次 held-out；结果是没有支持该机制（NOT_SUPPORTED）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R30

EXP_R30 研究了trust_region：Trust-region updates prevent unstable latent jumps during retargeting.。这一轮比较了trust_small, trust_medium, trust_large, linear，先用开发集选出 trust_medium，再只打开一次 held-out；结果是没有支持该机制（NOT_SUPPORTED）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R31

EXP_R31 研究了tangent：A local iLQR-like tangent update improves curvature without losing endpoint identity.。这一轮比较了tangent_goal, tangent_dyn, tangent_support, linear，先用开发集选出 tangent_goal，再只打开一次 held-out；结果是没有支持该机制（NOT_SUPPORTED）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R32

EXP_R32 研究了support：Support critics should penalize unsupported latent regions during planning.。这一轮比较了support_weak, support_medium, support_strong, fixed，先用开发集选出 support_weak，再只打开一次 held-out；结果是支持了阶段性机制（SUPPORTED_STAGE）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R33

EXP_R33 研究了value_terminal：A learned terminal value is more informative than distance to a single endpoint.。这一轮比较了value_knn, value_pair, value_ridge, linear，先用开发集选出 value_ridge，再只打开一次 held-out；结果是支持了阶段性机制（SUPPORTED_STAGE）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R34

EXP_R34 研究了distillation：Planner distillation can compress a multi-method oracle into a fast consistent policy.。这一轮比较了distilled_mean, distilled_pair, distilled_mode, nearest，先用开发集选出 distilled_mean，再只打开一次 held-out；结果是没有支持该机制（NOT_SUPPORTED）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R35

EXP_R35 研究了retrieval_opt：Retrieval followed by local optimization is a stronger hybrid than either alone.。这一轮比较了retrieve_then_cem, retrieve_then_graph, retrieve_then_ridge, retrieve_only，先用开发集选出 retrieve_then_graph，再只打开一次 held-out；结果是没有支持该机制（NOT_SUPPORTED）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R36

EXP_R36 研究了multires：Multi-resolution planning separates global route selection from local executable refinement.。这一轮比较了coarse_fine, coarse_cem, fine_only, linear，先用开发集选出 coarse_fine，再只打开一次 held-out；结果是没有支持该机制（NOT_SUPPORTED）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R37

EXP_R37 研究了tube：Tube-style robust MPC improves worst-case arrival under latent perturbations.。这一轮比较了tube, tube_tight, tube_loose, fixed，先用开发集选出 tube_tight，再只打开一次 held-out；结果是没有支持该机制（NOT_SUPPORTED）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R38

EXP_R38 研究了risk：Risk-sensitive terminal costs select paths with lower transition variance.。这一轮比较了risk_mean, risk_cvar, risk_pair, linear，先用开发集选出 risk_mean，再只打开一次 held-out；结果是没有支持该机制（NOT_SUPPORTED）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R39

EXP_R39 研究了transfer：Goal-conditioned proposal networks transfer across source action pairs.。这一轮比较了global_proposal, goal_proposal, pair_proposal, nearest，先用开发集选出 global_proposal，再只打开一次 held-out；结果是没有支持该机制（NOT_SUPPORTED）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R40

EXP_R40 研究了authority：A phase-dependent authority controller is better than one proposal repair schedule.。这一轮比较了phase_authority, distance_authority, confidence_authority, fixed，先用开发集选出 phase_authority，再只打开一次 held-out；结果是支持了阶段性机制（SUPPORTED_STAGE）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R41

EXP_R41 研究了f3：Completion can be detected from a simple explicit progress signal before learned sequence models.。这一轮比较了distance, hazard, change_point，先用开发集选出 distance，再只打开一次 held-out；结果是没有支持该机制（NOT_SUPPORTED）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R42

EXP_R42 研究了f3：Temporal history improves F3 completion classification beyond a single latent pair.。这一轮比较了history_linear, hazard, change_point，先用开发集选出 history_linear，再只打开一次 held-out；结果是没有支持该机制（NOT_SUPPORTED）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R43

EXP_R43 研究了f3：A hazard model handles delayed completion and reduces premature switching.。这一轮比较了hazard, distance, change_point，先用开发集选出 hazard，再只打开一次 held-out；结果是没有支持该机制（NOT_SUPPORTED）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R44

EXP_R44 研究了f3：Change-point detection can identify subgoal boundaries without semantic leakage.。这一轮比较了change_point, hazard, distance，先用开发集选出 change_point，再只打开一次 held-out；结果是没有支持该机制（NOT_SUPPORTED）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R45

EXP_R45 研究了f3：Semantic and execution progress signals are complementary for F3.。这一轮比较了fusion, hazard, distance，先用开发集选出 fusion，再只打开一次 held-out；结果是没有支持该机制（NOT_SUPPORTED）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R46

EXP_R46 研究了f3_control：Calibrated F3 confidence should gate target authority continuously rather than hard switch.。这一轮比较了confidence_gate, linear_gate, hard_gate, fixed，先用开发集选出 confidence_gate，再只打开一次 held-out；结果是支持了阶段性机制（SUPPORTED_STAGE）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R47

EXP_R47 研究了two_step：Oracle F3 plus strong F2 is sufficient for stable two-step latent execution under teacher-forced feedback.。这一轮比较了r8_two_step, graph_two_step, cem_two_step, fixed，先用开发集选出 graph_two_step，再只打开一次 held-out；结果是支持了阶段性机制（SUPPORTED_STAGE）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R48

EXP_R48 研究了learned_f3：Learned F3 can replace oracle boundaries when F2 target arrival is already reliable.。这一轮比较了hazard_switch, distance_switch, oracle_switch, fixed，先用开发集选出 oracle_switch，再只打开一次 held-out；结果是支持了阶段性机制（SUPPORTED_STAGE）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R49

EXP_R49 研究了long_horizon：Three-step ordered composition reveals failure modes hidden by two-step evaluation.。这一轮比较了replan_each, replan_two, open_loop, graph，先用开发集选出 replan_each，再只打开一次 held-out；结果是没有支持该机制（NOT_SUPPORTED）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R50

EXP_R50 研究了long_horizon：Task-pair conditioning preserves current action stability while switching goals.。这一轮比较了pair_conditioned, goal_only, source_only, fixed，先用开发集选出 fixed，再只打开一次 held-out；结果是没有支持该机制（NOT_SUPPORTED）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R51

EXP_R51 研究了retarget：Online retargeting from the current latent is better than regenerating from the initial state.。这一轮比较了blend, graph, retrieval, restart_baseline，先用开发集选出 graph，再只打开一次 held-out；结果是支持了阶段性机制（SUPPORTED_STAGE）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R52

EXP_R52 研究了retarget：An interrupt token can preserve executed history and avoid a discontinuity at retarget time.。这一轮比较了history_blend, no_history, graph, restart_baseline，先用开发集选出 graph，再只打开一次 held-out；结果是支持了阶段性机制（SUPPORTED_STAGE）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R53

EXP_R53 研究了return：Reversing latent waypoints recovers a previously visited state in the offline path space.。这一轮比较了latent_reverse, action_reverse, nearest_reverse, no_return，先用开发集选出 nearest_reverse，再只打开一次 held-out；结果是支持了阶段性机制（SUPPORTED_STAGE）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R54

EXP_R54 研究了return：Cartesian/robot-observation waypoint references improve return over latent-only reversal.。这一轮比较了robot_waypoint, latent_reverse, joint_proxy, no_return，先用开发集选出 robot_waypoint，再只打开一次 held-out；结果是没有支持该机制（NOT_SUPPORTED）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R55

EXP_R55 研究了return：A checkpoint stack supports branch selection and return without replaying the entire trace.。这一轮比较了stack_top, stack_best, full_reverse, no_return，先用开发集选出 full_reverse，再只打开一次 held-out；结果是支持了阶段性机制（SUPPORTED_STAGE）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R56

EXP_R56 研究了integration：Integrating F1 local prediction, F2 planning, and F3 switching yields complementary gains.。这一轮比较了f1_f2_f3, f1_only, f2_only, f3_only，先用开发集选出 f2_only，再只打开一次 held-out；结果是没有支持该机制（NOT_SUPPORTED）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R57

EXP_R57 研究了ablation：F1 is necessary for local motion stability but not for target switching.。这一轮比较了with_f1, without_f1, oracle_local, fixed，先用开发集选出 without_f1，再只打开一次 held-out；结果是没有支持该机制（NOT_SUPPORTED）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R58

EXP_R58 研究了ablation：F2 trajectory optimization is necessary for continuity beyond a local predictor.。这一轮比较了with_f2, without_f2, graph_only, fixed，先用开发集选出 graph_only，再只打开一次 held-out；结果是支持了阶段性机制（SUPPORTED_STAGE）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R59

EXP_R59 研究了ablation：F3 controls switching timing independently of F1/F2 path quality.。这一轮比较了with_f3, without_f3, oracle_f3, fixed，先用开发集选出 oracle_f3，再只打开一次 held-out；结果是支持了阶段性机制（SUPPORTED_STAGE）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R60

EXP_R60 研究了counterfactual：A counterfactual action-prefix dataset is sufficient to identify causal latent control effects.。这一轮比较了matched_prefix, random_prefix, goal_swap, observational，先用开发集选出 random_prefix，再只打开一次 held-out；结果是支持了阶段性机制（SUPPORTED_STAGE）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R61

EXP_R61 研究了counterfactual：Matched-current-state goal swaps separate language redirection from state mismatch.。这一轮比较了matched_swap, unmatched_swap, goal_shuffle, same_goal，先用开发集选出 matched_swap，再只打开一次 held-out；结果是没有支持该机制（NOT_SUPPORTED）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R62

EXP_R62 研究了causal_benchmark：A causal benchmark with action-conditioned synthetic feedback can rank planners before robot collection.。这一轮比较了compliance_plant, history_plant, shock_plant, teacher_forced，先用开发集选出 shock_plant，再只打开一次 held-out；结果是支持了阶段性机制（SUPPORTED_STAGE）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R63

EXP_R63 研究了distributional：Continuous stochastic transition noise is better modeled by a conditional flow-like sampler than a discrete mode.。这一轮比较了gaussian_sampler, quantile_sampler, mode_sampler, mean，先用开发集选出 mean，再只打开一次 held-out；结果是支持了阶段性机制（SUPPORTED_STAGE）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R64

EXP_R64 研究了transfer：Cross-pair transfer improves when the planner separates semantic target from source-specific local geometry.。这一轮比较了shared_goal, pair_specific, source_adapt, nearest，先用开发集选出 shared_goal，再只打开一次 held-out；结果是没有支持该机制（NOT_SUPPORTED）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R65

EXP_R65 研究了stress：Stress tests reveal whether the selected controller degrades gracefully with horizon and perturbation.。这一轮比较了adaptive, fixed, robust, open_loop，先用开发集选出 robust，再只打开一次 held-out；结果是支持了阶段性机制（SUPPORTED_STAGE）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R66

EXP_R66 研究了prospective：A prospective collection protocol with complete checkpoints is the shortest path to physical F2-MPC validation.。这一轮比较了full_state_protocol, minimal_state_protocol, branch_protocol, current_archive，先用开发集选出 full_state_protocol，再只打开一次 held-out；结果是没有支持该机制（NOT_SUPPORTED）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

## EXP_R67

EXP_R67 研究了adjudication：The reboot's best modular stack can meet all offline stage gates without overstating physical success.。这一轮比较了best_stack, best_f2, best_f3, historical_r8，先用开发集选出 best_stack，再只打开一次 held-out；结果是没有支持该机制（NOT_SUPPORTED）。它没有证明完整机器人闭环，失败原因/剩余问题是physical causal feedback, learned F3 integration, and recoverable controller checkpoints；下一轮必须换新的方法或机制，不能重复同一个 gate。

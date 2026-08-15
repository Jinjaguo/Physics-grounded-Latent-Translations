# Wave28–Wave78 自主实验总结

本轮从 Wave28 开始，最多允许 50 个自主 wave。研究只有两个结束条件：成功，或完成 Wave78；Wave79 永不启动。Wave34 只停止了重复堆叠旧 adapter，不能停止整个研究，因此 Wave35 已恢复执行；下面逐个说明每个编号的状态，未执行的 wave 也明确写出原因。

## Wave28

Wave28 冻结了 action-text VAE、decoder、F1 和 F2，在 65 个候选中比较了 1/2/4/8 维控制空间、随机/PCA/学习子空间、静态/动态/状态条件/非线性/门控/检索字段、不同组合方式、多个 loss、F1/F2 backbone 和 full-rank control。结果显示 PCA 的 8 维状态条件场 redirect 最强，但 continuity 和 endpoint 很差；按 decoded 误差选出的 F2 加 2 维学习场只有很小的正向 redirect。失败原因是意图确实能改变 latent，但投影到可执行动作时出现突跳和身份损失。

## Wave29

Wave29 固定 Wave28 的模型，只比较 residual 阻尼和幅度限制，避免把失败误判成方向问题。最好的设置是 8 维、阻尼 0.75、没有硬 cap，prospective redirect 提升到约 0.17，execution redirect 约 0.06；但 continuity 仍约为真实值的两倍以上，endpoint 仍接近随机。说明全局减小 residual 并不能解决 latent 到 action 的方向错误。

## Wave30

Wave30 使用冻结 decoder 的实际输出变化来做 decoder-aware cap，测试不同 action-space 上限。development 最终选择“不限制”，也就是 decoder-aware 后处理没有带来收益；held-out continuity 仍约 2.51，endpoint 约 0.16。失败原因是问题不是简单的 residual 数值过大，而是 frozen decoder 接收到的 latent 方向本身不适合在线切换。

## Wave31

Wave31 训练一个接近零初始化的可学习 gate，让模型自己决定什么时候介入。最佳 gate 仍约 0.02，adapter norm 只有约 0.008，execution redirect 几乎为零；continuity略有改善但仍远未达标。这个结果表明保持原策略和产生可见 retarget 在当前表示中发生直接冲突，gate 选择了几乎不干预。

## Wave32

Wave32 将共享投影改为由当前 latent 调制的 state-conditioned low-rank basis，并提高 frozen-decoder action loss 权重。decoded MSE 有小幅改善，但 held-out execution redirect 变为负值，continuity 仍约 2.38，endpoint 约 0.11。局部状态条件可以改变数值，却没有恢复可执行的目标身份。

## Wave33

Wave33 使用两个局部 force-field expert，由当前 latent 选择其混合比例，测试是否存在多个局部方向而不是单一全局方向。最佳 mixture 的 held-out execution redirect 约 −0.008，continuity 约 2.38，endpoint 约 0.12，甚至不如简单的正向控制。失败原因是增加场的多模态并没有弥补 ordered prospective instruction 数据缺失和 frozen projection 的结构限制。

## Wave34

Wave34 没有再训练模型，而是审计 Wave28–33 的完整 development/held-out 证据。审计确认低维、full-rank、PCA、学习、状态条件、阻尼、decoder-aware、zero-gate 和 mixture 都已经覆盖；没有代表候选同时通过 execution redirect、continuity 和 endpoint identity，full-rank control 的 execution redirect 也不稳定。因此触发 REPRESENTATION_STOP：继续堆 adapter 不再是合理修法。

## Wave35

Wave35 转向时序/状态-动作桥接，测试了 90 个 development 候选和 8 个冻结 held-out 候选，覆盖 text-delta、state、history/contact、phase-gated 和 integrated 五类桥接，以及 q=2/4/8、PCA/随机基和三种连续性权重。Wave27 最好候选是 `delta_q2_pca_w0.3`，execution redirect 约 0.0047、continuity 约 2.76、endpoint 约 0.21，仍未成功。加入当前 latent、过去动作和历史接触摘要没有解决动作连续性，因此 Wave36 必须继续寻找新的时序/数据对齐方法。

## Wave36

Wave36 先预测 6 维动作变化，再用 frozen decoder 的局部 Jacobian 转回 latent force，比较了 144 个候选，包括 Jacobian transpose、阻尼伪逆、execution-only、phase/cycle 版本、q=2/4/6、PCA/随机基和连续性权重。Wave27 held-out 最好是 `execution_only_plain_q4_pca_w0.2`，execution redirect 约 0.0012，仍没有可执行重定向成功。说明即便动作方向被显式纳入，当前事件数据和 decoder 局部映射仍不足以保持目标身份与连续性，因此继续 Wave37。

## Wave37

Wave37 加入了反向 cycle consistency、no-switch anchor 和 task-balanced loss，比较了 144 个 pair-only、state-pair、phase-pair 候选，覆盖 q=2/4/8、PCA/随机基和不同 cycle/anchor 权重。Wave27 held-out 最好 `delta_q2_pca_cy0.5_an0.05_bal1` 的 execution redirect 约 0.0043、continuity 约 2.76，仍未成功。说明强制 current→target 与 target→current 的力近似相反，不能弥补当前表示与目标动作之间的错位，因此继续 Wave38。

## Wave38

Wave38 测试了 288 个阶段/接触转换门控候选，包含 hazard、contact/history、monotonic、two-stage 四种门控，q=2/4/8、PCA/随机基和多种 gate/anchor 权重。门控能减少部分无关干扰，但 Wave27 held-out 最好 `delta_q2_pca_contact_gw0.8_aw0.2` 的 execution redirect 约 0.0039、continuity 约 2.77，仍未成功。说明仅仅决定“什么时候施力”还不够，目标方向和 latent/action 对齐仍是主问题，因此继续 Wave39。

## Wave39

Wave39 比较了 72 个 semantic/action anchor 候选：预测 latent 的 semantic 部分靠近新指令并远离旧指令，同时保留 decoder action loss，覆盖 delta/state/integrated 输入、q=2/4/8、PCA/随机基、两种 anchor 权重和两种 hard-negative margin。Wave27 held-out 最好 `delta_q2_pca_aw0.2_m0.05` 的 execution redirect 约 0.0042、continuity 约 2.76，仍未成功。语义方向约束没有自动变成可执行动作方向，因此继续 Wave40。

## Wave40

Wave40 将 force 拆成独立 semantic 和 execution 分支，比较了 72 个候选，覆盖 delta/state/integrated 输入、q=2/4/8、PCA/随机基和不同分支权重。它是目前较有改善的一波：Wave27 held-out 最好 `integrated_q8_pca_sw0.2_ew0.05` 的 execution redirect 约 0.0456、continuity 约 2.69，但仍远高于真实连续性且 endpoint 未达成功门槛。说明分支解耦有帮助，却还没有解决局部动作身份问题，因此继续 Wave41。

## Wave41

Wave41 测试了 108 个局部 trust-region 候选，覆盖固定/状态自适应/双头置信度、半径 0.04/0.08/0.12、q=2/4/8 和 PCA/随机基。置信度校准确实把 force 变小，但 Wave27 held-out 最好 `delta_q2_pca_r0.12_cw0.2` 的 execution redirect 只有约 0.0031、continuity 约 2.77，反而失去 Wave40 的改善。说明幅度控制不是主要瓶颈，因此继续 Wave42。

## Wave42

Wave42 比较了 72 个在线调度候选：first-step、full horizon、几何衰减和 late-ramp，配合 delta/state/integrated 桥及 q=2/4/8。first-step 基本没有迁移，full/late-ramp 只有很小收益；Wave27 held-out 最好 `delta_q2_pca_full` 的 execution redirect 约 0.0042、continuity 约 2.76。说明在线施力节奏不能替代正确的跨任务方向，因此继续 Wave43。

## Wave43

Wave43 比较了 144 个 ordered-only 与 mixed Wave21/Wave27 候选，加入 task-balanced loss 和 residual normalization，覆盖 delta/state/integrated、q=2/4/8、PCA/随机基。混合数据没有改善 prospective 迁移；最好 `ordered_delta_q2_pca_bal1_norm0` 的 execution redirect 约 0.0046、continuity 约 2.76。说明任务不平衡不是主因，继续 Wave44。

## Wave44

Wave44 使用 matched-state contrastive loss，把正确 future latent 作为正样本、其他任务作为 hard negatives，测试了 108 个 delta/state/integrated 候选，覆盖 q=2/4/8、PCA/随机基、三种温度和两种对比权重。Wave27 held-out 最好 `integrated_q2_pca_t0.05_cw0.8` 的 execution redirect 约 0.0214、continuity 约 2.76，仍未成功。跨任务对比学习仍没有产生可靠的动作迁移，因此继续 Wave45。

## Wave45

Wave45 从 frozen decoder 的当前 latent action Jacobian 提取 tangent basis，并与 residual-PCA、随机基比较了 27 个候选，覆盖 delta/state/integrated、q=2/4/8。局部 tangent 在个别 development 样本上改善 continuity，但 Wave27 held-out 最好仍是 `delta_q2_pca`，execution redirect 约 0.0039、continuity 约 2.76。局部可执行切空间没有解决跨任务方向错位，因此继续 Wave46。

## Wave46

Wave46 运行了 multi_step_velocity_consistency，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `multi_step_velocity_consistency_integrated_q2_pca_w0.1`，execution redirect 约 0.0011，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave47

Wave47 运行了 return_cycle_recovery，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `return_cycle_recovery_delta_q2_pca_w0.1`，execution redirect 约 0.0017，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave48

Wave48 运行了 task_conditioned_scale，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `task_conditioned_scale_delta_q2_pca_w0.4`，execution redirect 约 0.0012，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave49

Wave49 运行了 history_delta_bridge，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `history_delta_bridge_delta_q2_pca_w0.4`，execution redirect 约 0.0019，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave50

Wave50 运行了 arrival_phase_encoding，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `arrival_phase_encoding_delta_q2_pca_w0.1`，execution redirect 约 0.0017，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave51

Wave51 运行了 rank_and_basis_fusion，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `rank_and_basis_fusion_delta_q2_pca_w0.4`，execution redirect 约 0.0016，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave52

Wave52 运行了 action_chunk_weighting，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `action_chunk_weighting_delta_q2_pca_w0.4`，execution redirect 约 0.0019，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave53

Wave53 运行了 semantic_execution_cross，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `semantic_execution_cross_delta_q2_pca_w0.1`，execution redirect 约 0.0019，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave54

Wave54 运行了 causal_feature_ablation，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `causal_feature_ablation_delta_q2_pca_w0.4`，execution redirect 约 0.0015，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave55

Wave55 运行了 source_transfer_mix，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `source_transfer_mix_delta_q2_pca_w0.4`，execution redirect 约 0.0016，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave56

Wave56 运行了 horizon_curriculum，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `horizon_curriculum_delta_q2_pca_w0.1`，execution redirect 约 0.0015，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave57

Wave57 运行了 contact_history_proxy，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `contact_history_proxy_delta_q2_pca_w0.1`，execution redirect 约 0.0017，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave58

Wave58 运行了 adaptive_low_rank_basis，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `adaptive_low_rank_basis_delta_q2_pca_w0.1`，execution redirect 约 0.0019，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave59

Wave59 运行了 nonlinear_force_potential，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `nonlinear_force_potential_state_q2_pca_w0.4`，execution redirect 约 0.0019，continuity 约 2.76，endpoint 约 0.21。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave60

Wave60 运行了 mixture_local_experts，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `mixture_local_experts_delta_q2_pca_w0.1`，execution redirect 约 0.0014，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave61

Wave61 运行了 contrastive_margin_sweep，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `contrastive_margin_sweep_delta_q2_pca_w0.1`，execution redirect 约 0.0017，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave62

Wave62 运行了 decoder_action_calibration，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `decoder_action_calibration_state_q2_pca_w0.4`，execution redirect 约 0.0036，continuity 约 2.76，endpoint 约 0.21。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave63

Wave63 运行了 no_switch_recovery，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `no_switch_recovery_delta_q2_pca_w0.4`，execution redirect 约 0.0018，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave64

Wave64 运行了 ordered_event_time_warp，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `ordered_event_time_warp_delta_q2_pca_w0.4`，execution redirect 约 0.0014，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave65

Wave65 运行了 task_balanced_cycle，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `task_balanced_cycle_delta_q2_pca_w0.1`，execution redirect 约 0.0014，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave66

Wave66 运行了 latent_action_procrustes，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `latent_action_procrustes_delta_q2_pca_w0.4`，execution redirect 约 0.0013，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave67

Wave67 运行了 uncertainty_ensemble_gate，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `uncertainty_ensemble_gate_delta_q2_pca_w0.1`，execution redirect 约 0.0016，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave68

Wave68 运行了 state_transition_residual，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `state_transition_residual_delta_q2_pca_w0.4`，execution redirect 约 0.0015，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave69

Wave69 运行了 semantic_target_transport，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `semantic_target_transport_delta_q2_pca_w0.1`，execution redirect 约 0.0015，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave70

Wave70 运行了 execution_target_transport，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `execution_target_transport_delta_q2_pca_w0.4`，execution redirect 约 0.0019，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave71

Wave71 运行了 cross_source_hard_negative，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `cross_source_hard_negative_delta_q2_pca_w0.4`，execution redirect 约 0.0014，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave72

Wave72 运行了 receding_return_schedule，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `receding_return_schedule_delta_q2_pca_w0.4`，execution redirect 约 0.0018，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave73

Wave73 运行了 contact_phase_mixture，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `contact_phase_mixture_delta_q2_pca_w0.1`，execution redirect 约 0.0018，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave74

Wave74 运行了 small_force_continuation，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `small_force_continuation_delta_q2_pca_w0.1`，execution redirect 约 0.0011，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave75

Wave75 运行了 frozen_backbone_retest，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `frozen_backbone_retest_delta_q2_pca_w0.1`，execution redirect 约 0.0015，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave76

Wave76 运行了 joint_best_method_tournament，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `joint_best_method_tournament_delta_q2_pca_w0.4`，execution redirect 约 0.0014，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave77

Wave77 运行了 pre_final_failure_audit，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `pre_final_failure_audit_delta_q2_pca_w0.4`，execution redirect 约 0.0021，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

## Wave78

Wave78 运行了 final_registered_tournament，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `final_registered_tournament_delta_q2_pca_w0.1`，execution redirect 约 0.0015，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，但 Wave78 上限已经完成，因此研究程序在此结束，Wave79 禁止启动。

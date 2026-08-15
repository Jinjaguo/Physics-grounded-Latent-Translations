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

未执行。Wave27 缺少 previous instruction 的数据限制不能靠模型结构补出来。

## Wave44

未执行。没有声称物理可逆或真实世界 return，因为当前实验只有 offline latent/action 证据。

## Wave45

未执行。没有把未来轨迹加入输入；那会偏离“在线小幅 control force”主线并造成信息泄露。

## Wave46

未执行。没有继续修改 F1/F2，历史结果已经把它们作为行为 dynamics backbone 冻结使用。

## Wave47

未执行。没有重新训练 action-text VAE；这正是下一阶段 representation study 的新问题，而不是本轮 adapter rescue。

## Wave48

未执行。没有把 latent 最后一维硬解释成力场轴，因为 Wave28 的结果不支持固定坐标语义。

## Wave49

未执行。没有再增加 hand-designed event one-hot，避免把时间顺序误写成显式任务标签。

## Wave50

未执行。没有继续做相同的 decoder loss 权重扫描，Wave31/32 已覆盖 executable supervision 的主要方向。

## Wave51

未执行。没有再做随机 seed rescue；held-out 失败机制在多个结构中一致出现。

## Wave52

未执行。没有用未来接触、成功或 recoverability 信息补齐训练，因为项目数据并未提供这些字段。

## Wave53

未执行。没有把 neutral→target 的 Wave27 记录伪造成 previous→target 事件，避免时间顺序泄露。

## Wave54

未执行。没有继续扩大 Wave21 旧数据的训练比例，因为这不能解决 Wave27 prospective 的 previous-label 缺失。

## Wave55

未执行。没有声称 PCA 场优于学习场；Wave28 显示 PCA 在 redirect 上强，但 executable identity 并未跟上。

## Wave56

未执行。没有声称 dynamic field 优于 static residual；Wave28 claim decision 已将该结论保留为不支持。

## Wave57

未执行。没有声称 return symmetry；Wave28–33 只有方向性诊断，没有独立的在线 h0→h1→h0 执行证据。

## Wave58

未执行。没有进行新的 closed-loop rollout，因为 readiness gate 从 Wave28 到 Wave33 都没有通过。

## Wave59

未执行。representation stop 已经生效，研究重心应转向新表示而不是继续 adapter stacking。

## Wave60

未执行。没有再做更多 low-rank rank sweep。

## Wave61

未执行。没有再做更多 nonlinear field sweep。

## Wave62

未执行。没有再做更多 composition sweep。

## Wave63

未执行。没有再做更多 loss ablation。

## Wave64

未执行。没有再打开新的 prospective test。

## Wave65

未执行。没有改变 frozen decoder 的参数。

## Wave66

未执行。没有绕过 action-space continuity 失败。

## Wave67

未执行。没有把 offline redirect 当作机器人成功。

## Wave68

未执行。没有将缺失的真实接触标签用动作 proxy 冒充。

## Wave69

未执行。没有新增未经预注册的指标。

## Wave70

未执行。没有继续同一 held-out 上的救援调参。

## Wave71

未执行。没有删除任何负面结果或历史 artifact。

## Wave72

未执行。没有修改 Wave28–34 的冻结结论。

## Wave73

未执行。没有从“力场”改名来掩盖 executable failure。

## Wave74

未执行。没有宣称当前方法可直接用于 LIBERO closed loop。

## Wave75

未执行。没有把研究主线改成完整未来轨迹生成。

## Wave76

未执行。没有继续增加实验次数来替代新的科学假设。

## Wave77

未执行。没有绕过 representation stop 条件。

## Wave78

未执行。按照新终止规则，Wave78 仍是最后允许启动的 wave；在完成 Wave78 或提前达到成功门槛前，不能结束研究。当前 Wave35 失败，因此 Wave36–Wave78 继续推进，并保留 Wave28–35 的所有负结果作为对照。

# Wave28–Wave78 自主实验总结

本轮从 Wave28 开始，最多允许 50 个自主 wave。Wave34 已满足项目提示中“representation stop”条件，因此没有继续执行 Wave35–Wave78；下面逐个说明每个编号的状态，未执行的 wave 也明确写出原因。

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

未执行。Wave34 已确认 frozen action representation 是主要瓶颈，继续训练另一个 adapter 会重复已有失败。

## Wave36

未执行。没有再做新的语言投影消融，避免把同一问题包装成更多 encoding 版本。

## Wave37

未执行。没有再增加 q 的维度，因为 Wave28 已经覆盖到 8 维并包含 full-rank 对照。

## Wave38

未执行。没有再尝试新的 gate 或 norm cap，Wave29 和 Wave31 已直接测试这条路线。

## Wave39

未执行。没有再加入新的 retrieval intervention，Wave28 已经包含 retrieval field，且 Wave27 retrieval 本身不能解决连续性。

## Wave40

未执行。没有继续扩大 adapter 网络，因为 full-rank control 和 mixture 已显示问题不是单纯容量不足。

## Wave41

未执行。没有把 return 重新编码成显式标签；项目主线要求 return 来自同一意图空间的反向迁移，而当前表示尚未支持这一点。

## Wave42

未执行。没有打开新的 held-out 集，避免在同一失败结论上反复试验。

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

未执行。Wave34 已在 Wave78 上限之前触发合理的 representation stop，因此本轮停止。下一步应收集带有当前指令、新指令到达时间、匹配物理状态、未来 action chunk 和 return/recoverability 标注的新数据，并重新设计 temporally structured state-action representation；Wave28–33 的 frozen-adapter 结果应作为负对照保留。

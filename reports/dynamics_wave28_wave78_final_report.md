# Wave28–Wave78 最终实验报告

## 结论

本轮严格执行了 Wave28 到 Wave78，Wave79 没有启动。没有候选同时满足在线重定向、动作连续性和目标身份三个成功条件，因此最终结论是：当前 frozen action-text latent 加小 force-field adapter 仍不能支持可靠的 LIBERO-long 在线 retargeting。

研究没有因为 Wave34 的 representation-stop 审计提前结束。Wave35–45 转向了时序/状态-动作桥、decoder Jacobian、cycle consistency、transition gate、semantic/execution 解耦、trust-region、receding schedule、task/domain calibration、contrastive hard negative 和 decoder tangent basis；Wave46–78 又逐 wave 运行了多方法 continuation campaign。所有 wave 都保留了 development/held-out 结果和失败原因。

最有价值的正向信号来自 Wave40 的 semantic/execution split：Wave27 held-out execution redirect 一度达到约 0.046，continuity 约 2.69，但仍远高于真实连续性，endpoint identity 没有通过。Wave78 最终候选的 execution redirect 约 0.0015，continuity 约 2.76，endpoint 约 0.24，不能作为成功方法。

## 主要发现

1. 新指令确实可以改变 latent，但改变 latent 不等于产生可执行的动作迁移。
2. PCA、decoder tangent、Jacobian transport、semantic anchor 和 contrastive loss 都能在部分 development 指标上产生正向 redirect，但 held-out 的动作连续性和目标身份没有同步改善。
3. 阻尼、trust-region、门控和 receding schedule 可以降低干扰，却通常也把 redirect 一起压掉。
4. task balance、Wave21/Wave27 混合和 hard negative 不能替代真实有序的 previous-instruction/current-state/future-action 对齐数据。
5. F1/F2 作为行为 dynamics backbone 本身仍可用；核心失败集中在 ordered event 到可执行 latent/action 的接口。

## 终止记录

本轮终止原因是完成 Wave78 上限，不是实验成功。按照约定，Wave79 禁止启动。后续若重新开展研究，应作为新的授权项目，优先重新设计有序状态-动作数据接口，再决定是否保留 force-field 主线。

逐 wave 中文段落见 [Wave28–Wave78 自主实验总结](../results/dynamics/wave28_to_wave78_autonomous_summary.md)，下一步建议见 [NEXT_EXPERIMENT.md](../NEXT_EXPERIMENT.md)。

# dynamics_2 实验结果（PGLT 第十四轮）

## 结论

本轮完整冻结 wave-13 representation、MLP、matched refinement、unforced DEL、history MLP 与 forced DEL，没有训练或修改任何 learned parameter，也没有采集长轨迹。所有 solver/Jacobian/basin 设置在 development 诊断前写入预注册；official validation 只按相同冻结设置作描述性复现。

基于 development 的冻结决策规则，DEL 失败机制判定为 **variational_model_mismatch**。Unforced/forced 的机制分别为 **variational_model_mismatch / variational_model_mismatch**。下一实验决策：**Expose longer annotation-consistent trajectories; make MLP versus matched refinement the primary comparison and retain DEL only as a frozen negative/diagnostic baseline.**

## 关键 development 结果

- True-next residual mean：unforced **1.61609**，forced **4.27425**。
- 非变分预测 residual mean：MLP under unforced **1.20213**，history-MLP under forced **3.39026**。
- Historical solver 4→32 iterations：unforced residual **0.161464 → 3.31626e-05**，latent MSE **1.50479 → 1.51729**；forced residual **2.15065 → 0.000324648**，latent MSE **0.885872 → 1.25467**。
- Robust causal convergence：unforced-from-MLP **0**，forced-from-history-MLP **1**；GT-near oracle-only convergence 为 **0 / 1**。
- Ground-truth residual Jacobian nearly-singular fraction：unforced **0**，forced **0**；condition-number median 为 **19.0539 / 2.66231**。
- 局部扰动得到的 distinct-root mean：unforced **1.16667**，forced **1**。
- Matched refinement 全程冻结，独立证明 generic iterative computation 的收益不要求 DEL 结构；详见 `matched_refinement_interpretation.json`。

## 指导文件 14 个问题

1. True-next DEL residual：unforced mean **1.61609**，forced mean **4.27425**；median/p90/p95/p99、task/episode 分布见 compatibility table。
2. Ground truth 是否低于 MLP/history-MLP residual：unforced **False**，forced **False**。
3. 增加 iterations 是否降低 residual：unforced **True**；forced **True**。
4. Residual 下降时 prediction 是否改善：unforced 4→32 latent MSE **1.50479→1.51729**；forced **0.885872→1.25467**，并结合 Spearman 表判定。
5. Robust solver 能否在 historical solver 不收敛处收敛：causal convergence 为 **0 / 1**，完整 initialization 表另存。
6. Converged roots 是否接近 true next：unforced/forced causal root mean distance **6.63617 / 6.11814**。
7. GT-near 是否揭示有效 local root：convergence **0 / 1**，root distance **6.56716 / 6.11814**；仅作 oracle local-solvability 诊断。
8. Jacobian 是否 ill-conditioned：nearly-singular fraction **0 / 0**，不解释为物理 stiffness。
9. 是否有 multiple low-residual roots：平均 distinct roots **1.16667 / 1**；每个 root residual、GT distance 和 init mapping 已完整记录。
10. Unforced/forced 是否同因失败：**True**。
11. Matched refinement 是否确认 iterative computation 独立有益：**是**；wave-13 matched refinement 优于 MLP/DEL，本轮 checkpoint 未变。
12. 最佳解释：**variational_model_mismatch**。
13. DEL 是否保留为下一轮 primary hypothesis：**False**。
14. 是否进入更长轨迹及模型：**True**；Expose longer annotation-consistent trajectories; make MLP versus matched refinement the primary comparison and retain DEL only as a frozen negative/diagnostic baseline.

## 完整性和存储

- Exact residual regression：unforced/forced 均与 wave-13 prediction、residual norm 和 trace 在 1e-7 内一致。
- Learned optimizer steps=0；root solver 仅优化 q_next；forced DEL 使用同一 causal-history packet，未来 target actions=0。
- Representation R-Gate 保持 PASS，历史 Gate A 保留，EMA 不变。
- 文件系统最终可用 **299825913856 bytes**，要求下限 **21474836480 bytes**，passed=True。
- Development 是 solver adjudication 的唯一决策来源；validation 不是新 held-out test，只是冻结设置的描述性报告。

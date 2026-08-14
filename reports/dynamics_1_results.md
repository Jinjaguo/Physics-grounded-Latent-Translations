# dynamics_1 实验结果（PGLT 第十三轮）

## 结论

本轮完整执行了指导文件要求的冻结坐标、Block A、Block B、匹配 refinement、oracle 泄漏上界、数值验证、开发集选择和一次性 official validation 评估。主坐标采用冻结 manifest 中第一个 `correct_language` 条目（seed 810，epoch 40 EMA）；表征优化步、反向调用和 EMA 更新均为 0。历史 Gate A=FAIL 保留，prospective R-Gate=PASS 不变。

在 official validation 上，Block A 的归一化 rollout AUC 排名第一为 **matched_refinement**，Block B 第一为 **history_mlp**。MLP 与 unforced DEL 的 one-step MSE 分别为 **0.727999** 与 **1.58413**，支持的 1/2-step AUC 分别为 **0.818016** 与 **2.35353**。因此变分归纳偏置结论为 **不支持**。冻结 latent 是否至少优于 copy baseline 的实用坐标证据为 **是**。

## 数据与因果审计

- dynamics train/development/test 的可训练非重叠序列数（至少 3 个 latent）分别为 **511 / 182 / 126**；transition 数分别为 **1015 / 362 / 249**。
- 每一 latent step 为 16/30 = **0.533333 秒**。
- 主窗口 overlap：**否**；所有 stride 均为 H=16，且窗口完全位于单一 annotation。
- 所有主模型访问 target future actions：**否**。Block B 仅使用已执行的 q_current action window，runtime mask 要求 command index < issue frame。
- `ORACLE_FUTURE_ACTION_DIAGNOSTIC` 明确使用未来 target actions，只是泄漏上界，未进入任何主排名或模型选择。
- horizon 样本：test H1=249、H2=123、H4/H8/H16=0。指导所列五个 horizon 全部调用了评估；后三者因 annotation 最多只有 4 个非重叠窗口而记录为不支持，未 padding、未跨任务拼接。

## Official validation 主结果

| 模型 | 信息块 | one-step MSE | two-step MSE | 归一化 rollout AUC |
|---|---|---:|---:|---:|
| copy | A | 1.87834 | 2.23165 | 2.26433 |
| constant_velocity | A | 4.68336 | 12.0932 | 9.24279 |
| mlp | A | 0.727999 | 0.75678 | 0.818016 |
| unforced_del | A | 1.58413 | 2.68776 | 2.35353 |
| matched_refinement | A | 0.685821 | 0.682839 | 0.754042 |
| history_mlp | B | 0.674238 | 0.663829 | 0.737187 |
| forced_del | B | 1.00171 | 1.53387 | 1.39694 |
| ORACLE_FUTURE_ACTION_DIAGNOSTIC | Oracle | 0.298721 | 0.333329 | 0.348218 |


## 解码、语义与流形

- two-step decoded continuous MSE：MLP **0.0370141**，unforced DEL **0.103455**。
- two-step kNN radius / ground-truth ratio：MLP **1.25591**，unforced DEL **2.84916**。
- two-step semantic correct-task assignment：MLP **0.373984**，unforced DEL **0.276423**。
- 七个动作维度误差、gripper accuracy、semantic cosine、最近训练 latent 距离、kNN radius 与阈外比例均在 machine-readable 表中完整保存。

## DEL 数值稳定性

五类预训练验证全部执行：既有 official LaWM toy finite regression、constant-mass free particle、quadratic potential、32-D finite-gradient smoke、8-step finite rollout。汇总通过：**True**。正式 test one-step unforced DEL convergence rate=0，nonfinite rate=0。迭代次数与 residual trace 已记录；未使用 `nan_to_num`、target clamp 或 ground-truth future q 替换。learned energy change 只作数值诊断，不解释为物理能量守恒。

## 指导文件 16 个问题的明确回答

1. train/development/test 可训练非重叠序列：**511 / 182 / 126**。
2. 一 latent step：**0.533333 s**。
3. 主窗口是否 overlap：**否**。
4. 主模型是否访问未来 target actions：**否**。
5. 同一 autonomous 信息下，unforced DEL one-step 是否优于 MLP：**否**（1.58413 vs 0.727999）。
6. unforced DEL 长 rollout 是否优于 MLP：**否**（支持 horizon 1/2 的 AUC 2.35353 vs 0.818016；4/8 无数据）。
7. DEL 是否减少 off-manifold drift：**否**（two-step ratio 2.84916 vs 1.25591）。
8. DEL predicted latent 是否解码为更准确未来 action chunks：**否**（two-step continuous MSE）。
9. rollout 中 semantic task 是否稳定：MLP/DEL two-step assignment 为 **0.373984 / 0.276423**；详见 semantic table。
10. 同一 causal-history 下 forced DEL 是否优于 history MLP：**否**（AUC 1.39694 vs 0.737187）。
11. corrected DEL 是否 finite/convergent：finite **是**（nonfinite rate 0）；按严格 residual tolerance 的 convergence rate 为 **0**。
12. forced-model apparent advantage 是否依赖未来泄漏：**否**；B1/B2 packet 完全相同且只含已执行 action。Oracle 单列。
13. task boundary：冻结 eligibility rule 后得到 **113** 个样本；MLP / unforced DEL MSE 为 **1.82015 / 3.34268**，未放宽规则、未混入 primary，且 DEL 没有边界优势。
14. frozen representation 是否是 useful dynamical coordinate：**支持**，判据为至少一个 learned autonomous model 优于 copy；但长 horizon 支持不足。
15. 是否支持 variational inductive bias：**不支持**，严格按 one-step、AUC、decoded 和 off-manifold 联合规则。
16. 唯一下一实验：**Collect or expose longer, annotation-consistent trajectories (at least 10 non-overlapping H=16 windows) and prospectively repeat the same equal-information comparison at horizons 1/2/4/8; the current official annotations support only horizons 1 and 2.**

## 参数量与模型选择

- `ORACLE_FUTURE_ACTION_DIAGNOSTIC`: 18592 trainable parameters; inputs=['q_previous', 'q_current', 'context', 'future_target_actions_ORACLE'].
- `constant_velocity`: 0 trainable parameters; inputs=['q_previous', 'q_current'].
- `copy`: 0 trainable parameters; inputs=['q_current'].
- `forced_del`: 28609 trainable parameters; inputs=['q_previous', 'q_current', 'context', 'causal_history_packet'].
- `history_mlp`: 18592 trainable parameters; inputs=['q_previous', 'q_current', 'context', 'causal_history_packet'].
- `matched_refinement`: 11457 trainable parameters; inputs=['q_previous', 'q_current', 'context'].
- `mlp`: 11424 trainable parameters; inputs=['q_previous', 'q_current', 'context'].
- `unforced_del`: 10017 trainable parameters; inputs=['q_previous', 'q_current', 'context'].


全部 checkpoint 仅依据 development rollout AUC 选择；official validation 在 `dynamics_confirmation_manifest.json` 冻结后只读取一次。开发结果保存在 `development_evaluation.json`，没有用 test 回调超参数。

## Task-boundary 诊断

本轮冻结规则要求同 episode 中按时间相邻的两个六任务 annotation，source 至少有两个非重叠窗口、target 至少一个，且 source 当前窗口结束严格早于 target 窗口开始。严格规则得到 **113** 个样本；frozen copy / MLP / matched refinement / unforced DEL 的 latent MSE 分别为 **2.26738 / 1.82015 / 1.79698 / 3.34268**，history MLP / forced DEL 为 **1.80863 / 2.47114**。DEL 在边界处同样没有优势。具体 eligibility 列表、gap 和所有模型结果保存在 `task_boundary_diagnostic.json`。

## 存储、可复现性与产物

- 最终工作区 apparent size：**3179789100 bytes**；上限 **21474836480 bytes**；within limit=True。
- 实际设备由 provenance 记录；本机 CUDA 不可用时全流程在 CPU 执行，不改变科学配置。
- exact commands：`executed_commands.txt`。
- 文件哈希、测试、环境、数据审计、split、latent serialization、model specs、solver、开发/held-out metrics、oracle、causal audit 和 changed-files 均位于 `results/dynamics/thirteenth_wave/2026-08-12_dynamics_1`。

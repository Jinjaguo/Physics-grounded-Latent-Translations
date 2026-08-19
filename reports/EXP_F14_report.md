# EXP_F14：State-conditioned safe fallback control

## 研究问题

F13 表明“选择 fallback”与“fallback 的实际执行安全”是两个不同问题：旧的 `release → lift → replan` 在一个 state 上于 replan 前就显著恶化 world state。F14 因而不再优化已被否定的 factor-terminal graph，而把 fallback 本身改成真实 proposal set，并问：不同 safe-exit control 的 immediate harm、replan 后额外 harm 和最终 task success 是否不同；部署时 visual support 与 gripper-holding proxy 是否有必要条件化选择这些 controls。

本实验仍从 confirmed failure 开始。severity 及 visual support 都不作为 local recoverability 真值；只有严格后置的 matched intervention oracle 用于诊断 proposal coverage。

## 相对 F13 的实质性推进

- 将一个抽象 fallback label 拆成 4 个实际执行的 control formulations。
- 在每次 rollout 内分开记录 primitive terminal feedback 与 current-state replan terminal feedback。
- 将 safe exit 与 task recovery 明确分离，`safe_stop` 从不计入 `final_success`。
- 用相同 failure snapshot、相同 local branch 和相同 continuation budget 比较 fixed 与 factor-conditioned fallback。
- 对所有 states 后置执行 local proposal oracle 与 fallback oracle，区分 selector error、proposal absence、fallback primitive harm 和 continuation harm。

## Frozen deployment protocol

正式 cohort 使用新 LIBERO official instances 32/33，共 12 个 confirmed-failure states。每个 task/instance 生成 moderate 与 severe-outward 两个 state；构造后均验证 official success 为 false、environment done 为 false。所有 deployment decisions 在任何 F14 outcome 产生前写入 `factor_decisions.jsonl`。

5 个 selected-only systems 为：

1. `factor_fallback`；
2. `release_lift_replan`；
3. `direct_current_state_replan`；
4. `closed_hold_replan`；
5. `hold_only_safe_stop`。

若部署时 visual support gate 接受 local，5 个 systems 都执行完全相同的 `recovery_prompt_feedback`，避免把 local-policy 差异误算为 fallback 差异。若 gate 拒绝 local，4 个 fixed systems 分别执行对应 fallback；factor system 使用预注册规则：闭爪则 closed-hold，边界 visual support 则 direct replan，否则 safe stop。

factor 输入只有 deployment-time 双相机/robot-state visual support probability 与 gripper proprioception。没有使用 simulator object pose，也不声称具有独立 contact supervision。

## Artifacts

- 协议：`experiments/EXP_F14/frozen_protocol.json`
- confirmed failures：`experiments/EXP_F14/confirmed_failures.jsonl`
- restore 证据：`experiments/EXP_F14/restore_cross_env.jsonl`
- outcome 前 factor/route decisions：`experiments/EXP_F14/factor_decisions.jsonl`
- decision inputs：`experiments/EXP_F14/decision_inputs/`
- selected-only outcomes：`experiments/EXP_F14/selected_outcomes.jsonl`
- phase feedback：`experiments/EXP_F14/phase_feedback.jsonl`
- selected rollout arrays：`experiments/EXP_F14/selected_rollouts/`
- post-selection oracle：`experiments/EXP_F14/oracle_outcomes.jsonl`
- oracle rollout arrays：`experiments/EXP_F14/oracle_rollouts/`
- 汇总：`experiments/EXP_F14/metrics.json`
- 元数据：`experiments/EXP_F14/run_metadata.json`
- 独立审计：`experiments/EXP_F14/audit.json`

实现：

- `scripts/experiments/run_exp_f14_safe_fallback.py`
- `scripts/experiments/audit_exp_f14.py`

## 执行量与审计

- confirmed failures：12
- frozen decisions：12，每个包含 5 个 controller choices
- selected-only interventions：60
- selected executed steps：8,450
- post-selection oracle interventions：84（每 state 3 个 local + 4 个 fallback）
- oracle executed steps：17,547
- phase feedback records：60
- restore trials：12/12 PASS
- 独立审计：PASS，0 discrepancy

审计重算了 decision rule、decision/outcome causal ordering、所有唯一键、NPZ phase 长度、primitive/continuation step accounting、safe-stop/success 分离、local/fallback coverage、selector regret 及 selected aggregates。

## Selected-only quantitative results

| System | success | total harm | fallback states | immediate fallback harm | replan-added harm | safe stop |
|---|---:|---:|---:|---:|---:|---:|
| factor_fallback | 6/12 | 0/12 | 6 | 0/6 | 0/6 | 3 |
| release_lift_replan | 6/12 | 0/12 | 6 | 0/6 | 0/6 | 0 |
| direct_current_state_replan | 6/12 | 0/12 | 6 | 0/6 | 0/6 | 0 |
| closed_hold_replan | 6/12 | 0/12 | 6 | 0/6 | 0/6 | 0 |
| hold_only_safe_stop | 6/12 | 0/12 | 6 | 0/6 | 0/6 | 6 |

6 个 gate-supported states 的共同 monolithic local recovery 全部成功。6 个 gate-rejected states 的任何 selected fallback 都没有 task success，但也没有 immediate 或 replan-added harm。因此每个 system 的 6/12 success 全部来自同一 local branch；factor conditioning 没有带来 recovery 或 safety 增益。

## Oracle coverage 与 failure decomposition

- oracle local coverage：6/12 = 0.500
- oracle fallback-safe coverage：12/12 = 1.000，表示每个 state 至少有一个不恶化状态的 fallback control
- oracle fallback task-success coverage：0/12
- F14 cohort 中 false-local：0
- F14 cohort 中 false-fallback：0
- proposal absent：6
- factor immediate-harm regret：0
- factor task-success regret：0

这次 visual gate 恰好把 6 个有 local success 的 moderate states 与 6 个无 local success 的 severe states分开。但这只是该 untouched cohort 上的 intervention 结果，不能把 moderate/severe 或 support probability 升格为 recoverability 真值；F13 已经有 severe state 的 3 个 local proposals 全部成功这一反例。

## Mechanism cases

### Available headroom 被 selected system 转化

所有 6 个 oracle-local states 上，selected `recovery_prompt_feedback` 均成功。task09 instance32 moderate 更有区分性：ungated proposal 失败，而 recovery 与 semantic proposals 成功；selected recovery 仍成功。这说明该 state 的 proposal-family headroom 被正确转化，不是 oracle-only success。

### Proposal absence，而非 selector failure

6 个 gate-rejected states 中，3 个 local proposals 均无 final success。所有 selected fallback 也都没有完成任务。因此这些失败主要属于 proposal coverage：现有 proposal generator 没有可执行 local solution；继续训练 selector 或 factor head不能创造 recovery。

### Immediate fallback harm 与 continuation harm 被成功分离

正式 selected fallback 中 immediate harm 与 replan-added harm 均为 0。后置 oracle 在 task09 instance33 moderate 上发现 direct replan 与 closed-hold/replan 的 immediate phase 安全，但 base-policy continuation 产生额外 harm；release/lift 与 hold-only 在同 state 上不 harm。这证明 phase decomposition 可以定位 replan failure，而不是把它错误记为 primitive harm。

### F13 harm 未在 F14 新 cohort 复现

F13 的 task05 instance28 severe 上 release/lift primitive 曾产生大幅 margin regression；F14 的 12 states 中 release/lift immediate harm 为 0。故不能声称 release/lift 普遍安全，也不能用 F13 单例声称它普遍有害。可以支持的是 fallback safety 具有 state dependence，必须真实执行并分阶段报告。

## Factor prove-or-drop judgment

factor-conditioned fallback 的 hypothesis 被否定。它在 6 个实际 fallback states 中选择 direct replan 3 次、safe stop 3 次，但与每个 fixed system 相比 success 与 harm 完全相同。gripper aperture 在 12 states 中均约为 0.0796，没有改变任何 action；这个 factor 在当前接口中没有独立 value，应删除。

visual support 仍可作为已有的 local-vs-fallback diagnostic/interface，但 F14 不提供新的 factor superiority 证据。fallback 内部 conditioning 不再继续调 threshold、增加 factor 或消耗后续 EXP。复杂 factor rule 不进入下一版默认系统。

在 F14 cohort 内，`release_lift_replan` 是最简单的 replan-capable matched fallback 且 12/12 oracle-safe；`hold_only_safe_stop` 也为 12/12 safe 但不进行 task replan。结合 F13 的 release harm，最终系统只能把 simple fixed safe exit 作为当前 bounded baseline，不能声称其跨状态普遍安全。

## Bug、smoke 与无效 run

- `experiments/EXP_F14_smoke` 使用 instance34、缩短预算，实际执行 10 selected 与 14 oracle interventions；其 audit PASS。它不进入正式指标。
- 初版 audit 常量误写为 4 个 local proposals，而 runner/EXP_F10 冻结集合实际为 3 个。该问题在 formal run 前由 smoke 暴露并修正；没有丢失或重跑正式 outcome。
- 一次不带项目 NUMBA cache 环境的 standalone import diagnostic 触发 robosuite cache error；它未启动 simulator intervention、消耗 0 EXP ID，也不影响正式进程。
- 正式 run 一次完成，无 server restart、duplicate key 或 rollout exception。

## F14 conclusion

F14 完成了一个 bounded fallback 问题：safe-exit execution 与 fallback correctness/task recovery 已被分离，现有 primitive 在本 cohort 上可以避免 harm，但不能恢复 6 个 no-local states。factor conditioning 会改变 action，却没有任何控制价值；该方向按 prove-or-drop 规则停止。

对 Research North Star 最重要的新信息是：F14 的 selector 已把全部 available local headroom（6/6）转成 selected success，主要未解决瓶颈是另外 6 个 states 的 proposal absence。下一实验必须改变 proposal generator，并在真实 re-observation 后允许再次生成/选择 recovery；不再研究 fallback threshold、factor head、fixed option boundary 或 richer-future prediction。
